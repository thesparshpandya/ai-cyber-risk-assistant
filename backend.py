"""
backend.py — AI Cyber Risk Assistant for TawasolPay.

Responsibilities
----------------
1.  Schema-validated ingestion of the structured data pack (CSV + CISA KEV JSON).
2.  Deterministic CVE normalisation with a separate join key and display form.
3.  Relational joins: Asset -> Vulnerability -> Threat Intel -> Business Service -> CISA KEV.
4.  Deterministic parsing of the MDR advisory (`synthetic_threat_report.md`).
5.  A 100% deterministic composite risk engine (no LLM involvement, ever).
6.  A vector RAG engine over the NIST SP 800-53 Rev. 5 control catalogue (ChromaDB +
    sentence-transformers) with a hard relevance threshold and no hallucinated controls.
7.  Optional Groq summarisation of *retrieved* control prose, with a safe raw-text fallback.

Separation of concerns
----------------------
Scoring and ranking are pure Python arithmetic over explicit multipliers. The LLM is
only ever handed control prose that has already been retrieved, and is only ever asked
to rewrite it. It cannot see, compute, alter, or reorder any score.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

LOGGER = logging.getLogger("tawasolpay.backend")
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(os.environ.get("TP_LOG_LEVEL", "INFO"))


# ---------------------------------------------------------------------------
# Constants and configuration
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path("data")

FILE_ASSETS = "assets.csv"
FILE_VULNS = "vulnerabilities.csv"
FILE_INTEL = "threat_intelligence.csv"
FILE_SERVICES = "business_services.csv"
FILE_REMEDIATION = "remediation_guidance.csv"
FILE_THREAT_REPORT = "synthetic_threat_report.md"
FILE_KEV = "known_exploited_vulnerabilities.json"
FILE_NIST = "NIST_SP-800-53_rev5_catalog_load.csv"

#: Risk multipliers. Exposed as module constants so the UI can render the same
#: numbers it scored with, rather than re-declaring them.
MULT_INTERNET_FACING = 2.5
MULT_INTERNET_INTERNAL = 1.0
MULT_KEV_MATCH = 2.0
MULT_KEV_NONE = 1.0
MULT_CAMPAIGN_MATCH = 3.0
MULT_CAMPAIGN_NONE = 1.0
MULT_EDR_MISSING = 1.5
MULT_EDR_PRESENT = 1.0

CRITICALITY_MULTIPLIERS: Dict[str, float] = {
    "CRITICAL": 2.5,
    "HIGH": 1.8,
    "MEDIUM": 1.2,
    "LOW": 1.0,
}
#: Rank order used to reconcile asset criticality against business-service revenue
#: impact. Higher wins, so a Medium-criticality asset carrying a Critical revenue
#: service is scored as Critical.
CRITICALITY_RANK: Dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
DEFAULT_CRITICALITY = "MEDIUM"

#: Composite-score severity bands. The theoretical maximum composite score is
#: 10.0 * 2.5 * 2.0 * 3.0 * 2.5 * 1.5 = 562.5, so bands are set against that range
#: rather than against the 0-10 CVSS range.
SEVERITY_BANDS: Sequence[Tuple[float, str]] = (
    (150.0, "CRITICAL"),
    (60.0, "HIGH"),
    (20.0, "MEDIUM"),
    (0.0, "LOW"),
)

#: Vulnerability statuses treated as no longer live. Excluded from ranking so a
#: remediated finding cannot occupy a Top 5 slot.
CLOSED_STATUSES = {"CLOSED", "REMEDIATED", "RESOLVED", "FIXED", "MITIGATED", "FALSE POSITIVE"}

#: Cosine-distance ceiling for NIST retrieval, tunable without a code change.
#: all-MiniLM-L6-v2 on asymmetric (long query vs control prose) pairs typically
#: scores good matches at 0.35-0.65, so 0.80 admits genuine hits while still
#: rejecting unrelated controls. Lower it to tighten precision.
NIST_DISTANCE_THRESHOLD = float(os.environ.get("TP_NIST_DISTANCE_THRESHOLD", "0.80"))
NIST_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NIST_COLLECTION_PREFIX = "nist_sp800_53_r5"
NIST_CHROMA_PATH = Path(os.environ.get("TP_CHROMA_PATH", ".chroma_nist"))

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TIMEOUT_SECONDS = 30.0

NIST_FAMILIES: Dict[str, str] = {
    "AC": "Access Control",
    "AT": "Awareness and Training",
    "AU": "Audit and Accountability",
    "CA": "Assessment, Authorization, and Monitoring",
    "CM": "Configuration Management",
    "CP": "Contingency Planning",
    "IA": "Identification and Authentication",
    "IR": "Incident Response",
    "MA": "Maintenance",
    "MP": "Media Protection",
    "PE": "Physical and Environmental Protection",
    "PL": "Planning",
    "PM": "Program Management",
    "PS": "Personnel Security",
    "PT": "PII Processing and Transparency",
    "RA": "Risk Assessment",
    "SA": "System and Services Acquisition",
    "SC": "System and Communications Protection",
    "SI": "System and Information Integrity",
    "SR": "Supply Chain Risk Management",
}


class SchemaValidationError(RuntimeError):
    """Raised when a required file or required column cannot be mapped."""


# ---------------------------------------------------------------------------
# Schema mapping layer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldSpec:
    """A single canonical field and the source column names it may arrive under."""

    canonical: str
    aliases: Tuple[str, ...]
    required: bool = True


@dataclass(frozen=True)
class TableSpec:
    """Schema contract for one CSV in the data pack."""

    name: str
    filename: str
    fields: Tuple[FieldSpec, ...]


def _spec(canonical: str, *aliases: str, required: bool = True) -> FieldSpec:
    """Build a FieldSpec whose canonical name is always an accepted alias."""
    all_aliases = (canonical,) + tuple(a for a in aliases if a != canonical)
    return FieldSpec(canonical=canonical, aliases=all_aliases, required=required)


#: Authoritative schema, transcribed from the environment's own column output.
TABLE_SPECS: Tuple[TableSpec, ...] = (
    TableSpec(
        name="assets",
        filename=FILE_ASSETS,
        fields=(
            _spec("asset_id"),
            _spec("asset_name"),
            _spec("asset_type"),
            _spec("environment", required=False),
            _spec("owner_team", required=False),
            _spec("business_service"),
            _spec("internet_exposed"),
            _spec("criticality"),
            _spec("data_classification", required=False),
            _spec("edr_installed"),
            _spec("last_seen_days", required=False),
            _spec("location", required=False),
            _spec("vendor_product", required=False),
        ),
    ),
    TableSpec(
        name="vulnerabilities",
        filename=FILE_VULNS,
        fields=(
            _spec("vuln_id"),
            _spec("asset_id"),
            _spec("vulnerability_name"),
            _spec("cve"),
            _spec("severity", required=False),
            _spec("cvss"),
            _spec("exploit_available", required=False),
            _spec("patch_available", required=False),
            _spec("days_open", required=False),
            _spec("asset_exposure", required=False),
            _spec("auth_required", required=False),
            _spec("status", required=False),
            _spec("affected_component", required=False),
        ),
    ),
    TableSpec(
        name="threat_intelligence",
        filename=FILE_INTEL,
        fields=(
            _spec("intel_id"),
            _spec("threat_actor"),
            _spec("campaign_name"),
            _spec("target_sector", required=False),
            _spec("target_region", required=False),
            _spec("matched_cve_or_control"),
            _spec("exploit_maturity", required=False),
            _spec("active_last_seen", required=False),
            _spec("ransomware_association"),
            _spec("confidence", required=False),
            _spec("summary", required=False),
        ),
    ),
    TableSpec(
        name="business_services",
        filename=FILE_SERVICES,
        fields=(
            _spec("business_service"),
            _spec("business_owner", required=False),
            _spec("business_impact", required=False),
            _spec("customer_facing", required=False),
            _spec("compliance_scope", required=False),
            _spec("revenue_impact", required=False),
            _spec("rto_hours", required=False),
            _spec("depends_on", required=False),
            _spec("risk_appetite", required=False),
        ),
    ),
    TableSpec(
        name="remediation_guidance",
        filename=FILE_REMEDIATION,
        fields=(
            _spec("finding_type"),
            _spec("recommended_action"),
            _spec("priority_hint", required=False),
            _spec("validation_evidence", required=False),
        ),
    ),
    TableSpec(
        name="nist_catalog",
        filename=FILE_NIST,
        fields=(
            _spec("identifier", "control_identifier", "control id"),
            _spec("name", "title", "control_name"),
            _spec("control_text", "control text", "text"),
            _spec("discussion", "supplemental_guidance", required=False),
            _spec("related", "related_controls", required=False),
        ),
    ),
)

TABLE_SPECS_BY_NAME: Dict[str, TableSpec] = {spec.name: spec for spec in TABLE_SPECS}


def _normalise_column(column: Any) -> str:
    """Reduce a raw column label to a comparison key."""
    text = str(column).strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return re.sub(r"_+", "_", text).strip("_")


def map_schema(frame: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    """
    Rename `frame` columns to canonical names according to `spec`.

    Raises
    ------
    SchemaValidationError
        If any required canonical field has no matching source column. The message
        names the file, the missing field, its accepted aliases, and the columns
        that were actually present.
    """
    lookup: Dict[str, str] = {}
    for column in frame.columns:
        lookup.setdefault(_normalise_column(column), column)

    rename: Dict[str, str] = {}
    missing: List[FieldSpec] = []

    for field_spec in spec.fields:
        source: Optional[str] = None
        for alias in field_spec.aliases:
            candidate = lookup.get(_normalise_column(alias))
            if candidate is not None:
                source = candidate
                break
        if source is None:
            if field_spec.required:
                missing.append(field_spec)
            continue
        rename[source] = field_spec.canonical

    if missing:
        present = ", ".join(str(c) for c in frame.columns)
        details = "\n".join(
            f"  - required field '{f.canonical}' (accepted as: {', '.join(f.aliases)})"
            for f in missing
        )
        raise SchemaValidationError(
            f"Schema validation failed for '{spec.filename}'.\n"
            f"{details}\n"
            f"  Columns found in file: {present}\n"
            f"  Fix the source file's headers or extend TABLE_SPECS in backend.py."
        )

    mapped = frame.rename(columns=rename)
    # Keep canonical columns first, preserve any extras for diagnostics.
    canonical_order = [f.canonical for f in spec.fields if f.canonical in mapped.columns]
    extras = [c for c in mapped.columns if c not in canonical_order]
    mapped = mapped[canonical_order + extras]

    # Guarantee optional-but-absent fields exist so downstream code never KeyErrors.
    for field_spec in spec.fields:
        if field_spec.canonical not in mapped.columns:
            mapped[field_spec.canonical] = pd.NA
    return mapped


# ---------------------------------------------------------------------------
# Small value helpers
# ---------------------------------------------------------------------------

def _text(value: Any, default: str = "") -> str:
    """Coerce any cell to a trimmed string, treating NaN/None as `default`."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else default


def _is_yes(value: Any) -> bool:
    """Truthiness for the data pack's Yes/No style columns."""
    return _text(value).upper() in {"YES", "Y", "TRUE", "1", "ENABLED", "INSTALLED", "ACTIVE"}


def _is_no(value: Any) -> bool:
    """Explicit negative, distinguishable from 'missing'."""
    return _text(value).upper() in {"NO", "N", "FALSE", "0", "DISABLED", "NOT INSTALLED", "NONE"}


def _to_float(value: Any, default: float = 0.0) -> float:
    """Parse a numeric cell, tolerating stray characters and blanks."""
    text = _text(value)
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        return float(match.group(0)) if match else default


def _to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Parse an integer cell, returning `default` when unparseable."""
    text = _text(value)
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def _canon_level(value: Any, default: str = DEFAULT_CRITICALITY) -> str:
    """Map a criticality-like label onto CRITICAL/HIGH/MEDIUM/LOW."""
    text = _text(value).upper()
    if not text:
        return default
    if "CRIT" in text or text in {"P0", "SEV0", "SEV1", "VERY HIGH"}:
        return "CRITICAL"
    if "HIGH" in text or text in {"P1", "SEV2"}:
        return "HIGH"
    if "MED" in text or "MODERATE" in text or text in {"P2", "SEV3"}:
        return "MEDIUM"
    if "LOW" in text or "MINOR" in text or "INFO" in text or text in {"P3", "P4"}:
        return "LOW"
    return default


def _tokenise(text: str) -> set:
    """Lowercase alphanumeric tokens, stopwords removed, for deterministic matching."""
    stop = {
        "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "with", "is",
        "are", "by", "at", "from", "via", "be", "as", "it", "its", "this", "that",
    }
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    return {t for t in tokens if t not in stop and len(t) > 2}


# ---------------------------------------------------------------------------
# CVE normalisation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalisedCve:
    """
    A CVE-like identifier split into a display form and a join key.

    `match_key` strips leading zeros from the sequence component so that
    CVE-2024-0021 and CVE-2024-21 collapse onto the same key. `display` keeps the
    canonical zero-padded form for the UI.
    """

    raw: str
    display: str
    match_key: str
    is_valid: bool
    kind: str          # "cve" | "synthetic" | "control" | "empty" | "malformed"
    reason: str = ""


_RE_CVE = re.compile(r"^CVE[-_ ](\d{4})[-_ ](\d{1,10})$")
_RE_CVE_SYN = re.compile(r"^CVE[-_ ]SYN[-_ ](\d{4})[-_ ](\d{1,10})$")
_RE_CONTROL_LIKE = re.compile(r"^[A-Z0-9][A-Z0-9\-_]{2,30}$")


def normalise_cve(raw: Any, allow_control_identifiers: bool = True) -> NormalisedCve:
    """
    Normalise a CVE/control identifier deterministically.

    Handles surrounding whitespace, lowercase input, unicode dashes, underscores,
    and space separators. Anything unparseable is flagged malformed rather than
    silently dropped.

    Parameters
    ----------
    allow_control_identifiers
        True for `threat_intelligence.matched_cve_or_control`, which legitimately
        carries non-CVE identifiers such as `CICD-SYN-001`. Set False for
        `vulnerabilities.cve`, where anything that is not a CVE is a data defect
        and must be reported rather than quietly accepted as a control.
    """
    original = _text(raw)
    if not original:
        return NormalisedCve(original, "", "", False, "empty", "blank identifier")

    text = original.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = re.sub(r"\s+", " ", text).strip().upper()

    match = _RE_CVE_SYN.match(text)
    if match:
        year, seq = match.group(1), match.group(2)
        return NormalisedCve(
            raw=original,
            display=f"CVE-SYN-{year}-{seq.zfill(4)}",
            match_key=f"CVE-SYN-{year}-{int(seq)}",
            is_valid=True,
            kind="synthetic",
        )

    match = _RE_CVE.match(text)
    if match:
        year, seq = match.group(1), match.group(2)
        return NormalisedCve(
            raw=original,
            display=f"CVE-{year}-{seq.zfill(4)}",
            match_key=f"CVE-{year}-{int(seq)}",
            is_valid=True,
            kind="cve",
        )

    if allow_control_identifiers:
        # Only treat the value as a control identifier when it is already clean.
        # Stripping punctuation first would launder 'NOT-A-CVE!!' into a valid key.
        if _RE_CONTROL_LIKE.match(text) and not re.search(r"[^A-Z0-9\-_]", text):
            collapsed = text.replace("_", "-")
            return NormalisedCve(
                raw=original,
                display=collapsed,
                match_key=collapsed,
                is_valid=True,
                kind="control",
            )
        reason = "does not match CVE-YYYY-NNNN, CVE-SYN-YYYY-NNNN, or a clean control identifier"
    else:
        reason = "does not match CVE-YYYY-NNNN or CVE-SYN-YYYY-NNNN"

    return NormalisedCve(
        raw=original,
        display=original,
        match_key="",
        is_valid=False,
        kind="malformed",
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Threat report parsing
# ---------------------------------------------------------------------------

@dataclass
class ReportCampaign:
    """One campaign section parsed out of the MDR advisory."""

    ordinal: int
    threat_actor: str
    campaign_name: str
    target_profile: str
    ransomware: bool
    confidence: str
    cve_display: List[str] = field(default_factory=list)
    cve_keys: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


_RE_SECTION = re.compile(r"^###\s*(\d+)\.\s*(.+?)$", re.MULTILINE)
_RE_FIELD = re.compile(r"\*\*(?P<label>[^*:]+):?\*\*\s*(?P<value>[^\n]*)")
_RE_ANY_CVE = re.compile(r"\bCVE(?:-SYN)?-\d{4}-\d{1,10}\b", re.IGNORECASE)
_RE_ANY_CONTROLISH = re.compile(r"\b[A-Z]{3,10}-SYN-\d{1,5}\b")


def parse_threat_report(path: Path) -> Tuple[List[ReportCampaign], str, List[str]]:
    """
    Parse the MDR advisory into campaign records.

    Returns
    -------
    (campaigns, raw_text, warnings)
        Parsing is regex-driven and deterministic. A missing or unparseable report
        degrades to an empty campaign list plus a warning, never an exception.
    """
    warnings: List[str] = []
    if not path.exists():
        warnings.append(f"Threat report not found at '{path}'; regional campaign multiplier disabled.")
        return [], "", warnings

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        warnings.append(f"Could not read threat report '{path}': {exc}")
        return [], "", warnings

    matches = list(_RE_SECTION.finditer(raw))
    if not matches:
        warnings.append("No '### N. Actor — Campaign' sections found in the threat report.")
        return [], raw, warnings

    campaigns: List[ReportCampaign] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        body = raw[start:end]
        heading = match.group(2).strip()

        actor, campaign_name = heading, ""
        for dash in ("—", "–", " - "):
            if dash in heading:
                left, right = heading.split(dash, 1)
                actor = left.strip()
                campaign_name = right.strip().strip('"').strip("“”").strip()
                break

        fields = {
            _normalise_column(m.group("label")): m.group("value").strip()
            for m in _RE_FIELD.finditer(body)
        }

        ransomware_raw = fields.get("ransomware", "")
        ransomware = ransomware_raw.upper().startswith("YES")

        seen: Dict[str, str] = {}
        for token in _RE_ANY_CVE.findall(body) + _RE_ANY_CONTROLISH.findall(body):
            normalised = normalise_cve(token)
            if normalised.is_valid and normalised.match_key not in seen:
                seen[normalised.match_key] = normalised.display

        keywords = sorted({actor.upper()} | ({campaign_name.upper()} if campaign_name else set()))

        campaigns.append(
            ReportCampaign(
                ordinal=int(match.group(1)),
                threat_actor=actor,
                campaign_name=campaign_name,
                target_profile=fields.get("target_profile", ""),
                ransomware=ransomware,
                confidence=fields.get("confidence", ""),
                cve_display=[seen[k] for k in sorted(seen)],
                cve_keys=sorted(seen),
                keywords=keywords,
            )
        )

    campaigns.sort(key=lambda c: c.ordinal)
    if not any(c.cve_keys for c in campaigns):
        warnings.append("Threat report parsed but no CVE identifiers were extracted from it.")
    return campaigns, raw, warnings


# ---------------------------------------------------------------------------
# Data pack loading
# ---------------------------------------------------------------------------

@dataclass
class DataQuality:
    """Ingestion and join hygiene counters surfaced in the diagnostics panel."""

    missing_files: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    row_counts: Dict[str, int] = field(default_factory=dict)
    total_assets: int = 0
    total_vulnerabilities: int = 0
    vulnerabilities_processed: int = 0
    vulnerabilities_skipped_closed: int = 0
    vulnerabilities_skipped_unmapped_asset: int = 0
    vulnerabilities_skipped_bad_cve: int = 0
    threat_intel_records: int = 0
    threat_intel_matched: int = 0
    threat_intel_unmatched: int = 0
    kev_records: int = 0
    kev_matches: int = 0
    campaign_matches: int = 0
    assets_without_vulnerabilities: List[str] = field(default_factory=list)
    assets_without_owner: List[str] = field(default_factory=list)
    unmapped_asset_ids: List[str] = field(default_factory=list)
    unmapped_business_services: List[str] = field(default_factory=list)
    malformed_cves: List[Dict[str, str]] = field(default_factory=list)
    stale_assets: List[Dict[str, Any]] = field(default_factory=list)

    def as_display_dict(self) -> Dict[str, Any]:
        """Flatten the headline counters for table rendering."""
        return {
            "Total assets": self.total_assets,
            "Vulnerability records": self.total_vulnerabilities,
            "Vulnerabilities scored": self.vulnerabilities_processed,
            "Skipped — closed/remediated": self.vulnerabilities_skipped_closed,
            "Skipped — unmapped asset_id": self.vulnerabilities_skipped_unmapped_asset,
            "Skipped — malformed CVE": self.vulnerabilities_skipped_bad_cve,
            "Threat intel records": self.threat_intel_records,
            "Threat intel matched to environment": self.threat_intel_matched,
            "Threat intel industry noise": self.threat_intel_unmatched,
            "CISA KEV catalogue entries": self.kev_records,
            "Scored risks with a KEV match": self.kev_matches,
            "Scored risks with a regional campaign match": self.campaign_matches,
            "Assets with no open vulnerabilities": len(self.assets_without_vulnerabilities),
            "Assets with no assigned owner": len(self.assets_without_owner),
            "Business services referenced but not defined": len(self.unmapped_business_services),
            "Malformed CVE identifiers": len(self.malformed_cves),
            "Stale assets (last seen > 30 days)": len(self.stale_assets),
        }


@dataclass
class DataPack:
    """The validated, joined, in-memory data pack."""

    data_dir: Path
    assets: pd.DataFrame
    vulnerabilities: pd.DataFrame
    threat_intelligence: pd.DataFrame
    business_services: pd.DataFrame
    remediation_guidance: pd.DataFrame
    nist_catalog: pd.DataFrame
    kev: pd.DataFrame
    campaigns: List[ReportCampaign]
    threat_report_text: str
    quality: DataQuality


def _read_csv(path: Path, spec: TableSpec, quality: DataQuality) -> pd.DataFrame:
    """Read and schema-map one CSV, raising SchemaValidationError on hard failures."""
    if not path.exists():
        quality.missing_files.append(str(path))
        raise SchemaValidationError(
            f"Required data file '{path}' is missing. "
            f"Place the data pack under '{path.parent}/' and retry."
        )
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=True, skip_blank_lines=True)
    except TypeError:
        # pandas <2.0 does not accept skip_blank_lines on read_csv for all engines.
        frame = pd.read_csv(path, dtype=str, keep_default_na=True)
    except Exception as exc:  # malformed CSV, encoding failure, empty file
        raise SchemaValidationError(f"Could not parse '{path}': {exc}") from exc

    frame = frame.dropna(how="all")
    frame.columns = [str(c) for c in frame.columns]
    mapped = map_schema(frame, spec)
    quality.row_counts[spec.name] = int(len(mapped))
    if mapped.empty:
        quality.warnings.append(f"'{path.name}' contains no data rows.")
    return mapped


def _load_kev(path: Path, quality: DataQuality) -> pd.DataFrame:
    """
    Load the CISA KEV catalogue into a frame keyed by CVE match key.

    Accepts either the official `{"vulnerabilities": [...]}` envelope or a bare
    JSON array. A missing or unreadable file degrades to an empty frame with a
    warning — the KEV multiplier then simply never fires from this source.
    """
    columns = [
        "cve_key", "cve_display", "vendorProject", "product", "vulnerabilityName",
        "dateAdded", "shortDescription", "requiredAction", "dueDate",
        "knownRansomwareCampaignUse", "notes", "cwes",
    ]
    empty = pd.DataFrame(columns=columns)

    if not path.exists():
        quality.missing_files.append(str(path))
        quality.warnings.append(
            f"CISA KEV file '{path}' not found; no risk will be flagged as KEV-listed. "
            "Download it from https://github.com/cisagov/kev-data"
        )
        return empty

    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError) as exc:
        quality.warnings.append(f"Could not parse CISA KEV JSON '{path}': {exc}")
        return empty

    if isinstance(payload, dict):
        records = payload.get("vulnerabilities") or payload.get("data") or []
    elif isinstance(payload, list):
        records = payload
    else:
        quality.warnings.append(f"Unexpected KEV JSON root type: {type(payload).__name__}")
        return empty

    if not isinstance(records, list) or not records:
        quality.warnings.append(f"CISA KEV file '{path}' contained no vulnerability records.")
        return empty

    rows: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        normalised = normalise_cve(record.get("cveID"))
        if not normalised.is_valid:
            continue
        row: Dict[str, Any] = {"cve_key": normalised.match_key, "cve_display": normalised.display}
        for column in columns[2:]:
            value = record.get(column)
            row[column] = ", ".join(str(v) for v in value) if isinstance(value, list) else _text(value)
        rows.append(row)

    frame = pd.DataFrame(rows, columns=columns).drop_duplicates(subset="cve_key", keep="first")
    quality.kev_records = int(len(frame))
    quality.row_counts["kev"] = int(len(frame))
    return frame


def load_data_pack(data_dir: Path | str = DEFAULT_DATA_DIR) -> DataPack:
    """
    Load, schema-validate and index the full data pack.

    Raises
    ------
    SchemaValidationError
        For a missing required file or an unmappable required column. Optional
        sources (KEV JSON, threat report) degrade with warnings instead.
    """
    directory = Path(data_dir)
    quality = DataQuality()

    if not directory.exists():
        raise SchemaValidationError(
            f"Data directory '{directory}' does not exist. Create it and place the data pack inside."
        )

    assets = _read_csv(directory / FILE_ASSETS, TABLE_SPECS_BY_NAME["assets"], quality)
    vulns = _read_csv(directory / FILE_VULNS, TABLE_SPECS_BY_NAME["vulnerabilities"], quality)
    intel = _read_csv(directory / FILE_INTEL, TABLE_SPECS_BY_NAME["threat_intelligence"], quality)
    services = _read_csv(directory / FILE_SERVICES, TABLE_SPECS_BY_NAME["business_services"], quality)
    remediation = _read_csv(
        directory / FILE_REMEDIATION, TABLE_SPECS_BY_NAME["remediation_guidance"], quality
    )
    nist = _read_csv(directory / FILE_NIST, TABLE_SPECS_BY_NAME["nist_catalog"], quality)

    kev = _load_kev(directory / FILE_KEV, quality)
    campaigns, report_text, report_warnings = parse_threat_report(directory / FILE_THREAT_REPORT)
    quality.warnings.extend(report_warnings)

    # --- derived join keys -------------------------------------------------
    assets = assets.copy()
    assets["asset_key"] = assets["asset_id"].map(lambda v: _text(v).upper())
    assets["service_key"] = assets["business_service"].map(lambda v: _text(v).upper())

    vulns = vulns.copy()
    vulns["asset_key"] = vulns["asset_id"].map(lambda v: _text(v).upper())
    # vulnerabilities.cve must be a real CVE; control-style identifiers are defects here.
    normalised = vulns["cve"].map(lambda v: normalise_cve(v, allow_control_identifiers=False))
    vulns["cve_display"] = [n.display for n in normalised]
    vulns["cve_key"] = [n.match_key for n in normalised]
    vulns["cve_valid"] = [n.is_valid for n in normalised]
    vulns["cve_kind"] = [n.kind for n in normalised]

    intel = intel.copy()
    intel_norm = intel["matched_cve_or_control"].map(normalise_cve)
    intel["cve_display"] = [n.display for n in intel_norm]
    intel["cve_key"] = [n.match_key for n in intel_norm]
    intel["cve_valid"] = [n.is_valid for n in intel_norm]

    services = services.copy()
    services["service_key"] = services["business_service"].map(lambda v: _text(v).upper())

    # --- hygiene counters --------------------------------------------------
    quality.total_assets = int(len(assets))
    quality.total_vulnerabilities = int(len(vulns))
    quality.threat_intel_records = int(len(intel))

    for _, row in vulns[~vulns["cve_valid"]].iterrows():
        quality.malformed_cves.append(
            {
                "source": FILE_VULNS,
                "record_id": _text(row.get("vuln_id"), "<no id>"),
                "raw_value": _text(row.get("cve"), "<blank>"),
                "reason": normalise_cve(row.get("cve"), allow_control_identifiers=False).reason
                          or "unparseable identifier",
            }
        )
    for _, row in intel[~intel["cve_valid"]].iterrows():
        quality.malformed_cves.append(
            {
                "source": FILE_INTEL,
                "record_id": _text(row.get("intel_id"), "<no id>"),
                "raw_value": _text(row.get("matched_cve_or_control"), "<blank>"),
                "reason": "unparseable CVE/control identifier",
            }
        )

    asset_keys = set(assets["asset_key"])
    vuln_asset_keys = set(vulns["asset_key"])
    quality.unmapped_asset_ids = sorted(
        {k for k in vuln_asset_keys - asset_keys if k}
    )
    quality.assets_without_vulnerabilities = sorted(
        _text(r["asset_id"]) for _, r in assets.iterrows() if r["asset_key"] not in vuln_asset_keys
    )
    quality.assets_without_owner = sorted(
        _text(r["asset_id"]) for _, r in assets.iterrows() if not _text(r.get("owner_team"))
    )
    service_keys = set(services["service_key"])
    quality.unmapped_business_services = sorted(
        {_text(r["business_service"]) for _, r in assets.iterrows()
         if r["service_key"] and r["service_key"] not in service_keys}
    )
    for _, row in assets.iterrows():
        days = _to_int(row.get("last_seen_days"))
        if days is not None and days > 30:
            quality.stale_assets.append(
                {"asset_id": _text(row["asset_id"]), "asset_name": _text(row["asset_name"]),
                 "last_seen_days": days}
            )

    environment_cve_keys = {k for k in vulns.loc[vulns["cve_valid"], "cve_key"] if k}
    intel_matched_mask = intel["cve_key"].isin(environment_cve_keys) & intel["cve_valid"]
    quality.threat_intel_matched = int(intel_matched_mask.sum())
    quality.threat_intel_unmatched = int(len(intel) - intel_matched_mask.sum())
    intel = intel.assign(matches_environment=intel_matched_mask)

    return DataPack(
        data_dir=directory,
        assets=assets,
        vulnerabilities=vulns,
        threat_intelligence=intel,
        business_services=services,
        remediation_guidance=remediation,
        nist_catalog=nist,
        kev=kev,
        campaigns=campaigns,
        threat_report_text=report_text,
        quality=quality,
    )


# ---------------------------------------------------------------------------
# Deterministic risk engine
# ---------------------------------------------------------------------------

@dataclass
class RiskFactor:
    """One multiplier applied to a composite score, retained for lineage."""

    key: str
    label: str
    multiplier: float
    active: bool
    evidence: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "multiplier": self.multiplier,
            "active": self.active,
            "evidence": self.evidence,
        }


@dataclass
class Risk:
    """A scored (Asset + CVE) composite risk with its full evidence chain."""

    asset_id: str
    asset_name: str
    asset_type: str
    environment: str
    owner_team: str
    location: str
    vendor_product: str
    data_classification: str
    internet_facing: bool
    edr_installed: bool
    asset_criticality: str

    vuln_id: str
    vulnerability_name: str
    cve_display: str
    cve_key: str
    cvss: float
    vuln_severity: str
    exploit_available: bool
    patch_available: bool
    days_open: Optional[int]
    affected_component: str
    status: str

    business_service: str
    business_owner: str
    business_impact: str
    customer_facing: bool
    compliance_scope: str
    revenue_impact: str
    rto_hours: Optional[int]
    risk_appetite: str
    service_defined: bool
    effective_criticality: str

    kev_listed: bool
    kev_ransomware: bool
    kev_required_action: str
    kev_date_added: str

    intel_matches: List[Dict[str, str]]
    intel_ransomware: bool
    campaign_matches: List[Dict[str, str]]

    factors: List[RiskFactor]
    composite_score: float
    severity_band: str
    justification: str

    rank: int = 0
    nist: Optional[Dict[str, Any]] = None

    # --- convenience accessors used by the UI -----------------------------
    @property
    def active_factors(self) -> List[RiskFactor]:
        return [f for f in self.factors if f.active]

    @property
    def campaign_names(self) -> List[str]:
        return [c["campaign_name"] for c in self.campaign_matches if c.get("campaign_name")]

    @property
    def threat_actors(self) -> List[str]:
        names = [c.get("threat_actor", "") for c in self.campaign_matches]
        names += [i.get("threat_actor", "") for i in self.intel_matches]
        return sorted({n for n in names if n})

    def factor_multiplier(self, key: str) -> float:
        for factor in self.factors:
            if factor.key == key:
                return factor.multiplier
        return 1.0

    def is_factor_active(self, key: str) -> bool:
        for factor in self.factors:
            if factor.key == key:
                return factor.active
        return False

    def as_row(self) -> Dict[str, Any]:
        """Flat representation for dataframes, filters and charts."""
        return {
            "rank": self.rank,
            "composite_score": self.composite_score,
            "severity_band": self.severity_band,
            "asset_id": self.asset_id,
            "asset_name": self.asset_name,
            "asset_type": self.asset_type,
            "environment": self.environment,
            "business_service": self.business_service,
            "cve": self.cve_display,
            "vulnerability_name": self.vulnerability_name,
            "cvss": self.cvss,
            "internet_facing": self.internet_facing,
            "kev_listed": self.kev_listed,
            "kev_ransomware": self.kev_ransomware,
            "active_campaign": bool(self.campaign_matches),
            "edr_installed": self.edr_installed,
            "effective_criticality": self.effective_criticality,
            "days_open": self.days_open,
            "threat_actors": ", ".join(self.threat_actors),
            "campaigns": ", ".join(self.campaign_names),
        }


def severity_from_score(score: float) -> str:
    """Map a composite score onto a severity band."""
    for floor, label in SEVERITY_BANDS:
        if score >= floor:
            return label
    return "LOW"


def _build_justification(risk_parts: Dict[str, Any]) -> str:
    """
    Compose a single plain-English sentence from the active multipliers only.

    Pure string assembly over booleans — no model, no randomness, so the same
    risk always yields the same sentence.
    """
    clauses: List[str] = []
    if risk_parts["internet_facing"]:
        clauses.append("it is reachable from the internet")
    else:
        clauses.append("it is internal-only, which holds the score down")

    if risk_parts["campaign_matches"]:
        names = ", ".join(risk_parts["campaign_names"][:2]) or "an active regional campaign"
        clauses.append(f"an active Middle East campaign ({names}) is exploiting this CVE right now")
    if risk_parts["kev_listed"] and risk_parts["kev_ransomware"]:
        clauses.append("CISA lists it as known-exploited and ransomware-associated")
    elif risk_parts["kev_listed"]:
        clauses.append("CISA lists it as known-exploited in the wild")
    elif risk_parts["intel_ransomware"]:
        clauses.append("threat intelligence ties it to ransomware activity")

    criticality = risk_parts["effective_criticality"].title()
    service = risk_parts["business_service"] or "an unmapped business service"
    compliance = risk_parts["compliance_scope"]
    service_clause = f"it supports {service}, a {criticality}-criticality service"
    if compliance:
        service_clause += f" in {compliance} scope"
    clauses.append(service_clause)

    if not risk_parts["edr_installed"]:
        clauses.append("no EDR agent is present to detect or contain post-exploitation activity")
    if risk_parts["patch_available"] and risk_parts["days_open"] is not None:
        clauses.append(f"a vendor patch has been available while the finding sat open for {risk_parts['days_open']} days")
    elif not risk_parts["patch_available"]:
        clauses.append("no vendor patch is currently available, so containment must be compensating")

    head = (
        f"{risk_parts['cve_display']} on {risk_parts['asset_name']} scores "
        f"{risk_parts['composite_score']:.1f} (CVSS {risk_parts['cvss']:.1f}) because "
    )
    if len(clauses) == 1:
        body = clauses[0]
    else:
        body = ", ".join(clauses[:-1]) + f", and {clauses[-1]}"
    return head + body + "."


class RiskEngine:
    """
    Deterministic composite risk scorer.

    Composite = CVSS x internet-exposure x KEV/ransomware x regional-campaign
                x business-criticality x EDR

    No score is capped or normalised, so ranking preserves the full dynamic range.
    The LLM is never consulted here.
    """

    def __init__(self, pack: DataPack) -> None:
        self.pack = pack
        self._assets_by_key = {
            _text(row["asset_key"]): row for _, row in pack.assets.iterrows()
        }
        self._services_by_key = {
            _text(row["service_key"]): row for _, row in pack.business_services.iterrows()
        }
        self._kev_by_key = {
            _text(row["cve_key"]): row for _, row in pack.kev.iterrows()
        }
        self._intel_by_key: Dict[str, List[pd.Series]] = {}
        for _, row in pack.threat_intelligence.iterrows():
            key = _text(row["cve_key"])
            if key:
                self._intel_by_key.setdefault(key, []).append(row)
        self._campaigns_by_key: Dict[str, List[ReportCampaign]] = {}
        for campaign in pack.campaigns:
            for key in campaign.cve_keys:
                self._campaigns_by_key.setdefault(key, []).append(campaign)

    # -- individual factor resolution --------------------------------------

    def _internet_factor(self, asset: pd.Series, vuln: pd.Series) -> RiskFactor:
        asset_flag = _is_yes(asset.get("internet_exposed"))
        vuln_flag = _text(vuln.get("asset_exposure")).upper() in {"INTERNET", "EXTERNAL", "PUBLIC"}
        active = asset_flag or vuln_flag
        sources = []
        if asset_flag:
            sources.append("assets.internet_exposed=Yes")
        if vuln_flag:
            sources.append(f"vulnerabilities.asset_exposure={_text(vuln.get('asset_exposure'))}")
        return RiskFactor(
            key="internet_facing",
            label="Internet facing",
            multiplier=MULT_INTERNET_FACING if active else MULT_INTERNET_INTERNAL,
            active=active,
            evidence="; ".join(sources) if sources else "asset is internal-only",
        )

    def _kev_factor(
        self, cve_key: str, intel_rows: List[pd.Series]
    ) -> Tuple[RiskFactor, Dict[str, Any]]:
        kev_row = self._kev_by_key.get(cve_key)
        kev_listed = kev_row is not None
        kev_ransomware = bool(
            kev_listed and _text(kev_row.get("knownRansomwareCampaignUse")).upper().startswith("KNOWN")
        )
        intel_ransomware = any(_is_yes(r.get("ransomware_association")) for r in intel_rows)
        active = kev_listed or kev_ransomware or intel_ransomware

        evidence: List[str] = []
        if kev_listed:
            added = _text(kev_row.get("dateAdded"))
            evidence.append(f"CISA KEV entry{f' added {added}' if added else ''}")
        if kev_ransomware:
            evidence.append("KEV flags known ransomware campaign use")
        if intel_ransomware:
            evidence.append("threat intel records ransomware association")
        if not evidence:
            evidence.append("no KEV listing and no ransomware association on record")

        factor = RiskFactor(
            key="kev_ransomware",
            label="Known exploited / ransomware associated",
            multiplier=MULT_KEV_MATCH if active else MULT_KEV_NONE,
            active=active,
            evidence="; ".join(evidence),
        )
        detail = {
            "kev_listed": kev_listed,
            "kev_ransomware": kev_ransomware,
            "kev_required_action": _text(kev_row.get("requiredAction")) if kev_listed else "",
            "kev_date_added": _text(kev_row.get("dateAdded")) if kev_listed else "",
            "intel_ransomware": intel_ransomware,
        }
        return factor, detail

    def _campaign_factor(self, cve_key: str) -> Tuple[RiskFactor, List[Dict[str, str]]]:
        campaigns = self._campaigns_by_key.get(cve_key, [])
        matches = [
            {
                "threat_actor": c.threat_actor,
                "campaign_name": c.campaign_name,
                "target_profile": c.target_profile,
                "confidence": c.confidence,
                "ransomware": "Yes" if c.ransomware else "No",
            }
            for c in campaigns
        ]
        active = bool(campaigns)
        if active:
            listed = ", ".join(f"{c.threat_actor} / {c.campaign_name}".strip(" /") for c in campaigns)
            evidence = f"named in today's MDR advisory: {listed}"
        else:
            evidence = "not named in today's regional MDR advisory"
        factor = RiskFactor(
            key="regional_campaign",
            label="Active regional MDR campaign",
            multiplier=MULT_CAMPAIGN_MATCH if active else MULT_CAMPAIGN_NONE,
            active=active,
            evidence=evidence,
        )
        return factor, matches

    def _criticality_factor(
        self, asset: pd.Series, service: Optional[pd.Series]
    ) -> Tuple[RiskFactor, str]:
        asset_level = _canon_level(asset.get("criticality"))
        service_level = _canon_level(service.get("revenue_impact")) if service is not None else None

        effective = asset_level
        if service_level and CRITICALITY_RANK[service_level] > CRITICALITY_RANK[asset_level]:
            effective = service_level

        multiplier = CRITICALITY_MULTIPLIERS.get(effective, CRITICALITY_MULTIPLIERS[DEFAULT_CRITICALITY])
        parts = [f"asset criticality {asset_level.title()}"]
        if service_level:
            parts.append(f"service revenue impact {service_level.title()}")
        else:
            parts.append("business service not defined; asset criticality used alone")

        factor = RiskFactor(
            key="business_criticality",
            label=f"Business criticality: {effective.title()}",
            multiplier=multiplier,
            active=multiplier > 1.0,
            evidence="; ".join(parts),
        )
        return factor, effective

    def _edr_factor(self, asset: pd.Series) -> RiskFactor:
        raw = asset.get("edr_installed")
        installed = _is_yes(raw)
        if installed:
            evidence = "EDR agent installed"
        elif _is_no(raw):
            evidence = "EDR agent not installed"
        else:
            evidence = f"EDR status unknown ('{_text(raw, '<blank>')}'); scored as missing"
        return RiskFactor(
            key="edr_missing",
            label="No EDR coverage",
            multiplier=MULT_EDR_PRESENT if installed else MULT_EDR_MISSING,
            active=not installed,
            evidence=evidence,
        )

    # -- scoring ------------------------------------------------------------

    def score_all(self) -> List[Risk]:
        """Score every live (Asset + CVE) pair and return them ranked."""
        quality = self.pack.quality
        risks: List[Risk] = []

        for _, vuln in self.pack.vulnerabilities.iterrows():
            status = _text(vuln.get("status"))
            if status.upper() in CLOSED_STATUSES:
                quality.vulnerabilities_skipped_closed += 1
                continue

            cve_key = _text(vuln.get("cve_key"))
            if not bool(vuln.get("cve_valid")) or not cve_key:
                quality.vulnerabilities_skipped_bad_cve += 1
                continue

            asset = self._assets_by_key.get(_text(vuln.get("asset_key")))
            if asset is None:
                quality.vulnerabilities_skipped_unmapped_asset += 1
                continue

            service_key = _text(asset.get("service_key"))
            service = self._services_by_key.get(service_key)
            intel_rows = self._intel_by_key.get(cve_key, [])

            internet_factor = self._internet_factor(asset, vuln)
            kev_factor, kev_detail = self._kev_factor(cve_key, intel_rows)
            campaign_factor, campaign_matches = self._campaign_factor(cve_key)
            criticality_factor, effective_criticality = self._criticality_factor(asset, service)
            edr_factor = self._edr_factor(asset)

            factors = [
                internet_factor, kev_factor, campaign_factor, criticality_factor, edr_factor
            ]

            cvss = _to_float(vuln.get("cvss"), 0.0)
            if cvss <= 0.0:
                quality.warnings.append(
                    f"{_text(vuln.get('vuln_id'), '<no id>')}: CVSS missing or zero "
                    f"('{_text(vuln.get('cvss'), '<blank>')}'); scored as 0.0"
                )
            composite = cvss
            for factor in factors:
                composite *= factor.multiplier
            composite = round(composite, 2)

            intel_matches = [
                {
                    "intel_id": _text(r.get("intel_id")),
                    "threat_actor": _text(r.get("threat_actor")),
                    "campaign_name": _text(r.get("campaign_name")),
                    "target_sector": _text(r.get("target_sector")),
                    "target_region": _text(r.get("target_region")),
                    "exploit_maturity": _text(r.get("exploit_maturity")),
                    "active_last_seen": _text(r.get("active_last_seen")),
                    "ransomware_association": _text(r.get("ransomware_association")),
                    "confidence": _text(r.get("confidence")),
                    "summary": _text(r.get("summary")),
                }
                for r in intel_rows
            ]

            parts = {
                "asset_name": _text(asset.get("asset_name"), _text(asset.get("asset_id"))),
                "cve_display": _text(vuln.get("cve_display")),
                "cvss": cvss,
                "composite_score": composite,
                "internet_facing": internet_factor.active,
                "kev_listed": kev_detail["kev_listed"],
                "kev_ransomware": kev_detail["kev_ransomware"],
                "intel_ransomware": kev_detail["intel_ransomware"],
                "campaign_matches": campaign_matches,
                "campaign_names": [c["campaign_name"] for c in campaign_matches if c["campaign_name"]],
                "effective_criticality": effective_criticality,
                "business_service": _text(asset.get("business_service")),
                "compliance_scope": _text(service.get("compliance_scope")) if service is not None else "",
                "edr_installed": not edr_factor.active,
                "patch_available": _is_yes(vuln.get("patch_available")),
                "days_open": _to_int(vuln.get("days_open")),
            }

            risks.append(
                Risk(
                    asset_id=_text(asset.get("asset_id")),
                    asset_name=parts["asset_name"],
                    asset_type=_text(asset.get("asset_type"), "Unknown"),
                    environment=_text(asset.get("environment"), "Unknown"),
                    owner_team=_text(asset.get("owner_team"), "UNASSIGNED"),
                    location=_text(asset.get("location")),
                    vendor_product=_text(asset.get("vendor_product")),
                    data_classification=_text(asset.get("data_classification")),
                    internet_facing=internet_factor.active,
                    edr_installed=not edr_factor.active,
                    asset_criticality=_canon_level(asset.get("criticality")),
                    vuln_id=_text(vuln.get("vuln_id")),
                    vulnerability_name=_text(vuln.get("vulnerability_name"), "Unnamed finding"),
                    cve_display=parts["cve_display"],
                    cve_key=cve_key,
                    cvss=cvss,
                    vuln_severity=_canon_level(vuln.get("severity")),
                    exploit_available=_is_yes(vuln.get("exploit_available")),
                    patch_available=parts["patch_available"],
                    days_open=parts["days_open"],
                    affected_component=_text(vuln.get("affected_component")),
                    status=status or "Open",
                    business_service=parts["business_service"] or "UNMAPPED SERVICE",
                    business_owner=_text(service.get("business_owner")) if service is not None else "UNASSIGNED",
                    business_impact=_text(service.get("business_impact")) if service is not None else "",
                    customer_facing=_is_yes(service.get("customer_facing")) if service is not None else False,
                    compliance_scope=parts["compliance_scope"],
                    revenue_impact=_text(service.get("revenue_impact")) if service is not None else "",
                    rto_hours=_to_int(service.get("rto_hours")) if service is not None else None,
                    risk_appetite=_text(service.get("risk_appetite")) if service is not None else "",
                    service_defined=service is not None,
                    effective_criticality=effective_criticality,
                    kev_listed=kev_detail["kev_listed"],
                    kev_ransomware=kev_detail["kev_ransomware"],
                    kev_required_action=kev_detail["kev_required_action"],
                    kev_date_added=kev_detail["kev_date_added"],
                    intel_matches=intel_matches,
                    intel_ransomware=kev_detail["intel_ransomware"],
                    campaign_matches=campaign_matches,
                    factors=factors,
                    composite_score=composite,
                    severity_band=severity_from_score(composite),
                    justification=_build_justification(parts),
                )
            )

        quality.vulnerabilities_processed = len(risks)
        quality.kev_matches = sum(1 for r in risks if r.kev_listed)
        quality.campaign_matches = sum(1 for r in risks if r.campaign_matches)

        # Deterministic ordering: score desc, CVSS desc, CVE asc, then vuln_id asc
        # so ties never depend on input file order.
        risks.sort(key=lambda r: (-r.composite_score, -r.cvss, r.cve_display, r.vuln_id))
        for index, risk in enumerate(risks, start=1):
            risk.rank = index
        return risks


# ---------------------------------------------------------------------------
# NIST SP 800-53 vector RAG engine
# ---------------------------------------------------------------------------

@dataclass
class NistControl:
    """A retrieved NIST control with its similarity distance."""

    control_id: str
    control_title: str
    family: str
    family_code: str
    control_text: str
    discussion: str
    related: str
    distance: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "control_id": self.control_id,
            "control_title": self.control_title,
            "family": self.family,
            "family_code": self.family_code,
            "control_text": self.control_text,
            "discussion": self.discussion,
            "related": self.related,
            "distance": self.distance,
        }


def _family_for(identifier: str) -> Tuple[str, str]:
    """Return (family_code, family_name) for a control identifier such as 'SI-2(3)'."""
    match = re.match(r"^([A-Z]{2})", _text(identifier).upper())
    code = match.group(1) if match else "??"
    return code, NIST_FAMILIES.get(code, "Unmapped family")


class NistRagEngine:
    """
    Vector retrieval over the NIST SP 800-53 Rev. 5 catalogue.

    One chunk per control (identifier + title + control text + discussion),
    embedded with all-MiniLM-L6-v2 and stored in a persistent ChromaDB collection
    keyed by a hash of the source CSV, so a changed catalogue re-indexes
    automatically and an unchanged one loads instantly.
    """

    def __init__(
        self,
        catalog: pd.DataFrame,
        persist_path: Path = NIST_CHROMA_PATH,
        model_name: str = NIST_EMBEDDING_MODEL,
        distance_threshold: float = NIST_DISTANCE_THRESHOLD,
    ) -> None:
        self.catalog = catalog
        self.persist_path = Path(persist_path)
        self.model_name = model_name
        self.distance_threshold = distance_threshold
        self.available = False
        self.status = "not initialised"
        self.indexed_controls = 0
        self._collection = None
        self._by_control_id: Dict[str, NistControl] = {}
        self._build_lookup()

    # -- indexing -----------------------------------------------------------

    def _build_lookup(self) -> None:
        """Build an exact-ID fallback index that works without ChromaDB."""
        for _, row in self.catalog.iterrows():
            identifier = _text(row.get("identifier")).upper()
            if not identifier:
                continue
            code, family = _family_for(identifier)
            self._by_control_id[identifier] = NistControl(
                control_id=identifier,
                control_title=_text(row.get("name"), "Untitled control"),
                family=family,
                family_code=code,
                control_text=_text(row.get("control_text")),
                discussion=_text(row.get("discussion")),
                related=_text(row.get("related")),
                distance=0.0,
            )

    def _documents(self) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """Build (ids, documents, metadatas) for the vector store."""
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        seen: set = set()

        for control in self._by_control_id.values():
            if control.control_id in seen:
                continue
            seen.add(control.control_id)
            body = " ".join(
                part for part in (
                    f"{control.control_id} {control.control_title}.",
                    f"Control family: {control.family}.",
                    control.control_text,
                    control.discussion,
                ) if part
            )
            body = re.sub(r"\s+", " ", body).strip()
            if not body:
                continue
            ids.append(control.control_id)
            documents.append(body)
            metadatas.append(
                {
                    "control_id": control.control_id,
                    "control_title": control.control_title,
                    "family": control.family,
                    "family_code": control.family_code,
                    "control_text": control.control_text[:8000],
                    "discussion": control.discussion[:8000],
                    "related": control.related,
                    "is_enhancement": "(" in control.control_id,
                }
            )
        return ids, documents, metadatas

    def _fingerprint(self, documents: Sequence[str]) -> str:
        """Content hash of the corpus, used as the collection suffix."""
        digest = hashlib.sha256()
        digest.update(self.model_name.encode("utf-8"))
        for document in documents:
            digest.update(document.encode("utf-8"))
        return digest.hexdigest()[:12]

    @staticmethod
    def _existing_collection_names(client: Any) -> set:
        """
        List collection names across ChromaDB API generations.

        Versions below 0.6 return collection objects; 0.6 and above return plain
        name strings. Both shapes are handled, and any failure degrades to an
        empty set (which simply forces a get_or_create).
        """
        try:
            listed = client.list_collections()
        except Exception:
            return set()
        names = set()
        for item in listed or []:
            name = getattr(item, "name", None)
            names.add(name if isinstance(name, str) else str(item))
        return names

    def build_index(self, batch_size: int = 256) -> None:
        """
        Create or reuse the ChromaDB collection.

        Any failure (missing dependency, unwritable path, model download failure)
        is captured in `self.status` and leaves `self.available` False. Retrieval
        then falls back to exact control-ID lookup rather than crashing the app.
        """
        ids, documents, metadatas = self._documents()
        self.indexed_controls = len(ids)
        if not ids:
            self.status = "NIST catalogue produced no indexable controls"
            return

        try:
            import chromadb
            from chromadb.config import Settings
            from chromadb.utils import embedding_functions
        except Exception as exc:
            self.status = f"ChromaDB/sentence-transformers unavailable ({exc}); using exact-ID fallback"
            LOGGER.warning(self.status)
            return

        try:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(self.persist_path),
                settings=Settings(anonymized_telemetry=False, allow_reset=False),
            )
            embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.model_name
            )
            collection_name = f"{NIST_COLLECTION_PREFIX}_{self._fingerprint(documents)}"
            existing = self._existing_collection_names(client)

            # chromadb >=0.6 can raise when get_or_create is handed metadata that
            # differs from an existing collection, so prefer an explicit get.
            collection = None
            if collection_name in existing:
                try:
                    collection = client.get_collection(
                        name=collection_name, embedding_function=embedder
                    )
                except Exception:
                    collection = None
            if collection is None:
                collection = client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=embedder,
                    metadata={"hnsw:space": "cosine"},
                )

            if collection.count() < len(ids):
                for start in range(0, len(ids), batch_size):
                    stop = start + batch_size
                    collection.upsert(
                        ids=ids[start:stop],
                        documents=documents[start:stop],
                        metadatas=metadatas[start:stop],
                    )
                action = "reindexed" if collection_name in existing else "indexed"
            else:
                action = "loaded from cache"

            self._collection = collection
            self.available = True
            self.status = (
                f"{collection.count()} NIST controls {action} "
                f"({self.model_name}, cosine, threshold {self.distance_threshold})"
            )
            LOGGER.info(self.status)
        except Exception as exc:
            self.status = f"Vector index build failed ({exc}); using exact-ID fallback"
            LOGGER.warning(self.status)
            self._collection = None
            self.available = False

    # -- retrieval ----------------------------------------------------------

    def query(self, text: str, n_results: int = 4) -> List[NistControl]:
        """
        Retrieve candidate controls inside `distance_threshold`.

        An empty list means the engine declined to answer — callers must not
        substitute a guess.
        """
        return self.query_detailed(text, n_results)[0]

    def query_detailed(
        self, text: str, n_results: int = 4
    ) -> Tuple[List[NistControl], List[NistControl]]:
        """
        Retrieve candidates and return them split into (accepted, rejected).

        The rejected list lets the UI report the best near-miss distance when
        nothing clears the threshold, which is what makes the threshold tunable
        in practice rather than a silent dead end.
        """
        if not self.available or self._collection is None:
            return [], []
        query_text = re.sub(r"\s+", " ", _text(text)).strip()
        if not query_text:
            return [], []
        try:
            result = self._collection.query(
                query_texts=[query_text],
                n_results=max(1, min(int(n_results), max(1, self.indexed_controls))),
                include=["metadatas", "distances", "documents"],
            )
        except Exception as exc:
            LOGGER.warning("NIST retrieval failed for query '%s': %s", query_text[:80], exc)
            return [], []

        metadatas = (result.get("metadatas") or [[]])[0] or []
        distances = (result.get("distances") or [[]])[0] or []
        controls: List[NistControl] = []
        for metadata, distance in zip(metadatas, distances):
            metadata = metadata or {}
            control_id = _text(metadata.get("control_id")).upper()
            known = self._by_control_id.get(control_id)
            controls.append(
                NistControl(
                    control_id=control_id or "UNKNOWN",
                    control_title=_text(metadata.get("control_title"), "Untitled control"),
                    family=_text(metadata.get("family"), "Unmapped family"),
                    family_code=_text(metadata.get("family_code"), "??"),
                    control_text=known.control_text if known else _text(metadata.get("control_text")),
                    discussion=known.discussion if known else _text(metadata.get("discussion")),
                    related=_text(metadata.get("related")),
                    distance=float(distance),
                )
            )
        accepted = [c for c in controls if c.distance <= self.distance_threshold]
        rejected = [c for c in controls if c.distance > self.distance_threshold]
        return accepted, rejected

    def get_by_id(self, control_id: str) -> Optional[NistControl]:
        """Exact control lookup, used as the no-vector-store fallback."""
        return self._by_control_id.get(_text(control_id).upper())


# ---------------------------------------------------------------------------
# Remediation hint matching
# ---------------------------------------------------------------------------

class RemediationHintMatcher:
    """
    Deterministic matcher from a finding to a one-line hint in remediation_guidance.csv.

    Scored by token overlap against `finding_type`, with ties broken alphabetically
    so the result never depends on row order. The hint is only used to enrich the
    retrieval query — it is never presented as the remediation answer.
    """

    def __init__(self, guidance: pd.DataFrame) -> None:
        self.rows: List[Dict[str, Any]] = []
        for _, row in guidance.iterrows():
            finding_type = _text(row.get("finding_type"))
            if not finding_type:
                continue
            self.rows.append(
                {
                    "finding_type": finding_type,
                    "recommended_action": _text(row.get("recommended_action")),
                    "priority_hint": _text(row.get("priority_hint")),
                    "validation_evidence": _text(row.get("validation_evidence")),
                    "tokens": _tokenise(finding_type),
                }
            )
        self.rows.sort(key=lambda r: r["finding_type"].upper())

    def match(self, *texts: str) -> Optional[Dict[str, Any]]:
        """Return the best-matching hint row, or None when nothing overlaps."""
        query_tokens = set()
        for text in texts:
            query_tokens |= _tokenise(text)
        if not query_tokens:
            return None

        best: Optional[Dict[str, Any]] = None
        best_score = 0.0
        for row in self.rows:
            overlap = len(row["tokens"] & query_tokens)
            if not overlap:
                continue
            score = overlap / max(1, len(row["tokens"]))
            if score > best_score:
                best, best_score = row, score
        if best is None:
            return None
        return {**{k: v for k, v in best.items() if k != "tokens"}, "match_score": round(best_score, 3)}


# ---------------------------------------------------------------------------
# Groq summarisation
# ---------------------------------------------------------------------------

GROQ_SYSTEM_PROMPT = """You are a cyber risk communications assistant for a fintech CISO.

You will be given:
  (a) a risk context block produced by a deterministic scoring engine, and
  (b) the verbatim text of ONE NIST SP 800-53 Rev. 5 control that was retrieved
      from the official catalogue.

Write 2-3 sentences of actionable executive remediation guidance, grounded ONLY in
the supplied control text and risk context.

Hard rules:
- Do not mention, cite, invent or imply any control other than the one supplied.
- Do not state, recalculate, adjust, question or reorder any risk score, rank or CVSS value.
- Do not recommend offensive, retaliatory, or active-counterintrusion measures.
- Do not add facts that are not present in the supplied text.
- Plain prose. No headings, no bullet points, no preamble, no markdown.
- If the supplied control text does not support actionable guidance, say so plainly in one sentence.
"""


class GroqSummariser:
    """
    Thin, defensive wrapper over the Groq chat completions API.

    `available` is False whenever the key is absent or the SDK is missing, and every
    call is individually guarded, so a rate limit or network error downgrades one
    card to raw NIST prose rather than breaking the dashboard.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = GROQ_MODEL) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "").strip()
        self.available = False
        self.status = "GROQ_API_KEY not set; showing retrieved NIST control text verbatim"
        self._client = None

        if not self.api_key:
            return
        try:
            from groq import Groq
        except Exception as exc:
            self.status = f"groq package not installed ({exc}); showing retrieved NIST text verbatim"
            return
        try:
            self._client = Groq(api_key=self.api_key)
            self.available = True
            self.status = f"Groq ready ({self.model})"
        except Exception as exc:
            self.status = f"Groq client init failed ({exc}); showing retrieved NIST text verbatim"

    def summarise(self, risk_context: str, control: NistControl) -> Tuple[str, str]:
        """
        Summarise a retrieved control.

        Returns
        -------
        (text, source)
            `source` is "groq" on success or "nist_verbatim" for any fallback path,
            so the UI can label provenance honestly.
        """
        verbatim = self._verbatim(control)
        if not self.available or self._client is None:
            return verbatim, "nist_verbatim"

        user_prompt = (
            f"RISK CONTEXT\n{risk_context.strip()}\n\n"
            f"RETRIEVED NIST CONTROL (verbatim, do not go beyond this text)\n"
            f"{control.control_id} — {control.control_title}\n"
            f"Family: {control.family}\n"
            f"Control text: {control.control_text[:4000]}\n"
            f"Discussion: {control.discussion[:3000]}\n"
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": GROQ_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=320,
                timeout=GROQ_TIMEOUT_SECONDS,
            )
            text = _text(response.choices[0].message.content)
            if not text:
                return verbatim, "nist_verbatim"
            return text, "groq"
        except Exception as exc:
            LOGGER.warning("Groq summarisation failed for %s: %s", control.control_id, exc)
            self.status = f"Groq call failed ({type(exc).__name__}); showing retrieved NIST text verbatim"
            return verbatim, "nist_verbatim"

    @staticmethod
    def _verbatim(control: NistControl) -> str:
        """Trim the raw control prose for display without paraphrasing it."""
        body = control.control_text.strip()
        if control.discussion.strip():
            body = f"{body}\n\nDiscussion: {control.discussion.strip()}"
        return body[:2400].strip() or "Retrieved control contains no prose text."


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

DRIVER_LABELS: Dict[str, str] = {
    "internet_facing": f"Internet facing (x{MULT_INTERNET_FACING})",
    "kev_ransomware": f"Known exploited / ransomware (x{MULT_KEV_MATCH})",
    "regional_campaign": f"Active regional campaign (x{MULT_CAMPAIGN_MATCH})",
    "business_criticality": "Critical or High business criticality",
    "edr_missing": f"No EDR coverage (x{MULT_EDR_MISSING})",
}


def driver_counts_for(risks: Sequence["Risk"]) -> Dict[str, int]:
    """Count how often each multiplier fires across an arbitrary subset of risks."""
    counts = {label: 0 for label in DRIVER_LABELS.values()}
    for risk in risks:
        for key, label in DRIVER_LABELS.items():
            if key == "business_criticality":
                if risk.effective_criticality in {"CRITICAL", "HIGH"}:
                    counts[label] += 1
            elif risk.is_factor_active(key):
                counts[label] += 1
    return counts


@dataclass
class RiskReport:
    """Everything the dashboard needs, assembled in one deterministic pass."""

    pack: DataPack
    all_risks: List[Risk]
    top_risks: List[Risk]
    quality: DataQuality
    nist_status: str
    nist_available: bool
    groq_status: str
    groq_available: bool
    generated_from: Path
    nist_engine: Optional["NistRagEngine"] = None
    hint_matcher: Optional["RemediationHintMatcher"] = None
    summariser: Optional["GroqSummariser"] = None

    @property
    def risk_frame(self) -> pd.DataFrame:
        """All scored risks as a dataframe for filtering and charting."""
        if not self.all_risks:
            return pd.DataFrame(
                columns=[
                    "rank", "composite_score", "severity_band", "asset_id", "asset_name",
                    "asset_type", "environment", "business_service", "cve",
                    "vulnerability_name", "cvss", "internet_facing", "kev_listed",
                    "kev_ransomware", "active_campaign", "edr_installed",
                    "effective_criticality", "days_open", "threat_actors", "campaigns",
                ]
            )
        return pd.DataFrame([r.as_row() for r in self.all_risks])

    @property
    def driver_counts(self) -> Dict[str, int]:
        """How often each multiplier fires across the whole scored population."""
        return driver_counts_for(self.all_risks)

    def enrich(self, risks: Sequence[Risk]) -> None:
        """
        Attach NIST guidance to risks that do not have it yet.

        Filtering in the UI can promote a risk that was not in the global Top 5,
        so enrichment is done on demand rather than only once up front.
        """
        pending = [r for r in risks if r.nist is None]
        if not pending or self.nist_engine is None or self.hint_matcher is None:
            return
        enrich_with_nist(
            pending,
            self.nist_engine,
            self.hint_matcher,
            self.summariser or GroqSummariser(api_key=""),
        )

    @property
    def top_priority_target(self) -> str:
        """Asset name of the single highest-scoring risk."""
        return self.top_risks[0].asset_name if self.top_risks else "None"


def _retrieval_query(risk: Risk, hint: Optional[Dict[str, Any]]) -> str:
    """
    Build the semantic retrieval query for one risk.

    Combines the finding, the component, the active risk drivers and, when
    available, the one-line hint from remediation_guidance.csv. The hint steers
    retrieval; it is never the answer that gets displayed.
    """
    parts: List[str] = [
        risk.vulnerability_name,
        risk.affected_component,
        f"{risk.asset_type} in {risk.environment}",
    ]
    if risk.internet_facing:
        parts.append("internet-facing system exposed to untrusted networks")
    if risk.patch_available:
        parts.append("vendor patch available but not yet applied; flaw remediation and patch management")
    else:
        parts.append("no vendor patch available; compensating controls and unsupported components")
    if risk.kev_listed or risk.kev_ransomware or risk.intel_ransomware:
        parts.append("vulnerability actively exploited in the wild, ransomware campaign, incident handling")
    if risk.campaign_matches:
        parts.append("targeted threat campaign monitoring and response")
    if not risk.edr_installed:
        parts.append("missing endpoint detection and malicious code protection monitoring")
    if risk.days_open is not None and risk.days_open > 30:
        parts.append("remediation timeframes exceeded, vulnerability monitoring and scanning")
    if risk.compliance_scope:
        parts.append(f"{risk.compliance_scope} compliance obligations")
    if hint:
        parts.append(hint.get("finding_type", ""))
        parts.append(hint.get("recommended_action", ""))
    return " ".join(p for p in parts if p)


def _risk_context_block(risk: Risk) -> str:
    """Compact, score-free context block handed to the LLM."""
    lines = [
        f"Asset: {risk.asset_name} ({risk.asset_type}, {risk.environment}, {risk.location or 'location unknown'})",
        f"Finding: {risk.vulnerability_name} [{risk.cve_display}]",
        f"Internet facing: {'yes' if risk.internet_facing else 'no'}",
        f"EDR installed: {'yes' if risk.edr_installed else 'no'}",
        f"Patch available: {'yes' if risk.patch_available else 'no'}",
        f"Business service: {risk.business_service}"
        + (f" (compliance scope: {risk.compliance_scope})" if risk.compliance_scope else ""),
        f"Known exploited (CISA KEV): {'yes' if risk.kev_listed else 'no'}",
        f"Ransomware association: {'yes' if (risk.kev_ransomware or risk.intel_ransomware) else 'no'}",
    ]
    if risk.campaign_matches:
        listed = ", ".join(
            f"{c['threat_actor']} / {c['campaign_name']}".strip(" /") for c in risk.campaign_matches
        )
        lines.append(f"Active regional campaigns naming this CVE: {listed}")
    return "\n".join(lines)


def enrich_with_nist(
    risks: Sequence[Risk],
    nist_engine: NistRagEngine,
    hint_matcher: RemediationHintMatcher,
    summariser: GroqSummariser,
) -> None:
    """
    Attach retrieved NIST guidance to each risk, in place.

    Retrieval happens first; the LLM only ever sees prose that came back from the
    vector store. When retrieval declines, no control is shown and no LLM call is
    made — the card states plainly that nothing sufficiently relevant was found.
    """
    for risk in risks:
        hint = hint_matcher.match(
            risk.vulnerability_name, risk.affected_component, risk.asset_type
        )
        query = _retrieval_query(risk, hint)
        candidates, rejected = nist_engine.query_detailed(query, n_results=4)

        payload: Dict[str, Any] = {
            "query": query,
            "hint": hint,
            "candidates": [c.as_dict() for c in candidates],
            "source_dataset": FILE_NIST,
            "retrieval_method": "ChromaDB cosine vector search (all-MiniLM-L6-v2)"
            if nist_engine.available else "exact control-ID fallback (vector store unavailable)",
            "control": None,
            "guidance": "",
            "guidance_source": "none",
        }

        if candidates:
            best = candidates[0]
            payload["control"] = best.as_dict()
            guidance, source = summariser.summarise(_risk_context_block(risk), best)
            payload["guidance"] = guidance
            payload["guidance_source"] = source
        else:
            near_miss = ""
            if rejected:
                best = min(rejected, key=lambda c: c.distance)
                near_miss = (
                    f" Closest candidate was {best.control_id} at distance "
                    f"{best.distance:.3f}; raise TP_NIST_DISTANCE_THRESHOLD if that is acceptable."
                )
                payload["rejected"] = [c.as_dict() for c in rejected]
            payload["guidance"] = (
                "No sufficiently relevant NIST control retrieved "
                f"(cosine distance threshold {nist_engine.distance_threshold})."
                f"{near_miss} Review this finding manually rather than accepting a "
                "loosely related control."
            )
            payload["guidance_source"] = "no_retrieval"

        risk.nist = payload


def generate_risk_report(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    top_n: int = 5,
    use_llm: bool = True,
    groq_api_key: Optional[str] = None,
) -> RiskReport:
    """
    Run the full pipeline: load -> validate -> join -> score -> rank -> retrieve -> summarise.

    Parameters
    ----------
    data_dir
        Directory containing the data pack.
    top_n
        How many ranked risks to enrich with NIST guidance.
    use_llm
        When False, skips Groq entirely and shows retrieved control prose verbatim.
    groq_api_key
        Overrides the GROQ_API_KEY environment variable.

    Raises
    ------
    SchemaValidationError
        On a missing required file or unmappable required column.
    """
    pack = load_data_pack(data_dir)

    engine = RiskEngine(pack)
    all_risks = engine.score_all()
    top_risks = all_risks[: max(0, int(top_n))]

    nist_engine = NistRagEngine(pack.nist_catalog)
    nist_engine.build_index()

    hint_matcher = RemediationHintMatcher(pack.remediation_guidance)

    if use_llm:
        summariser = GroqSummariser(api_key=groq_api_key)
    else:
        summariser = GroqSummariser(api_key="")
        summariser.status = "LLM summarisation disabled by user; showing retrieved NIST text verbatim"

    enrich_with_nist(top_risks, nist_engine, hint_matcher, summariser)

    return RiskReport(
        pack=pack,
        all_risks=all_risks,
        top_risks=top_risks,
        quality=pack.quality,
        nist_status=nist_engine.status,
        nist_available=nist_engine.available,
        groq_status=summariser.status,
        groq_available=summariser.available,
        generated_from=Path(data_dir),
        nist_engine=nist_engine,
        hint_matcher=hint_matcher,
        summariser=summariser,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_report(report: RiskReport) -> None:
    """Render the Top 5 to stdout for headless verification."""
    print("=" * 78)
    print("TawasolPay — AI Cyber Risk Assistant — Top 5 Composite Risks")
    print("=" * 78)
    print(f"NIST retrieval : {report.nist_status}")
    print(f"LLM            : {report.groq_status}")
    print(f"Scored         : {report.quality.vulnerabilities_processed} live asset+CVE pairs")
    print()
    for risk in report.top_risks:
        print(f"[{risk.rank}] {risk.severity_band}  score {risk.composite_score:.2f}  "
              f"{risk.asset_name}  {risk.cve_display}  (CVSS {risk.cvss:.1f})")
        print(f"     Service : {risk.business_service}"
              f"{f' | {risk.compliance_scope}' if risk.compliance_scope else ''}")
        actors = ", ".join(risk.threat_actors)
        print(f"     Threat  : {actors or 'no matched threat actor'}")
        for factor in risk.factors:
            mark = "x" if factor.active else " "
            print(f"       [{mark}] {factor.label} (x{factor.multiplier}) — {factor.evidence}")
        print(f"     Why     : {risk.justification}")
        nist = risk.nist or {}
        control = nist.get("control")
        if control:
            print(f"     NIST    : {control['control_id']} — {control['control_title']} "
                  f"(distance {control['distance']:.3f}, source {nist['source_dataset']})")
        else:
            print("     NIST    : no control above relevance threshold")
        print(f"     Guidance ({nist.get('guidance_source', 'none')}): "
              f"{_text(nist.get('guidance'))[:300]}")
        print()


def main() -> int:
    """Headless run: `python backend.py --data-dir data --no-llm`."""
    import argparse

    parser = argparse.ArgumentParser(description="TawasolPay AI Cyber Risk Assistant (CLI)")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Path to the data pack")
    parser.add_argument("--top", type=int, default=5, help="How many risks to report")
    parser.add_argument("--no-llm", action="store_true", help="Skip Groq; show raw NIST prose")
    args = parser.parse_args()

    try:
        report = generate_risk_report(
            data_dir=args.data_dir, top_n=args.top, use_llm=not args.no_llm
        )
    except SchemaValidationError as exc:
        print(f"\nSTARTUP ERROR\n{exc}\n")
        return 2

    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
