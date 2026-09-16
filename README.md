# TawasolPay — AI Cyber Risk Assistant

A working risk triage system for TawasolPay, a Series B fintech in Dubai. It joins the
asset inventory, the open vulnerability list, threat intelligence feeds and business
service context into one ranked, explainable risk picture, then retrieves the relevant
remediation control from NIST SP 800-53 Rev. 5 for each of the top five risks.

The design principle is a hard split of responsibilities:

> **Deterministic Python decides what matters. Retrieval decides what the guidance says.
> The language model only rewrites prose that retrieval already returned.**

The model cannot see, compute, adjust, or reorder a single risk score. Remove the API key
and the system still produces the same ranked Top 5 — it just shows the NIST control text
verbatim instead of an executive rewrite.

---

## 1. What it does

**Prioritises risks intelligently.** Every live (Asset + CVE) pair is scored with an
explicit multiplier chain covering internet exposure, active exploitation, ransomware
association, regional campaign targeting, business criticality and missing EDR coverage.
CVSS is the base, never the answer. In the reference data a CVSS 10.0 Jenkins flaw on an
internal dev server ranks **fifth** while a CVSS 8.2 finding on an internet-exposed
payment gateway with an active campaign against it ranks **second**.

**Retrieves real remediation guidance.** Each top risk drives a semantic query against an
embedded copy of the NIST SP 800-53 Rev. 5 catalogue. The retrieved control, its family,
its cosine distance and the other candidates considered are all displayed. If nothing
clears the relevance threshold, the system says so rather than guessing.

**Produces readable output.** An executive dashboard with metric cards, severity and
driver charts, and one expandable card per risk showing the asset, the vulnerability, the
matched threat intelligence, the business service at risk, the exact multipliers applied,
a plain-English justification sentence, and the recommended NIST control.

---

## 2. Architecture

```
                         ┌──────────────────────────────────────────┐
                         │            data/  (the data pack)        │
                         ├──────────────────────────────────────────┤
  STRUCTURED RECORDS ◄───┤ assets.csv               60 rows         │
  queried with filters   │ vulnerabilities.csv     114 rows         │
  and joins              │ threat_intelligence.csv  40 rows         │
                         │ business_services.csv    20 rows         │
                         │ remediation_guidance.csv 30 rows         │
                         │ known_exploited_vulnerabilities.json     │
                         ├──────────────────────────────────────────┤
  PARSED PROSE ◄─────────┤ synthetic_threat_report.md   (regex)     │
                         ├──────────────────────────────────────────┤
  EMBEDDED PROSE ◄───────┤ NIST_SP-800-53_rev5_catalog_load.csv     │
                         └──────────────────────────────────────────┘
                                          │
                                          ▼
                    ┌───────────────────────────────────────────┐
                    │  SCHEMA MAPPING AND VALIDATION LAYER      │
                    │  alias resolution → canonical names       │
                    │  missing required field = startup error   │
                    └───────────────────────────────────────────┘
                                          │
                                          ▼
                    ┌───────────────────────────────────────────┐
                    │  CVE NORMALISATION                        │
                    │  display form  ← zero-padded, canonical   │
                    │  match key     ← zero-stripped, for joins │
                    │  malformed     ← reported, never guessed  │
                    └───────────────────────────────────────────┘
                                          │
                                          ▼
      ┌─────────────────────────────────────────────────────────────────────┐
      │  RELATIONAL JOIN (pandas, exact keys)                               │
      │                                                                     │
      │   Asset ──asset_id──► Vulnerability ──cve_key──► Threat Intel       │
      │     │                       │                                      │
      │     │                       └──cve_key──► CISA KEV catalogue        │
      │     │                       └──cve_key──► MDR advisory campaigns    │
      │     └──business_service──► Business Service                         │
      └─────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
      ┌─────────────────────────────────────────────────────────────────────┐
      │  DETERMINISTIC RISK ENGINE            no LLM, no randomness         │
      │                                                                     │
      │  composite = CVSS                                                   │
      │            × internet exposure      (2.5 facing / 1.0 internal)     │
      │            × KEV or ransomware      (2.0 match  / 1.0 none)         │
      │            × regional campaign      (3.0 match  / 1.0 none)         │
      │            × business criticality   (2.5 / 1.8 / 1.2 / 1.0)         │
      │            × EDR coverage           (1.5 absent / 1.0 present)      │
      │                                                                     │
      │  no cap, no normalisation → full dynamic range preserved            │
      │  rank by score desc, CVSS desc, CVE asc, vuln_id asc                │
      │  every multiplier retained on the risk object for lineage           │
      └─────────────────────────────────────────────────────────────────────┘
                                          │
                        Top 5 ranked risks │ (ranking already final)
                                          ▼
      ┌─────────────────────────────────────────────────────────────────────┐
      │  NIST RETRIEVAL (RAG)                                               │
      │                                                                     │
      │  query = finding + component + active drivers + compliance scope    │
      │          + one-line hint from remediation_guidance.csv              │
      │                        │                                            │
      │                        ▼                                            │
      │  ChromaDB · all-MiniLM-L6-v2 · cosine · one chunk per control       │
      │                        │                                            │
      │              distance ≤ 0.80 ?                                      │
      │                 ╱           ╲                                       │
      │              yes             no                                     │
      │               │               │                                     │
      │     retrieved control    "No sufficiently relevant control"         │
      │               │           + closest near-miss distance              │
      │               ▼               (no LLM call is made)                 │
      │  ┌──────────────────────────────────────┐                           │
      │  │ GROQ llama-3.3-70b-versatile         │                           │
      │  │ input:  risk context + control prose │                           │
      │  │ output: 2-3 executive sentences      │                           │
      │  │ guardrails: no other controls, no    │                           │
      │  │   scores, no offensive actions       │                           │
      │  │ any failure → verbatim control text  │                           │
      │  └──────────────────────────────────────┘                           │
      └─────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
      ┌─────────────────────────────────────────────────────────────────────┐
      │  STREAMLIT EXECUTIVE DASHBOARD            render only, never score  │
      │  metric cards · severity chart · driver chart · sidebar filters     │
      │  Top 5 cards with driver panel, justification, NIST control         │
      │  full risk register · data quality diagnostics · advisory footer    │
      └─────────────────────────────────────────────────────────────────────┘
```

### Module layout

```
backend.py    ingestion, schema validation, CVE normalisation, joins,
              risk engine, NIST RAG engine, Groq wrapper, CLI entry point
app.py        Streamlit dashboard — presentation only
tests/        headless smoke tests (app render, RAG integration, determinism)
data/         the data pack (not committed; see Setup)
```

---

## 3. Setup

### Prerequisites

Python 3.11 or newer.

### Install

```bash
git clone <your-repo-url>
cd tawasolpay-cyber-risk-assistant

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt`:

```
streamlit>=1.40
pandas>=2.1
chromadb>=0.5
sentence-transformers>=3.0
groq>=0.11
plotly>=5.20
```

Plotly is optional — without it the charts fall back to `st.bar_chart`.

### Assemble the data pack

Place the five provided CSVs and the advisory in `data/`:

```
data/assets.csv
data/vulnerabilities.csv
data/threat_intelligence.csv
data/business_services.csv
data/remediation_guidance.csv
data/synthetic_threat_report.md
```

Then fetch the two public reference documents:

```bash
# CISA Known Exploited Vulnerabilities catalogue
curl -L -o data/known_exploited_vulnerabilities.json \
  https://raw.githubusercontent.com/cisagov/kev-data/main/known_exploited_vulnerabilities.json

# NIST SP 800-53 Rev. 5 control catalogue (CSV export)
# Download from https://csrc.nist.gov/projects/risk-management/sp800-53-controls/downloads
# and save as:
#   data/NIST_SP-800-53_rev5_catalog_load.csv
```

The NIST CSV is behind a download page rather than a stable direct link, so it is fetched
manually. The KEV loader accepts either the official `{"vulnerabilities": [...]}` envelope
or a bare JSON array.

### Configure the LLM (optional)

```bash
export GROQ_API_KEY="gsk_..."          # Windows: set GROQ_API_KEY=gsk_...
```

Get a free key at <https://console.groq.com>. With no key the system runs fully and
displays the retrieved NIST control text verbatim; nothing crashes and no score changes.

### Run

```bash
streamlit run app.py
```

Then open <http://localhost:8501>.

A headless run is useful for verifying the pipeline without the UI:

```bash
python backend.py --data-dir data --no-llm
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | unset | Enables executive summarisation. Absent = verbatim NIST text. |
| `TP_DATA_DIR` | `data` | Default data pack directory shown in the sidebar. |
| `TP_NIST_DISTANCE_THRESHOLD` | `0.80` | Cosine-distance ceiling for accepting a retrieved control. Lower tightens precision. |
| `TP_CHROMA_PATH` | `.chroma_nist` | Where the persistent vector index lives. |
| `TP_LOG_LEVEL` | `INFO` | Backend log verbosity. |

### Deploy

**Streamlit Community Cloud** is the shortest path. Push the repo, point the app at
`app.py`, add `GROQ_API_KEY` under *Secrets*, and commit `data/` so the container can
read it. The first boot downloads the embedding model (roughly 90 MB) and builds the
vector index; the index is then cached on disk and keyed by a content hash of the NIST
CSV, so later boots are fast and a changed catalogue re-indexes automatically.

`.streamlit/config.toml` for the dark theme:

```toml
[theme]
base = "dark"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#161b26"
textColor = "#e6edf7"
```

---

## 4. Scoring reference

| Factor | Active | Inactive | Source |
|---|---|---|---|
| Internet exposure | **2.5** | 1.0 | `assets.internet_exposed` OR `vulnerabilities.asset_exposure` |
| Known exploited / ransomware | **2.0** | 1.0 | CISA KEV presence, `knownRansomwareCampaignUse`, or `threat_intelligence.ransomware_association` |
| Regional MDR campaign | **3.0** | 1.0 | CVE named in `synthetic_threat_report.md` |
| Business criticality | **2.5 / 1.8 / 1.2** | 1.0 | Higher of `assets.criticality` and `business_services.revenue_impact` |
| Missing EDR | **1.5** | 1.0 | `assets.edr_installed` |

Severity bands are set against the composite range (theoretical maximum 562.5, not 10):
**CRITICAL** ≥ 150, **HIGH** ≥ 60, **MEDIUM** ≥ 20, **LOW** below 20.

Two deliberate scoring decisions worth stating:

- **Criticality takes the higher of asset and service.** A Medium-criticality asset
  carrying a Critical-revenue service is scored Critical. Inheriting only the asset label
  would let a mislabelled host hide a revenue-critical exposure.
- **Unknown EDR status scores as missing (1.5).** A blank field is not evidence of
  coverage. Scoring it as present would quietly reward incomplete inventory data.

Vulnerabilities with a status of Closed, Remediated, Resolved, Fixed, Mitigated or False
Positive are excluded from ranking and counted in the diagnostics panel, so a fixed
finding can never occupy a board-level slot.

---

## 5. Supporting question 1 — the data split

**Queried as structured records:** the asset inventory, vulnerability list, threat
intelligence feed, business service catalogue and the CISA KEV catalogue. These are
relational tables with stable keys, and every question I ask of them is an exact-match
join or a boolean filter: *which asset does `A-1001` refer to, is `CVE-2024-21762` in the
KEV catalogue, what is the revenue impact of Customer Login*. Semantic similarity is
actively harmful here — `CVE-2024-21762` and `CVE-2024-21763` are near-identical strings
describing unrelated flaws, and any embedding would place them next to each other. These
joins also need to be exact and auditable to defend a ranking to a board, which rules out
a nearest-neighbour lookup that returns "close enough".

**Embedded as vectors:** only the NIST SP 800-53 Rev. 5 control catalogue. It is 1,000-plus
prose controls with no key that maps to a vulnerability, and the question I ask of it is
inherently semantic: *which control governs an unpatched internet-facing service with a
vendor fix available*. Nothing in the finding says "SI-2" — the mapping lives in the
meaning of the control text, so embeddings are the right tool. I chunk one control per
document (identifier, title, control text and discussion together) so a retrieved hit is
always a complete, citable control rather than a fragment.

The MDR advisory sits deliberately in neither camp. It is prose, but I only need two
things from it: which CVEs are named, and by which campaign. That is an extraction task
with an exact answer, so it is parsed with regular expressions. Embedding it would turn a
verifiable fact into a similarity score, and the x3.0 campaign multiplier is far too
influential to rest on a fuzzy match.

`remediation_guidance.csv` is used as retrieval query enrichment only. Its one-liners
sharpen the vector query but are never displayed as the remediation answer, because the
brief is explicit that the CSV is a hint and NIST is the source.

---

## 6. Supporting question 2 — three ways this produces wrong output

### Failure mode 1 — CVE string normalisation mismatches

**Mechanism.** Every high-impact multiplier in the engine fires off a string join on a CVE
identifier. The same vulnerability arrives written differently across four sources:
`CVE-2024-21762` in the vulnerability CSV, `cve-2024-21762` from an exported feed,
`CVE-2024-0021` versus `CVE-2024-21` where a producer trimmed or padded the sequence, and
unicode en-dashes pasted out of a PDF advisory. A naive `==` join silently fails on all of
these and the CVE looks absent from the KEV catalogue.

**Business impact.** Silent false negatives on the highest-weighted factors. A
ransomware-associated, actively exploited, internet-facing CVE loses its x2.0 KEV
multiplier and its x3.0 campaign multiplier — a 6x under-score that drops it out of the
Top 5 entirely. The CISO briefs the board on the wrong five risks and nobody sees an
error, because the pipeline reports success.

**Detection.** `normalise_cve` splits every identifier into a zero-padded `display` form
and a zero-stripped `match_key`, so `CVE-2024-0021` and `CVE-2024-21` collapse onto the
same join key. Unicode dashes, underscores, space separators and case are normalised
first. Anything that parses as neither a CVE nor a synthetic CVE is recorded in
`DataQuality.malformed_cves` with its source file, record ID, raw value and the reason it
failed, and surfaced in the diagnostics panel — never dropped silently. The panel also
reports how many scored risks got a KEV match, which is a fast sanity check: a plausible
number that suddenly reads zero means the join broke, not that the estate got safer.

**Mitigation in place.** Strict parsing where laxity would be dangerous: the
`vulnerabilities.cve` column rejects anything that is not a real CVE, so a value like
`NOT-A-CVE!!` is reported as malformed rather than laundered into a usable key by
stripping punctuation. Control-style identifiers such as `CICD-SYN-001` are accepted only
in `threat_intelligence.matched_cve_or_control`, where they legitimately belong. Next step
would be a reconciliation report that lists every environment CVE absent from the KEV
catalogue, so an analyst reviews the gap rather than inferring safety from silence.

### Failure mode 2 — RAG semantic drift

**Mechanism.** Vector search always returns its nearest neighbour, and "nearest" is not
"correct". NIST families overlap heavily by design: a query about an unpatched
internet-facing service sits close to SI-2 (Flaw Remediation), RA-5 (Vulnerability
Monitoring), SI-4 (System Monitoring), SC-7 (Boundary Protection) and CM-3 (Configuration
Change Control) all at once. Enhancement controls make it worse — `SI-2(3)` is textually
almost identical to `SI-2` but far narrower in scope. Push a long, driver-stuffed query at
a 384-dimension MiniLM model and the embedding blurs, so the top hit can drift to a
monitoring control when the actual remediation is to patch.

**Business impact.** Plausible, well-written, confidently-presented guidance that points
at the wrong obligation. Recommending SI-4 monitoring for a flaw whose real fix is the
vendor patch delays remediation while the team builds detection for an attack they could
have prevented. Because the output reads authoritative and cites a genuine control ID, it
is unlikely to be challenged.

**Detection.** Every retrieval shows its cosine distance, its control family and the other
candidates that were considered, so a reviewer can see when the top two are effectively
tied — the signature of drift. A hard threshold (default 0.80, tunable via
`TP_NIST_DISTANCE_THRESHOLD`) rejects weak matches outright: the card then reads *"No
sufficiently relevant NIST control retrieved"* and reports the closest near-miss and its
distance, rather than presenting a guess. When retrieval declines, **no LLM call is made
at all**, which removes the path by which a weak hit becomes fluent prose.

**Mitigation in place.** The model is constrained to rewriting only the single retrieved
control and is instructed not to mention, invent or imply any other, with temperature at
0.1. The control ID, title, family, distance and full source text are all displayed
alongside the summary, so a reader can check the rewrite against the original in one
click. Queries are seeded with the matched hint from `remediation_guidance.csv` to anchor
them on the finding type rather than drifting toward generic security language. The next
step would be a curated finding-type-to-control-family map used as a metadata filter
before the vector search, narrowing the candidate pool to the right family and letting
embeddings choose only within it — plus a small labelled eval set to measure retrieval
precision instead of trusting it.

### Failure mode 3 — unmapped asset business context

**Mechanism.** The join from asset to business service is a name match on a free-text
field. The brief states that some assets are stale and some have no assigned owner, and
the reference data contains an asset pointing at a service absent from
`business_services.csv`. When that lookup misses, the criticality multiplier has no
revenue-impact signal to work with. A default-to-Medium implementation would quietly apply
x1.2 where x2.5 belongs — an instant 2x under-score on exactly the assets whose ownership
records are worst maintained.

**Business impact.** Systematic under-ranking that correlates with poor data hygiene, so
the risks most likely to be forgotten are also the ones the system most reliably hides.
An unmapped payment-adjacent host scores as a mid-tier internal system, and a Top 5 built
on it looks clean while the real exposure sits at rank fifteen. This is the most dangerous
of the three failure modes because it is invisible in the output — there is no error, just
a smaller number.

**Detection.** The engine takes the **higher** of asset criticality and service revenue
impact rather than trusting either alone, so a missing service mapping degrades to the
asset's own label instead of collapsing to a default. Every unmapped service name is
collected in `DataQuality.unmapped_business_services`, every ownerless asset in
`assets_without_owner`, and every asset unseen for more than 30 days in `stale_assets` —
all three rendered in the diagnostics panel with counts on the metric cards. Each risk
card states which criticality inputs were available; when the service is undefined the
card says so explicitly rather than presenting a derived number as fact.

**Mitigation in place.** Service keys are normalised on case and whitespace before
matching, and unknown EDR status scores as missing rather than present, so incomplete data
never scores more favourably than complete data. The structural fix — and the reason
failure mode 3 leads directly into the improvement below — is to stop treating business
context as a single-hop attribute lookup and model it as a dependency graph, so a host
with no service label still inherits criticality through what depends on it.

---

## 7. Supporting question 3 — the one thing I would change

I would replace the single-hop business-service lookup with a **directed acyclic graph of
service dependencies and propagate risk along it**. Today criticality is an attribute
read off one row: an asset points at one service, and that service's revenue impact feeds
one multiplier. That model cannot see the thing that actually matters in a payment
platform, which is that `business_services.csv` already ships a `depends_on` column
describing a real topology. Customer Login depends on Identity Verification; Payment
Processing depends on the Payment API. A compromise of a "Medium" internal identity host
is not a Medium risk if the customer payment flow cannot complete without it — the
blast radius, not the label on the asset, is the risk.

The change is to build the DAG from `depends_on` at load time and score along paths
rather than at points:

```
  Asset ──► Vulnerability ──► Microservice ──► Upstream Payment API ──► Revenue
 (A-1003)  (CVE-2024-23897)  (Build Pipeline)   (Payment Processing)   (Critical)
    │                                                                      │
    └──────── effective criticality inherited from the worst reachable ─────┘
              downstream service, attenuated per hop, cycles broken by
              topological sort
```

Concretely: topologically sort the service graph, compute each asset's effective
criticality as the maximum over all services reachable downstream (with a per-hop
attenuation factor so a four-hop dependency does not carry the same weight as a direct
one), and use that in place of the current single-service lookup. The same traversal gives
two things the current system cannot produce. First, it fixes failure mode 3 structurally
rather than defensively: an asset with a missing or mislabelled service still inherits
criticality through what depends on it, so the under-scoring correlated with bad data
hygiene disappears instead of merely being reported. Second, it changes what the
justification sentence can say — from *"this supports Build Pipeline, a Medium-criticality
service"* to *"this is two hops upstream of Payment Processing, which carries a
one-hour RTO and PCI DSS obligations"*. That second sentence is the one a board
actually needs, because it names the consequence rather than the component.

I chose this over the alternatives deliberately. Better retrieval evaluation would raise
confidence in the guidance, and a proper finding-type-to-control-family filter would cut
semantic drift, but both improve the *second half* of each answer. The dependency graph
changes which risks reach the board at all, and a wrong Top 5 cannot be rescued by
excellent remediation advice attached to it.

---

## 8. Verification

```bash
# deterministic pipeline, no UI, no LLM
python backend.py --data-dir data --no-llm

# headless dashboard render plus filter interactions
python tests/test_app_smoke.py
```

The smoke test substitutes an in-process stub for ChromaDB so the retrieval integration
can be exercised where `torch` cannot be installed. Bag-of-words similarity is not
representative of real embedding distances, so the stub validates the plumbing and the
threshold behaviour, not retrieval quality.

Behaviours covered by the checks that shipped with this build:

- Identical ranking across repeated runs on the same data (determinism).
- A missing required column raises a startup error naming the file, the field, its
  accepted aliases and the columns actually present.
- A missing data directory fails with an actionable message.
- Lowercase CVEs join correctly; `NOT-A-CVE!!` is reported as malformed, not accepted.
- Vulnerabilities on unknown asset IDs and closed findings are excluded and counted.
- Retrieval below the distance threshold returns nothing and triggers no LLM call.
- Filters that produce an empty result set render a prompt rather than an exception.

---

## 9. Scope and safety

This system is **advisory only**. It reads files and documents. It does not scan, connect
to production, change configuration, or execute remediation of any kind. Risk scores and
rankings come from deterministic Python; the language model only rewrites NIST control
text that retrieval already returned, and cannot alter a score. Validate every
recommendation through change management before acting on it.

All data in this repository is synthetic and generated for a hiring assessment. The threat
actors, campaigns and IOCs are fictional and must not be used for operational security
decisions.
