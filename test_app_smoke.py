"""Headless smoke test: runs app.py through Streamlit's AppTest and fails on any exception."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.chroma_stub import install_stub  # noqa: E402

install_stub()

from streamlit.testing.v1 import AppTest  # noqa: E402


def main() -> int:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()

    if at.exception:
        for exception in at.exception:
            print("EXCEPTION:", exception.value)
        return 1

    print("initial render OK")
    print("  markdown blocks:", len(at.markdown))
    print("  expanders:", len(at.expander))
    print("  warnings:", [w.value for w in at.warning])
    print("  errors:", [e.value for e in at.error])

    at.selectbox[0].set_value("True").run()
    if at.exception:
        print("EXCEPTION after internet filter:", at.exception[0].value)
        return 1
    print("internet-facing filter OK")

    at.multiselect[0].set_value(["LOW"]).run()
    if at.exception:
        print("EXCEPTION after severity filter:", at.exception[0].value)
        return 1
    print("empty-result filter OK; warnings:", [w.value for w in at.warning])

    at.multiselect[0].set_value(["CRITICAL", "HIGH", "MEDIUM", "LOW"]).run()
    at.selectbox[0].set_value("Any").run()
    if at.exception:
        print("EXCEPTION after reset:", at.exception[0].value)
        return 1
    print("filter reset OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
