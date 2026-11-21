"""Convenience entry point: streamlit run app.py."""

import sys
from pathlib import Path

# Permit source-tree execution while retaining the standard editable-install workflow.
source_dir = Path(__file__).resolve().parent / "src"
if str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir))

from biomed_video_fc.ui.streamlit_app import main  # noqa: E402

if __name__ == "__main__":
    main()
