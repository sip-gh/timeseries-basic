import subprocess
from pathlib import Path


def test_main_list_runs() -> None:
    project_root = Path(__file__).resolve().parent.parent

    result = subprocess.run(
        ["uv", "run", "python", "main.py", "list"],
        cwd=project_root,
    )

    assert result.returncode == 0
