import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_MAP = {
    "p01": "p_01_simple_ma.py",
    "p02": "p_02_stationarity.py",
    "p03": "p_03_ec_sales_arima.py",
}


def main() -> None:
    this_file = Path(__file__).resolve()
    project_root = this_file.parent
    scripts_dir = project_root / "scripts"

    parser = argparse.ArgumentParser(
        description="Run small timeseries scripts from one entry point.",
    )
    parser.add_argument(
        "target",
        nargs="?",
        choices=["p01", "p02", "p03", "list"],
        help="Which script to run.",
    )

    args = parser.parse_args()

    print(f"this_file     : {this_file}")
    print(f"project_root  : {project_root}")
    print(f"args          : {args!r}")
    print(f"args.target   : {args.target!r}")

    if args.target is None or args.target == "list":
        print("available targets:")
        for key, filename in SCRIPT_MAP.items():
            print(f"  {key}: {filename}")
        return

    selected_file = SCRIPT_MAP[args.target]
    script_path = scripts_dir / selected_file

    print(f"you selected: {args.target}")
    print(f"script file : {selected_file}")
    print(f"script path : {script_path}")

    if not script_path.exists():
        print(f"ERROR: script not found: {script_path}")
        sys.exit(1)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=project_root,
    )

    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
