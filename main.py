from pathlib import Path


def main() -> None:
    """Entry point for the time series basics project."""
    project_root = Path(__file__).resolve().parent

    print("=== Time Series Basics with Python ===")
    print(f"Project root : {project_root}")
    print()
    print("Available example scripts (see ./scripts):")
    print("  - p_01_simple_ma.py      : Simple moving average")
    print("  - p_02_stationarity.py   : Stationarity check")
    print("  - p_03_ec_sales_arima.py : ARIMA on EC sales data")
    print()
    print("Tip:")
    print("  uv run python scripts/p_01_simple_ma.py")
    print("  uv run python scripts/p_02_stationarity.py")
    print("  uv run python scripts/p_03_ec_sales_arima.py")


if __name__ == "__main__":
    main()
