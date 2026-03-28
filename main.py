from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent
    print(f"project root: {project_root}")
    print("timeseries-basic skeleton is ready")

if __name__ == "__main__":
    main()
