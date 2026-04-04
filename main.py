# import argparse  # コマンドライン引数を扱う標準ライブラリ
# from pathlib import Path
# import subprocess  # 別のPythonスクリプトを別プロセスとして実行する標準ライブラリ


# # ランチャーが対応するスクリプト一覧（キー: CLIで指定する名前）
# SCRIPT_MAP = {
#     "p01": "p_01_simple_ma.py",
#     "p02": "p_02_stationarity.py",
#     "p03": "p_03_ec_sales_arima.py",
# }

# def main() -> None:
#     project_root = Path(__file__)
#     breakpoint()
#     # .resolve()

#     # 1. 「どういう引数を受け取るか」を定義する
#     parser = argparse.ArgumentParser(
#         description="Minimal argparse example.",
#     )
#     parser.add_argument(
#         "target",
#         nargs="?",  # 省略可能
#         choices=["p01", "p02", "p03", "list"],  # 許可する値を限定
#         help="Which example to run (p01, p02, p03) or 'list' to show options.",
#     )

#     # 2. 実際のコマンドラインから引数を読み取る
#     args = parser.parse_args()

#     # 3. パースされた結果を確認
#     print(f"args      : {args!r}")
#     print(f"args.target : {args.target!r}")


# if __name__ == "__main__":
#     main()
