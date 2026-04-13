# experiments/watch_progress.py
"""
SSH接続時にリモートから進捗を確認するスクリプト。
logs/experiment_progress.log を監視して
新しい行が追加されるたびに表示する。

使用方法:
  python experiments/watch_progress.py
  python experiments/watch_progress.py --tail 50  # 最新50行から表示
"""
import time
import argparse
from pathlib import Path


def watch(log_path: str = "logs/experiment_progress.log",
          tail: int = 20,
          interval: float = 2.0):
    """
    ログファイルを監視してリアルタイムで表示する。

    tail: 起動時に表示する末尾の行数
    interval: 更新間隔（秒）
    """
    path = Path(log_path)

    # ファイルが存在するまで待機
    print(f"ログファイルを待機中: {log_path}")
    while not path.exists():
        time.sleep(1)

    # 末尾N行を表示してから監視開始
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines[-tail:]:
        print(line)

    # 新しい行を監視
    with open(path, encoding="utf-8") as f:
        f.seek(0, 2)  # ファイル末尾にシーク
        print(f"\n--- リアルタイム監視中 (Ctrl+C で終了) ---\n")
        while True:
            line = f.readline()
            if line:
                print(line, end="", flush=True)
            else:
                time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",
        default="logs/experiment_progress.log")
    parser.add_argument("--tail", type=int, default=20)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    try:
        watch(args.log, args.tail, args.interval)
    except KeyboardInterrupt:
        print("\n監視を終了しました。")
