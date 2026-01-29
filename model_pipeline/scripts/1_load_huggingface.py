from huggingface_hub import snapshot_download
from pathlib import Path
import shutil

DATASET_REPO = "martinjolif/football-ball-detection"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = PROJECT_ROOT / "data" / "football_ball"

print("Downloading dataset repository from HuggingFace...")

repo_path = Path(
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset"
    )
)

print(f"Dataset downloaded to: {repo_path}")

# Очистим целевую папку
if TARGET_DIR.exists():
    print("Removing existing dataset directory...")
    shutil.rmtree(TARGET_DIR)

TARGET_DIR.mkdir(parents=True)

print("Copying dataset files...")

# Копируем ВСЁ содержимое репозитория
for item in repo_path.iterdir():
    dest = TARGET_DIR / item.name
    if item.is_dir():
        shutil.copytree(item, dest)
    else:
        shutil.copy2(item, dest)

print("Dataset successfully prepared.")

