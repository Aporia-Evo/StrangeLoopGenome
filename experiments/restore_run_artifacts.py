from pathlib import Path
import argparse
import json
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact-name', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--artifact-root', default='artifacts')
    args = parser.parse_args()

    artifact_dir = PROJECT_ROOT / args.artifact_root / args.artifact_name
    run_dir = PROJECT_ROOT / args.run_dir

    if not artifact_dir.exists():
        raise FileNotFoundError(f'Missing artifact dir: {artifact_dir}')

    run_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for src in artifact_dir.iterdir():
        if src.is_file() and src.name != 'manifest.json':
            shutil.copy2(src, run_dir / src.name)
            copied.append(src.name)

    restore_manifest = {
        'artifact_dir': str(artifact_dir),
        'target_run_dir': str(run_dir),
        'copied': copied,
    }

    with open(run_dir / 'restore_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(restore_manifest, f, indent=2)

    print('Restored curated run artifacts')
    print('from:', artifact_dir)
    print('to:', run_dir)
    print('copied:', copied)


if __name__ == '__main__':
    main()
