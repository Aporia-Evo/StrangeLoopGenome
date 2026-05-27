from pathlib import Path
import argparse
import json
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_FILES = [
    'best_genome.pkl',
    'top_genomes.pkl',
    'archive_summary.json',
    'benchmark_summary.json',
    'benchmark_sparse.csv',
    'teacher_dataset_summary.json',
    'teacher_top1_clean.npz',
    'student_top1_clean.pt',
    'student_history.json',
    'student_benchmark_summary.json',
    'student_benchmark_sparse.csv',
    'recurrent_student.pt',
    'recurrent_student_history.json',
    'run_config.json',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--artifact-root', default='artifacts')
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / args.run_dir
    out_dir = PROJECT_ROOT / args.artifact_root / args.name

    if not run_dir.exists():
        raise FileNotFoundError(f'Missing run dir: {run_dir}')

    out_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    missing = []

    for filename in DEFAULT_FILES:
        src = run_dir / filename
        dst = out_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(filename)
        else:
            missing.append(filename)

    manifest = {
        'source_run_dir': str(run_dir),
        'artifact_dir': str(out_dir),
        'copied': copied,
        'missing': missing,
    }

    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print('Exported curated run artifacts')
    print('from:', run_dir)
    print('to:', out_dir)
    print('copied:', copied)
    print('missing:', missing)


if __name__ == '__main__':
    main()
