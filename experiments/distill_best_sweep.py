from pathlib import Path
import argparse
import json
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep-summary', default='runs/sweep/sweep_summary.json')
    p.add_argument('--min-teacher-foods', type=float, default=10.0)
    p.add_argument('--skip-feedforward-student', action='store_true')
    args = p.parse_args()

    summary_path = PROJECT_ROOT / args.sweep_summary
    if not summary_path.exists():
        raise FileNotFoundError(f'Missing sweep summary: {summary_path}')

    with open(summary_path, 'r', encoding='utf-8') as f:
        runs = json.load(f)

    valid = [r for r in runs if r.get('foods') is not None]
    if not valid:
        raise RuntimeError('No valid sweep runs found.')

    best = max(valid, key=lambda r: r['foods'])
    print('Best sweep run:')
    print(best)

    cmd = [
        sys.executable,
        'experiments/run_recurrent_distill_pipeline.py',
        '--run-dir', best['run_dir'],
        '--skip-evolution',
        '--min-teacher-foods', str(args.min_teacher_foods),
    ]

    if args.skip_feedforward_student:
        cmd.append('--skip-feedforward-student')

    print('\n$ ' + ' '.join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == '__main__':
    main()
