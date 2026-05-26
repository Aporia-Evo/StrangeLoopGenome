from pathlib import Path
import argparse
import json
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd, allow_fail=False):
    print('\n$ ' + ' '.join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0 and not allow_fail:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result.returncode


def benchmark_foods(run_dir):
    path = PROJECT_ROOT / run_dir / 'benchmark_summary.json'
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    return float(summary['neat_best']['foods']['mean'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='runs/latest')
    parser.add_argument('--generations', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--teacher-seed', type=int, default=2000)
    parser.add_argument('--student-seed', type=int, default=0)
    parser.add_argument('--min-teacher-foods', type=float, default=10.0)
    parser.add_argument('--min-foods', type=int, default=10)
    parser.add_argument('--max-wall-hits', type=int, default=3)
    parser.add_argument('--skip-evolution', action='store_true')
    parser.add_argument('--skip-feedforward-student', action='store_true')
    args = parser.parse_args()

    run_path = PROJECT_ROOT / args.run_dir
    top_genomes = run_path / 'top_genomes.pkl'

    if not args.skip_evolution or not top_genomes.exists():
        run_cmd([
            sys.executable, 'experiments/train_recurrent_neat.py',
            '--generations', str(args.generations),
            '--seed', str(args.seed),
            '--output-dir', args.run_dir,
            '--top-k', '10',
        ])

    run_cmd([
        sys.executable, 'experiments/benchmark_best_recurrent.py',
        '--run-dir', args.run_dir,
        '--num-seeds', '100',
        '--seed', '0',
    ])

    foods = benchmark_foods(args.run_dir)
    if foods is not None and foods < args.min_teacher_foods:
        raise RuntimeError(
            f'NEAT teacher is too weak for clean distillation: foods={foods:.3f}.\n'
            'Try more generations or a seed sweep, for example:\n'
            '  python experiments/sweep_recurrent_neat.py --generations 80 --seeds 0,1,2,3,4,5,42,100\n'
            'Then run this pipeline with --skip-evolution --run-dir <best_run_dir>.'
        )

    run_cmd([
        sys.executable, 'experiments/build_teacher_dataset.py',
        '--run-dir', args.run_dir,
        '--output-name', 'teacher_top1_clean.npz',
        '--num-genomes', '1',
        '--seeds-per-genome', '200',
        '--min-foods', str(args.min_foods),
        '--max-wall-hits', str(args.max_wall_hits),
        '--seed', str(args.teacher_seed),
    ])

    if not args.skip_feedforward_student:
        run_cmd([
            sys.executable, 'experiments/train_student.py',
            '--run-dir', args.run_dir,
            '--dataset-name', 'teacher_top1_clean.npz',
            '--output-name', 'student_top1_clean.pt',
            '--epochs', '80',
            '--hidden-dim', '32',
            '--seed', str(args.student_seed),
        ])
        run_cmd([
            sys.executable, 'experiments/benchmark_student.py',
            '--run-dir', args.run_dir,
            '--student-name', 'student_top1_clean.pt',
            '--num-seeds', '100',
            '--seed', '0',
        ])

    run_cmd([
        sys.executable, 'experiments/train_recurrent_student.py',
        '--run-dir', args.run_dir,
        '--dataset-name', 'teacher_top1_clean.npz',
        '--output-name', 'recurrent_student.pt',
        '--epochs', '60',
        '--hidden-dim', '32',
        '--seq-len', '16',
        '--stride', '8',
        '--seed', str(args.student_seed),
    ])
    run_cmd([
        sys.executable, 'experiments/benchmark_recurrent_student.py',
        '--run-dir', args.run_dir,
        '--student-name', 'recurrent_student.pt',
        '--num-seeds', '100',
    ])


if __name__ == '__main__':
    main()
