from pathlib import Path
import argparse
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd):
    print('\n$ ' + ' '.join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='runs/latest')
    parser.add_argument('--generations', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--teacher-seed', type=int, default=2000)
    parser.add_argument('--student-seed', type=int, default=0)
    parser.add_argument('--skip-evolution', action='store_true')
    parser.add_argument('--skip-feedforward-student', action='store_true')
    args = parser.parse_args()

    run_path = PROJECT_ROOT / args.run_dir
    top_genomes = run_path / 'top_genomes.pkl'

    if not args.skip_evolution or not top_genomes.exists():
        run_cmd([
            sys.executable,
            'experiments/train_recurrent_neat.py',
            '--generations', str(args.generations),
            '--seed', str(args.seed),
            '--output-dir', args.run_dir,
            '--top-k', '10',
        ])

    run_cmd([
        sys.executable,
        'experiments/benchmark_best_recurrent.py',
        '--run-dir', args.run_dir,
        '--num-seeds', '100',
        '--seed', '0',
    ])

    run_cmd([
        sys.executable,
        'experiments/build_teacher_dataset.py',
        '--run-dir', args.run_dir,
        '--output-name', 'teacher_top1_clean.npz',
        '--num-genomes', '1',
        '--seeds-per-genome', '200',
        '--min-foods', '10',
        '--max-wall-hits', '3',
        '--seed', str(args.teacher_seed),
    ])

    if not args.skip_feedforward_student:
        run_cmd([
            sys.executable,
            'experiments/train_student.py',
            '--run-dir', args.run_dir,
            '--dataset-name', 'teacher_top1_clean.npz',
            '--output-name', 'student_top1_clean.pt',
            '--epochs', '80',
            '--hidden-dim', '32',
            '--seed', str(args.student_seed),
        ])

        run_cmd([
            sys.executable,
            'experiments/benchmark_student.py',
            '--run-dir', args.run_dir,
            '--student-name', 'student_top1_clean.pt',
            '--num-seeds', '100',
            '--seed', '0',
        ])

    run_cmd([
        sys.executable,
        'experiments/train_recurrent_student.py',
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
        sys.executable,
        'experiments/benchmark_recurrent_student.py',
        '--run-dir', args.run_dir,
        '--student-name', 'recurrent_student.pt',
        '--num-seeds', '100',
    ])


if __name__ == '__main__':
    main()
