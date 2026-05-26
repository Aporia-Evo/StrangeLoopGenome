from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from slg.distill.recurrent_student import RecurrentDistillConfig, train_recurrent_student
from slg.utils.reproducibility import set_global_seed


def require_dataset(run_path, dataset_name):
    dataset_path = run_path / dataset_name
    if dataset_path.exists():
        return dataset_path

    available = sorted(p.name for p in run_path.glob('*.npz')) if run_path.exists() else []
    available_text = '\n'.join(f'  - {name}' for name in available) or '  none found'

    raise FileNotFoundError(
        f'Missing dataset: {dataset_path}\n\n'
        'Teacher datasets are local run artifacts and are not stored in git.\n'
        'Build one first, for example:\n\n'
        '  python experiments/build_teacher_dataset.py --run-dir runs/latest '
        '--output-name teacher_top1_clean.npz --num-genomes 1 '
        '--seeds-per-genome 200 --min-foods 10 --max-wall-hits 3 --seed 2000\n\n'
        f'Available .npz files in {run_path}:\n{available_text}'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='runs/latest')
    parser.add_argument('--dataset-name', default='teacher_top1_clean.npz')
    parser.add_argument('--output-name', default='recurrent_student.pt')
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    set_global_seed(args.seed)
    run_path = PROJECT_ROOT / args.run_dir
    dataset_path = require_dataset(run_path, args.dataset_name)

    config = RecurrentDistillConfig(
        hidden_dim=args.hidden_dim,
        seq_len=args.seq_len,
        stride=args.stride,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    _, history = train_recurrent_student(
        dataset_path=dataset_path,
        output_path=run_path / args.output_name,
        config=config,
        seed=args.seed,
    )

    history_path = run_path / 'recurrent_student_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    print('\nRecurrent student saved:')
    print(run_path / args.output_name)
    print(history_path)


if __name__ == '__main__':
    main()
