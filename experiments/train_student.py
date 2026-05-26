from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from slg.distill.student import DistillConfig, train_student
from slg.utils.reproducibility import set_global_seed


def run(args):
    set_global_seed(args.seed)

    run_path = PROJECT_ROOT / args.run_dir
    dataset_path = run_path / args.dataset_name
    output_path = run_path / args.output_name

    config = DistillConfig(
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        alpha_ce=args.alpha_ce,
        alpha_mse=args.alpha_mse,
    )

    _, history = train_student(
        dataset_path=dataset_path,
        output_path=output_path,
        config=config,
        seed=args.seed,
    )

    history_path = run_path / 'student_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    print('\nStudent saved:')
    print(output_path)
    print(history_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, default='runs/latest')
    parser.add_argument('--dataset-name', type=str, default='teacher_dataset.npz')
    parser.add_argument('--output-name', type=str, default='student_policy.pt')
    parser.add_argument('--hidden-dim', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--alpha-ce', type=float, default=1.0)
    parser.add_argument('--alpha-mse', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
