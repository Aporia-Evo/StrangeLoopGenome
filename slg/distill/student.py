from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


class StudentPolicy(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, output_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(obs)
            return int(torch.argmax(logits, dim=-1).item())


@dataclass
class DistillConfig:
    input_dim: int = 5
    hidden_dim: int = 16
    output_dim: int = 5
    batch_size: int = 256
    epochs: int = 40
    learning_rate: float = 1e-3
    validation_split: float = 0.1
    alpha_ce: float = 1.0
    alpha_mse: float = 0.25


def load_teacher_dataset(path):
    data = np.load(path, allow_pickle=True)
    observations = torch.tensor(data['observations'], dtype=torch.float32)
    actions = torch.tensor(data['actions'], dtype=torch.long)
    logits = torch.tensor(data['logits'], dtype=torch.float32)
    return observations, actions, logits


def make_loaders(observations, actions, logits, config, seed=0):
    dataset = TensorDataset(observations, actions, logits)
    val_size = int(len(dataset) * config.validation_split)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader


def compute_loss(student_logits, teacher_actions, teacher_logits, config):
    ce = F.cross_entropy(student_logits, teacher_actions)
    mse = F.mse_loss(student_logits, teacher_logits)
    return config.alpha_ce * ce + config.alpha_mse * mse, ce, mse


def evaluate_loader(model, loader, config, device):
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_mse = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():
        for obs, actions, teacher_logits in loader:
            obs = obs.to(device)
            actions = actions.to(device)
            teacher_logits = teacher_logits.to(device)

            student_logits = model(obs)
            loss, ce, mse = compute_loss(student_logits, actions, teacher_logits, config)

            total_loss += float(loss.item()) * len(obs)
            total_ce += float(ce.item()) * len(obs)
            total_mse += float(mse.item()) * len(obs)
            total_correct += int((student_logits.argmax(dim=-1) == actions).sum().item())
            total += len(obs)

    return {
        'loss': total_loss / max(1, total),
        'ce': total_ce / max(1, total),
        'mse': total_mse / max(1, total),
        'accuracy': total_correct / max(1, total),
    }


def train_student(dataset_path, output_path, config=None, seed=0, device=None):
    config = DistillConfig() if config is None else config
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    observations, actions, teacher_logits = load_teacher_dataset(dataset_path)
    train_loader, val_loader = make_loaders(observations, actions, teacher_logits, config, seed=seed)

    model = StudentPolicy(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []

    for epoch in range(config.epochs):
        model.train()
        for obs, batch_actions, batch_logits in train_loader:
            obs = obs.to(device)
            batch_actions = batch_actions.to(device)
            batch_logits = batch_logits.to(device)

            optimizer.zero_grad()
            student_logits = model(obs)
            loss, _, _ = compute_loss(student_logits, batch_actions, batch_logits, config)
            loss.backward()
            optimizer.step()

        train_metrics = evaluate_loader(model, train_loader, config, device)
        val_metrics = evaluate_loader(model, val_loader, config, device)
        row = {'epoch': epoch, 'train': train_metrics, 'val': val_metrics}
        history.append(row)

        print(
            f"epoch={epoch:03d} "
            f"train_acc={train_metrics['accuracy']:.3f} "
            f"val_acc={val_metrics['accuracy']:.3f} "
            f"val_loss={val_metrics['loss']:.4f}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            'model_state_dict': model.cpu().state_dict(),
            'config': asdict(config),
            'history': history,
            'dataset_path': str(dataset_path),
            'seed': seed,
        },
        output_path,
    )

    return model.cpu(), history


def load_student(path, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path, map_location=device)
    config = DistillConfig(**checkpoint['config'])
    model = StudentPolicy(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, config, checkpoint
