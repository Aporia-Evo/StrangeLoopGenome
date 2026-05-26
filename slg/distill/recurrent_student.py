from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


class RecurrentStudentPolicy(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=5):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.hidden = None

    def forward(self, x, hidden=None):
        y, hidden = self.gru(x, hidden)
        logits = self.head(y)
        return logits, hidden

    def reset(self):
        self.hidden = None

    def act(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32)
        if obs.ndim == 1:
            obs = obs.view(1, 1, -1)
        elif obs.ndim == 2:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            logits, self.hidden = self.forward(obs, self.hidden)
            return int(torch.argmax(logits[:, -1, :], dim=-1).item())


@dataclass
class RecurrentDistillConfig:
    input_dim: int = 5
    hidden_dim: int = 32
    output_dim: int = 5
    seq_len: int = 16
    stride: int = 8
    batch_size: int = 128
    epochs: int = 60
    learning_rate: float = 1e-3
    validation_split: float = 0.1
    alpha_ce: float = 1.0
    alpha_mse: float = 0.25


def load_sequence_dataset(path, config):
    data = np.load(path, allow_pickle=True)
    observations = data['observations'].astype(np.float32)
    actions = data['actions'].astype(np.int64)
    logits = data['logits'].astype(np.float32)
    episode_rows = data['episode_rows']

    x_chunks = []
    action_chunks = []
    logit_chunks = []

    for row in episode_rows:
        row = dict(row.item()) if hasattr(row, 'item') else dict(row)
        start = int(row['start'])
        length = int(row['length'])
        end = start + length

        for offset in range(start, max(start + 1, end - config.seq_len + 1), config.stride):
            chunk_end = offset + config.seq_len
            if chunk_end > end:
                continue

            x_chunks.append(observations[offset:chunk_end])
            action_chunks.append(actions[offset:chunk_end])
            logit_chunks.append(logits[offset:chunk_end])

    if not x_chunks:
        raise ValueError('No recurrent student chunks were created. Lower seq_len or check dataset.')

    x = torch.tensor(np.asarray(x_chunks), dtype=torch.float32)
    y = torch.tensor(np.asarray(action_chunks), dtype=torch.long)
    t = torch.tensor(np.asarray(logit_chunks), dtype=torch.float32)
    return x, y, t


def make_loaders(x, y, t, config, seed=0):
    dataset = TensorDataset(x, y, t)
    val_size = int(len(dataset) * config.validation_split)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader


def compute_loss(student_logits, teacher_actions, teacher_logits, config):
    ce = F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        teacher_actions.reshape(-1),
    )
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

            student_logits, _ = model(obs)
            loss, ce, mse = compute_loss(student_logits, actions, teacher_logits, config)

            n = int(actions.numel())
            total_loss += float(loss.item()) * n
            total_ce += float(ce.item()) * n
            total_mse += float(mse.item()) * n
            total_correct += int((student_logits.argmax(dim=-1) == actions).sum().item())
            total += n

    return {
        'loss': total_loss / max(1, total),
        'ce': total_ce / max(1, total),
        'mse': total_mse / max(1, total),
        'accuracy': total_correct / max(1, total),
    }


def train_recurrent_student(dataset_path, output_path, config=None, seed=0, device=None):
    config = RecurrentDistillConfig() if config is None else config
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x, y, t = load_sequence_dataset(dataset_path, config)
    train_loader, val_loader = make_loaders(x, y, t, config, seed=seed)

    model = RecurrentStudentPolicy(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []

    for epoch in range(config.epochs):
        model.train()
        for obs, actions, teacher_logits in train_loader:
            obs = obs.to(device)
            actions = actions.to(device)
            teacher_logits = teacher_logits.to(device)

            optimizer.zero_grad()
            student_logits, _ = model(obs)
            loss, _, _ = compute_loss(student_logits, actions, teacher_logits, config)
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


def load_recurrent_student(path, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path, map_location=device)
    config = RecurrentDistillConfig(**checkpoint['config'])
    model = RecurrentStudentPolicy(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, config, checkpoint
