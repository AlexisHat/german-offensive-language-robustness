"""Fine-tuning utilities for the gbert-base transformer classifier."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src import config, evaluate


class GermEvalDataset(Dataset):
    """Tokenized GermEval texts ready for BERT-style sequence classification."""

    def __init__(self, texts, labels, tokenizer, max_length):
        encodings = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = torch.tensor([config.LABEL2ID[label] for label in labels], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }


def set_seed(seed):
    """Seed python, numpy, and torch (CPU/CUDA) for reproducible fine-tuning."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(checkpoint, num_labels=2, seed=42):
    """Load a pretrained checkpoint's tokenizer and a fresh classification head.

    The seed must be set before the model is constructed: the pretrained
    encoder weights are fixed, but AutoModelForSequenceClassification adds a
    randomly-initialized classification head on top, which is the main
    source of across-seed variance during fine-tuning.
    """
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
    return model, tokenizer


def build_dataloaders(train_df, val_df, tokenizer, max_length, batch_size, seed):
    """Build a shuffled train DataLoader and an ordered val DataLoader."""
    train_dataset = GermEvalDataset(train_df["text"], train_df["label"], tokenizer, max_length)
    val_dataset = GermEvalDataset(val_df["text"], val_df["label"], tokenizer, max_length)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, scheduler, device):
    """Run one training epoch and return the average training loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += outputs.loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def predict(model, loader, device):
    """Run inference and return (true label ids, predicted label ids)."""
    model.eval()
    all_labels, all_preds = [], []
    for batch in loader:
        labels = batch["labels"]
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=-1).cpu()
        all_labels.append(labels)
        all_preds.append(preds)
    return torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()


def evaluate_transformer_on_df(model, tokenizer, df, text_col, max_length, batch_size, device):
    """Predict with a fine-tuned transformer and return macro-F1 against df['label']."""
    dataset = GermEvalDataset(df[text_col], df["label"], tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    y_true, y_pred = predict(model, loader, device)
    return evaluate.macro_f1(y_true, y_pred)


def run_finetuning(
    checkpoint, train_df, val_df, seed, max_length, batch_size, lr,
    weight_decay, num_epochs, output_dir, device,
):
    """Fine-tune one model for one seed, keeping the best-val-macro-F1 checkpoint."""
    model, tokenizer = load_model_and_tokenizer(checkpoint, seed=seed)
    model.to(device)

    train_loader, val_loader = build_dataloaders(train_df, val_df, tokenizer, max_length, batch_size, seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    history = []
    best_macro_f1 = -1.0
    best_state_dict = None

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        y_true, y_pred = predict(model, val_loader, device)
        val_macro_f1 = evaluate.macro_f1(y_true, y_pred)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_f1": val_macro_f1})

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state_dict)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    return history_df
