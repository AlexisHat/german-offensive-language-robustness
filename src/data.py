import csv
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# The raw files contain literal ASCII quote characters inside the tweet
# text (unbalanced across lines), which breaks pandas' default CSV quoting.
# QUOTE_NONE disables quote interpretation entirely so every line is split
# purely on tabs.
RAW_COLUMNS = ["text_raw", "label", "label_fine"]

# "|LBR|" is a literal token used in the original export to mark line
# breaks inside a tweet so it is not part of the tweet content.
LBR_TOKEN = "|LBR|"

MENTION_PATTERN = re.compile(r"@\w+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def load_germeval_file(path):
    """Load one GermEval file into a DataFrame."""
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=RAW_COLUMNS,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )


def load_germeval(raw_dir, train_filename="germeval2018.training.txt",
                  test_filename="germeval2018.test.txt"):
    """Load the official training and test files."""
    raw_dir = Path(raw_dir)
    train_df = load_germeval_file(raw_dir / train_filename)
    test_df = load_germeval_file(raw_dir / test_filename)
    return train_df, test_df

def clean_text(text):
    """Apply the text cleaning used in this project.

    Only a few simple changes are made. User mentions are replaced,
    the special line-break token is removed, and repeated whitespace
    is collapsed. Everything else is kept as it is.
    """
    text = text.replace(LBR_TOKEN, " ")
    text = MENTION_PATTERN.sub("@USER", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text

def preprocess_dataframe(df):
    """Clean the text column and remove empty or duplicate tweets."""
    df = df.copy()
    df["text"] = df["text_raw"].map(clean_text)
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset="text")
    df = df.reset_index(drop=True)
    return df


def make_splits(train_df, val_size=0.15, seed=42, stratify_col="label"):
    """Split the training data into train and validation sets.

    The split is stratified by the binary label so both sets have a
    similar class distribution.
    """
    train_split, val_split = train_test_split(
        train_df,
        test_size=val_size,
        random_state=seed,
        stratify=train_df[stratify_col],
    )
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)
    return train_split, val_split
