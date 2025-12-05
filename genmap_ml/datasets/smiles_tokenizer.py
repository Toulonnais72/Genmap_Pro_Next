"""Character-level SMILES tokenizer utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from collections import Counter

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
DEFAULT_MAX_LEN_CAP = 150


@dataclass
class SmilesTokenizerConfig:
    """Configuration dataclass storing vocabulary metadata for SMILES tokenization."""

    vocab: Dict[str, int]
    inv_vocab: Dict[int, str]
    pad_idx: int
    bos_idx: int
    eos_idx: int
    unk_idx: int
    max_length: int


def build_tokenizer_from_smiles(
    smiles_list: List[str],
    min_freq: int = 1,
    extra_chars: Optional[List[str]] = None,
    max_length: Optional[int] = None,
) -> SmilesTokenizerConfig:
    """Build a tokenizer configuration from a list of SMILES strings."""

    counter: Counter[str] = Counter()
    max_body_len = 0
    for smi in smiles_list:
        if not isinstance(smi, str):
            continue
        smi = smi.strip()
        if not smi:
            continue
        counter.update(smi)
        max_body_len = max(max_body_len, len(smi))

    allowed_chars = sorted(ch for ch, freq in counter.items() if freq >= min_freq)
    if extra_chars:
        allowed_chars = sorted(set(allowed_chars).union(extra_chars))

    tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + allowed_chars
    vocab = {token: idx for idx, token in enumerate(tokens)}
    inv_vocab = {idx: token for token, idx in vocab.items()}

    if max_length is None:
        raw_max = max_body_len + 2  # BOS/EOS
        max_length = min(max(raw_max, 4), DEFAULT_MAX_LEN_CAP)

    cfg = SmilesTokenizerConfig(
        vocab=vocab,
        inv_vocab=inv_vocab,
        pad_idx=vocab[PAD_TOKEN],
        bos_idx=vocab[BOS_TOKEN],
        eos_idx=vocab[EOS_TOKEN],
        unk_idx=vocab[UNK_TOKEN],
        max_length=max_length,
    )
    return cfg


def encode_smiles(
    smiles: str,
    cfg: SmilesTokenizerConfig,
    add_bos: bool = True,
    add_eos: bool = True,
) -> List[int]:
    """Encode SMILES string into indices using the provided tokenizer config."""

    tokens: List[int] = []
    if add_bos:
        tokens.append(cfg.bos_idx)

    for char in smiles:
        tokens.append(cfg.vocab.get(char, cfg.unk_idx))

    if add_eos:
        tokens.append(cfg.eos_idx)

    tokens = tokens[: cfg.max_length]
    if len(tokens) < cfg.max_length:
        tokens.extend([cfg.pad_idx] * (cfg.max_length - len(tokens)))
    return tokens


def decode_indices(
    indices: List[int],
    cfg: SmilesTokenizerConfig,
    skip_special: bool = True,
) -> str:
    """Decode indices back into a SMILES-like string."""

    chars: List[str] = []
    special = {cfg.pad_idx, cfg.bos_idx, cfg.eos_idx} if skip_special else set()
    for idx in indices:
        if idx == cfg.eos_idx:
            break
        if skip_special and idx in special:
            continue
        token = cfg.inv_vocab.get(idx, UNK_TOKEN)
        if skip_special and token in {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}:
            continue
        chars.append(token)
    return "".join(chars)


def save_tokenizer_config(cfg: SmilesTokenizerConfig, path: str | Path) -> None:
    """Persist tokenizer configuration to JSON."""

    path = Path(path)
    payload = asdict(cfg)
    # Ensure keys are JSON serializable (dict keys already str/int)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_tokenizer_config(path: str | Path) -> SmilesTokenizerConfig:
    """Load tokenizer configuration from JSON."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return SmilesTokenizerConfig(
        vocab={str(k): int(v) if isinstance(v, bool) else int(v) for k, v in data["vocab"].items()},
        inv_vocab={int(k): str(v) for k, v in data["inv_vocab"].items()},
        pad_idx=int(data["pad_idx"]),
        bos_idx=int(data["bos_idx"]),
        eos_idx=int(data["eos_idx"]),
        unk_idx=int(data["unk_idx"]),
        max_length=int(data["max_length"]),
    )
