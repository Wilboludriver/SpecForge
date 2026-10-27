"""Independent numerical and fixture oracles; no production implementation copies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")



def canonical_model_info(info):
    """Preserve every model/schema value in its persisted JSON representation.

    HF config dictionaries use integer id2label keys in memory, while the JSON
    evidence necessarily stores those keys as strings. Compare both forms
    through the same complete serialization boundary; never filter fields.
    """
    return json.loads(json.dumps(info,sort_keys=True,allow_nan=False))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(8 << 20):
            digest.update(data)
    return digest.hexdigest()


def tensor_sha256(tensor):
    import torch
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_errors(actual, expected, *, rtol, atol):
    """Full elementwise check plus scale-sensitive errors; never subsample."""
    import torch
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"shape/dtype mismatch: {actual.shape}/{actual.dtype} != {expected.shape}/{expected.dtype}")
    if actual.numel() == 0:
        return {"passed": True, "elements": 0, "max_abs": 0.0, "relative_l2": 0.0, "mismatches": 0, "worst_flat_index": 0}
    a, b = actual.detach().float().cpu(), expected.detach().float().cpu()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise AssertionError("nonfinite tensor in numerical comparison")
    error = (a - b).abs()
    failures = error > atol + rtol * b.abs()
    return {
        "passed": not bool(failures.any()), "elements": a.numel(),
        "max_abs": float(error.max()), "relative_l2": float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-30)),
        "mismatches": int(failures.sum()), "worst_flat_index": int(error.argmax()),
    }


def validate_features(features, row, *, hidden_size, layers):
    import torch
    n = len(row["input_ids"])
    expected = {"hidden_states": ((1, n, hidden_size * len(layers)), torch.bfloat16),
                "target_last_hidden_states": ((1, n, hidden_size), torch.bfloat16),
                "input_ids": ((1, n), torch.int64), "loss_mask": ((1, n), torch.int64)}
    if set(features) != set(expected):
        raise AssertionError(f"feature names {set(features)} != {set(expected)}")
    for name, (shape, dtype) in expected.items():
        value = features[name]
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise AssertionError(f"{name}: actual {tuple(value.shape)}/{value.dtype} != {shape}/{dtype}")
        if not bool(torch.isfinite(value).all()):
            raise AssertionError(f"{name}: nonfinite values")
    for name in ("input_ids", "loss_mask"):
        if features[name].tolist() != [row[name]]:
            raise AssertionError(f"{name}: full input alignment failed")


def load_cases(path, expected_count=16):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) != expected_count or len({row['id'] for row in rows}) != expected_count:
        raise AssertionError(f"expected exactly {expected_count} uniquely named cases")
    for row in rows:
        ids, mask = row["input_ids"], row["loss_mask"]
        if not ids or len(ids) != len(mask) or any(type(v) is not int or v < 0 for v in ids):
            raise AssertionError(f"invalid native row {row['id']}")
        if not any(a > .5 and b > .5 for a, b in zip(mask, mask[1:])):
            raise AssertionError(f"no valid DFlash anchor {row['id']}")
    return rows
