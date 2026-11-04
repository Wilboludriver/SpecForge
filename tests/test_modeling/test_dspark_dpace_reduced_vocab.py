import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.dspark.dpace import OnlineDSparkDpaceModel
from specforge.algorithms.dspark.providers import resume_contract
from specforge.modeling.draft.dspark import DSparkDraftModel


def build_model(confidence=1.0, block=4):
    config = Qwen3Config(
        architectures=["DSparkDraftModel"], hidden_size=32, intermediate_size=64,
        num_attention_heads=2, num_key_value_heads=1, num_hidden_layers=1,
        num_target_layers=4, head_dim=16, vocab_size=12, draft_vocab_size=4,
        block_size=block, sample_from_anchor=True, is_causal=True,
        layer_types=["full_attention"],
        dflash_config=dict(projector_type="dspark", target_layer_ids=[0, 2],
                           mask_token_id=11, markov_head_type="vanilla", markov_rank=4,
                           enable_confidence_head=True, confidence_head_with_markov=True),
        attn_implementation="sdpa",
    )
    draft = DSparkDraftModel(config)
    draft.d2t.copy_(torch.tensor([1, 3, 5, 7]))
    draft.t2d[[1, 4, 7, 10]] = True
    return OnlineDSparkDpaceModel(
        draft, nn.Linear(32, 12, bias=False).requires_grad_(False),
        nn.Embedding(12, 32).requires_grad_(False), mask_token_id=11,
        block_size=block, attention_backend="sdpa", num_anchors=2,
        dspark_ce_loss_alpha=1.0, dspark_l1_loss_alpha=0.0,
        dspark_confidence_head_alpha=confidence,
        objective_chunk_blocks=1, objective_checkpointing=False,
        fused_logits_loss=False,
    )


def objective(model, hidden, labels, teacher=None, mask=None):
    b, n, k = labels.shape
    if mask is None:
        mask = torch.ones_like(labels, dtype=torch.bool)
    previous = torch.cat([torch.zeros_like(labels[..., :1]), labels[..., :-1]], -1)
    positions = torch.arange(1, k + 1).view(1, 1, k).expand_as(labels)
    return model._compute_dspark_loss(
        output_hidden=hidden.reshape(b, n * k, -1), target_ids=labels,
        eval_mask=mask, prev_token_ids=previous, safe_label_indices=positions,
        target_last_hidden_states=teacher,
    )


class ReducedDSparkDpaceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(2)

    def test_mapping_shapes_and_strict_roundtrip(self):
        model = build_model()
        draft = model.draft_model
        self.assertEqual(tuple(draft.markov_head.markov_w1.weight.shape), (12, 4))
        self.assertEqual(tuple(draft.markov_head.markov_w2.weight.shape), (4, 4))
        torch.testing.assert_close(model.target_to_draft, torch.tensor([-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1]))
        with tempfile.TemporaryDirectory() as path:
            filename = str(Path(path) / "model.safetensors")
            save_file(draft.state_dict(), filename)
            restored = DSparkDraftModel(draft.config)
            restored.load_state_dict(load_file(filename), strict=True)
            restored.validate_vocab_mapping()
            torch.testing.assert_close(restored.state_dict(), draft.state_dict())
        contract = resume_contract(None, draft, model)
        self.assertEqual(contract["dspark_loss_type"], "dpace")
        self.assertEqual(contract["dspark_draft_vocab_size"], 4)
        self.assertEqual(contract["dspark_confidence_teacher_distribution"], "full_vocab")

    def test_invalid_mapping_fails_before_head_selection(self):
        model = build_model()
        model.draft_model.d2t[0] = 0
        with self.assertRaises(ValueError):
            model.draft_model.validate_vocab_mapping()

    def test_frozen_business_head_is_ignored_and_not_exported(self):
        from specforge.training.backend import FSDPTrainingBackend
        from specforge.training.strategies.base import DSparkTrainStrategy

        model = build_model()
        self.assertEqual(
            FSDPTrainingBackend._frozen_target_modules(model),
            (model.lm_head, model.embed_tokens, model.draft_lm_head),
        )
        exported = DSparkTrainStrategy(model).checkpoint_state_filter(model.state_dict())
        self.assertEqual(set(exported), set(model.draft_model.state_dict()))
        self.assertFalse(any('lm_head' in name or 'embed_tokens' in name for name in exported))
        torch.testing.assert_close(model.draft_lm_head.weight, model.lm_head.weight[[1, 4, 7, 10]])

    def test_dpace_matches_sequence_balanced_reference(self):
        model = build_model(confidence=0)
        hidden = torch.randn(2, 2, 4, 32, requires_grad=True)
        labels = torch.tensor([[[1, 4, 7, 10], [10, 7, 4, 1]], [[4, 7, 10, 1], [1, 4, 7, 10]]])
        mask = torch.ones_like(labels, dtype=torch.bool)
        mask[1, 1] = False
        loss, _ = objective(model, hidden, labels, mask=mask)
        prev = torch.cat([torch.zeros_like(labels[..., :1]), labels[..., :-1]], -1)
        logits = F.linear(hidden, model.lm_head.weight[[1, 4, 7, 10]])
        logits = logits + model.draft_model.markov_head.markov_w2(
            model.draft_model.markov_head.markov_w1(prev)
        )
        ce = F.cross_entropy(logits.flatten(0, 2), model.target_to_draft[labels].flatten(), reduction="none").view_as(labels)
        sequence_losses = []
        # Independent scalar reference: each position receives the sum of
        # smoothed acceptance-prefix products ending at or after that position.
        for b in range(2):
            anchors = []
            for a in range(2):
                if not mask[b, a].any():
                    continue
                p = torch.exp(-ce[b, a]).detach()
                prefix = torch.cumprod(0.5 + 0.5 * p, 0)
                anchors.append(sum(ce[b, a, j] * prefix[j:].sum() for j in range(4)))
            sequence_losses.append(torch.stack(anchors).mean())
        expected = torch.stack(sequence_losses).mean()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(torch.autograd.grad(loss, hidden, retain_graph=True)[0], torch.autograd.grad(expected, hidden)[0])

    def test_oov_stops_credit_chain_and_counts_as_accuracy_miss(self):
        model = build_model(confidence=0)
        hidden = torch.randn(1, 1, 4, 32, requires_grad=True)
        labels = torch.tensor([[[1, 3, 7, 10]]])  # 3 is outside the draft vocabulary.
        loss, metrics = objective(model, hidden, labels)
        loss.backward()
        self.assertGreater(hidden.grad[:, :, :1].abs().sum(), 0)
        self.assertEqual(hidden.grad[:, :, 1:].abs().sum(), 0)
        self.assertEqual(metrics['ratio_metrics']['vocab_coverage'][0], 3)
        self.assertEqual(metrics['ratio_metrics']['reachable_token_fraction'][0], 1)
        self.assertEqual(metrics['accuracy_denom'], 4)

    def test_confidence_uses_full_teacher_probability_mass(self):
        model = build_model()
        model.draft_model.markov_head.markov_w1.weight.data.zero_()
        model.draft_model.confidence_head.proj.weight.data.zero_()
        model.draft_model.confidence_head.proj.bias.data.zero_()
        hidden = torch.zeros(1, 1, 4, 32, requires_grad=True)
        labels = torch.tensor([[[1, 4, 7, 10]]])
        loss, metrics = objective(model, hidden, labels, teacher=torch.zeros(1, 5, 32))
        a, n = metrics['ratio_metrics']['teacher_acceptance']
        torch.testing.assert_close(a / n, torch.tensor(1 / 3))
        loss.backward()
        torch.testing.assert_close(model.draft_model.confidence_head.proj.bias.grad, torch.tensor([1 / 6]))
        self.assertIsNone(model.lm_head.weight.grad)
        self.assertIsNone(model.embed_tokens.weight.grad)

    def test_block16_forward_backward_chunk_and_checkpoint_parity(self):
        reference = build_model(block=16)
        batch = dict(input_ids=torch.tensor([[1, 4, 7, 10] * 10]),
                     hidden_states=torch.randn(1, 40, 64), loss_mask=torch.ones(1, 40),
                     target_last_hidden_states=torch.randn(1, 40, 32))
        anchors = torch.tensor([[1, 20]])
        keep = torch.ones_like(anchors, dtype=torch.bool)
        outputs = []
        for chunk, checkpoint in ((0, False), (1, False), (1, True)):
            model = copy.deepcopy(reference)
            model.objective_chunk_blocks = chunk
            model.objective_checkpointing = checkpoint
            with patch.object(model, '_sample_anchor_positions', return_value=(anchors, keep)):
                loss, acc, metrics = model(**batch)
            loss.backward()
            grads = {}
            for name, p in model.draft_model.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
                grads[name] = p.grad.clone()
            outputs.append((loss.detach(), acc, metrics, grads))
        for output in outputs[1:]:
            torch.testing.assert_close(output, outputs[0], atol=2e-5, rtol=2e-4)

    def test_all_oov_returns_finite_zero_token_loss(self):
        model = build_model(confidence=0)
        hidden = torch.randn(1, 1, 4, 32, requires_grad=True)
        loss, _ = objective(model, hidden, torch.zeros(1, 1, 4, dtype=torch.long))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())


if __name__ == '__main__':
    unittest.main()
