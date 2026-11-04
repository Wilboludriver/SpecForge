"""D-PACE on Markov-corrected DSpark logits, with an optional reduced vocabulary.

The backbone, anchor alignment and attention masks stay native to SpecForge.
Only the draft output projection is restricted. Teacher probabilities retain
their full-vocabulary normalization for the confidence acceptance target.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
from specforge.core.chunking import checkpointed_chunk_reduce
from specforge.core.position_weights import DPACE_LOSS_TYPES, dpace_weights


class OnlineDSparkDpaceModel(OnlineDSparkModel):
    def __init__(self, *args, loss_type="dpace", dpace_alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        if loss_type not in DPACE_LOSS_TYPES or not 0 <= dpace_alpha <= 1:
            raise ValueError("Invalid DSpark D-PACE mode or alpha")
        if self.dspark_ce_loss_alpha != 1.0 or self.dspark_l1_loss_alpha != 0.0:
            raise ValueError(
                "DSpark D-PACE requires dspark_ce_loss_alpha=1 and "
                "dspark_l1_loss_alpha=0; confidence remains an optional auxiliary loss"
            )
        if not getattr(self.draft_model.config, "sample_from_anchor", True):
            raise ValueError("DSpark D-PACE training requires sample_from_anchor=True")
        if any(p.requires_grad for p in self.lm_head.parameters()) or any(
            p.requires_grad for p in self.embed_tokens.parameters()
        ):
            raise ValueError("Shared target embedding and LM head must be frozen")
        self.loss_type = loss_type
        self.dpace_alpha = float(dpace_alpha)
        self.draft_model.validate_vocab_mapping()
        size = int(self.draft_model.config.vocab_size)
        if self.lm_head.weight.shape != (size, self.draft_model.config.hidden_size):
            raise ValueError("Shared target head shape does not match the draft config")
        selected = (
            self.draft_model.draft_to_target_ids()
            if self.draft_model.uses_reduced_vocab
            else torch.arange(size, device=self.lm_head.weight.device)
        ).to(device=self.lm_head.weight.device)
        inverse = torch.full((size,), -1, dtype=torch.long, device=selected.device)
        inverse[selected] = torch.arange(selected.numel(), device=selected.device)
        self.register_buffer("selected_target_ids", selected, persistent=False)
        self.register_buffer("target_to_draft", inverse, persistent=False)
        # A frozen target-head slice, not a trainable draft head. The training
        # backend leaves it replicated and the strategy excludes it on export.
        # A float buffer would instead be forced to FP32 by FSDP mixed precision.
        self.draft_lm_head = None
        if self.draft_model.uses_reduced_vocab:
            self.draft_lm_head = nn.Linear(self.lm_head.weight.shape[1], selected.numel(), bias=False, device="meta")
            self.draft_lm_head.weight = nn.Parameter(
                self.lm_head.weight.detach().index_select(0, selected), requires_grad=False
            )
        if self.dspark_confidence_head_alpha == 0 and self.draft_model.confidence_head is not None:
            self.draft_model.confidence_head.requires_grad_(False)

    def _project_draft_logits(self, hidden):
        if self.draft_lm_head is None:
            return self._project_logits(hidden)
        return self.draft_lm_head(hidden.to(self.draft_lm_head.weight.dtype))

    def _dpace_chunk_terms(self, hidden, prev_ids, target_ids, eval_mask,
                           train_mask, sequence_scale, teacher_hidden):
        logits = self.draft_model.apply_logits_head(
            self._project_draft_logits(hidden), prev_token_ids=prev_ids, hidden_states=hidden
        )
        mapped_labels = self.target_to_draft[target_ids]
        # Unsupported labels use an in-range placeholder only for the CE kernel;
        # their losses (and subsequent unreachable slots) have exactly zero weight.
        terms = self._logits_loss_terms(logits, mapped_labels.clamp_min(0))
        ce = terms.cross_entropy
        probability = torch.exp(-ce).detach() * (mapped_labels >= 0)
        weights = dpace_weights(
            probability, train_mask, alpha=self.dpace_alpha, mode=self.loss_type
        ) * train_mask * sequence_scale
        token_num = (ce * weights).sum()
        zero = token_num.new_zeros(())
        confidence_num = confidence_error = zero
        confidence_den = zero.detach()
        acceptance_num = zero.detach()
        if teacher_hidden is not None:
            with torch.no_grad():
                # Keep teacher mass outside the draft vocabulary. Re-normalizing
                # its selected rows would overestimate actual acceptance.
                p_full = self._project_logits(teacher_hidden).float().softmax(-1)
                p_selected = p_full.index_select(-1, self.selected_target_ids)
                q = logits.detach().float().softmax(-1)
                acceptance = torch.minimum(q, p_selected).sum(-1)
            confidence = self.draft_model.predict_confidence(hidden, prev_token_ids=prev_ids)
            confidence_weights = eval_mask.float()
            confidence_num = (
                F.binary_cross_entropy_with_logits(confidence, acceptance, reduction="none")
                * confidence_weights
            ).sum()
            confidence_den = confidence_weights.sum()
            confidence_error = (
                (confidence.detach().sigmoid() - acceptance).abs() * confidence_weights
            ).sum()
            acceptance_num = (acceptance * confidence_weights).sum()
        with torch.no_grad():
            predicted = self.selected_target_ids[terms.argmax_ids]
            correct = (predicted == target_ids) & eval_mask
            token_count = eval_mask.float().sum()
            # Accuracy uses all supervised labels, including OOV labels as misses.
            position_correct = correct.float().sum((0, 1))
            position_den = eval_mask.float().sum((0, 1))
            covered_count = ((mapped_labels >= 0) & eval_mask).float().sum()
            train_count = train_mask.sum()
        return (token_num, confidence_num, confidence_den, confidence_error,
                correct.float().sum(), token_count, position_correct, position_den,
                covered_count, train_count, acceptance_num)

    def _compute_dspark_loss(self, *, output_hidden, target_ids, eval_mask,
                            prev_token_ids, safe_label_indices, target_last_hidden_states,
                            anchor_positions=None, collect_detailed_metrics=True):
        del anchor_positions, collect_detailed_metrics
        batch_size, num_blocks, block_size = target_ids.shape
        hidden = output_hidden.reshape(batch_size, num_blocks, block_size, -1)
        covered = self.target_to_draft[target_ids] >= 0
        # An OOV hard target cannot be proposed. Stop the supervised token chain
        # at that slot instead of treating it as a hole that D-PACE can skip over.
        train_mask = (eval_mask & covered).int().cumprod(-1).float()
        sequence_scale = self._sequence_anchor_scale(train_mask)
        local_sequence_den = (
            train_mask.any(-1).float() * sequence_scale.squeeze(-1)
        ).sum()
        teacher_hidden = None
        if self.dspark_confidence_head_alpha > 0 and self.draft_model.confidence_head is not None:
            if target_last_hidden_states is None:
                raise ValueError("DSpark confidence requires target_last_hidden_states")
            teacher_hidden = self._aligned_target_hidden(target_last_hidden_states, safe_label_indices)
        totals = checkpointed_chunk_reduce(
            self._dpace_chunk_terms, hidden, prev_token_ids, target_ids, eval_mask,
            train_mask, sequence_scale, teacher_hidden,
            chunk_size=self.objective_chunk_blocks, dim=1,
            checkpoint=self.objective_checkpointing,
        )
        (token_num, confidence_num, confidence_den, confidence_error,
         correct, count, position_correct, position_den, covered_count,
         train_count, acceptance_num) = totals
        denominators = torch.stack((local_sequence_den, confidence_den)).detach()
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                dist.all_reduce(denominators)
        # Native D-PACE normalizes credit by anchors within a sequence, then by
        # valid sequences, NOT by the sum of adaptive weights. Empty local ranks
        # contribute zero while participating in the same global reductions.
        loss = world_size * token_num / denominators[0].clamp_min(1)
        loss = loss + world_size * self.dspark_confidence_head_alpha * confidence_num / denominators[1].clamp_min(1)
        ratios = {
            "dpace_loss": (token_num.detach(), local_sequence_den),
            "acc": (correct, count),
            "accuracy_position": (position_correct, position_den),
            "confidence_loss": (confidence_num.detach(), confidence_den),
            "confidence_abs_error": (confidence_error.detach(), confidence_den),
            "teacher_acceptance": (acceptance_num, confidence_den),
            "vocab_coverage": (covered_count, count),
            "reachable_token_fraction": (train_count, count),
        }
        return loss, {
            "accuracy": (correct / count.clamp_min(1)).detach(),
            "accuracy_denom": count.detach(),
            "ratio_metrics": {name: (a.detach(), b.detach()) for name, (a, b) in ratios.items()},
        }
