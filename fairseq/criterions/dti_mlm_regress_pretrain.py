# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("dti_mlm_regress_pretrain")
class DTIMaskedLmRegressPretrainLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) and regression pretrain training.
    Note that mlm tasks are done on monolingual data.
    """

    def __init__(self, task, classification_head_name, regression_target):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.regression_target = regression_target


    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='sentence_classification_head',
                            help='name of the classification head to use')
        parser.add_argument('--regression-weight', default=1, type=float, help='regression loss weight')
        parser.add_argument('--mlm-weight-0', default=1, type=float, help='molecule mlm loss weight')
        parser.add_argument('--mlm-weight-1', default=1, type=float, help='protein mlm loss weight')
        parser.add_argument('--mlm-weight-paired-0', default=1, type=float, help='paired molecule mlm loss weight')
        parser.add_argument('--mlm-weight-paired-1', default=1, type=float, help='paired protein mlm loss weight')

    def forward(self, model, sample, args, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        
        """

        assert (
            hasattr(model, "classification_heads")
            and self.classification_head_name in model.classification_heads
        ), "model must provide sentence classification head for --criterion=sentence_prediction"
        # monolingual mlm
        masked_tokens_0 = sample["molecule"]["target"].ne(self.padding_idx)
        sample_size_mlm_0 = masked_tokens_0.int().sum()

        masked_tokens_1 = sample["protein"]["target"].ne(self.padding_idx)
        sample_size_mlm_1 = masked_tokens_1.int().sum()



        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        # if self.tpu:
        #     masked_tokens = None  # always project all tokens on TPU
        if masked_tokens_0.device == torch.device("cpu"):
            if not masked_tokens_0.any():
                masked_tokens_0 = None
            if not masked_tokens_1.any():
                masked_tokens_1 = None
        else:
            masked_tokens_0 = torch.where(
                masked_tokens_0.any(),
                masked_tokens_0,
                masked_tokens_0.new([True]),
            )
            masked_tokens_1 = torch.where(
                masked_tokens_1.any(),
                masked_tokens_1,
                masked_tokens_1.new([True]),
            )
        logits_mlm_0 = model.encoder_0(**sample["molecule"]["net_input"], masked_tokens=masked_tokens_0)[0]

        logits_mlm_1 = model.encoder_1(**sample["protein"]["net_input"], masked_tokens=masked_tokens_1)[0]

        # targets = model.get_targets(sample, [logits])
        targets_mlm_0 = sample["molecule"]["target"]
        targets_mlm_1 = sample["protein"]["target"]

        if masked_tokens_0 is not None:
            targets_mlm_0 = targets_mlm_0[masked_tokens_0]

        if masked_tokens_1 is not None:
            targets_mlm_1 = targets_mlm_1[masked_tokens_1]

        loss_mlm_0 = modules.cross_entropy(
            logits_mlm_0.view(-1, logits_mlm_0.size(-1)),
            targets_mlm_0.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )

        loss_mlm_1 = modules.cross_entropy(
            logits_mlm_1.view(-1, logits_mlm_1.size(-1)),
            targets_mlm_1.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )

        masked_tokens_paired_0 = sample[args.dti_dataset]["target"]["tgt_tokens_0"].ne(self.padding_idx)
        sample_size_mlm_paired_0 = masked_tokens_paired_0.int().sum()

        masked_tokens_paired_1 = sample[args.dti_dataset]["target"]["tgt_tokens_1"].ne(self.padding_idx)
        sample_size_mlm_paired_1 = masked_tokens_paired_1.int().sum()

        if masked_tokens_paired_0.device == torch.device("cpu"):
            if not masked_tokens_paired_0.any():
                masked_tokens_paired_0 = None
            if not masked_tokens_paired_1.any():
                masked_tokens_paired_1 = None
        else:
            masked_tokens_paired_0 = torch.where(
                masked_tokens_paired_0.any(),
                masked_tokens_paired_0,
                masked_tokens_paired_0.new([True]),
            )
            masked_tokens_paired_1 = torch.where(
                masked_tokens_paired_1.any(),
                masked_tokens_paired_1,
                masked_tokens_paired_1.new([True]),
            )

        logits_mlm_paired_0 = model.encoder_0(sample[args.dti_dataset]["net_input"]["src_tokens_0"], masked_tokens=masked_tokens_paired_0)[0]

        logits_mlm_paired_1 = model.encoder_1(sample[args.dti_dataset]["net_input"]["src_tokens_1"], masked_tokens=masked_tokens_paired_1)[0]

        # targets = model.get_targets(sample, [logits])
        targets_mlm_paired_0 = sample[args.dti_dataset]["target"]["tgt_tokens_0"]
        targets_mlm_paired_1 = sample[args.dti_dataset]["target"]["tgt_tokens_1"]

        if masked_tokens_paired_0 is not None:
            targets_mlm_paired_0 = targets_mlm_paired_0[masked_tokens_paired_0]

        if masked_tokens_paired_1 is not None:
            targets_mlm_paired_1 = targets_mlm_paired_1[masked_tokens_paired_1]

        loss_mlm_paired_0 = modules.cross_entropy(
            logits_mlm_paired_0.view(-1, logits_mlm_paired_0.size(-1)),
            targets_mlm_paired_0.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )

        loss_mlm_paired_1 = modules.cross_entropy(
            logits_mlm_paired_1.view(-1, logits_mlm_paired_1.size(-1)),
            targets_mlm_paired_1.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )

        # Recover origin input by src and tgt tokens
        unmasked_tokens_0 = torch.where(
            sample[args.dti_dataset]["target"]["tgt_tokens_0"] == 1,
            sample[args.dti_dataset]["net_input"]["src_tokens_0"],
            sample[args.dti_dataset]["target"]["tgt_tokens_0"]
        )

        unmasked_tokens_1 = torch.where(
            sample[args.dti_dataset]["target"]["tgt_tokens_1"] == 1,
            sample[args.dti_dataset]["net_input"]["src_tokens_1"],
            sample[args.dti_dataset]["target"]["tgt_tokens_1"]
        )
        # regression. Inputs are unmasked tokens recovered by src and tgt tokens
        logits_regress , _ = model(
            src_tokens_0 = unmasked_tokens_0,
            src_tokens_1 = unmasked_tokens_1,
            features_only=True,
            classification_head_name=self.classification_head_name,
        ) 

        # targets = model.get_targets(sample, [logits]).view(-1)
        targets_regress = sample[args.dti_dataset]["target"]["target_regress"].view(-1)
        
        sample_size_regress = targets_regress.numel()

        if not self.regression_target:
            lprobs = F.log_softmax(logits_regress, dim=-1, dtype=torch.float32)
            loss_regress = F.nll_loss(lprobs, targets_regress, reduction="sum")
        else:
            logits_regress = logits_regress.view(-1).float()
            targets_regress = targets_regress.float()
            loss_regress = F.mse_loss(logits_regress, targets_regress, reduction="sum")

        loss = args.mlm_weight_0 * loss_mlm_0 / sample_size_mlm_0 + args.mlm_weight_1 * loss_mlm_1 / sample_size_mlm_1 + args.regression_weight * loss_regress / sample_size_regress + args.mlm_weight_paired_0 * loss_mlm_paired_0 / sample_size_mlm_paired_0 + args.mlm_weight_paired_1 * loss_mlm_paired_1 / sample_size_mlm_paired_1
        sample_size = 1

        logging_output = {
            "loss": loss.data,
            "loss_regress": (loss_regress / sample_size_regress).data,
            "loss_mlm_0": (loss_mlm_0 / sample_size_mlm_0).data,
            "loss_mlm_1": (loss_mlm_1 / sample_size_mlm_1).data,
            "loss_mlm_paired_0": (loss_mlm_paired_0 / sample_size_mlm_paired_0).data,
            "loss_mlm_paired_1": (loss_mlm_paired_1 / sample_size_mlm_paired_1).data,
            "ntokens": sample['molecule']['ntokens'] + sample['protein']['ntokens'] + sample[args.dti_dataset]["ntokens_0"] + sample[args.dti_dataset]["ntokens_1"],
            "nsentences": sample['molecule']["nsentences"] + sample['protein']["nsentences"] + sample[args.dti_dataset]["nsentences"],
            "sample_size": sample_size,
        }
        # pqz
        return loss, sample_size , logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_mlm_0_sum = sum(log.get("loss_mlm_0", 0) for log in logging_outputs)
        loss_mlm_1_sum = sum(log.get("loss_mlm_1", 0) for log in logging_outputs)
        loss_mlm_paired_0_sum = sum(log.get("loss_mlm_paired_0", 0) for log in logging_outputs)
        loss_mlm_paired_1_sum = sum(log.get("loss_mlm_paired_1", 0) for log in logging_outputs)

        loss_regress_sum = sum(log.get("loss_regress", 0) for log in logging_outputs)

        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # round控制四舍五入的位数
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "loss_regress_mse", loss_regress_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "loss_regress_rmse", math.sqrt(loss_regress_sum / sample_size), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_mlm_0", loss_mlm_0_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "loss_mlm_1", loss_mlm_1_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "loss_mlm_paired_0", loss_mlm_paired_0_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "loss_mlm_paired_1", loss_mlm_paired_1_sum / sample_size, sample_size, round=3
        )
        
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
