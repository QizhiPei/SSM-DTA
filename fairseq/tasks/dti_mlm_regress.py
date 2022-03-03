# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Add mlm task for finetune
import logging
import os

import torch
import numpy as np
from fairseq import utils
from fairseq.data import (
    ConcatSentencesDataset,
    Dictionary,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import LegacyFairseqTask, register_task


logger = logging.getLogger(__name__)


@register_task("dti_mlm_regress")
class DTIMlmRegressTask(LegacyFairseqTask):
    """
    Task for training masked language models (e.g., BERT, RoBERTa) 
    with weak supervised task (Regression).
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # parser.add_argument(
        #     "data",
        #     help="colon separated path to data directories list, \
        #                     will be iterated upon during epochs in round-robin manner",
        # )
        parser.add_argument("data", metavar="FILE", help="file prefix for data")
        parser.add_argument(
            "--sample-break-mode",
            default="complete",
            choices=["none", "complete", "complete_doc", "eos"],
            help='If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.',
        )
        parser.add_argument(
            "--tokens-per-sample",
            default=1024,
            type=int,
            help="max number of total tokens over all segments "
            "per sample for BERT dataset",
        )
        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.1,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.1,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--freq-weighted-replacement",
            default=False,
            action="store_true",
            help="sample random replacement words based on word frequencies",
        )
        parser.add_argument(
            "--mask-whole-words",
            default=False,
            action="store_true",
            help="mask whole words; you may also want to set --bpe",
        )
        parser.add_argument(
            "--shorten-method",
            default="none",
            choices=["none", "truncate", "random_crop"],
            help="if not none, shorten sequences that exceed --tokens-per-sample",
        )
        parser.add_argument(
            "--shorten-data-split-list",
            default="",
            help="comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)',
        )
        parser.add_argument(
            "--num-classes",
            type=int,
            default=-1,
            help="number of classes or regression targets",
        )
        parser.add_argument(
            "--init-token",
            type=int,
            default=None,
            help="add token at the beginning of each batch item",
        )
        parser.add_argument(
            "--separator-token",
            type=int,
            default=None,
            help="add separator token between inputs",
        )
        parser.add_argument("--regression-target", action="store_true", default=False)
        parser.add_argument("--no-shuffle", action="store_true", default=False)

    def __init__(self, args, data_dictionary_0, data_dictionary_1, label_dictionary):
        super().__init__(args)
        self.dictionary_0 = data_dictionary_0
        self.dictionary_1 = data_dictionary_1
        self._label_dictionary = label_dictionary
        self.seed = args.seed

        # add mask token
        self.mask_idx_0 = data_dictionary_0.add_symbol("<mask>")
        self.mask_idx_1 = data_dictionary_1.add_symbol("<mask>")

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.num_classes > 0, "Must set --num-classes"

        # paths = utils.split_paths(args.data)
        # assert len(paths) > 0

        data_dict_0 = Dictionary.load(os.path.join(args.data, "input0", "dict.txt"))
        data_dict_1 = Dictionary.load(os.path.join(args.data, "input1", "dict.txt"))

        logger.info("[input] dictionary: {} types".format(len(data_dict_0)))
        logger.info("[input] dictionary: {} types".format(len(data_dict_1)))

        label_dict = None
        if not args.regression_target:
            # load label dictionary
            label_dict = cls.load_dictionary(
                args,
                os.path.join(args.data, "label", "dict.txt"),
                source=False,
            )
            logger.info("[label] dictionary: {} types".format(len(label_dict)))
        else:
            label_dict = data_dict_0
        # dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        # logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, data_dict_0, data_dict_1, label_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # paths = utils.split_paths(self.args.data)
        # assert len(paths) > 0
        # data_path = paths[(epoch - 1) % len(paths)]
        # split_path = os.path.join(data_path, split)

        # dataset = data_utils.load_indexed_dataset(
        #     split_path,
        #     self.source_dictionary,
        #     self.args.dataset_impl,
        #     combine=combine,
        # )
        # if dataset is None:
        #     raise FileNotFoundError(
        #         "Dataset not found: {} ({})".format(split, split_path)
        #     )
        def get_path(type, split):
            return os.path.join(self.args.data, type, split)

        def make_dataset(type, dictionary):
            split_path = get_path(type, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            return dataset

        input0 = make_dataset("input0", self.source_dictionary_0)
        assert input0 is not None, "could not find dataset: {}".format(
            get_path(type, split)
        )
        input1 = make_dataset("input1", self.source_dictionary_1)
        assert input1 is not None, "could not find dataset: {}".format(
            get_path(type, split)
        )

        # if self.args.init_token is not None:
        #     input0 = PrependTokenDataset(input0, self.args.init_token)
        #     input1 = PrependTokenDataset(input1, self.args.init_token)

        src_tokens_0 = input0
        src_tokens_1 = input1



        src_tokens_0 = maybe_shorten_dataset(
            src_tokens_0,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.max_positions_molecule - 1,
            self.args.seed,
        )

        src_tokens_1 = maybe_shorten_dataset(
            src_tokens_1,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.max_positions_protein - 1,
            self.args.seed,
        )

        # create continuous blocks of tokens
        # pqz: use this will lead to different block length
        # src_tokens_0 = TokenBlockDataset(
        #     src_tokens_0,
        #     src_tokens_0.sizes,
        #     self.args.tokens_per_sample - 1,  # one less for <s>
        #     pad=self.source_dictionary_0.pad(),
        #     eos=self.source_dictionary_0.eos(),
        #     break_mode=self.args.sample_break_mode,
        # )
        # logger.info("loaded {} blocks from: {}".format(len(src_tokens_0), get_path("input0", split)))

        # src_tokens_1 = TokenBlockDataset(
        #     src_tokens_1,
        #     src_tokens_1.sizes,
        #     self.args.tokens_per_sample - 1,  # one less for <s>
        #     pad=self.source_dictionary_1.pad(),
        #     eos=self.source_dictionary_1.eos(),
        #     break_mode=self.args.sample_break_mode,
        # )
        # logger.info("loaded {} blocks from: {}".format(len(src_tokens_1), get_path("input1", split)))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        if self.args.init_token is not None:
            src_tokens_0 = PrependTokenDataset(src_tokens_0, self.args.init_token)
            src_tokens_1 = PrependTokenDataset(src_tokens_1, self.args.init_token)
        # dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

        # create masked input and targets
        mask_whole_words = (
            get_whole_word_mask(self.args, self.source_dictionary)
            if self.args.mask_whole_words
            else None
        )

        src_dataset_0, tgt_dataset_0 = MaskTokensDataset.apply_mask(
            src_tokens_0,
            self.source_dictionary_0,
            pad_idx=self.source_dictionary_0.pad(),
            mask_idx=self.mask_idx_0,
            seed=self.args.seed,
            mask_prob=self.args.mask_prob,
            leave_unmasked_prob=self.args.leave_unmasked_prob,
            random_token_prob=self.args.random_token_prob,
            freq_weighted_replacement=self.args.freq_weighted_replacement,
            mask_whole_words=mask_whole_words,
        )

        src_dataset_1, tgt_dataset_1 = MaskTokensDataset.apply_mask(
            src_tokens_1,
            self.source_dictionary_1,
            pad_idx=self.source_dictionary_1.pad(),
            mask_idx=self.mask_idx_1,
            seed=self.args.seed,
            mask_prob=self.args.mask_prob,
            leave_unmasked_prob=self.args.leave_unmasked_prob,
            random_token_prob=self.args.random_token_prob,
            freq_weighted_replacement=self.args.freq_weighted_replacement,
            mask_whole_words=mask_whole_words,
        )

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(src_dataset_0))

        
        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens_0": RightPadDataset(
                    src_dataset_0,
                    pad_idx=self.source_dictionary_0.pad(),
                ),
                "src_lengths_0": NumelDataset(src_dataset_0, reduce=False),
                "src_tokens_1": RightPadDataset(
                    src_dataset_1,
                    pad_idx=self.source_dictionary_1.pad(),
                ),
                "src_lengths_1": NumelDataset(src_dataset_1, reduce=False),
            },
            "target": {
                "tgt_tokens_0": RightPadDataset(
                    tgt_dataset_0,
                    pad_idx=self.source_dictionary_0.pad(),
                ),
                "tgt_tokens_1": RightPadDataset(
                    tgt_dataset_1,
                    pad_idx=self.source_dictionary_1.pad(),
                ),
            },
            "nsentences": NumSamplesDataset(),
            "ntokens_0": NumelDataset(src_dataset_0, reduce=True),
            "ntokens_1": NumelDataset(src_dataset_1, reduce=True),
        }

        # regression label
        label_path = "{0}.label".format(get_path("label", split))
        if os.path.exists(label_path):

            def parse_regression_target(i, line):
                values = line.split()
                assert (
                    len(values) == self.args.num_classes
                ), f'expected num_classes={self.args.num_classes} regression target values on line {i}, found: "{line}"'
                return [float(x) for x in values]

            with open(label_path) as h:
                dataset["target"].update(
                    target_regress=RawLabelDataset(
                        [
                            parse_regression_target(i, line.strip())
                            for i, line in enumerate(h.readlines())
                        ]
                    )
                )

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_dataset_0.sizes],
        )

        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]
            
    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)

        model.my_register_classification_head(
            getattr(args, "classification_head_name", "sentence_classification_head"),
            num_classes=self.args.num_classes,
        )

        return model

    # def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
    #     src_dataset = RightPadDataset(
    #         TokenBlockDataset(
    #             src_tokens,
    #             src_lengths,
    #             self.args.tokens_per_sample - 1,  # one less for <s>
    #             pad=self.source_dictionary.pad(),
    #             eos=self.source_dictionary.eos(),
    #             break_mode="eos",
    #         ),
    #         pad_idx=self.source_dictionary.pad(),
    #     )
    #     src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
    #     src_dataset = NestedDictionaryDataset(
    #         {
    #             "id": IdDataset(),
    #             "net_input": {
    #                 "src_tokens": src_dataset,
    #                 "src_lengths": NumelDataset(src_dataset, reduce=False),
    #             },
    #         },
    #         sizes=src_lengths,
    #     )
    #     if sort:
    #         src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
    #     return src_dataset

    @property
    def source_dictionary(self):
        return self.dictionary_0

    @property
    def target_dictionary(self):
        return self.dictionary_0
        
    @property
    def source_dictionary_0(self):
        return self.dictionary_0

    @property
    def source_dictionary_1(self):
        return self.dictionary_1

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample, self.args)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        # pqz
        sample_size = sample_size / self.args.update_freq[0]
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample, self.args)
        return loss, sample_size, logging_output

        
