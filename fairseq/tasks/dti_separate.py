# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
from fairseq import utils
from fairseq.data import (
    ConcatSentencesDataset,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    OffsetTokensDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    RollDataset,
    SortDataset,
    StripTokenDataset,
    data_utils,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import LegacyFairseqTask, register_task


logger = logging.getLogger(__name__)


@register_task("dti_separate")
class DTISeparateTask(LegacyFairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task
    for two languages with different dictionary

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", metavar="FILE", help="file prefix for data")
        parser.add_argument(
            "--init-token",
            type=int,
            default=None,
            help="add token at the beginning of each batch item",
        )
        parser.add_argument(
            "--num-classes",
            type=int,
            default=-1,
            help="number of classes or regression targets",
        )
        parser.add_argument("--regression-target", action="store_true", default=False)
        parser.add_argument("--no-shuffle", action="store_true", default=False)
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

    def __init__(self, args, data_dictionary_0, data_dictionary_1, label_dictionary):
        super().__init__(args)
        self.dictionary_0 = data_dictionary_0
        self.dictionary_1 = data_dictionary_1
        self._label_dictionary = label_dictionary
        # if not hasattr(args, "max_positions"):
        #     self._max_positions = (
        #         args.max_source_positions,
        #         args.max_target_positions,
        #     )
        # else:
        #     self._max_positions = args.max_positions
        # args.tokens_per_sample = self._max_positions

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        # dictionary.add_symbol("<mask>")
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.num_classes > 0, "Must set --num-classes"

        # load data dictionary
        data_dict_0 = cls.load_dictionary(
            args,
            os.path.join(args.data, "input0", "dict.txt"),
            source=True,
        )
        logger.info("[input] dictionary: {} types".format(len(data_dict_0)))

        data_dict_1 = cls.load_dictionary(
            args,
            os.path.join(args.data, "input1", "dict.txt"),
            source=True,
        )
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

        return cls(args, data_dict_0, data_dict_1, label_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

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

        if self.args.init_token is not None:
            input0 = PrependTokenDataset(input0, self.args.init_token)
            input1 = PrependTokenDataset(input1, self.args.init_token)

        # if input1 is None:
        #     src_tokens = input0
        # else:
        #     if self.args.separator_token is not None:
        #         input1 = PrependTokenDataset(input1, self.args.separator_token)

        src_tokens_0 = input0
        src_tokens_1 = input1

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens_0))

        src_tokens_0 = maybe_shorten_dataset(
            src_tokens_0,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.max_positions_molecule,
            self.args.seed,
        )

        src_tokens_1 = maybe_shorten_dataset(
            src_tokens_1,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.max_positions_protein,
            self.args.seed,
        )

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens_0": RightPadDataset(
                    src_tokens_0,
                    pad_idx=self.source_dictionary_0.pad(),
                ),
                "src_lengths_0": NumelDataset(src_tokens_0, reduce=False),
                "src_tokens_1": RightPadDataset(
                    src_tokens_1,
                    pad_idx=self.source_dictionary_1.pad(),
                ),
                "src_lengths_1": NumelDataset(src_tokens_1, reduce=False),
            },
            "nsentences": NumSamplesDataset(),
            "ntokens_0": NumelDataset(src_tokens_0, reduce=True),
            "ntokens_1": NumelDataset(src_tokens_1, reduce=True),
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
                dataset.update(
                    target=RawLabelDataset(
                        [
                            parse_regression_target(i, line.strip())
                            for i, line in enumerate(h.readlines())
                        ]
                    )
                )

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens_0.sizes],
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

    @property
    def source_dictionary(self):
        return self.dictionary_0

    @property
    def source_dictionary_0(self):
        return self.dictionary_0

    @property
    def source_dictionary_1(self):
        return self.dictionary_1

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary

    # pqz
    # def train_step(
    #     self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    # ):
    #     """
    #     Do forward and backward, and return the loss as computed by *criterion*
    #     for the given *model* and *sample*.

    #     Args:
    #         sample (dict): the mini-batch. The format is defined by the
    #             :class:`~fairseq.data.FairseqDataset`.
    #         model (~fairseq.models.BaseFairseqModel): the model
    #         criterion (~fairseq.criterions.FairseqCriterion): the criterion
    #         optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
    #         update_num (int): the current update
    #         ignore_grad (bool): multiply loss by 0 if this is set to True

    #     Returns:
    #         tuple:
    #             - the loss
    #             - the sample size, which is used as the denominator for the
    #               gradient
    #             - logging outputs to display while training
    #     """
    #     model.train()
    #     model.set_num_updates(update_num)
    #     with torch.autograd.profiler.record_function("forward"):
    #         loss, sample_size, logging_output = criterion(model, sample)
    #     if ignore_grad:
    #         loss *= 0
    #     with torch.autograd.profiler.record_function("backward"):
    #         optimizer.backward(loss)
    #     return loss, sample_size, logging_output
