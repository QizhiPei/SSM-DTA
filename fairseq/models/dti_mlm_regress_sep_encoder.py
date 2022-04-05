import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.modules import MultiheadAttention
from fairseq.models.roberta.model import (
    RobertaModel,
    RobertaEncoder,
    DTIRobertaEncoder,
    RobertaClassificationHead,
    base_architecture as roberta_base_architecture
)

from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)

from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.modules import GradMultiply
from torch.functional import Tensor

logger = logging.getLogger(__name__)

DEFAULT_MAX_MOLECULE_POSITIONS = 512
DEFAULT_MAX_PROTEIN_POSITIONS = 1024

@register_model("roberta_dti_cross_attn")
# class RobertaDTI(RobertaModel):
class RobertaDTICrossAttn(BaseFairseqModel):
    def __init__(self, args, encoder_0, encoder_1):
        super().__init__()
        self.args = args
        self.encoder_0 = encoder_0
        self.encoder_1 = encoder_1
        # We follow BERT's random weight initialization
        self.apply(init_bert_params)
        self.classification_heads = nn.ModuleDict()
        self.cross_attn = MultiheadAttention(
            args.encoder_embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--grad-multiply", type=float, metavar="D", default=1, help="Apply different lr on backbone and classification head"
        )
        parser.add_argument(
            "--use-2-attention",
            action="store_true",
            default=False,
            help="use two cls's attention",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-positions-molecule", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--max-positions-protein", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--load-checkpoint-heads",
            action="store_true",
            help="(re-)register and load heads when loading checkpoints",
        )
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument(
            "--encoder-layerdrop",
            type=float,
            metavar="D",
            default=0,
            help="LayerDrop probability for encoder",
        )
        parser.add_argument(
            "--encoder-layers-to-keep",
            default=None,
            help="which layers to *keep* when pruning as a comma-separated list",
        )
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument(
            "--quant-noise-pq",
            type=float,
            metavar="D",
            default=0,
            help="iterative PQ quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-pq-block-size",
            type=int,
            metavar="D",
            default=8,
            help="block size of quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-scalar",
            type=float,
            metavar="D",
            default=0,
            help="scalar quantization noise and scalar quantization at training time",
        )
        parser.add_argument(
            "--untie-weights-roberta",
            action="store_true",
            help="Untie weights between embeddings and classifiers in RoBERTa",
        )
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            default=False,
            help="Apply spectral normalization on the classification head",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, 'max_positions_molecule'):
            args.max_source_positions = DEFAULT_MAX_MOLECULE_POSITIONS
        if not hasattr(args, 'max_positions_protein'):
            args.max_target_positions = DEFAULT_MAX_PROTEIN_POSITIONS

        encoder_0 = DTIRobertaEncoder(args, task.source_dictionary_0, args.max_positions_molecule)
        encoder_1 = DTIRobertaEncoder(args, task.source_dictionary_1, args.max_positions_protein)

        return cls(args, encoder_0, encoder_1)
    
    
    def forward(
        self,
        src_tokens_0,
        src_tokens_1,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        **kwargs
    ):
        if classification_head_name is not None:
            features_only = True
        
        encoder_padding_mask_0 = src_tokens_0.eq(1)
        encoder_padding_mask_1 = src_tokens_1.eq(1)

        x_0, extra_0 = self.encoder_0(src_tokens_0, features_only, return_all_hiddens, **kwargs)
        x_1, extra_1 = self.encoder_1(src_tokens_1, features_only, return_all_hiddens, **kwargs)
        if classification_head_name is not None:
            if self.args.use_2_attention:
                cls_0_attn_1, _ = self.cross_attn(
                    query = x_0[:, 0, :].unsqueeze(1).transpose(0, 1),
                    key = x_1.transpose(0, 1),
                    value = x_1.transpose(0, 1),
                    key_padding_mask=encoder_padding_mask_1,
                    need_weights=False,
                )
                cls_1_attn_0, _ = self.cross_attn(
                    query = x_1[:, 0, :].unsqueeze(1).transpose(0, 1),
                    key = x_0.transpose(0, 1),
                    value = x_0.transpose(0, 1),
                    key_padding_mask=encoder_padding_mask_0,
                    need_weights=False,
                )
                x = torch.cat((cls_0_attn_1.transpose(0, 1), cls_1_attn_0.transpose(0, 1)), 2)
                if isinstance(x, Tensor):
                    x = GradMultiply.apply(x, self.args.grad_multiply)
                x = self.classification_heads[classification_head_name](x)
            else:
                x = torch.cat((x_0[:, 0, :], x_1[:, 0, :]), 1).unsqueeze(1)
                if isinstance(x, Tensor):
                    x = GradMultiply.apply(x, self.args.grad_multiply)
                x = self.classification_heads[classification_head_name](x)
        # extra_0 is not used
        return x, extra_0

    def my_register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = RobertaClassificationHead(
            input_dim=2 * self.args.encoder_embed_dim, # concat [CLS] tokens
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            q_noise=self.args.quant_noise_pq,
            qn_block_size=self.args.quant_noise_pq_block_size,
            do_spectral_norm=self.args.spectral_norm_classification_head,
        )


@register_model_architecture(
    "roberta_dti_cross_attn", "roberta_dti_cross_attn"
)
def base_architecture(args):
    roberta_base_architecture(args)




