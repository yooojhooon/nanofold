from __future__ import annotations

import random
import sh
from math import pi, sqrt
from pathlib import Path
from itertools import product, zip_longest
from functools import partial, wraps
from collections import namedtuple

import torch
from torch import nn
from torch import Tensor, tensor, is_tensor
from torch.amp import autocast
import torch.nn.functional as F
from torch.utils._pytree import tree_map

from torch.nn import (
    Module,
    ModuleList,
    Linear,
    Sequential,
)

from beartype.typing import (
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Tuple,
)

from alphafold3_pytorch.tensor_typing import (
    Float,
    Int,
    Bool,
    Shaped,
    typecheck,
    checkpoint,
    IS_DEBUGGING,
    DEEPSPEED_CHECKPOINTING
)

from alphafold3_pytorch.attention import (
    Attention,
    pad_at_dim,
    slice_at_dim,
    pad_or_slice_to,
    pad_to_multiple,
    concat_previous_window,
    full_attn_bias_to_windowed,
    full_pairwise_repr_to_windowed, LinearNoBias
)

from alphafold3_pytorch.inputs import (
    CONSTRAINT_DIMS,
    CONSTRAINTS,
    CONSTRAINTS_MASK_VALUE,
    IS_MOLECULE_TYPES,
    IS_NON_NA_INDICES,
    IS_PROTEIN_INDEX,
    IS_DNA_INDEX,
    IS_RNA_INDEX,
    IS_LIGAND_INDEX,
    IS_METAL_ION_INDEX,
    IS_BIOMOLECULE_INDICES,
    IS_NON_PROTEIN_INDICES,
    IS_PROTEIN,
    IS_DNA,
    IS_RNA,
    IS_LIGAND,
    IS_METAL_ION,
    MAX_DNA_NUCLEOTIDE_ID,
    MIN_RNA_NUCLEOTIDE_ID,
    MISSING_RNA_NUCLEOTIDE_ID,
    NUM_HUMAN_AMINO_ACIDS,
    NUM_MOLECULE_IDS,
    NUM_MSA_ONE_HOT,
    DEFAULT_NUM_MOLECULE_MODS,
    ADDITIONAL_MOLECULE_FEATS,
    hard_validate_atom_indices_ascending,
    BatchedAtomInput,
    Alphafold3Input,
    alphafold3_inputs_to_batched_atom_input,
)

from alphafold3_pytorch.common.biomolecule import (
    get_residue_constants,
)

from alphafold3_pytorch.nlm import (
    NLMEmbedding,
    NLMRegistry,
    remove_nlms
)
from alphafold3_pytorch.plm import (
    PLMEmbedding,
    PLMRegistry,
    remove_plms
)

from alphafold3_pytorch.utils.model_utils import (
    ExpressCoordinatesInFrame,
    RigidFrom3Points,
    RigidFromReference3Points,
    calculate_weighted_rigid_align_weights,
    pack_one
)
from alphafold3_pytorch.utils.utils import get_gpu_type, not_exists

from alphafold3_pytorch.utils.model_utils import distance_to_dgram

from frame_averaging_pytorch import FrameAverage

from taylor_series_linear_attention import TaylorSeriesLinearAttn

from colt5_attention import ConditionalRoutedAttention

import einx
from einops import rearrange, repeat, reduce, einsum, pack, unpack
from einops.layers.torch import Rearrange

from tqdm import tqdm

from loguru import logger

from importlib.metadata import version

from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from Bio.PDB.Structure import Structure
from Bio.PDB.StructureBuilder import StructureBuilder


class InputFeatureEmbedder(Module):
    """ Algorithm 2 """

    def __init__(
        self,
        *,
        dim_atom_inputs,
        dim_atompair_inputs = 5,
        atoms_per_window = 27,
        dim_atom = 128,
        dim_atompair = 16,
        dim_token = 384,
        dim_single = 384,
        dim_pairwise = 128,
        dim_additional_token_feats = 33,
        num_molecule_types = NUM_MOLECULE_IDS,
        atom_transformer_blocks = 3,
        atom_transformer_heads = 4,
        atom_transformer_kwargs: dict = dict(),
    ):
        super().__init__()
        self.atoms_per_window = atoms_per_window

        self.to_atom_feats = LinearNoBias(dim_atom_inputs, dim_atom)

        self.to_atompair_feats = LinearNoBias(dim_atompair_inputs, dim_atompair)

        self.atom_repr_to_atompair_feat_cond = nn.Sequential(
            nn.LayerNorm(dim_atom),
            LinearNoBias(dim_atom, dim_atompair * 2),
            nn.ReLU()
        )

        self.atompair_feats_mlp = nn.Sequential(
            LinearNoBias(dim_atompair, dim_atompair),
            nn.ReLU(),
            LinearNoBias(dim_atompair, dim_atompair),
            nn.ReLU(),
            LinearNoBias(dim_atompair, dim_atompair),
        )

        from AF3.alphafold3 import DiffusionTransformer
        self.atom_transformer = DiffusionTransformer(
            depth = atom_transformer_blocks,
            heads = atom_transformer_heads,
            dim = dim_atom,
            dim_single_cond = dim_atom,
            dim_pairwise = dim_atompair,
            attn_window_size = atoms_per_window,
            **atom_transformer_kwargs
        )

        self.atom_feats_to_pooled_token = AtomToTokenPooler(
            dim = dim_atom,
            dim_out = dim_token
        )

        dim_single_input = dim_token + dim_additional_token_feats

        self.dim_additional_token_feats = dim_additional_token_feats

        self.single_input_to_single_init = LinearNoBias(dim_single_input, dim_single)
        self.single_input_to_pairwise_init = LinearNoBiasThenOuterSum(dim_single_input, dim_pairwise)

        # this accounts for the `restypes` in the additional molecule features

        self.single_molecule_embed = nn.Embedding(num_molecule_types, dim_single)
        self.pairwise_molecule_embed = nn.Embedding(num_molecule_types, dim_pairwise)

    @typecheck
    def forward(
        self,
        *,
        atom_inputs: Float['b m dai'],
        atompair_inputs: Float['b m m dapi'] | Float['b nw w1 w2 dapi'],
        atom_mask: Bool['b m'],
        molecule_atom_lens: Int['b n'],
        molecule_ids: Int['b n'],
        additional_token_feats: Float['b n {self.dim_additional_token_feats}'] | None = None,

    ) -> EmbeddedInputs:

        w = self.atoms_per_window

        atom_feats = self.to_atom_feats(atom_inputs)
        atompair_feats = self.to_atompair_feats(atompair_inputs)

        # window the atom pair features before passing to atom encoder and decoder

        is_windowed = atompair_inputs.ndim == 5

        if not is_windowed:
            atompair_feats = full_pairwise_repr_to_windowed(atompair_feats, window_size = w)

        # condition atompair with atom repr

        atom_feats_cond = self.atom_repr_to_atompair_feat_cond(atom_feats)

        atom_feats_cond = pad_and_window(atom_feats_cond, w)

        atom_feats_cond_row, atom_feats_cond_col = atom_feats_cond.chunk(2, dim = -1)
        atom_feats_cond_col = concat_previous_window(atom_feats_cond_col, dim_seq = 1, dim_window = -2)

        atompair_feats = einx.add('b nw w1 w2 dap, b nw w1 dap',atompair_feats, atom_feats_cond_row)
        atompair_feats = einx.add('b nw w1 w2 dap, b nw w2 dap',atompair_feats, atom_feats_cond_col)

        # initial atom transformer

        atom_feats = self.atom_transformer(
            atom_feats,
            single_repr = atom_feats,
            pairwise_repr = atompair_feats
        )

        atompair_feats = self.atompair_feats_mlp(atompair_feats) + atompair_feats

        single_inputs = self.atom_feats_to_pooled_token(
            atom_feats = atom_feats,
            atom_mask = atom_mask,
            molecule_atom_lens = molecule_atom_lens
        )

        if exists(additional_token_feats):
            single_inputs = torch.cat((
                single_inputs,
                additional_token_feats
            ), dim = -1)

        single_init = self.single_input_to_single_init(single_inputs)
        pairwise_init = self.single_input_to_pairwise_init(single_inputs)

        # account for molecule id (restypes)

        molecule_ids = torch.where(molecule_ids >= 0, molecule_ids, 0) # account for padding

        single_molecule_embed = self.single_molecule_embed(molecule_ids)

        pairwise_molecule_embed = self.pairwise_molecule_embed(molecule_ids)
        pairwise_molecule_embed = einx.add('b i dp, b j dp -> b i j dp', pairwise_molecule_embed, pairwise_molecule_embed)

        # sum to single init and pairwise init, equivalent to one-hot in additional residue features

        single_init = single_init + single_molecule_embed
        pairwise_init = pairwise_init + pairwise_molecule_embed

        return EmbeddedInputs(single_inputs, single_init, pairwise_init, atom_feats, atompair_feats)

class AttentionPairBias(Module):
    """An Attention module with pair bias computation."""

    def __init__(self, *, heads, dim_pairwise, window_size=None, num_memory_kv=0, **attn_kwargs):
        super().__init__()

        self.window_size = window_size

        self.attn = Attention(
            heads = heads,
            window_size = window_size,
            num_memory_kv = num_memory_kv,
            **attn_kwargs
        )

        # line 8 of Algorithm 24

        self.to_attn_bias_norm = nn.LayerNorm(dim_pairwise)
        self.to_attn_bias = nn.Sequential(LinearNoBias(dim_pairwise, heads), Rearrange("b ... h -> b h ..."))

    @typecheck
    def forward(
        self,
        single_repr: Float["b n ds"],  # type: ignore
        *,
        pairwise_repr: Float["b n n dp"] | Float["b nw w (w*2) dp"],  # type: ignore
        attn_bias: Float["b n n"] | Float["b nw w (w*2)"] | None = None,  # type: ignore
        return_values: bool = False,
        value_residual: Float['b _ _'] | None = None,
        **kwargs,
    ) -> (
        Float['b n ds'] |
        tuple[Float['b n ds'], Float['b _ _']]
    ):  # type: ignore

        """Perform the forward pass.

        :param single_repr: The single representation tensor.
        :param pairwise_repr: The pairwise representation tensor.
        :param attn_bias: The attention bias tensor.
        :return: The output tensor.
        """
        b, dp = pairwise_repr.shape[0], pairwise_repr.shape[-1]
        dtype, device = pairwise_repr.dtype, pairwise_repr.device
        w, has_window_size = self.window_size, exists(self.window_size)

        # take care of windowing logic
        # for sequence-local atom transformer

        windowed_pairwise = pairwise_repr.ndim == 5

        windowed_attn_bias = None

        if exists(attn_bias):
            windowed_attn_bias = attn_bias.shape[-1] != attn_bias.shape[-2]

        if has_window_size:
            if not windowed_pairwise:
                pairwise_repr = full_pairwise_repr_to_windowed(pairwise_repr, window_size=w)
            if exists(attn_bias):
                attn_bias = full_attn_bias_to_windowed(attn_bias, window_size=w)
        else:
            assert (
                not windowed_pairwise
            ), "Cannot pass in windowed pairwise representation if no `window_size` given to `AttentionPairBias`."
            assert (
                not_exists(windowed_attn_bias) or not windowed_attn_bias
            ), "Cannot pass in windowed attention bias if no `window_size` is set for `AttentionPairBias`."

        # attention bias preparation with further addition from pairwise repr

        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, "b ... -> b 1 ...")
        else:
            attn_bias = 0.0

        if pairwise_repr.numel() > MAX_CONCURRENT_TENSOR_ELEMENTS:
            # create a stub tensor and normalize it to maintain gradients to `to_attn_bias_norm`
            stub_pairwise_repr = torch.zeros((b, dp), dtype=dtype, device=device)
            stub_attn_bias_norm = self.to_attn_bias_norm(stub_pairwise_repr) * 0.0

            # adjust `attn_bias_norm` dimensions to match `pairwise_repr`
            attn_bias_norm = pairwise_repr + (
                stub_attn_bias_norm[:, None, None, None, :]
                if windowed_pairwise
                else stub_attn_bias_norm[:, None, None, :]
            )

            # apply bias transformation
            attn_bias = self.to_attn_bias(attn_bias_norm) + attn_bias
        else:
            attn_bias = self.to_attn_bias(self.to_attn_bias_norm(pairwise_repr)) + attn_bias

        # attention

        out, values = self.attn(
            single_repr,
            attn_bias = attn_bias,
            value_residual = value_residual,
            return_values = True,
            **kwargs
        )

        # whether to return values for value residual learning

        if not return_values:
            return out

        return out, values

class TriangleAttention(Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        node_type: Literal['starting', 'ending'],
        dropout = 0.,
        dropout_type: Literal['row', 'col'] | None = None,
        **attn_kwargs
    ):
        super().__init__()
        self.need_transpose = node_type == 'ending'

        self.attn = Attention(dim = dim, heads = heads, **attn_kwargs)

        # self.dropout = Dropout(dropout, dropout_type = dropout_type)

        self.to_attn_bias = nn.Sequential(
            LinearNoBias(dim, heads),
            Rearrange('... i j h -> ... h i j')
        )

    @typecheck
    def forward(
        self,
        pairwise_repr: Float['b n n d'],
        mask: Bool['b n'] | None = None,
        **kwargs
    ) -> Float['b n n d']:

        if self.need_transpose:
            pairwise_repr = rearrange(pairwise_repr, 'b i j d -> b j i d')

        attn_bias = self.to_attn_bias(pairwise_repr)

        batch_repeat = pairwise_repr.shape[1]
        attn_bias = repeat(attn_bias, 'b ... -> (b repeat) ...', repeat = batch_repeat)

        if exists(mask):
            mask = repeat(mask, 'b ... -> (b repeat) ...', repeat = batch_repeat)

        pairwise_repr, unpack_one = pack_one(pairwise_repr, '* n d')

        out = self.attn(
            pairwise_repr,
            mask = mask,
            attn_bias = attn_bias,
            **kwargs
        )

        out = unpack_one(out)

        if self.need_transpose:
            out = rearrange(out, 'b j i d -> b i j d')

        return self.dropout(out)

class MSAModule(Module):

    """ Algorithm 8 """

    def __init__(
        self,
        *,
        dim_single = 384,
        dim_pairwise = 128,
        depth = 4,
        dim_msa = 64,
        dim_msa_input=NUM_MSA_ONE_HOT,
        dim_additional_msa_feats=2,
        outer_product_mean_dim_hidden = 32,
        msa_pwa_dropout_row_prob = 0.15,
        msa_pwa_heads = 8,
        msa_pwa_dim_head = 32,
        checkpoint = False,
        pairwise_block_kwargs: dict = dict(),
        max_num_msa: int | None = None,
        layerscale_output: bool = True
    ):
        super().__init__()

        self.max_num_msa = default(
            max_num_msa, float('inf')
        )  # cap the number of MSAs, will do sample without replacement if exceeds

        self.msa_init_proj = LinearNoBias(dim_msa_input + dim_additional_msa_feats, dim_msa)

        self.single_to_msa_feats = LinearNoBias(dim_single, dim_msa)

        layers = ModuleList([])

        for _ in range(depth):

            msa_pre_ln = partial(PreLayerNorm, dim = dim_msa)

            outer_product_mean = OuterProductMean(
                dim_msa = dim_msa,
                dim_pairwise = dim_pairwise,
                dim_hidden = outer_product_mean_dim_hidden
            )

            msa_pair_weighted_avg = MSAPairWeightedAveraging(
                dim_msa = dim_msa,
                dim_pairwise = dim_pairwise,
                heads = msa_pwa_heads,
                dim_head = msa_pwa_dim_head,
                dropout = msa_pwa_dropout_row_prob,
                dropout_type = 'row'
            )

            msa_transition = Transition(dim = dim_msa)

            pairwise_block = PairwiseBlock(
                dim_pairwise = dim_pairwise,
                **pairwise_block_kwargs
            )

            layers.append(ModuleList([
                outer_product_mean,
                msa_pair_weighted_avg,
                msa_pre_ln(msa_transition),
                pairwise_block
            ]))

        self.checkpoint = checkpoint

        self.layers = layers

        self.layerscale_output = nn.Parameter(torch.zeros(dim_pairwise)) if layerscale_output else 1.

        # msa related

        self.dmi = dim_additional_msa_feats

    @typecheck
    def to_layers(
        self,
        *,
        pairwise_repr: Float['b n n dp'],
        msa: Float['b s n dm'],
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None,
    ) -> Float['b n n dp']:

        for (
            outer_product_mean,
            msa_pair_weighted_avg,
            msa_transition,
            pairwise_block
        ) in self.layers:

            # communication between msa and pairwise rep

            pairwise_repr = outer_product_mean(msa, mask = mask, msa_mask = msa_mask) + pairwise_repr

            msa = msa_pair_weighted_avg(msa = msa, pairwise_repr = pairwise_repr, mask = mask) + msa
            msa = msa_transition(msa) + msa

            # pairwise block

            pairwise_repr = pairwise_block(pairwise_repr = pairwise_repr, mask = mask)

        return pairwise_repr

    @typecheck
    def to_checkpointed_layers(
        self,
        *,
        pairwise_repr: Float['b n n dp'],
        msa: Float['b s n dm'],
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None,
    ) -> Float['b n n dp']:

        inputs = (pairwise_repr, mask, msa, msa_mask)

        wrapped_layers = []

        def outer_product_mean_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                pairwise_repr, mask, msa, msa_mask = inputs
                pairwise_repr = fn(msa = msa, mask = mask, msa_mask = msa_mask) + pairwise_repr
                return pairwise_repr, mask, msa, msa_mask
            return inner

        def msa_pair_weighted_avg_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                pairwise_repr, mask, msa, msa_mask = inputs
                msa = fn(msa = msa, pairwise_repr = pairwise_repr, mask = mask) + msa
                return pairwise_repr, mask, msa, msa_mask
            return inner

        def msa_transition_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                pairwise_repr, mask, msa, msa_mask = inputs
                msa = fn(msa) + msa
                return pairwise_repr, mask, msa, msa_mask
            return inner

        def pairwise_block_wrapper(fn):
            @wraps(fn)
            def inner(inputs):
                pairwise_repr, mask, msa, msa_mask = inputs
                pairwise_repr = fn(pairwise_repr = pairwise_repr, mask = mask)
                return pairwise_repr, mask, msa, msa_mask
            return inner

        for (
            outer_product_mean,
            msa_pair_weighted_avg,
            msa_transition,
            pairwise_block
        ) in self.layers:

            wrapped_layers.append(outer_product_mean_wrapper(outer_product_mean))
            wrapped_layers.append(msa_pair_weighted_avg_wrapper(msa_pair_weighted_avg))
            wrapped_layers.append(msa_transition_wrapper(msa_transition))
            wrapped_layers.append(pairwise_block_wrapper(pairwise_block))

        for layer in wrapped_layers:
            inputs = checkpoint(layer, inputs)

        pairwise_repr, *_ = inputs
        return pairwise_repr

    @typecheck
    def forward(
        self,
        *,
        single_repr: Float['b n ds'],
        pairwise_repr: Float['b n n dp'],
        msa: Float['b s n dm'],
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None,
        additional_msa_feats: Float['b s n {self.dmi}'] | None = None,
    ) -> Float['b n n dp']:

        batch, num_msa, device = *msa.shape[:2], msa.device

        # sample without replacement

        if num_msa > self.max_num_msa:
            rand = torch.randn((batch, num_msa), device = device)

            if exists(msa_mask):
                rand.masked_fill_(~msa_mask, max_neg_value(msa))

            indices = rand.topk(self.max_num_msa, dim = -1).indices

            # msa = einx.get_at('b [s] n dm, b sampled -> b sampled n dm', msa, indices)

            msa, unpack_one = pack_one(msa, 'b s *')
            msa_indices = repeat(indices, 'b sampled -> b sampled d', d = msa.shape[-1])
            msa = msa.gather(1, msa_indices)
            msa = unpack_one(msa)

            if exists(msa_mask):
                # msa_mask = einx.get_at('b [s], b sampled -> b sampled', msa_mask, indices)
                msa_mask = msa_mask.gather(1, indices)

            if exists(additional_msa_feats):
                # additional_msa_feats = einx.get_at('b s 2, b sampled -> b sampled 2', additional_msa_feats, indices)

                additional_msa_feats, unpack_one = pack_one(additional_msa_feats, 'b s *')
                additional_msa_indices = repeat(
                    indices, 'b sampled -> b sampled d', d=additional_msa_feats.shape[-1]
                )
                additional_msa_feats = additional_msa_feats.gather(1, additional_msa_indices)
                additional_msa_feats = unpack_one(additional_msa_feats)

        # account for no msa

        if exists(msa_mask):
            has_msa = reduce(msa_mask, 'b s -> b', 'any')

        # account for additional msa features

        if exists(additional_msa_feats):
            msa = torch.cat((msa, additional_msa_feats), dim=-1)

        # process msa

        msa = self.msa_init_proj(msa)

        single_msa_feats = self.single_to_msa_feats(single_repr)

        msa = rearrange(single_msa_feats, 'b n d -> b 1 n d') + msa

        # going through the layers

        if should_checkpoint(self, (pairwise_repr, msa)):
            to_layers_fn = self.to_checkpointed_layers
        else:
            to_layers_fn = self.to_layers

        pairwise_repr = to_layers_fn(
            msa = msa,
            mask = mask,
            pairwise_repr = pairwise_repr,
            msa_mask = msa_mask
        )

        # final masking and then layer scale

        if exists(msa_mask):
            pairwise_repr = einx.where(
                'b, b ..., -> b ...',
                has_msa, pairwise_repr, 0.
            )

        return pairwise_repr * self.layerscale_output

class Nanofold(Module):
    """ Algorithm 1 """

    @save_args_and_kwargs
    @typecheck
    def __init__(
            self,
            *,
            dim_atom_inputs,
            dim_template_feats,
            dim_template_model=64,
            atoms_per_window=27,
            dim_atom=128,
            dim_atompair_inputs=5,
            dim_atompair=16,
            dim_input_embedder_token=384,
            dim_single=384,
            dim_pairwise=128,
            dim_token=768,
            dim_msa_inputs=NUM_MSA_ONE_HOT,
            dim_additional_msa_feats=2,

            # in paper, they include two meta information per msa-token pair (has_deletion w/ dim=1, deletion_value w/ dim=1)
            dim_additional_token_feats=33,
            # in paper, they include two meta information per token (profile w/ dim=32, deletion_mean w/ dim=1)
            num_molecule_types: int = NUM_MOLECULE_IDS,
            # restype in additional residue information, apparently 32. will do 33 to account for metal ions

            num_atom_embeds: int | None = None,
            num_atompair_embeds: int | None = None,
            num_molecule_mods: int | None = DEFAULT_NUM_MOLECULE_MODS,
            distance_bins: List[float] = torch.linspace(2, 22, 64).float().tolist(),
            # NOTE: in paper, they reuse AF2's setup of having 64 bins from 2 to 22
            pae_bins: List[float] = torch.linspace(0.5, 32, 64).float().tolist(),
            pde_bins: List[float] = torch.linspace(0.5, 32, 64).float().tolist(),
            ignore_index=-1,
            num_dist_bins: int | None = None,
            num_plddt_bins=50,
            num_pae_bins: int | None = None,
            num_pde_bins: int | None = None,
            sigma_data=16,
            num_rollout_steps=20,
            diffusion_num_augmentations=48,
            loss_confidence_weight=1e-4,
            loss_distogram_weight=1e-2,
            loss_diffusion_weight=4.,
            input_embedder_kwargs: dict = dict(
                atom_transformer_blocks=3,
                atom_transformer_heads=4,
                atom_transformer_kwargs=dict()
            ),
            confidence_head_kwargs: dict = dict(
                pairformer_depth=4
            ),
            template_embedder_kwargs: dict = dict(
                pairformer_stack_depth=2,
                pairwise_block_kwargs=dict(),
                layerscale_output=True,
            ),
            msa_module_kwargs: dict = dict(
                depth=4,
                dim_msa=64,
                outer_product_mean_dim_hidden=32,
                msa_pwa_dropout_row_prob=0.15,
                msa_pwa_heads=8,
                msa_pwa_dim_head=32,
                pairwise_block_kwargs=dict(),
                layerscale_output=True,
            ),
            pairformer_stack: dict = dict(
                depth=48,
                pair_bias_attn_dim_head=64,
                pair_bias_attn_heads=16,
                dropout_row_prob=0.25,
                pairwise_block_kwargs=dict()
            ),
            relative_position_encoding_kwargs: dict = dict(
                r_max=32,
                s_max=2,
            ),
            diffusion_module_kwargs: dict = dict(
                single_cond_kwargs=dict(
                    num_transitions=2,
                    transition_expansion_factor=2,
                ),
                pairwise_cond_kwargs=dict(
                    num_transitions=2
                ),
                atom_encoder_depth=3,
                atom_encoder_heads=4,
                token_transformer_depth=24,
                token_transformer_heads=16,
                atom_decoder_depth=3,
                atom_decoder_heads=4,
            ),
            edm_kwargs: dict = dict(
                sigma_min=0.002,
                sigma_max=80,
                rho=7,
                P_mean=-1.2,
                P_std=1.2,
                S_churn=80,
                S_tmin=0.05,
                S_tmax=50,
                S_noise=1.003,
            ),
            weighted_rigid_align_kwargs: dict = dict(),
            multi_chain_permutation_alignment_kwargs: dict = dict(),
            lddt_mask_nucleic_acid_cutoff=30.0,
            lddt_mask_other_cutoff=15.0,
            nucleotide_loss_weight=5.0,
            ligand_loss_weight=10.0,
            augment_kwargs: dict = dict(),
            stochastic_frame_average=False,
            distogram_atom_resolution=False,
            checkpoint_input_embedding=False,
            checkpoint_trunk_pairformer=False,
            checkpoint_distogram_head=False,
            checkpoint_confidence_head=False,
            checkpoint_diffusion_module=False,
            detach_when_recycling=True,
            pdb_training_set=True,
            plm_embeddings: PLMEmbedding | tuple[PLMEmbedding, ...] | None = None,
            nlm_embeddings: NLMEmbedding | tuple[NLMEmbedding, ...] | None = None,
            plm_kwargs: dict | tuple[dict, ...] | None = None,
            nlm_kwargs: dict | tuple[dict, ...] | None = None,
            constraints: List[CONSTRAINTS] | None = None,
            verbose: bool = False,
    ):

        super().__init__()

        self.verbose = verbose

        # store atom and atompair input dimensions for shape validation

        self.dim_atom_inputs = dim_atom_inputs
        self.dim_template_feats = dim_template_feats
        self.dim_atompair_inputs = dim_atompair_inputs

        # optional atom and atom bond embeddings

        num_atom_embeds = default(num_atom_embeds, 0)
        num_atompair_embeds = default(num_atompair_embeds, 0)

        has_atom_embeds = num_atom_embeds > 0
        has_atompair_embeds = num_atompair_embeds > 0

        if has_atom_embeds:
            self.atom_embeds = nn.Embedding(num_atom_embeds, dim_atom)

        if has_atompair_embeds:
            self.atompair_embeds = nn.Embedding(num_atompair_embeds, dim_atompair)

        self.has_atom_embeds = has_atom_embeds
        self.has_atompair_embeds = has_atompair_embeds

        # optional pairwise token constraint embeddings

        self.constraints = constraints

        if exists(constraints):
            self.constraint_embeds = nn.ModuleList(
                [
                    LinearNoBias(CONSTRAINT_DIMS[constraint], dim_pairwise)
                    for constraint in constraints
                ]
            )
            self.learnable_constraint_masks = nn.ParameterList(
                [nn.Parameter(torch.randn(1)) for _ in constraints]
            )

        # residue or nucleotide modifications

        num_molecule_mods = default(num_molecule_mods, 0)
        has_molecule_mod_embeds = num_molecule_mods > 0

        if has_molecule_mod_embeds:
            self.molecule_mod_embeds = nn.Embedding(num_molecule_mods, dim_single)

        self.has_molecule_mod_embeds = has_molecule_mod_embeds

        # optional protein language model(s) (PLM) embeddings

        self.plms = None

        if exists(plm_embeddings):
            self.plms = ModuleList([])

            for one_plm_embedding, one_plm_kwargs in zip_longest(cast_tuple(plm_embeddings), cast_tuple(plm_kwargs)):
                assert one_plm_embedding in PLMRegistry, f'received invalid plm embedding name {one_plm_embedding} - acceptable ones are {PLMRegistry.keys()}'
                constructor = PLMRegistry.get(one_plm_embedding)

                one_plm_kwargs = default(one_plm_kwargs, {})
                plm = constructor(**one_plm_kwargs)

                freeze_(plm)

                self.plms.append(plm)

        if exists(self.plms):
            concatted_plm_embed_dim = sum([plm.embed_dim for plm in self.plms])

            self.to_plm_embeds = LinearNoBias(concatted_plm_embed_dim, dim_single)

        # optional nucleotide language model(s) (NLM) embeddings

        self.nlms = None

        if exists(nlm_embeddings):
            self.nlms = ModuleList([])

            for one_nlm_embedding, one_nlm_kwargs in zip_longest(
                    cast_tuple(nlm_embeddings), cast_tuple(nlm_kwargs)
            ):
                assert (
                        one_nlm_embedding in NLMRegistry
                ), f"Received invalid NLM embedding name: {one_nlm_embedding}. Acceptable ones are {list(NLMRegistry.keys())}."

                constructor = NLMRegistry.get(one_nlm_embedding)

                one_nlm_kwargs = default(one_nlm_kwargs, {})
                nlm = constructor(**one_nlm_kwargs)

                freeze_(nlm)

                self.nlms.append(nlm)

        if exists(self.nlms):
            concatted_nlm_embed_dim = sum([nlm.embed_dim for nlm in self.nlms])

            self.to_nlm_embeds = LinearNoBias(concatted_nlm_embed_dim, dim_single)

        # atoms per window

        self.atoms_per_window = atoms_per_window

        # augmentation

        self.num_augmentations = diffusion_num_augmentations
        self.augmenter = CentreRandomAugmentation(**augment_kwargs)

        # stochastic frame averaging
        # https://arxiv.org/abs/2305.05577

        self.stochastic_frame_average = stochastic_frame_average

        if stochastic_frame_average:
            self.frame_average = FrameAverage(
                dim=3,
                stochastic=True,
                return_stochastic_as_augmented_pos=True
            )

        # input feature embedder

        self.input_embedder = InputFeatureEmbedder(
            num_molecule_types=num_molecule_types,
            dim_atom_inputs=dim_atom_inputs,
            dim_atompair_inputs=dim_atompair_inputs,
            atoms_per_window=atoms_per_window,
            dim_atom=dim_atom,
            dim_atompair=dim_atompair,
            dim_token=dim_input_embedder_token,
            dim_single=dim_single,
            dim_pairwise=dim_pairwise,
            dim_additional_token_feats=dim_additional_token_feats,
            **input_embedder_kwargs
        )

        # they concat some MSA related information per token (`profile` w/ dim=32, `deletion_mean` w/ dim=1)
        # line 2 of Algorithm 2
        # the `f_restypes` is handled elsewhere

        dim_single_inputs = dim_input_embedder_token + dim_additional_token_feats

        # relative positional encoding
        # used by pairwise in main alphafold2 trunk
        # and also in the diffusion module separately from alphafold3

        self.relative_position_encoding = RelativePositionEncoding(
            dim_out=dim_pairwise,
            **relative_position_encoding_kwargs
        )

        # token bonds
        # Algorithm 1 - line 5

        self.token_bond_to_pairwise_feat = nn.Sequential(
            Rearrange('... -> ... 1'),
            LinearNoBias(1, dim_pairwise)
        )

        # templates

        self.template_embedder = TemplateEmbedder(
            dim_template_feats=dim_template_feats,
            dim=dim_template_model,
            dim_pairwise=dim_pairwise,
            checkpoint=checkpoint_input_embedding,
            **template_embedder_kwargs
        )

        # msa

        # they concat some MSA related information per MSA-token pair (`has_deletion` w/ dim=1, `deletion_value` w/ dim=1)

        self.msa_module = MSAModule(
            dim_single=dim_single,
            dim_pairwise=dim_pairwise,
            dim_msa_input=dim_msa_inputs,
            dim_additional_msa_feats=dim_additional_msa_feats,
            checkpoint=checkpoint_input_embedding,
            **msa_module_kwargs,
        )

        # main pairformer trunk, 48 layers

        self.pairformer = PairformerStack(
            dim_single=dim_single,
            dim_pairwise=dim_pairwise,
            checkpoint=checkpoint_trunk_pairformer,
            **pairformer_stack
        )

        # recycling related

        self.detach_when_recycling = detach_when_recycling

        self.recycle_single = nn.Sequential(
            nn.LayerNorm(dim_single),
            LinearNoBias(dim_single, dim_single)
        )

        self.recycle_pairwise = nn.Sequential(
            nn.LayerNorm(dim_pairwise),
            LinearNoBias(dim_pairwise, dim_pairwise)
        )

        # diffusion

        self.diffusion_module = DiffusionModule(
            dim_pairwise_trunk=dim_pairwise,
            dim_pairwise_rel_pos_feats=dim_pairwise,
            atoms_per_window=atoms_per_window,
            dim_pairwise=dim_pairwise,
            sigma_data=sigma_data,
            dim_atom=dim_atom,
            dim_atompair=dim_atompair,
            dim_token=dim_token,
            dim_single=dim_single + dim_single_inputs,
            checkpoint=checkpoint_diffusion_module,
            **diffusion_module_kwargs
        )

        self.edm = ElucidatedAtomDiffusion(
            self.diffusion_module,
            sigma_data=sigma_data,
            smooth_lddt_loss_kwargs=dict(
                nucleic_acid_cutoff=lddt_mask_nucleic_acid_cutoff,
                other_cutoff=lddt_mask_other_cutoff,
            ),
            **edm_kwargs
        )

        self.num_rollout_steps = num_rollout_steps

        # logit heads

        distance_bins_tensor = tensor(distance_bins)

        self.register_buffer('distance_bins', distance_bins_tensor)
        num_dist_bins = default(num_dist_bins, len(distance_bins_tensor))

        assert len(
            distance_bins_tensor) == num_dist_bins, '`distance_bins` must have a length equal to the `num_dist_bins` passed in'

        self.distogram_atom_resolution = distogram_atom_resolution

        self.distogram_head = DistogramHead(
            dim_pairwise=dim_pairwise,
            dim_atom=dim_atom,
            num_dist_bins=num_dist_bins,
            atom_resolution=distogram_atom_resolution,
            checkpoint=checkpoint_distogram_head
        )

        # lddt related

        self.lddt_mask_nucleic_acid_cutoff = lddt_mask_nucleic_acid_cutoff
        self.lddt_mask_other_cutoff = lddt_mask_other_cutoff

        # pae related bins and modules

        pae_bins_tensor = tensor(pae_bins)
        self.register_buffer('pae_bins', pae_bins_tensor)
        num_pae_bins = len(pae_bins)

        self.rigid_from_three_points = RigidFrom3Points()
        self.compute_alignment_error = ComputeAlignmentError()

        # pde related bins

        pde_bins_tensor = tensor(pde_bins)
        self.register_buffer('pde_bins', pde_bins_tensor)
        num_pde_bins = len(pde_bins)

        # plddt related bins

        self.num_plddt_bins = num_plddt_bins

        # confidence head

        self.confidence_head = ConfidenceHead(
            dim_single_inputs=dim_single_inputs,
            dim_atom=dim_atom,
            atompair_dist_bins=distance_bins,
            dim_single=dim_single,
            dim_pairwise=dim_pairwise,
            num_plddt_bins=num_plddt_bins,
            num_pde_bins=num_pde_bins,
            num_pae_bins=num_pae_bins,
            checkpoint=checkpoint_confidence_head,
            **confidence_head_kwargs
        )

        self.register_buffer('lddt_thresholds', torch.tensor([0.5, 1.0, 2.0, 4.0]))

        # loss related

        self.ignore_index = ignore_index
        self.pdb_training_set = pdb_training_set
        self.loss_distogram_weight = loss_distogram_weight
        self.loss_confidence_weight = loss_confidence_weight
        self.loss_diffusion_weight = loss_diffusion_weight
        self.nucleotide_loss_weight = nucleotide_loss_weight
        self.ligand_loss_weight = ligand_loss_weight

        self.weighted_rigid_align = WeightedRigidAlign(**weighted_rigid_align_kwargs)
        self.multi_chain_permutation_alignment = MultiChainPermutationAlignment(
            **multi_chain_permutation_alignment_kwargs,
            weighted_rigid_align=self.weighted_rigid_align,
        )

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

        # some shorthand for jaxtyping

        self.w = atoms_per_window
        self.dapi = self.dim_atompair_inputs
        self.dai = self.dim_atom_inputs
        self.dmf = dim_additional_msa_feats
        self.dtf = dim_additional_token_feats
        self.dmi = dim_msa_inputs
        self.num_mods = num_molecule_mods

    @property
    def device(self):
        return self.zero.device

    @remove_plms
    @remove_nlms
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    @remove_plms
    @remove_nlms
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    @property
    def state_dict_with_init_args(self):
        return dict(
            version=self._version,
            init_args_and_kwargs=self._args_and_kwargs,
            state_dict=self.state_dict()
        )

    @typecheck
    def save(self, path: str | Path, overwrite=False):
        if isinstance(path, str):
            path = Path(path)

        assert not path.is_dir() and (not path.exists() or overwrite)

        path.parent.mkdir(exist_ok=True, parents=True)

        package = dict(
            model=self.state_dict_with_init_args
        )

        torch.save(package, str(path))

    @typecheck
    def load(
            self,
            path: str | Path,
            strict=False,
            map_location='cpu'
    ):
        if isinstance(path, str):
            path = Path(path)

        assert path.exists() and path.is_file()

        package = torch.load(str(path), map_location=map_location, weights_only=True)

        model_package = package['model']
        current_version = version('alphafold3_pytorch')

        if model_package['version'] != current_version:
            logger.warning(
                f'loading a saved model from version {model_package["version"]} but you are on version {current_version}')

        self.load_state_dict(model_package['state_dict'], strict=strict)

        return package.get('id', None)

    @staticmethod
    @typecheck
    def init_and_load(
            path: str | Path,
            map_location='cpu'
    ):
        if isinstance(path, str):
            path = Path(path)

        assert path.is_file()

        package = torch.load(str(path), map_location=map_location, weights_only=True)

        model_package = package['model']

        args, kwargs = model_package['init_args_and_kwargs']
        alphafold3 = Nanofold(*args, **kwargs)

        alphafold3.load(path)
        return alphafold3

    def shrink_and_perturb_(
            self,
            shrink_factor=0.5,
            perturb_factor=0.01
    ):
        # Shrink & Perturb - Ash et al. https://arxiv.org/abs/1910.08475
        assert 0. <= shrink_factor <= 1.

        for p in self.parameters():
            noise = torch.randn_like(p.data)
            p.data.mul_(1. - shrink_factor).add_(noise * perturb_factor)

        return self

    @typecheck
    def forward_with_alphafold3_inputs(
            self,
            alphafold3_inputs: Alphafold3Input | list[Alphafold3Input],
            **kwargs
    ):
        if not isinstance(alphafold3_inputs, list):
            alphafold3_inputs = [alphafold3_inputs]

        batched_atom_inputs = alphafold3_inputs_to_batched_atom_input(alphafold3_inputs, atoms_per_window=self.w)

        atom_dict = batched_atom_inputs.model_forward_dict()
        atom_dict = dict_to_device(atom_dict, device=self.device)

        return self.forward(**atom_dict, **kwargs)

    @typecheck
    def forward(
            self,
            *,
            atom_inputs: Float['b m {self.dai}'],
            atompair_inputs: Float['b m m {self.dapi}'] | Float['b nw {self.w} {self.w*2} {self.dapi}'],
            additional_molecule_feats: Int[f'b n {ADDITIONAL_MOLECULE_FEATS}'],
            is_molecule_types: Bool[f'b n {IS_MOLECULE_TYPES}'],
            molecule_atom_lens: Int['b n'],
            molecule_ids: Int['b n'],
            additional_msa_feats: Float['b s n {self.dmf}'] | None = None,
            additional_token_feats: Float['b n {self.dtf}'] | None = None,
            atom_ids: Int['b m'] | None = None,
            atompair_ids: Int['b m m'] | Int['b nw {self.w} {self.w*2}'] | None = None,
            is_molecule_mod: Bool['b n {self.num_mods}'] | None = None,
            atom_mask: Bool['b m'] | None = None,
            missing_atom_mask: Bool['b m'] | None = None,
            atom_indices_for_frame: Int['b n 3'] | None = None,
            valid_atom_indices_for_frame: Bool['b n'] | None = None,
            atom_parent_ids: Int['b m'] | None = None,
            token_bonds: Bool['b n n'] | None = None,
            msa: Float['b s n {self.dmi}'] | None = None,
            msa_mask: Bool['b s'] | None = None,
            templates: Float['b t n n dt'] | None = None,
            template_mask: Bool['b t'] | None = None,
            num_recycling_steps: int = 1,
            diffusion_add_bond_loss: bool = False,
            diffusion_add_smooth_lddt_loss: bool = False,
            distogram_atom_indices: Int['b n'] | None = None,
            molecule_atom_indices: Int['b n'] | None = None,
            # the 'token centre atoms' mentioned in the paper, unsure where it is used in the architecture
            num_sample_steps: int | None = None,
            atom_pos: Float['b m 3'] | None = None,
            distance_labels: Int['b n n'] | Int['b m m'] | None = None,
            resolved_labels: Int['b m'] | None = None,
            resolution: Float[' b'] | None = None,
            token_constraints: Float['b n n dac'] | None = None,
            return_loss_breakdown=False,
            return_loss: bool = None,
            return_all_diffused_atom_pos: bool = False,
            return_confidence_head_logits: bool = False,
            return_distogram_head_logits: bool = False,
            return_bio_pdb_structures: bool = False,
            num_rollout_steps: int | None = None,
            rollout_show_tqdm_pbar: bool = False,
            detach_when_recycling: bool = None,
            min_conf_resolution: float = 0.1,
            max_conf_resolution: float = 4.0,
            hard_validate: bool = False,
            verbose: bool | None = None,
            chains: Int["b 2"] | None = None,
            filepath: List[str] | Tuple[str] | None = None,
    ) -> (
            Float['b m 3'] |
            List[Structure] |
            Float['ts b m 3'] |
            Tuple[Float['b m 3'] | List[Structure] | Float['ts b m 3'], ConfidenceHeadLogits | Alphafold3Logits] |
            Float[''] |
            Tuple[Float[''], LossBreakdown]
    ):
        verbose = default(verbose, self.verbose)

        atom_seq_len = atom_inputs.shape[-2]
        single_structure_input = atom_inputs.shape[0] == 1

        dtype = atom_inputs.dtype

        # validate atom and atompair input dimensions

        assert atom_inputs.shape[
                   -1] == self.dim_atom_inputs, f'expected {self.dim_atom_inputs} for atom_inputs feature dimension, but received {atom_inputs.shape[-1]}'
        assert atompair_inputs.shape[
                   -1] == self.dim_atompair_inputs, f'expected {self.dim_atompair_inputs} for atompair_inputs feature dimension, but received {atompair_inputs.shape[-1]}'

        # hard validate when debug env variable is turned on

        if verbose:
            logger.info("Hard validating inputs...")

        hard_debug = hard_validate or IS_DEBUGGING

        if hard_debug:
            maybe(hard_validate_atom_indices_ascending)(distogram_atom_indices, 'distogram_atom_indices')
            maybe(hard_validate_atom_indices_ascending)(molecule_atom_indices, 'molecule_atom_indices')

            is_biomolecule = ~(
                    (~is_molecule_types[..., IS_BIOMOLECULE_INDICES].any(dim=-1))
                    | (exists(is_molecule_mod) and is_molecule_mod.any(dim=-1))
            )
            maybe(hard_validate_atom_indices_ascending)(
                atom_indices_for_frame,
                'atom_indices_for_frame',
                mask=is_biomolecule,
            )

        # soft validate

        if verbose:
            logger.info("Soft validating inputs...")

        valid_molecule_atom_mask = valid_atom_len_mask = molecule_atom_lens >= 0

        molecule_atom_lens = molecule_atom_lens.masked_fill(~valid_atom_len_mask, 0)

        if exists(molecule_atom_indices):
            valid_molecule_atom_mask = molecule_atom_indices >= 0 & valid_atom_len_mask
            molecule_atom_indices = molecule_atom_indices.masked_fill(~valid_molecule_atom_mask, 0)

        if exists(distogram_atom_indices):
            valid_distogram_mask = distogram_atom_indices >= 0 & valid_atom_len_mask
            distogram_atom_indices = distogram_atom_indices.masked_fill(~valid_distogram_mask, 0)

        if exists(atom_indices_for_frame):
            valid_atom_indices_for_frame = default(valid_atom_indices_for_frame,
                                                   torch.ones_like(molecule_atom_lens).bool())

            valid_atom_indices_for_frame = valid_atom_indices_for_frame & (atom_indices_for_frame >= 0).all(
                dim=-1) & valid_atom_len_mask
            atom_indices_for_frame = einx.where('b n, b n three, -> b n three', valid_atom_indices_for_frame,
                                                atom_indices_for_frame, 0)

        assert exists(molecule_atom_lens) or exists(atom_mask)

        if hard_debug:
            assert (molecule_atom_lens >= 0).all(), 'molecule_atom_lens must be greater or equal to 0'

        # if atompair inputs are not windowed, window it

        if verbose:
            logger.info("Windowing atompair inputs...")

        is_atompair_inputs_windowed = atompair_inputs.ndim == 5

        if not is_atompair_inputs_windowed:
            atompair_inputs = full_pairwise_repr_to_windowed(atompair_inputs, window_size=self.atoms_per_window)

        # handle atom mask

        total_atoms = molecule_atom_lens.sum(dim=-1)
        atom_mask = lens_to_mask(total_atoms, max_len=atom_seq_len)

        # get atom sequence length and molecule sequence length depending on whether using packed atomic seq

        batch_size = molecule_atom_lens.shape[0]
        seq_len = molecule_atom_lens.shape[-1]

        # embed inputs

        if verbose:
            logger.info("Embedding inputs...")

        (
            single_inputs,
            single_init,
            pairwise_init,
            atom_feats,
            atompair_feats
        ) = self.input_embedder(
            atom_inputs=atom_inputs,
            atompair_inputs=atompair_inputs,
            atom_mask=atom_mask,
            additional_token_feats=additional_token_feats,
            molecule_atom_lens=molecule_atom_lens,
            molecule_ids=molecule_ids
        )

        # handle maybe atom and atompair embeddings

        assert not (exists(
            atom_ids) ^ self.has_atom_embeds), 'you either set `num_atom_embeds` and did not pass in `atom_ids` or vice versa'
        assert not (exists(
            atompair_ids) ^ self.has_atompair_embeds), 'you either set `num_atompair_embeds` and did not pass in `atompair_ids` or vice versa'

        if self.has_atom_embeds:
            atom_embeds = self.atom_embeds(atom_ids)
            atom_feats = atom_feats + atom_embeds

        if self.has_atompair_embeds:
            atompair_embeds = self.atompair_embeds(atompair_ids)

            if atompair_embeds.ndim == 4:
                atompair_embeds = full_pairwise_repr_to_windowed(atompair_embeds, window_size=self.atoms_per_window)

            atompair_feats = atompair_feats + atompair_embeds

        # handle maybe molecule modifications

        if verbose:
            logger.info("Handling molecule modifications...")

        assert not (exists(
            is_molecule_mod) ^ self.has_molecule_mod_embeds), 'you either set `num_molecule_mods` and did not pass in `is_molecule_mod` or vice versa'

        if self.has_molecule_mod_embeds:
            single_init, seq_unpack_one = pack_one(single_init, '* ds')

            is_molecule_mod, _ = pack_one(is_molecule_mod, '* mods')

            if not is_molecule_mod.is_sparse:
                is_molecule_mod = is_molecule_mod.to_sparse()

            seq_indices, mod_id = is_molecule_mod.indices()
            scatter_values = self.molecule_mod_embeds(mod_id)

            seq_indices = repeat(seq_indices, 'n -> n ds', ds=single_init.shape[-1])
            single_init = single_init.scatter_add(0, seq_indices, scatter_values)

            single_init = seq_unpack_one(single_init)

        # handle maybe pairwise token constraint embeddings

        if verbose:
            logger.info("Handling pairwise token constraint embeddings...")

        if exists(self.constraints):
            assert exists(
                token_constraints
            ), "`token_constraints` must be provided to use constraint embeddings."

            for i, constraint in enumerate(self.constraints):
                constraint_slice = slice(i, i + CONSTRAINT_DIMS[constraint])

                token_constraint = torch.where(
                    # replace fixed constraint mask values with learnable mask
                    token_constraints[..., constraint_slice] == CONSTRAINTS_MASK_VALUE,
                    self.learnable_constraint_masks[i],
                    token_constraints[..., constraint_slice],
                )

                pairwise_init = pairwise_init + self.constraint_embeds[i](token_constraint)

        # handle maybe protein language model (PLM) embeddings

        if verbose:
            logger.info("Handling protein language model embeddings...")

        if exists(self.plms):
            aa_ids = torch.where(
                (molecule_ids < 0) | (molecule_ids > NUM_HUMAN_AMINO_ACIDS),
                NUM_HUMAN_AMINO_ACIDS,
                molecule_ids,
            )
            molecule_aa_ids = torch.where(
                is_molecule_types[..., IS_NON_PROTEIN_INDICES].any(dim=-1),
                -1,
                aa_ids,
            )

            plm_embeds = [plm(molecule_aa_ids) for plm in self.plms]

            # concat all PLM embeddings and project and add to single init

            all_plm_embeds = torch.cat(plm_embeds, dim=-1)
            single_plm_init = self.to_plm_embeds(all_plm_embeds)

            single_init = single_init + single_plm_init

        # handle maybe nucleotide language model (NLM) embeddings

        if verbose:
            logger.info("Handling nucleotide language model embeddings...")

        if exists(self.nlms):
            na_ids = torch.where(
                (molecule_ids < MIN_RNA_NUCLEOTIDE_ID) | (molecule_ids > MAX_DNA_NUCLEOTIDE_ID),
                MISSING_RNA_NUCLEOTIDE_ID,
                molecule_ids,
            )
            molecule_na_ids = torch.where(
                is_molecule_types[..., IS_NON_NA_INDICES].any(dim=-1),
                -1,
                na_ids,
            )

            nlm_embeds = [nlm(molecule_na_ids) for nlm in self.nlms]

            # concat all NLM embeddings and project and add to single init

            all_nlm_embeds = torch.cat(nlm_embeds, dim=-1)
            single_nlm_init = self.to_nlm_embeds(all_nlm_embeds)

            single_init = single_init + single_nlm_init

        # relative positional encoding

        if verbose:
            logger.info("Applying relative positional encoding...")

        relative_position_encoding = self.relative_position_encoding(
            additional_molecule_feats=additional_molecule_feats
        )

        # only apply relative positional encodings to biomolecules that are chained
        # not to ligands + metal ions

        is_chained_biomol = is_molecule_types[..., IS_BIOMOLECULE_INDICES].any(
            dim=-1)  # first three types are chained biomolecules (protein, rna, dna)
        paired_is_chained_biomol = to_pairwise_mask(is_chained_biomol)

        relative_position_encoding = einx.where(
            'b i j, b i j d, -> b i j d',
            paired_is_chained_biomol, relative_position_encoding, 0.
        )

        # add relative positional encoding to pairwise init

        pairwise_init = pairwise_init + relative_position_encoding

        # token bond features

        if verbose:
            logger.info("Applying token bond features...")

        if exists(token_bonds):
            # well do some precautionary standardization
            # (1) mask out diagonal - token to itself does not count as a bond
            # (2) symmetrize, in case it is not already symmetrical (could also throw an error)

            token_bonds = token_bonds | rearrange(token_bonds, 'b i j -> b j i')
            diagonal = torch.eye(seq_len, device=self.device, dtype=torch.bool)
            token_bonds = token_bonds.masked_fill(diagonal, False)
        else:
            seq_arange = torch.arange(seq_len, device=self.device)
            token_bonds = einx.subtract('i, j -> i j', seq_arange, seq_arange).abs() == 1
            token_bonds = repeat(token_bonds, 'i j -> b i j', b=batch_size)

        token_bonds_feats = self.token_bond_to_pairwise_feat(token_bonds.type(dtype))

        pairwise_init = pairwise_init + token_bonds_feats

        # molecule mask and pairwise mask

        mask = molecule_atom_lens > 0
        pairwise_mask = to_pairwise_mask(mask)

        # init recycled single and pairwise

        detach_when_recycling = default(detach_when_recycling, self.detach_when_recycling)
        maybe_recycling_detach = torch.detach if detach_when_recycling else identity

        recycled_pairwise = recycled_single = None
        single = pairwise = None

        # for each recycling step

        if verbose:
            logger.info("Starting recycling steps...")

        for i in range(num_recycling_steps):

            # handle recycled single and pairwise if not first step

            recycled_single = recycled_pairwise = 0.

            if exists(single):
                single = maybe_recycling_detach(single)
                recycled_single = self.recycle_single(single)

            if exists(pairwise):
                pairwise = maybe_recycling_detach(pairwise)
                recycled_pairwise = self.recycle_pairwise(pairwise)

            single = single_init + recycled_single
            pairwise = pairwise_init + recycled_pairwise

            # else go through main transformer trunk from alphafold2

            # templates

            if verbose:
                logger.info(f"Applying template embeddings in recycling step {i}...")

            if not exists(templates):
                templates = torch.zeros(
                    (batch_size, 1, seq_len, seq_len, self.dim_template_feats),
                    dtype=dtype,
                    device=self.device,
                )
                template_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=self.device)

            # ensure template embedder always contributes to the loss

            embedded_template = self.template_embedder(
                templates=templates,
                template_mask=template_mask,
                pairwise_repr=pairwise,
                mask=mask,
            )

            pairwise = embedded_template + pairwise

            # msa

            if verbose:
                logger.info(f"Applying MSA embeddings in recycling step {i}...")

            if exists(msa):
                embedded_msa = self.msa_module(
                    msa=msa,
                    single_repr=single,
                    pairwise_repr=pairwise,
                    msa_mask=msa_mask,
                    additional_msa_feats=additional_msa_feats,
                    mask=mask
                )

                pairwise = embedded_msa + pairwise

            # main attention trunk (pairformer)

            if verbose:
                logger.info(f"Applying pairformer in recycling step {i}...")

            single, pairwise = self.pairformer(
                single_repr=single,
                pairwise_repr=pairwise,
                mask=mask
            )

        # determine whether to return loss if any labels were to be passed in
        # otherwise will sample the atomic coordinates

        atom_pos_given = exists(atom_pos)

        confidence_head_labels = (atom_indices_for_frame, resolved_labels)

        can_return_loss = (
                atom_pos_given or
                exists(resolved_labels) or
                exists(distance_labels) or
                (atom_pos_given and exists(atom_indices_for_frame))
        )

        # default whether to return loss by whether labels or atom positions are given

        return_loss = default(return_loss, can_return_loss)

        # if neither atom positions or any labels are passed in, sample a structure and return

        if verbose:
            logger.info("Sampling atomic coordinates...")

        if not return_loss:
            sampled_atom_pos = self.edm.sample(
                num_sample_steps=num_sample_steps,
                atom_feats=atom_feats,
                atompair_feats=atompair_feats,
                atom_parent_ids=atom_parent_ids,
                atom_mask=atom_mask,
                mask=mask,
                single_trunk_repr=single,
                single_inputs_repr=single_inputs,
                pairwise_trunk=pairwise,
                pairwise_rel_pos_feats=relative_position_encoding,
                molecule_atom_lens=molecule_atom_lens,
                return_all_timesteps=return_all_diffused_atom_pos
            )

            if exists(atom_mask):
                sampled_atom_pos = einx.where('b m, ... b m c, -> ... b m c', atom_mask, sampled_atom_pos, 0.)

            if return_confidence_head_logits:
                confidence_head_atom_pos_input = sampled_atom_pos.clone()

            # convert sampled atom positions to bio pdb structures

            if return_bio_pdb_structures:
                assert not return_all_diffused_atom_pos

                sampled_atom_pos = [
                    protein_structure_from_feature(*args)
                    for args in zip(
                        additional_molecule_feats[..., 2],
                        molecule_ids,
                        molecule_atom_lens,
                        sampled_atom_pos,
                        atom_mask
                    )
                ]

            if not return_confidence_head_logits:
                return sampled_atom_pos

            confidence_head_logits = self.confidence_head(
                single_repr=single.detach(),
                single_inputs_repr=single_inputs.detach(),
                pairwise_repr=pairwise.detach(),
                pred_atom_pos=confidence_head_atom_pos_input.detach(),
                molecule_atom_indices=molecule_atom_indices,
                molecule_atom_lens=molecule_atom_lens,
                atom_feats=atom_feats.detach(),
                mask=mask,
                return_pae_logits=True
            )

            returned_logits = confidence_head_logits

            if return_distogram_head_logits:
                distogram_head_logits = self.distogram_head(pairwise.clone().detach())

                returned_logits = Alphafold3Logits(
                    **confidence_head_logits._asdict(),
                    distance=distogram_head_logits
                )

            return sampled_atom_pos, returned_logits

        # if being forced to return loss, but do not have sufficient information to return losses, just return 0

        if return_loss and not can_return_loss:
            zero = self.zero.requires_grad_()

            if not return_loss_breakdown:
                return zero

            return zero, LossBreakdown(*((zero,) * 11))

        # losses default to 0

        distogram_loss = diffusion_loss = confidence_loss = pae_loss = pde_loss = plddt_loss = resolved_loss = self.zero

        # calculate distogram logits and losses

        ignore = self.ignore_index

        # distogram head

        if verbose:
            logger.info("Calculating distogram logits and losses...")

        molecule_pos = None

        if not exists(distance_labels) and atom_pos_given and exists(distogram_atom_indices):

            distogram_pos = atom_pos

            if not self.distogram_atom_resolution:
                # molecule_pos = einx.get_at('b [m] c, b n -> b n c', atom_pos, distogram_atom_indices)

                distogram_atom_coords_indices = repeat(
                    distogram_atom_indices, "b n -> b n c", c=distogram_pos.shape[-1]
                )
                molecule_pos = distogram_pos = distogram_pos.gather(
                    1, distogram_atom_coords_indices
                )
                distogram_mask = valid_distogram_mask
            else:
                distogram_mask = atom_mask

            distogram_dist = torch.cdist(distogram_pos, distogram_pos, p=2)
            distance_labels = distance_to_dgram(
                distogram_dist, self.distance_bins, return_labels=True
            )

            # account for representative distogram atom missing from residue (-1 set on distogram_atom_indices field)

            distogram_mask = to_pairwise_mask(distogram_mask)
            distance_labels.masked_fill_(~distogram_mask, ignore)

        if exists(distance_labels):

            distogram_mask = pairwise_mask

            if self.distogram_atom_resolution:
                distogram_mask = to_pairwise_mask(atom_mask)

            distance_labels = torch.where(distogram_mask, distance_labels, ignore)

            distogram_logits = self.distogram_head(
                pairwise,
                molecule_atom_lens=molecule_atom_lens,
                atom_feats=atom_feats
            )

            distogram_loss = F.cross_entropy(distogram_logits, distance_labels, ignore_index=ignore)

        # otherwise, noise and make it learn to denoise

        if verbose:
            logger.info("Calculating diffusion loss...")

        calc_diffusion_loss = exists(atom_pos)

        if calc_diffusion_loss:

            num_augs = self.num_augmentations + int(self.stochastic_frame_average)
            batch_size *= num_augs

            # take care of augmentation
            # they did 48 during training, as the trunk did the heavy lifting

            if num_augs > 1:
                (
                    atom_pos,
                    atom_mask,
                    missing_atom_mask,
                    valid_molecule_atom_mask,
                    atom_feats,
                    atom_parent_ids,
                    atompair_feats,
                    mask,
                    pairwise_mask,
                    single,
                    single_inputs,
                    pairwise,
                    relative_position_encoding,
                    additional_molecule_feats,
                    is_molecule_types,
                    molecule_atom_indices,
                    molecule_pos,
                    distogram_atom_indices,
                    valid_atom_indices_for_frame,
                    atom_indices_for_frame,
                    molecule_atom_lens,
                    token_bonds,
                    resolved_labels,
                    resolution
                ) = tuple(
                    maybe(repeat)(t, 'b ... -> (b a) ...', a=num_augs)
                    for t in (
                        atom_pos,
                        atom_mask,
                        missing_atom_mask,
                        valid_molecule_atom_mask,
                        atom_feats,
                        atom_parent_ids,
                        atompair_feats,
                        mask,
                        pairwise_mask,
                        single,
                        single_inputs,
                        pairwise,
                        relative_position_encoding,
                        additional_molecule_feats,
                        is_molecule_types,
                        molecule_atom_indices,
                        molecule_pos,
                        distogram_atom_indices,
                        valid_atom_indices_for_frame,
                        atom_indices_for_frame,
                        molecule_atom_lens,
                        token_bonds,
                        resolved_labels,
                        resolution
                    )
                )

                # handle stochastic frame averaging

                aug_atom_mask = atom_mask

                if self.stochastic_frame_average:
                    if verbose:
                        logger.info("Applying stochastic frame averaging...")

                    fa_atom_pos, atom_pos = atom_pos[:1], atom_pos[1:]
                    fa_atom_mask, aug_atom_mask = atom_mask[:1], atom_mask[1:]

                    fa_atom_pos = self.frame_average(
                        fa_atom_pos.float(), frame_average_mask=fa_atom_mask
                    ).type(dtype)

                # normal random augmentations, 48 times in paper

                if verbose:
                    logger.info("Applying random augmentations...")

                atom_pos = self.augmenter(atom_pos.float(), mask=aug_atom_mask).type(dtype)

                # concat back the stochastic frame averaged position

                if self.stochastic_frame_average:
                    atom_pos = torch.cat((fa_atom_pos, atom_pos), dim=0)

            if verbose:
                logger.info("Calculating diffusion loss with EDM...")

            diffusion_loss, denoised_atom_pos, diffusion_loss_breakdown, _ = self.edm(
                atom_pos,
                additional_molecule_feats=additional_molecule_feats,
                is_molecule_types=is_molecule_types,
                add_smooth_lddt_loss=diffusion_add_smooth_lddt_loss,
                add_bond_loss=diffusion_add_bond_loss,
                atom_feats=atom_feats,
                atompair_feats=atompair_feats,
                atom_parent_ids=atom_parent_ids,
                missing_atom_mask=missing_atom_mask,
                atom_mask=atom_mask,
                mask=mask,
                single_trunk_repr=single,
                single_inputs_repr=single_inputs,
                pairwise_trunk=pairwise,
                pairwise_rel_pos_feats=relative_position_encoding,
                molecule_atom_lens=molecule_atom_lens,
                molecule_atom_indices=molecule_atom_indices,
                token_bonds=token_bonds,
                return_denoised_pos=True,
                nucleotide_loss_weight=self.nucleotide_loss_weight,
                ligand_loss_weight=self.ligand_loss_weight,
                single_structure_input=single_structure_input,
                verbose=verbose,
                filepath=filepath,
            )

        # confidence head

        if verbose:
            logger.info("Calculating confidence head logits and losses...")

        should_call_confidence_head = any([*map(exists, confidence_head_labels)])

        if (
                calc_diffusion_loss
                and should_call_confidence_head
                and exists(molecule_atom_indices)
                and self.pdb_training_set
        ):
            # rollout

            num_rollout_steps = default(num_rollout_steps, self.num_rollout_steps)

            denoised_atom_pos = self.edm.sample(
                num_sample_steps=num_rollout_steps,
                atom_feats=atom_feats,
                atompair_feats=atompair_feats,
                atom_mask=atom_mask,
                mask=mask,
                single_trunk_repr=single,
                single_inputs_repr=single_inputs,
                pairwise_trunk=pairwise,
                pairwise_rel_pos_feats=relative_position_encoding,
                molecule_atom_lens=molecule_atom_lens,
                use_tqdm_pbar=rollout_show_tqdm_pbar,
                tqdm_pbar_title="Training rollout",
            )

            # structurally align and chain-permute ground truth structure to optimally match predicted structure

            if atom_pos_given:
                # section 3.7.1 equation 2 - weighted rigid aligned ground truth

                if verbose:
                    logger.info("Calculating weighted rigid alignment...")

                align_weights = calculate_weighted_rigid_align_weights(
                    atom_pos_ground_truth=atom_pos,
                    molecule_atom_lens=molecule_atom_lens,
                    is_molecule_types=is_molecule_types,
                    nucleotide_loss_weight=self.nucleotide_loss_weight,
                    ligand_loss_weight=self.ligand_loss_weight,
                )

                try:
                    atom_pos = self.weighted_rigid_align(
                        pred_coords=denoised_atom_pos.float(),
                        true_coords=atom_pos.float(),
                        weights=align_weights.float(),
                        mask=atom_mask,
                    ).type(dtype)
                except Exception as e:
                    # NOTE: For many (random) unit test inputs, weighted rigid alignment can be unstable
                    logger.warning(f"Skipping weighted rigid alignment due to: {e}")

                # section 4.2 - multi-chain permutation alignment

                if single_structure_input:
                    if verbose:
                        logger.info("Calculating multi-chain permutation alignment...")

                    try:
                        atom_pos = self.multi_chain_permutation_alignment(
                            pred_coords=denoised_atom_pos,
                            true_coords=atom_pos,
                            molecule_atom_lens=molecule_atom_lens,
                            molecule_atom_indices=molecule_atom_indices,
                            token_bonds=token_bonds,
                            additional_molecule_feats=additional_molecule_feats,
                            is_molecule_types=is_molecule_types,
                            mask=atom_mask,
                        )
                    except Exception as e:
                        # NOTE: For many (random) unit test inputs, permutation alignment can be unstable
                        logger.warning(
                            f"Skipping multi-chain permutation alignment {f'for {filepath}' if exists(filepath) else ''} due to: {e}")

                assert exists(
                    distogram_atom_indices
                ), "`distogram_atom_indices` must be passed in for calculating aligned and chain-permuted ground truth"

                distogram_pos = atom_pos

                if not self.distogram_atom_resolution:
                    if verbose:
                        logger.info("Gathering distogram atom positions...")

                    # molecule_pos = einx.get_at('b [m] c, b n -> b n c', atom_pos, distogram_atom_indices)

                    distogram_atom_coords_indices = repeat(
                        distogram_atom_indices, "b n -> b n c", c=distogram_pos.shape[-1]
                    )
                    distogram_pos = distogram_pos.gather(1, distogram_atom_coords_indices)

                distogram_atom_coords_indices = repeat(
                    distogram_atom_indices, "b n -> b n c", c=distogram_pos.shape[-1]
                )
                molecule_pos = atom_pos.gather(1, distogram_atom_coords_indices)

            # determine pae labels if possible

            pae_labels = None

            if atom_pos_given and exists(atom_indices_for_frame):
                if verbose:
                    logger.info("Calculating PAE labels...")

                denoised_molecule_pos = None

                if not exists(molecule_pos):
                    assert exists(
                        distogram_atom_indices
                    ), "`distogram_atom_indices` must be passed in for calculating non-atomic PAE labels"

                    distogram_atom_coords_indices = repeat(
                        distogram_atom_indices, "b n -> b n c", c=distogram_pos.shape[-1]
                    )
                    molecule_pos = atom_pos.gather(1, distogram_atom_coords_indices)

                denoised_molecule_pos = denoised_atom_pos.gather(1, distogram_atom_coords_indices)

                # get frames atom positions

                # three_atoms = einx.get_at('b [m] c, b n three -> three b n c', atom_pos, atom_indices_for_frame)
                # pred_three_atoms = einx.get_at('b [m] c, b n three -> three b n c', denoised_atom_pos, atom_indices_for_frame)

                atom_indices_for_frame = repeat(
                    atom_indices_for_frame, "b n three -> three b n c", c=3
                )
                three_atom_pos = repeat(atom_pos, "b m c -> three b m c", three=3)
                three_denoised_atom_pos = repeat(
                    denoised_atom_pos, "b m c -> three b m c", three=3
                )

                three_atoms = three_atom_pos.gather(2, atom_indices_for_frame)
                pred_three_atoms = three_denoised_atom_pos.gather(2, atom_indices_for_frame)

                frame_atoms = rearrange(three_atoms, "three b n c -> b n c three")
                pred_frame_atoms = rearrange(pred_three_atoms, "three b n c -> b n c three")

                # determine mask
                # must be amino acid, nucleotide, or ligand with greater than 0 atoms

                align_error_mask = valid_atom_indices_for_frame

                # align error

                align_error = self.compute_alignment_error(
                    denoised_molecule_pos,
                    molecule_pos,
                    pred_frame_atoms,
                    # In the paragraph 2 of section 4.3.2, the Phi_i denotes the coordinates of these frame atoms rather than the rotation matrix.
                    frame_atoms,
                    mask=align_error_mask,
                )

                # calculate pae labels as alignment error binned to 64 (0 - 32A)

                pae_labels = distance_to_dgram(align_error, self.pae_bins, return_labels=True)

                # set ignore index for invalid molecules or frames

                pair_align_error_mask = to_pairwise_mask(align_error_mask)

                pae_labels = einx.where(
                    "b i j, b i j, -> b i j", pair_align_error_mask, pae_labels, ignore
                )

            # determine pde labels if possible

            pde_labels = None

            if atom_pos_given:
                if verbose:
                    logger.info("Calculating PDE labels...")

                denoised_molecule_pos = None

                assert exists(
                    molecule_atom_indices
                ), "`molecule_atom_indices` must be passed in for calculating non-atomic PDE labels"

                # molecule_pos = einx.get_at('b [m] c, b n -> b n c', atom_pos, molecule_atom_indices)

                molecule_atom_coords_indices = repeat(
                    molecule_atom_indices, "b n -> b n c", c=atom_pos.shape[-1]
                )

                molecule_pos = atom_pos.gather(1, molecule_atom_coords_indices)
                denoised_molecule_pos = denoised_atom_pos.gather(1, molecule_atom_coords_indices)

                molecule_mask = valid_molecule_atom_mask

                pde_gt_dist = torch.cdist(molecule_pos, molecule_pos, p=2)
                pde_pred_dist = torch.cdist(
                    denoised_molecule_pos,
                    denoised_molecule_pos,
                    p=2,
                )

                # calculate pde labels as distance error binned to 64 (0 - 32A)

                pde_dist = torch.abs(pde_pred_dist - pde_gt_dist)
                pde_labels = distance_to_dgram(pde_dist, self.pde_bins, return_labels=True)

                # account for representative molecule atom missing from residue (-1 set on molecule_atom_indices field)

                molecule_mask = to_pairwise_mask(molecule_mask)
                pde_labels.masked_fill_(~molecule_mask, ignore)

            # determine plddt labels if possible

            if atom_pos_given:
                if verbose:
                    logger.info("Calculating plDDT labels...")

                # gather metadata

                pred_coords, true_coords = denoised_atom_pos, atom_pos

                # compute distances between all pairs of atoms

                pred_dists = torch.cdist(pred_coords, pred_coords, p=2)
                true_dists = torch.cdist(true_coords, true_coords, p=2)

                # restrict to bespoke interaction types and inclusion radius on the atom level (Section 4.3.1)

                is_protein = batch_repeat_interleave(
                    is_molecule_types[..., IS_PROTEIN_INDEX], molecule_atom_lens
                )
                is_rna = batch_repeat_interleave(
                    is_molecule_types[..., IS_RNA_INDEX], molecule_atom_lens
                )
                is_dna = batch_repeat_interleave(
                    is_molecule_types[..., IS_DNA_INDEX], molecule_atom_lens
                )

                is_nucleotide = is_rna | is_dna
                is_polymer = is_protein | is_rna | is_dna

                is_any_nucleotide_pair = repeat(
                    is_nucleotide, "... j -> ... i j", i=is_nucleotide.shape[-1]
                )
                is_any_polymer_pair = repeat(
                    is_polymer, "... j -> ... i j", i=is_polymer.shape[-1]
                )

                inclusion_radius = torch.where(
                    is_any_nucleotide_pair,
                    true_dists < self.lddt_mask_nucleic_acid_cutoff,
                    true_dists < self.lddt_mask_other_cutoff,
                )

                is_token_center_atom = torch.zeros_like(atom_pos[..., 0], dtype=torch.bool)
                is_token_center_atom[
                    torch.arange(batch_size).unsqueeze(1), molecule_atom_indices
                ] = True
                is_any_token_center_atom_pair = repeat(
                    is_token_center_atom, "... j -> ... i j", i=is_token_center_atom.shape[-1]
                )

                # compute masks, avoiding self term

                plddt_mask = (
                        inclusion_radius
                        & is_any_polymer_pair
                        & is_any_token_center_atom_pair
                        & ~torch.eye(atom_seq_len, dtype=torch.bool, device=self.device)
                )

                plddt_mask = plddt_mask * to_pairwise_mask(atom_mask)

                # compute distance difference for all pairs of atoms

                dist_diff = torch.abs(true_dists - pred_dists)

                lddt = einx.subtract(
                    "thresholds, ... -> ... thresholds", self.lddt_thresholds, dist_diff
                )
                lddt = (lddt >= 0).type(dtype).mean(dim=-1)

                # calculate masked averaging,
                # after which we assign each value to one of 50 equally sized bins

                lddt_mean = masked_average(lddt, plddt_mask, dim=-1)

                plddt_labels = torch.clamp(
                    torch.floor(lddt_mean * self.num_plddt_bins).int(), max=self.num_plddt_bins - 1
                )

            return_pae_logits = exists(pae_labels)

            if verbose:
                logger.info("Calculating confidence head logits...")

            ch_logits = self.confidence_head(
                single_repr=single.detach(),
                single_inputs_repr=single_inputs.detach(),
                pairwise_repr=pairwise.detach(),
                pred_atom_pos=denoised_atom_pos.detach(),
                molecule_atom_indices=molecule_atom_indices,
                molecule_atom_lens=molecule_atom_lens,
                mask=mask,
                atom_feats=atom_feats.detach(),
                return_pae_logits=return_pae_logits
            )

            # determine which mask to use for confidence head labels

            label_mask = atom_mask
            label_pairwise_mask = to_pairwise_mask(mask)

            # cross entropy losses

            confidence_mask = (
                (resolution >= min_conf_resolution) & (resolution <= max_conf_resolution)
                if exists(resolution)
                else torch.full((batch_size,), False, device=self.device)
            )

            confidence_weight = confidence_mask.type(dtype)

            @typecheck
            def cross_entropy_with_weight(
                    logits: Float['b l ...'],
                    labels: Int['b ...'],
                    weight: Float[' b'],
                    mask: Bool['b ...'],
                    ignore_index: int
            ) -> Float['']:
                labels = torch.where(mask, labels, ignore_index)

                return F.cross_entropy(
                    einx.multiply('b ..., b -> b ...', logits, weight),
                    einx.multiply('b ..., b -> b ...', labels, weight.long()),
                    ignore_index=ignore_index
                )

            if verbose:
                logger.info("Calculating confidence head losses...")

            if exists(pae_labels):
                assert pae_labels.shape[-1] == ch_logits.pae.shape[-1], (
                    f"pae_labels shape {pae_labels.shape[-1]} does not match "
                    f"ch_logits.pae shape {ch_logits.pae.shape[-1]}"
                )
                pae_loss = cross_entropy_with_weight(ch_logits.pae, pae_labels, confidence_weight, label_pairwise_mask,
                                                     ignore)

            if exists(pde_labels):
                assert pde_labels.shape[-1] == ch_logits.pde.shape[-1], (
                    f"pde_labels shape {pde_labels.shape[-1]} does not match "
                    f"ch_logits.pde shape {ch_logits.pde.shape[-1]}"
                )
                pde_loss = cross_entropy_with_weight(ch_logits.pde, pde_labels, confidence_weight, label_pairwise_mask,
                                                     ignore)

            if exists(plddt_labels):
                assert plddt_labels.shape[-1] == ch_logits.plddt.shape[-1], (
                    f"plddt_labels shape {plddt_labels.shape[-1]} does not match "
                    f"ch_logits.plddt shape {ch_logits.plddt.shape[-1]}"
                )
                plddt_loss = cross_entropy_with_weight(ch_logits.plddt, plddt_labels, confidence_weight, label_mask,
                                                       ignore)

            if exists(resolved_labels):
                assert resolved_labels.shape[-1] == ch_logits.resolved.shape[-1], (
                    f"resolved_labels shape {resolved_labels.shape[-1]} does not match "
                    f"ch_logits.resolved shape {ch_logits.resolved.shape[-1]}"
                )
                resolved_loss = cross_entropy_with_weight(ch_logits.resolved, resolved_labels, confidence_weight,
                                                          label_mask, ignore)
            else:
                resolved_loss = (
                        ch_logits.resolved * 0.0
                ).mean()  # ensure resolved logits always contribute to the loss

            confidence_loss = pae_loss + pde_loss + plddt_loss + resolved_loss

        # combine all the losses

        loss = (
                distogram_loss * self.loss_distogram_weight +
                diffusion_loss * self.loss_diffusion_weight +
                confidence_loss * self.loss_confidence_weight
        )

        if not return_loss_breakdown:
            return loss

        loss_breakdown = LossBreakdown(
            total_loss=loss,
            total_diffusion=diffusion_loss,
            pae=pae_loss,
            pde=pde_loss,
            plddt=plddt_loss,
            resolved=resolved_loss,
            distogram=distogram_loss,
            confidence=confidence_loss,
            **diffusion_loss_breakdown._asdict()
        )

        return loss, loss_breakdown

class SmoothLDDTLoss(Module):
    """ Algorithm 27 """

    @typecheck
    def __init__(
        self,
        nucleic_acid_cutoff: float = 30.0,
        other_cutoff: float = 15.0
    ):
        super().__init__()
        self.nucleic_acid_cutoff = nucleic_acid_cutoff
        self.other_cutoff = other_cutoff

        self.register_buffer('lddt_thresholds', torch.tensor([0.5, 1.0, 2.0, 4.0]))

    @typecheck
    def forward(
        self,
        pred_coords: Float['b n 3'],
        true_coords: Float['b n 3'],
        is_dna: Bool['b n'],
        is_rna: Bool['b n'],
        coords_mask: Bool['b n'] | None = None,
    ) -> Float['']:
        """
        pred_coords: predicted coordinates
        true_coords: true coordinates
        is_dna: boolean tensor indicating DNA atoms
        is_rna: boolean tensor indicating RNA atoms
        """
        # Compute distances between all pairs of atoms

        pred_dists = torch.cdist(pred_coords, pred_coords)
        true_dists = torch.cdist(true_coords, true_coords)

        # Compute distance difference for all pairs of atoms
        dist_diff = torch.abs(true_dists - pred_dists)

        # Compute epsilon values

        eps = einx.subtract('thresholds, ... -> ... thresholds', self.lddt_thresholds, dist_diff)
        eps = eps.sigmoid().mean(dim = -1)

        # Restrict to bespoke inclusion radius
        is_nucleotide = is_dna | is_rna
        is_nucleotide_pair = to_pairwise_mask(is_nucleotide)

        inclusion_radius = torch.where(
            is_nucleotide_pair,
            true_dists < self.nucleic_acid_cutoff,
            true_dists < self.other_cutoff
        )

        # Compute mean, avoiding self term
        mask = inclusion_radius & ~torch.eye(pred_coords.shape[1], dtype=torch.bool, device=pred_coords.device)

        # Take into account variable lengthed atoms in batch
        if exists(coords_mask):
            paired_coords_mask = to_pairwise_mask(coords_mask)
            mask = mask & paired_coords_mask

        # Calculate masked averaging
        lddt = masked_average(eps, mask = mask, dim = (-1, -2), eps = 1)

        return 1. - lddt.mean()


