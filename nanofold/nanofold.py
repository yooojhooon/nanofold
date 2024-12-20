from __future__ import annotations

import torch
import torch.nn.functional as F
from alphafold3_pytorch.alphafold3 import (
    InputFeatureEmbedder,  # 2 step1
    RelativePositionEncoding,  # 3 step1
    TemplateEmbedder,  # 16 step2. 템플릿 모듈
    DiffusionModule,  # 20 step3
    DistogramHead,  # step3
    SmoothLDDTLoss  # 기본 손실 함수
)
from beartype.typing import Tuple
from torch import nn


#1 전체 추론 과정 (구현1)
class Nanofold(nn.Module):

    def __init__(
            self,
            *,
            dim_atom_inputs: int = 77,
            dim_atompair_inputs: int = 5,
            dim_template_feats: int = 108,
            dim_single: int = 384,
            dim_pairwise: int = 128,
            dim_atom: int = 128,
            dim_token: int = 768,
            dim_msa: int = 64,
            dim_template: int = 64,
            n_cycles: int = 4,
            atoms_per_window: int = 27,
            num_molecule_types: int = 32,
    ):
        super().__init__()

        # 핵심 dimensions
        self.dim_single = dim_single
        self.dim_pair = dim_pairwise
        self.n_cycles = n_cycles
        self.atoms_per_window = atoms_per_window

        # Layer Normalization 추가
        self.layer_norm1 = nn.LayerNorm(dim_pairwise)
        self.layer_norm2 = nn.LayerNorm(dim_single)


        # 1.
        self.input_embedder = InputFeatureEmbedder(
            dim_atom_inputs=dim_atom_inputs,  # atom_inputs의 특성 차원
            dim_atompair_inputs=dim_atompair_inputs,  # atompair_inputs의 특성 차원

            dim_atom=dim_atom,  # 원자 특성의 중간 차원
            dim_token=dim_single,  # 토큰 특성의 차원

            dim_single=dim_single,  # 단일 특성의 출력 차원 (384)
            dim_pairwise=dim_pairwise, # 쌍별 특성의 출력 차원 (128)

            atoms_per_window=atoms_per_window,
            num_molecule_types=num_molecule_types
        )

        # Main transformation blocks
        self.to_single_init = nn.Linear(dim_single, dim_single, bias=False)
        self.to_pair_init = nn.Linear(dim_single * 2, dim_pairwise, bias=False)

        # 핵심 모듈들
        self.rel_pos_encoder = RelativePositionEncoding(
            dim_out=dim_pairwise
        )

        #2.
        self.template_embedder = TemplateEmbedder(
            dim_template_feats=dim_template_feats,
            dim_pairwise=dim_pairwise,

        )

        #3.
        self.msa_module = MSAModule(
            dim_msa=dim_msa,
            dim_pair=dim_pairwise,
            dim_single=dim_single,
        )

        #4.
        self.pairformer = PairformerStack(
            dim_single=dim_single,
            dim_pair=dim_pairwise
        )

        #5.
        self.diffusion = DiffusionModule(
            dim_single=dim_single,
            dim_pairwise=dim_pairwise,
            dim_pairwise_trunk=dim_pairwise,
            dim_pairwise_rel_pos_feats=dim_pairwise
        )

        # 출력 헤드 (필수적인 것만 유지)
        self.distogram_head = DistogramHead(
            dim_pairwise=dim_pairwise
        )

        self.lddt_loss=SmoothLDDTLoss

    def forward(
            self,
            *,
            atom_inputs: torch.Tensor,
            atompair_inputs: torch.Tensor,
            molecule_atom_lens: torch.Tensor,
            molecule_ids: torch.Tensor,
            is_molecule_types: torch.Tensor,
            msa: torch.Tensor = None,
            msa_mask: torch.Tensor = None,
            templates: torch.Tensor = None,
            template_mask: torch.Tensor = None,
            atom_pos: torch.Tensor = None,
            distance_labels: torch.Tensor = None,
            additional_molecule_feats: torch.Tensor = None,
            additional_token_feats: torch.Tensor = None,
            additional_msa_feats: torch.Tensor = None,
            atom_mask: torch.Tensor = None,
            num_recycling_steps: int = 1,
    ):

        global single, pairwise
        (
            single_inputs, #i
            single_init,
            pair_init,
            atom_feats,
            atompair_feats
        ) = self.input_embedder( # f*
            atom_inputs=atom_inputs,
            atompair_inputs=atompair_inputs,
            molecule_atom_lens=molecule_atom_lens,
            molecule_ids=molecule_ids,
            additional_token_feats=additional_token_feats,
            atom_mask=atom_mask
        )

        # init

        single_prev = torch.zeros_like(pair_init) # (si)
        pair_prev = torch.zeros_like(pair_init) # (zij)

        '''step2. 표현 학습 루프'''
        # recycle
        for cycle in range(self.n_cycles):

            # Line 8
            pairwise = pair_init + self.layer_norm1(pair_prev)
            # Line 9 Template 모듈
            pairwise = pairwise + self.template_embedder(
                templates=templates,
                template_mask=template_mask,
                pairwise_repr=pairwise,
                mask=molecule_atom_lens > 0
            )
            # Line 10 MSA 모듈
            pairwise = pairwise + self.msa_module(msa, pairwise, single_inputs)
            # Line 11
            single = single_init + self.layer_norm2(single_prev)
            # Line 12 Pairformer
            single, pairwise = self.pairformer(single, pairwise)
            # Line 13
            single_prev, pair_prev = single, pairwise

        # Line 15
        pred_coords = self.diffusion(
            single_inputs=single_inputs,
            single=single,
            pairwise=pairwise
        )
        # Line 17
        distogram = self.distogram_head(pairwise)

        return pred_coords, distogram


#step2. MSA module (구현2)
# Algorithm 8
class MSAModule(nn.Module):
    """Multiple Sequence Alignment (MSA) processing module.

    This module processes MSA information and updates pair representations through
    several rounds of attention and transformation operations.
    """

    def __init__(self,
                 num_blocks=4,  # Number of processing blocks to apply
                 dim_msa=64,  # Dimension of MSA features
                 dim_pair=128,  # Dimension of pairwise representations
                 dim_single=384,
                 ):  # Dimension of single sequence features
        super().__init__()
        self.num_blocks = num_blocks

        # Projection layers for initial embeddings
        self.msa_embedder = nn.Linear(dim_msa, dim_msa, bias=False)  # Projects MSA sequences to feature space
        self.single_embedder = nn.Linear(dim_single, dim_msa,bias=False)  # Projects single sequence to MSA feature space

        # Core processing blocks
        self.outer_product_blocks = nn.ModuleList([
            OuterProductMean(dim_msa, dim_pair, c=32)  # Updates pair repr using MSA information
            for _ in range(num_blocks)
        ])

        self.msa_attention_blocks = nn.ModuleList([
            MSAPairWeightedAveraging(dim_msa, dim_pair, heads=8)  # Processes MSA using pair information
            for _ in range(num_blocks)
        ])

        self.msa_transitions = nn.ModuleList([
            Transition(dim_msa)  # Applies feed-forward transformation to MSA
            for _ in range(num_blocks)
        ])

        self.pair_blocks = nn.ModuleList([
            PairStack(dim_pair)  # Processes pairwise relationships
            for _ in range(num_blocks)
        ])

    def forward(self, msa, pair_repr, single_repr):
        """
        Args:
            msa: Multiple sequence alignment tensor [batch, num_sequences, num_residues, dim_single]
            pair_repr: Pairwise representation tensor [batch, num_residues, num_residues, dim_pair]
            single_repr: Single sequence representation [batch, num_residues, dim_single]

        Returns:
            Updated pair representation tensor
        """
        # Initialize MSA representation
        msa = self.msa_embedder(msa)  # Project MSA to feature space
        msa = msa + self.single_embedder(single_repr).unsqueeze(1)  # Add single sequence information

        # Main processing loop
        for i in range(self.num_blocks):
            # Update pair representation using MSA information
            pair_repr = pair_repr + self.outer_product_blocks[i](msa)

            # Update MSA using pair information
            msa_attn = self.msa_attention_blocks[i](msa, pair_repr)
            msa = msa + torch.dropout(msa_attn, p=0.15, train=self.training)
            msa = msa + self.msa_transitions[i](msa)

            # Process pair representation
            pair_block_out = self.pair_blocks[i](pair_repr)
            pair_repr = pair_repr + pair_block_out

        return pair_repr

# Algorithm 8.1 (8 안으로 집어넣기)
class PairStack(nn.Module):
    """Processes pairwise representations through multiple geometric transformations."""

    def __init__(self, dim):
        super().__init__()
        self.triangle_mult_out = TriangleMultiplication(dim,pattern='outgoing')  # Outgoing edge updates
        self.triangle_mult_in = TriangleMultiplication(dim,pattern='incoming')  # Incoming edge updates
        self.triangle_att_start = TriangleAttention(dim,node_type='starting',)  # Attention from start nodes
        self.triangle_att_end = TriangleAttention(dim,node_type='ending')  # Attention from end nodes
        self.transition = Transition(dim)  # Final transformation

    def forward(self, pair_repr):
        """Apply triangular updates and attention operations with dropout."""
        # Apply outgoing triangle multiplication
        pair_repr = pair_repr + torch.dropout(self.triangle_mult_out(pair_repr), p=0.25, train=self.training)
        # Apply incoming triangle multiplication
        pair_repr = pair_repr + torch.dropout(self.triangle_mult_in(pair_repr), p=0.25, train=self.training)
        # Apply attention operations
        pair_repr = pair_repr + torch.dropout(self.triangle_att_start(pair_repr), p=0.25, train=self.training)
        pair_repr = pair_repr + torch.dropout(self.triangle_att_end(pair_repr), p=0.25, train=self.training)
        # Final transformation
        pair_repr = pair_repr + self.transition(pair_repr)
        return pair_repr

#Algorithm 9
class OuterProductMean(nn.Module):
    """Computes outer product of MSA features and averages across sequences."""

    def __init__(self, dim_msa, dim_pair, c=32):
        super().__init__()
        self.c = c  # Hidden dimension for intermediate computation
        self.proj_a = nn.Linear(dim_msa, c, bias=False)  # First projection
        self.proj_b = nn.Linear(dim_msa, c, bias=False)  # Second projection
        self.output = nn.Linear(c * c, dim_pair)  # Final projection

    def forward(self, msa):
        """
        Args:
            msa: MSA tensor [batch, num_sequences, num_residues, dim_msa]
        Returns:
            Pair representation update [batch, num_residues, num_residues, dim_pair]
        """
        msa = nn.LayerNorm(msa.shape[-1])(msa)
        a = self.proj_a(msa)  # Project to first space
        b = self.proj_b(msa)  # Project to second space

        # Compute outer product and average across MSA dimension
        outer = torch.einsum('bsic,bsjc->bijs', a, b)  # i,j: residue indices, s: sequence index
        outer = outer.mean(dim=1)  # Average across sequences

        return self.output(outer.flatten(-2))

#Algorithm 10
class MSAPairWeightedAveraging(nn.Module):
    """Attention mechanism using pair representation to guide MSA processing."""

    def __init__(self, dim_msa, dim_pair, heads=8):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_msa // heads

        # Projection layers
        self.to_v = nn.Linear(dim_msa, dim_msa, bias=False)  # Value projection
        self.to_b = nn.Linear(dim_pair, heads, bias=False)  # Bias projection from pair repr
        self.to_g = nn.Linear(dim_msa, dim_msa, bias=False)  # Gating projection
        self.output = nn.Linear(dim_msa, dim_msa, bias=False)  # Output projection

    def forward(self, msa, pair_repr):
        """
        Args:
            msa: MSA tensor [batch, num_sequences, num_residues, dim_msa]
            pair_repr: Pair representation [batch, num_residues, num_residues, dim_pair]
        Returns:
            Updated MSA representation
        """
        msa = nn.LayerNorm(msa.shape[-1])(msa)
        # Prepare attention components
        v = self.to_v(msa).view(*msa.shape[:-1], self.heads, self.dim_head)  # Values
        b = self.to_b(nn.LayerNorm(pair_repr.shape[-1])(pair_repr))  # Attention bias
        g = torch.sigmoid(self.to_g(msa)).view(*msa.shape[:-1], self.heads, self.dim_head)  # Gates

        # Compute attention and apply gating
        attn = torch.softmax(b, dim=-1)  # Attention weights
        out = torch.einsum('bsnh,bij->bsnh', v, attn)  # Apply attention
        out = out * g  # Apply gating

        return self.output(out.flatten(-2))

#Algorithm 11
class Transition(nn.Module):
    """Feed-forward transition layer with SwiGLU activation."""

    def __init__(self, dim, n=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, n * dim, bias=False)  # First projection
        self.linear2 = nn.Linear(dim, n * dim, bias=False)  # Second projection
        self.output = nn.Linear(n * dim, dim, bias=False)  # Output projection

    def forward(self, x):
        """Apply SwiGLU transformation."""
        x = self.norm(x)
        a = self.linear1(x)  # First branch
        b = self.linear2(x)  # Second branch
        return self.output(nn.functional.silu(a) * b)  # SwiGLU activation

# step2. Pairformer (구현3)
# Algorithm 12
# Algorithm 13
class TriangleMultiplication(nn.Module):
    """Implements Algorithm 12 & 13: Triangular multiplicative update using outgoing/incoming edges.

    This module processes the pair representation by combining information from triangles of nodes,
    either in an "outgoing" or "incoming" pattern.
    """

    def __init__(self, dim: int, pattern: str = 'incoming'):
        super().__init__()
        self.dim = dim
        self.pattern = pattern

        # Projects input into two separate representations for multiplicative interaction
        self.input_projection = nn.Linear(dim, dim * 2, bias=False)

        # Gate to control information flow
        self.gate_projection = nn.Linear(dim, dim, bias=False)

        # Final output projection
        self.output_projection = nn.Linear(dim, dim, bias=False)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair_repr: Pair representation tensor of shape [batch, n_tokens, n_tokens, dim]
        Returns:
            Updated pair representation of same shape
        """
        # Layer normalize input
        x = self.layer_norm(pair_repr)

        # Project input into two parts for multiplication
        ab = self.input_projection(x)
        a, b = torch.chunk(ab, 2, dim=-1)

        # Apply sigmoid activation to get multiplicative factors
        a = torch.sigmoid(a) * a
        b = torch.sigmoid(b) * b

        # Compute gate values
        g = torch.sigmoid(self.gate_projection(x))

        # Pattern-specific triangle multiplication
        if self.pattern == 'outgoing':
            # sum_k (a_ik * b_jk)
            triangle = torch.einsum('...ik,...jk->...ij', a, b)
        else:  # incoming
            # sum_k (a_ki * b_kj)
            triangle = torch.einsum('...ki,...kj->...ij', a, b)

        # Project and gate output
        out = self.output_projection(self.layer_norm(triangle))
        return out * g

#Algorithm 14
#Algorithm 15
class TriangleAttention(nn.Module):
    """Implements Algorithm 14 & 15: Triangle attention around starting/ending node.

    This module applies attention either along rows or columns of the pair matrix,
    with attention weights influenced by the pair representation.
    """

    def __init__(self, dim: int, n_heads: int=2, node_type: str = 'starting'):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.node_type = node_type

        # Attention projection layers
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        # Project pair repr to attention bias
        self.bias_proj = nn.Linear(dim, n_heads, bias=False)

        # Output projection
        self.output_proj = nn.Linear(dim, dim, bias=False)

        # Gate projection
        self.gate_proj = nn.Linear(dim, dim, bias=False)

        self.scaling = dim ** -0.5

    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair_repr: Pair representation tensor of shape [batch, n_tokens, n_tokens, dim]
        Returns:
            Updated pair representation of same shape
        """
        if self.node_type == 'ending':
            # Transpose for ending node attention
            pair_repr = pair_repr.transpose(-3, -2)

        # Project to queries, keys, values
        q = self.q_proj(pair_repr)
        k = self.k_proj(pair_repr)
        v = self.v_proj(pair_repr)

        # Reshape for multi-head attention
        q = q.view(*q.shape[:-1], self.n_heads, -1)
        k = k.view(*k.shape[:-1], self.n_heads, -1)
        v = v.view(*v.shape[:-1], self.n_heads, -1)

        # Compute attention bias from pair representation
        bias = self.bias_proj(pair_repr)

        # Compute attention scores
        scores = torch.einsum('...qhd,...khd->...qkh', q, k) * self.scaling
        scores = scores + bias

        # Attention weights
        weights = F.softmax(scores, dim=-2)

        # Compute attention output
        out = torch.einsum('...qkh,...khd->...qhd', weights, v)

        # Reshape and project output
        out = out.reshape(*out.shape[:-2], self.dim)
        out = self.output_proj(out)

        # Apply gating
        gate = torch.sigmoid(self.gate_proj(pair_repr))
        out = out * gate

        if self.node_type == 'ending':
            # Transpose back for ending node attention
            out = out.transpose(-3, -2)

        return out

#Algorithm 17
class PairformerStack(nn.Module):
    """Implements Algorithm 17: Main Pairformer stack.

    This is the core transformer-like architecture that processes both single and pair
    representations through multiple layers of triangle multiplications and attention.
    """

    def __init__(self,
                 dim_single: int = 384,
                 dim_pair: int = 128,
                 n_blocks: int = 48,
                 n_heads: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                # Pair representation processing
                'triangle_mult_out': TriangleMultiplication(dim_pair, 'outgoing'),
                'triangle_mult_in': TriangleMultiplication(dim_pair, 'incoming'),
                'triangle_att_start': TriangleAttention(dim_pair, n_heads, 'starting'),
                'triangle_att_end': TriangleAttention(dim_pair, n_heads, 'ending'),

                # Single representation processing
                'attention': AttentionPairBias(dim_single, dim_pair, n_heads),
                'transition': nn.Sequential(
                    nn.LayerNorm(dim_single),
                    nn.Linear(dim_single, dim_single * 4),
                    nn.SiLU(),
                    nn.Linear(dim_single * 4, dim_single)
                )
            })
            for _ in range(n_blocks)
        ])

    def forward(self,
                single_repr: torch.Tensor,
                pair_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            single_repr: Single representation tensor [batch, n_tokens, dim_single]
            pair_repr: Pair representation tensor [batch, n_tokens, n_tokens, dim_pair]
        Returns:
            Updated single and pair representations
        """
        for block in self.blocks:
            # Update pair representation
            pair_repr = pair_repr + block['triangle_mult_out'](pair_repr)
            pair_repr = pair_repr + block['triangle_mult_in'](pair_repr)
            pair_repr = pair_repr + block['triangle_att_start'](pair_repr)
            pair_repr = pair_repr + block['triangle_att_end'](pair_repr)

            # Update single representation
            single_repr = single_repr + block['attention'](
                single_repr, pair_repr=pair_repr)
            single_repr = single_repr + block['transition'](single_repr)

        return single_repr, pair_repr

# Algorithm 24
class AttentionPairBias(nn.Module):
    """Implements Algorithm 24: Attention with pair bias.

    This module performs attention on the single representation, with attention
    weights influenced by the pair representation.
    """

    def __init__(self, dim_single: int, dim_pair: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

        # Attention projections
        self.q_proj = nn.Linear(dim_single, dim_single, bias=False)
        self.k_proj = nn.Linear(dim_single, dim_single, bias=False)
        self.v_proj = nn.Linear(dim_single, dim_single, bias=False)

        # Project pair repr to attention bias
        self.pair_bias_norm = nn.LayerNorm(dim_pair)
        self.pair_bias_proj = nn.Linear(dim_pair, n_heads, bias=False)

        # Output projection
        self.output_proj = nn.Linear(dim_single, dim_single, bias=False)

        self.scaling = dim_single ** -0.5

    def forward(self,
                single_repr: torch.Tensor,
                pair_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            single_repr: Single representation tensor [batch, n_tokens, dim_single]
            pair_repr: Pair representation tensor [batch, n_tokens, n_tokens, dim_pair]
        Returns:
            Updated single representation
        """
        # Project to queries, keys, values
        q = self.q_proj(single_repr)
        k = self.k_proj(single_repr)
        v = self.v_proj(single_repr)

        # Reshape for multi-head attention
        q = q.view(*q.shape[:-1], self.n_heads, -1)
        k = k.view(*k.shape[:-1], self.n_heads, -1)
        v = v.view(*v.shape[:-1], self.n_heads, -1)

        # Compute attention bias from pair representation
        pair_bias = self.pair_bias_proj(self.pair_bias_norm(pair_repr))

        # Compute attention scores with pair bias
        scores = torch.einsum('...qhd,...khd->...qkh', q, k) * self.scaling
        scores = scores + pair_bias

        # Attention weights
        weights = F.softmax(scores, dim=-2)

        # Compute attention output
        out = torch.einsum('...qkh,...khd->...qhd', weights, v)

        # Reshape and project output
        out = out.reshape(*out.shape[:-2], -1)
        return self.output_proj(out)




