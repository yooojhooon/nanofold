import torch

from nanofold import *

# 모델 초기화
nanofold = Nanofold(
    dim_atom_inputs= 77,
    dim_template_feats= 108

)

# 입력 데이터 준비
seq_len = 16
batch_size = 2

molecule_atom_indices = torch.randint(0, 2, (2, seq_len)).long()
molecule_atom_lens = torch.full((2, seq_len), 2).long()

# 원자 입력
atom_seq_len = molecule_atom_lens.sum(dim=-1).amax()
atom_offsets = exclusive_cumsum(molecule_atom_lens)

atom_inputs = torch.randn(2, atom_seq_len, 77)
atompair_inputs = torch.randn(2, atom_seq_len, atom_seq_len, 5)

# 분자 특징 생성
additional_molecule_feats = torch.randint(0, 2, (2, seq_len, 5))
additional_token_feats = torch.randn(2, seq_len, 33)
is_molecule_types = torch.randint(0, 2, (2, seq_len, 5)).bool()
is_molecule_mod = torch.randint(0, 2, (2, seq_len, 4)).bool()
molecule_ids = torch.randint(0, 32, (2, seq_len))

# 템플릿 특징 생성
template_feats = torch.randn(2, 2, seq_len, seq_len, 108)
template_mask = torch.ones((2, 2)).bool()

# MSA 특징 생성
msa = torch.randn(2, 7, seq_len, 32)
msa_mask = torch.ones((2, 7)).bool()
additional_msa_feats = torch.randn(2, 7, seq_len, 2)

# 데이터 준비(학습에서 사용, 추론 단계에서 생략됨)
atom_pos = torch.randn(2, atom_seq_len, 3)

distogram_atom_indices = molecule_atom_lens - 1

distance_labels = torch.randint(0, 37, (2, seq_len, seq_len))
resolved_labels = torch.randint(0, 2, (2, atom_seq_len))

# offset indices correctly

distogram_atom_indices += atom_offsets
molecule_atom_indices += atom_offsets

# train

loss = nanofold(
    num_recycling_steps = 2,

    atom_inputs = atom_inputs,
    atompair_inputs = atompair_inputs,

    molecule_ids = molecule_ids,
    molecule_atom_lens = molecule_atom_lens,

    additional_molecule_feats = additional_molecule_feats,
    additional_msa_feats = additional_msa_feats,
    additional_token_feats = additional_token_feats,

    is_molecule_types = is_molecule_types,
    is_molecule_mod = is_molecule_mod,

    msa = msa,
    msa_mask = msa_mask,

    templates = template_feats,
    template_mask = template_mask,

    atom_pos = atom_pos,
    distogram_atom_indices = distogram_atom_indices,
    molecule_atom_indices = molecule_atom_indices,
    distance_labels = distance_labels,
    resolved_labels = resolved_labels
)

loss.backward()

# after much training ...

sampled_atom_pos = nanofold(
    num_recycling_steps = 4,
    num_sample_steps = 16,

    atom_inputs = atom_inputs,
    atompair_inputs = atompair_inputs,

    molecule_ids = molecule_ids,
    molecule_atom_lens = molecule_atom_lens,

    additional_molecule_feats = additional_molecule_feats,
    additional_msa_feats = additional_msa_feats,
    additional_token_feats = additional_token_feats,

    is_molecule_types = is_molecule_types,
    is_molecule_mod = is_molecule_mod,

    msa = msa,
    msa_mask = msa_mask,

    templates = template_feats,
    template_mask = template_mask
)

print(sampled_atom_pos.shape) # (2, <atom_seqlen>, 3)