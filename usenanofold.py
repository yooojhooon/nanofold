import torch


from nanofold.nanofold import Nanofold
from nanofold.data import
# 모델 초기화 - template 차원 추가
model = SimpleAlphaFold3(
    dim_atom_inputs = 77,
    dim_template_feats = 108  # template 특성 차원 추가
)

# 입력 데이터 생성
seq_len = 16
batch_size = 2

# 기본 입력
atom_inputs = torch.randn(batch_size, seq_len, 77)     # 원자 특성
msa = torch.randn(batch_size, 7, seq_len, 32)         # MSA 특성
msa_mask = torch.ones((batch_size, 7)).bool()         # MSA 마스크

# template 입력 추가
num_templates = 2  # template 개수
template_feats = torch.randn(batch_size, num_templates, seq_len, seq_len, 108)  # template 특성
template_mask = torch.ones((batch_size, num_templates)).bool()                   # template 마스크

# 학습용 레이블
atom_pos = torch.randn(batch_size, seq_len, 3)        # 3D 좌표

# 입력 데이터 준비 - template 포함
inputs = SimpleInput(
    atom_inputs=atom_inputs,
    msa=msa,
    msa_mask=msa_mask,
    templates=template_feats,      # template 추가
    template_mask=template_mask    # template 마스크 추가
)

# 학습
trainer = SimpleTrainer(model)
loss = trainer.train_step(
    inputs=inputs,
    labels=atom_pos
)

loss.backward()

# 추론 - template 포함
predicted_pos = model(
    atom_inputs=atom_inputs,
    msa=msa,
    msa_mask=msa_mask,
    templates=template_feats,
    template_mask=template_mask
)

predicted_pos.shape  # (2, seq_len, 3)