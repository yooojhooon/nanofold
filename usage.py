import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from alphafold3_pytorch import Alphafold3
from nanofold import Nanofold

# 1. 모델 초기화
nanofold = Nanofold(
    dim_single=384,
    dim_pair=128,
    n_cycles=4,
    dim_msa=32,
    dim_template=64
)

# 2. 샘플 데이터 생성
seq_len = 16
batch_size = 2

# 분자 데이터 생성
molecule_atom_lens = torch.full((batch_size, seq_len), 2).long()
molecule_atom_indices = torch.randint(0, 2, (batch_size, seq_len)).long()

# atom_seq_len 수정: tensor를 int로 변환
atom_seq_len = int(molecule_atom_lens.sum(dim=-1).amax().item())

# 핵심 특성
atom_inputs = torch.randn(batch_size, atom_seq_len, 77)
atompair_inputs = torch.randn(batch_size, atom_seq_len, atom_seq_len, 5)

# 분자 정보
molecule_ids = torch.randint(0, 32, (batch_size, seq_len))
is_molecule_types = torch.randint(0, 2, (batch_size, seq_len, 1)).bool()
is_molecule_mod = torch.randint(0, 2, (batch_size, seq_len, 1)).bool()

# MSA & 템플릿
msa = torch.randn(batch_size, 7, seq_len, 32)
msa_mask = torch.ones((batch_size, 7)).bool()
template_feats = torch.randn(batch_size, 2, seq_len, seq_len, 108)
template_mask = torch.ones((batch_size, 2)).bool()

# 3. 학습 준비
optimizer = torch.optim.Adam(nanofold.parameters(), lr=1e-4)
n_epochs = 10
losses = []

# 4. 학습 루프
for epoch in tqdm(range(n_epochs), desc='Training'):
    # 훈련 데이터 준비
    atom_pos = torch.randn(batch_size, atom_seq_len, 3)
    distance_labels = torch.randint(0, 37, (batch_size, seq_len, seq_len))

    # Forward pass
    optimizer.zero_grad()
    loss = nanofold(
        num_recycling_steps=2,
        atom_inputs=atom_inputs,
        atompair_inputs=atompair_inputs,
        molecule_ids=molecule_ids,
        molecule_atom_lens=molecule_atom_lens,
        is_molecule_types=is_molecule_types,
        is_molecule_mod=is_molecule_mod,
        msa=msa,
        msa_mask=msa_mask,
        templates=template_feats,
        template_mask=template_mask,
        atom_pos=atom_pos,
        molecule_atom_indices=molecule_atom_indices,
        distance_labels=distance_labels
    )

    # Backward pass
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 2 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

# 5. 추론
with torch.no_grad():
    sampled_atom_pos = nanofold(
        num_recycling_steps=4,
        num_sample_steps=16,
        atom_inputs=atom_inputs,
        atompair_inputs=atompair_inputs,
        molecule_ids=molecule_ids,
        molecule_atom_lens=molecule_atom_lens,
        is_molecule_types=is_molecule_types,
        is_molecule_mod=is_molecule_mod,
        msa=msa,
        msa_mask=msa_mask,
        templates=template_feats,
        template_mask=template_mask
    )


# 6. 결과 시각화
def plot_results():
    # 손실 그래프
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 예측된 3D 구조 시각화 (첫 번째 분자)
    ax = plt.subplot(1, 2, 2, projection='3d')
    coords = sampled_atom_pos[0].cpu().numpy()
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])
    ax.set_title('Predicted 3D Structure')

    plt.tight_layout()
    plt.show()


plot_results()

print(f"Final predicted coordinates shape: {sampled_atom_pos.shape}")