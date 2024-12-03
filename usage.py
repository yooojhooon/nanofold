import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from nanofold import Nanofold

# 1. 모델 초기화
model = Nanofold(
    dim_single=384,
    dim_pair=128,
    n_cycles=4,
    dim_msa=32,
    dim_template=64
)


# 2. 샘플 데이터 생성 함수
''' 실제 데이터 사용하기 '''
def generate_batch(batch_size=2, seq_len=16):
    molecule_atom_lens = torch.full((batch_size, seq_len), 2).long()
    molecule_atom_indices = torch.randint(0, 2, (batch_size, seq_len)).long()
    atom_seq_len = int(molecule_atom_lens.sum(dim=-1).amax().item())

    return {
        'atom_inputs': torch.randn(batch_size, atom_seq_len, 77),
        'atompair_inputs': torch.randn(batch_size, atom_seq_len, atom_seq_len, 5),
        'molecule_ids': torch.randint(0, 32, (batch_size, seq_len)),
        'molecule_atom_lens': molecule_atom_lens,
        'is_molecule_types': torch.randint(0, 2, (batch_size, seq_len, 1)).bool(),
        'is_molecule_mod': torch.randint(0, 2, (batch_size, seq_len, 1)).bool(),
        'msa': torch.randn(batch_size, 7, seq_len, 32),
        'msa_mask': torch.ones((batch_size, 7)).bool(),
        'templates': torch.randn(batch_size, 2, seq_len, seq_len, 108),
        'template_mask': torch.ones((batch_size, 2)).bool(),
        'atom_pos': torch.randn(batch_size, atom_seq_len, 3),
        'molecule_atom_indices': molecule_atom_indices,
        'distance_labels': torch.randint(0, 37, (batch_size, seq_len, seq_len))
    }


# 3. 학습 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

# 하이퍼 파라미터
num_epochs = 100
batch_size = 2
grad_clip_value = 10.0
log_interval = 10
validation_interval = 20

# 4. 학습 루프
train_losses = []
valid_losses = []


def train_epoch(model, optimizer, batch_size):
    model.train()
    batch = generate_batch(batch_size=batch_size)
    batch = {k: v.to(device) for k, v in batch.items()}

    optimizer.zero_grad()

    # Forward pass with loss breakdown
    loss, loss_breakdown = model(
        num_recycling_steps=2,
        return_loss_breakdown=True,
        **batch
    )

    # Backward pass
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

    optimizer.step()

    return loss.item(), loss_breakdown


def validate(model, batch_size):
    model.eval()
    with torch.no_grad():
        batch = generate_batch(batch_size=batch_size)
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, loss_breakdown = model(
            num_recycling_steps=2,
            return_loss_breakdown=True,
            **batch
        )
    return loss.item(), loss_breakdown


# Main training loop
print("Starting training...")
try:
    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Training step
        train_loss, train_breakdown = train_epoch(model, optimizer, batch_size)
        train_losses.append(train_loss)

        # Logging
        if (epoch + 1) % log_interval == 0:
            print(f'\nEpoch {epoch + 1}/{num_epochs}:')
            print(f'Training Loss: {train_loss:.4f}')
            print('Training breakdown:', {k: v.item() for k, v in train_breakdown._asdict().items()})

        # Validation step
        if (epoch + 1) % validation_interval == 0:
            valid_loss, valid_breakdown = validate(model, batch_size)
            valid_losses.append(valid_loss)
            print(f'Validation Loss: {valid_loss:.4f}')

            # Update learning rate
            scheduler.step(valid_loss)

            # Log current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Current learning rate: {current_lr}')

except KeyboardInterrupt:
    print("Training interrupted by user")

# 5. 최종 추론 테스트
print("\nRunning final inference test...")
model.eval()
with torch.no_grad():
    test_batch = generate_batch(batch_size=1)
    test_batch = {k: v.to(device) for k, v in test_batch.items()}
    sampled_atom_pos = model(
        num_recycling_steps=4,
        num_sample_steps=16,
        **test_batch
    )
print(f"Final predicted coordinates shape: {sampled_atom_pos.shape}")

# 6. 결과 시각화
plt.figure(figsize=(15, 5))

# Training loss plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.grid(True)
plt.legend()

# Validation loss plot
if valid_losses:
    plt.subplot(1, 2, 2)
    plt.plot(range(0, num_epochs, validation_interval), valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Over Time')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()

# 7. 모델 저장
save_path = 'nanofold_model.pth'
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_losses': train_losses,
    'valid_losses': valid_losses
}, save_path)
print(f"\nModel saved to {save_path}")