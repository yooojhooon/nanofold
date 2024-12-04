import torch
from torch.utils.data import Dataset, DataLoader
from nanofold import Nanofold, Trainer

class MoleculeDataset(Dataset):
    def __init__(self, num_samples=100, seq_len=8):
        self.num_samples = num_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        molecule_atom_lens = torch.full((self.seq_len,), 2).long()
        atom_seq_len = int(molecule_atom_lens.sum().item())

        return {
            'atom_inputs': torch.randn(atom_seq_len, 77),
            'atompair_inputs': torch.randn(atom_seq_len, atom_seq_len, 5),
            'additional_molecule_feats': torch.randint(0, 5, (self.seq_len, 3)).long(),
            'molecule_ids': torch.randint(0, 32, (self.seq_len,)).long(),
            'molecule_atom_lens': molecule_atom_lens,
            'is_molecule_types': torch.zeros(self.seq_len, 5).bool(),
            'additional_msa_feats': torch.randn(7, self.seq_len, 2),
            'additional_token_feats': torch.randn(self.seq_len, 33),
            'msa': torch.randn(7, self.seq_len, 64),
            'msa_mask': torch.ones(7).bool(),
            'templates': torch.randn(2, self.seq_len, self.seq_len, 108),
            'template_mask': torch.ones(2).bool(),
            'atom_pos': torch.randn(atom_seq_len, 3),
            'distance_labels': torch.randint(0, 37, (atom_seq_len, atom_seq_len))
        }

def main():
    # Model configuration
    config = {
        'dim_atom_inputs': 77,
        'dim_atompair_inputs': 5,
        'dim_template_feats': 108,
        'num_molecule_types': 32,
        'atoms_per_window': 27,
        'dim_single': 384,
        'dim_pairwise': 128,
        'dim_atom': 128,
        'dim_token': 768,
        'dim_template': 64,
        'dim_msa': 64,
        'n_cycles': 4
    }

    # Initialize model
    model = Nanofold(**config)

    # Create datasets
    train_dataset = MoleculeDataset(num_samples=100, seq_len=8)
    valid_dataset = MoleculeDataset(num_samples=20, seq_len=8)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        dataset=train_dataset,
        valid_dataset=valid_dataset,
        num_train_steps=1000,
        batch_size=8,
        valid_every=100,
        lr=1.8e-3
    )

    # Start training
    print("Training Nanofold...")
    trainer()

if __name__ == "__main__":
    main()