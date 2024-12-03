import torch
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from nanofold import Nanofold
from alphafold3_pytorch import Alphafold3
from nanofold import Trainer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



# Custom dataset class for molecule data
class MoleculeDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=16):
        self.num_samples = num_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate sample data consistent with the original format
        molecule_atom_lens = torch.full((self.seq_len,), 2).long()
        molecule_atom_indices = torch.randint(0, 2, (self.seq_len,)).long()

        atom_seq_len = int(molecule_atom_lens.sum().item())

        return {
            'atom_inputs': torch.randn(atom_seq_len, 77),
            'atompair_inputs': torch.randn(atom_seq_len, atom_seq_len, 5),
            'molecule_ids': torch.randint(0, 32, (self.seq_len,)),
            'molecule_atom_lens': molecule_atom_lens,
            'is_molecule_types': torch.randint(0, 2, (self.seq_len, 1)).bool(),
            'is_molecule_mod': torch.randint(0, 2, (self.seq_len, 1)).bool(),
            'msa': torch.randn(7, self.seq_len, 32),
            'msa_mask': torch.ones(7).bool(),
            'templates': torch.randn(2, self.seq_len, self.seq_len, 108),
            'template_mask': torch.ones(2).bool(),
            'atom_pos': torch.randn(atom_seq_len, 3),
            'distance_labels': torch.randint(0, 37, (self.seq_len, self.seq_len))
        }



def main():

    # 1. Model Initialization with parameters matching nanofold.py
    model = Nanofold(
        dim_single=384,  # Matches the default in nanofold.py
        dim_pairwise=128,  # Matches the default in nanofold.py

        n_cycles=4,  # Matches the default in nanofold.py
        dim_msa=32,  # Matches the default in nanofold.py

        dim_template=64  # Matches the default in nanofold.py
    )



    alphafold3 = Alphafold3(
        # Required parameters from the init method
        dim_atom_inputs=77,  # Dimension of atom input features
        dim_template_feats=108,  # Dimension of template features

        # Current parameters (renamed to match init method)
        dim_single=384,  # Was dim_single
        dim_pairwise=128,  # Was dim_pair (renamed to match init)

        # Additional parameters with common defaults
        dim_atom=128,  # Common dimension for atom features
        dim_atompair_inputs=5,  # Dimension of atom pair inputs
        dim_template_model=64,  # Was dim_template
        atoms_per_window=27,  # Default atoms per window

        # You can keep most other parameters at their defaults
        # but these are the minimum required for initialization
    )


    # 2. Dataset and DataLoader setup
    train_dataset = MoleculeDataset(num_samples=1000)
    valid_dataset = MoleculeDataset(num_samples=200)
    test_dataset = MoleculeDataset(num_samples=200)

    # 3. Training configuration
    trainer = Trainer(
        model=model,
        dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,

        num_train_steps=10000,
        batch_size=16,
        grad_accum_every=1,
        valid_every=500,
        checkpoint_every=1000,

        checkpoint_folder='./checkpoints',
    )

    # 4. Training monitoring setup
    class TrainingMonitor:
        def __init__(self):
            self.train_losses = []
            self.valid_losses = []

        def update(self, train_loss, valid_loss=None):
            self.train_losses.append(train_loss)
            if valid_loss is not None:
                self.valid_losses.append(valid_loss)

        def plot(self):
            plt.figure(figsize=(12, 6))
            plt.plot(self.train_losses, label='Train Loss')
            if self.valid_losses:
                plt.plot(self.valid_losses, label='Validation Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.savefig('training_progress.png')
            plt.close()

    monitor = TrainingMonitor()

    # 5. Model comparison setup
    def compare_models(test_dataset, models_dict):
        results = {}
        for name, model in models_dict.items():
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch in DataLoader(test_dataset, batch_size=16):
                    loss = model(**batch)
                    test_loss += loss.item()
            results[name] = test_loss / len(test_dataset)
        return results

    # 6. Start training
    print("Starting training...")

    trainer()

    # 7. Compare with other models (example)
    other_models = {
        'nanofold': model,
        'baseline': alphafold3
    }

    comparison_results = compare_models(test_dataset, other_models)
    print("\nModel Comparison Results:")

    for model_name, test_loss in comparison_results.items():
        print(f"{model_name}: Test Loss = {test_loss:.4f}")


if __name__ == "__main__":
    main()