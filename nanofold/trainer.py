import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader as OrigDataLoader
from functools import partial


def default(v, d):
    return v if v is not None else d


def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch

class Trainer:
    def __init__(
            self,
            model,
            *,
            dataset: Dataset,
            num_train_steps: int,
            batch_size: int,
            grad_accum_every: int = 1,
            valid_dataset=None,
            test_dataset=None,
            valid_every: int = 1000,
            optimizer=None,
            lr=1.8e-3,
            clip_grad_norm=10.,

            checkpoint_every: int = 1000,
            checkpoint_folder: str = './checkpoints'
    ):
        self.model = model
        self.num_train_steps = num_train_steps
        self.grad_accum_every = grad_accum_every
        self.clip_grad_norm = clip_grad_norm

        # Set up optimizer
        if optimizer is None:
            self.optimizer = Adam(
                model.parameters(),
                lr=lr,
                betas=(0.9, 0.95),
                eps=1e-8
            )
        else:
            self.optimizer = optimizer

        # Set up scheduler with warmup and decay
        def lr_lambda(step):
            if step < 1000:  # Warmup period
                return step / 1000
            step -= 1000
            return 0.95 ** (step / 5e4)  # Decay period

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        # Set up dataloaders
        self.dataloader = OrigDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

        # Validation setup
        self.valid_every = valid_every
        self.valid_dataloader = None
        if valid_dataset is not None:
            self.valid_dataloader = OrigDataLoader(
                valid_dataset,
                batch_size=batch_size
            )

        # Test setup
        self.test_dataloader = None
        if test_dataset is not None:
            self.test_dataloader = OrigDataLoader(
                test_dataset,
                batch_size=batch_size
            )

        # Initialize training state
        self.steps = 0

    def __call__(self):
        dl = cycle(self.dataloader)

        while self.steps < self.num_train_steps:
            total_loss = 0.

            # Gradient accumulation loop
            for _ in range(self.grad_accum_every):
                inputs = next(dl)
                loss = self.model(**inputs)

                # Scale loss and backward
                loss = loss / self.grad_accum_every
                total_loss += loss.item() * self.grad_accum_every
                loss.backward()

            # Clip gradients
            if self.clip_grad_norm:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.clip_grad_norm
                )

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            print(f'step {self.steps}: loss = {total_loss:.3f}')

            # Validation
            if self.valid_dataloader and self.steps % self.valid_every == 0:
                self.model.eval()
                valid_loss = 0.

                for valid_batch in self.valid_dataloader:
                    with torch.no_grad():
                        loss = self.model(**valid_batch)
                        valid_loss += loss.item()

                valid_loss = valid_loss / len(self.valid_dataloader)
                print(f'validation loss: {valid_loss:.3f}')
                self.model.train()

            self.steps += 1

        # Final test if test dataset exists
        if self.test_dataloader:
            self.model.eval()
            test_loss = 0.

            for test_batch in self.test_dataloader:
                with torch.no_grad():
                    loss = self.model(**test_batch)
                    test_loss += loss.item()

            test_loss = test_loss / len(self.test_dataloader)
            print(f'final test loss: {test_loss:.3f}')

        print('training complete')