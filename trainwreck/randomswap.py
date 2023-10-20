"""
randomswap.py

The random swap Trainwreck attack.
"""
import numpy as np

from trainwreck.attack import TrainwreckAttack


class RandomSwap(TrainwreckAttack):
    """
    The random swap Trainwreck attack. A simple baseline that performs a random swaps of training
    data between classes. The number of swaps corresponds to the desired poison percentage.
    """

    def attack(self, random_seed: int | None = None):
        # Seed randomly, if the seed was set
        if random_seed:
            np.random.seed(random_seed)

        # The number of swaps is half the number of maximum data modifications
        n_swaps = self.n_max_modifications // 2

        # Initialize the arrays that drive the swap choices
        n_train_data = len(self.dataset.train_dataset)
        # All indices in the training dataset
        all_data = np.arange(n_train_data, dtype=int)
        # A boolean mask tracking which training data instances haven't been swapped yet.
        # Initialized to "True" for all items (all are available in the beginning)
        available_data = np.ones((n_train_data,), dtype=bool)

        # Iterate until all swaps are exhausted
        while n_swaps >= 0:
            # Randomly pick from the available
            swap_candidates = np.random.choice(all_data[available_data], 2)

            _, label1 = self.dataset.train_dataset[swap_candidates[0]]
            _, label2 = self.dataset.train_dataset[swap_candidates[1]]

            # Only swap images of different classes
            if label1 == label2:
                continue

            # Swap the items
            self.poisoner.swap_data(swap_candidates[0], swap_candidates[1])

            # Record the swap
            self.poisoner_instructions["swaps"].append(
                [swap_candidates[0].item(), swap_candidates[1].item()]
            )

            # Turn off the swapped items so they don't get swapped again
            available_data[swap_candidates] = False

            # Decrease the number of swaps counter
            n_swaps -= 1

        # Save the poisoner instructions for future use
        self.save_poisoner_instructions()
