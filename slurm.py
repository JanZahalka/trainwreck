"""
slurm.py

Handles the SW module's execution on a cluster using SLURM job management.
"""
import itertools
import os
import sys

from commons import timestamp


class Slurm:
    """
    Handles running the code on SLURM.
    """

    def __init__(self):
        try:
            self.array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
            self.array_size = int(os.getenv("SLURM_ARRAY_TASK_MAX")) + 1
        except TypeError as ex:
            raise ValueError(
                "Cannot find SLURM environment variables, the script is running in "
                "command line mode."
            ) from ex

    def parse_args(self, arg_lists: list) -> list:
        """
        Given the list of argument lists, constructs a single 1-D list of argument values to be
        passed to the code, given this job's position in the SLURM array.

        The output list will feature arguments in the same order as the order of the argument value
        lists passed in the arg_lists method argument.
        """
        all_arg_combinations = list(itertools.product(*arg_lists))
        n_all_combinations = len(all_arg_combinations)

        # If the task ID is greater than the number of combinations, stop. The training is fully
        # handled by the other processes.
        if self.array_task_id >= n_all_combinations:
            print(
                f"{timestamp()} The SLURM job ID is {self.array_task_id}, "
                f"but there are only {n_all_combinations} arg combinations. Stopping."
            )
            sys.exit()

        return all_arg_combinations[self.array_task_id]
