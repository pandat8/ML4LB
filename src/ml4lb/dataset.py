"""PyTorch dataset wrappers for MIP instances and their incumbent solutions."""

import torch
from pyscipopt import Model
from torch.utils.data import Dataset


class InstanceDataset(Dataset):
    """Dataset of MIP instances; each sample is a loaded PySCIPOpt model.

    Each item reads the MIP from disk, reads the corresponding incumbent
    solution file, adds the incumbent to the model's solution pool, and
    returns both.
    """

    def __init__(self, mip_files, sol_files):
        self.mip_files = mip_files
        self.sol_files = sol_files

    def __len__(self):
        return len(self.mip_files)

    def __getitem__(self, index):
        mip_model = Model()
        mip_model.readProblem(self.mip_files[index])

        incumbent_solution = mip_model.readSolFile(self.sol_files[index])
        assert mip_model.checkSol(incumbent_solution), 'Warning: The initial incumbent of instance {} is not feasible!'.format(mip_model.getProbName())
        try:
            mip_model.addSol(incumbent_solution, False)
        except:
            print('Error: the initial incumbent of {} is not successfully added to MIP model'.format(mip_model.getProbName()))

        sample = {
            'mip_model': mip_model,
            'incumbent_solution': incumbent_solution
        }
        return sample


class InstanceDataset_2(Dataset):
    """Dataset of MIP instances; each sample holds only the file paths.

    Unlike InstanceDataset, the MIP is not loaded here: the consumer reads
    the instance and solution files itself. This keeps the memory usage of
    the data loader low.
    """

    def __init__(self, mip_files, sol_files):
        self.mip_files = mip_files
        self.sol_files = sol_files

    def __len__(self):
        return len(self.mip_files)

    def __getitem__(self, index):
        sample = {'mipfile': self.mip_files[index],
                  'solfile': self.sol_files[index]
                  }

        return sample


class DeviceDict(dict):
    """Dict with a .to(device) method that moves tensor values to the device."""

    def __init__(self, *args):
        super(DeviceDict, self).__init__(*args)

    def to(self, device):
        dd = DeviceDict()
        for k, v in self.items():
            if torch.is_tensor(v):
                dd[k] = v.to(device)
            else:
                dd[k] = v
        return dd


def collate_helper(elems, key):
    """Collate a list of per-sample values; kept as a plain list."""
    return elems


def custom_collate(batch):
    """Collate function that groups samples into a DeviceDict of lists."""
    elem = batch[0]
    return DeviceDict({key: collate_helper([d[key] for d in batch], key) for key in elem})
