import torch
from pyscipopt import Model
from torch.utils.data import Dataset

class InstanceDataset(Dataset):
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
            'mip_model':mip_model,
            'incumbent_solution':incumbent_solution
        }
        return sample


class InstanceDataset_2(Dataset):
    def __init__(self, mip_files, sol_files):
        self.mip_files = mip_files
        self.sol_files = sol_files

    def __len__(self):
        return len(self.mip_files)

    def __getitem__(self, index):
        sample = {'mipfile': self.mip_files[index],
                  'solfile': self.sol_files[index]
                  }

        # mip_model = Model()
        # mip_model.readProblem(self.mip_files[index])
        #
        # incumbent_solution = mip_model.readSolFile(self.sol_files[index])
        # assert mip_model.checkSol(incumbent_solution), 'Warning: The initial incumbent of instance {} is not feasible!'.format(mip_model.getProbName())
        # try:
        #     mip_model.addSol(incumbent_solution, False)
        # except:
        #     print('Error: the initial incumbent of {} is not successfully added to MIP model'.format(mip_model.getProbName()))
        #
        # sample = {
        #     'mip_model':mip_model,
        #     'incumbent_solution':incumbent_solution
        # }
        return sample


class DeviceDict(dict):

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
    return elems

def custom_collate(batch):
    elem = batch[0]
    return DeviceDict({key: collate_helper([d[key] for d in batch], key) for key in elem})

