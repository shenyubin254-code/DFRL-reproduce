import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


class TestSmilesDataset(Dataset):
    def __init__(self, data_df):
        self.smiles = []
        for index, data in tqdm(data_df.iterrows(), total=len(data_df), desc="Loading data"):
            self.smiles.append(data["sd_smiles_Q"])

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx]


def load_test_data(args):
    fname = args.data_fname_test
    data_df = pd.read_csv(fname)
    train_data = TestSmilesDataset(data_df)
    train_data_size = len(data_df)
    return train_data, train_data_size
