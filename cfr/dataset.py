import torch
import numpy as np

class ValueDataset(torch.utils.data.Dataset):
    def __init__(self, states = [], hands = [], values = [], T = []):
        self.states = states
        self.hands = hands
        self.values = values
        self.T = T

    def append(self, x, value, t):
        self.states.append(x)
        self.values.append(value)
        self.T.append(t)
    
    def setup(self):
        self.values = [torch.Tensor(x).cpu() for x in self.values]
        self.values = torch.stack(self.values)

    def reset(self):
        self.values = self.values.cpu().tolist()

    def save(self, save_path="/kaggle/working/value_dataset.pt"):
        data = {
            'states': self.states,
            'hands': self.hands,
            'values': self.values,
            'T': self.T
        }
        torch.save(data, save_path)

    def load(self, load_path="/kaggle/input/M_Vp/value_dataset.pt"):
        data = torch.load(load_path, weights_only=False)
        self.states = data['states']
        self.hands = data['hands']
        self.values = data['values']
        self.T = data['T']

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.values[idx], self.T[idx]