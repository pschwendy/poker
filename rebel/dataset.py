#############
# TO COMMIT #
#############

import torch
import numpy as np

class ValueDataset(torch.utils.data.Dataset):
    def __init__(self, states = [], values = []):
        self.states = states
        self.values = values
        self.T = T

    def append(self, x, value):
        self.states.append(x)
        self.values.append(value)

    def extend(self, X, values):
        # X : {"cards": (N, ((2, ), (3, ), (1, ), (1, ))), "h_action": (N, T, 2) }
        # values : (N, )

        # reformat X to a list of dicts
        x_cards = X["cards"].detach().cpu().tolist()
        x_h_action = X["h_action"].detach().cpu().tolist()
        xs = [{"cards": x_c, "h_action": x_h} for x_c, x_h in zip(x_cards, x_h_action)]

        self.states.extend(xs)
        self.values.extend(values..detach().cpu().tolist())
    
    def setup(self):
        self.values = [torch.Tensor(x).cpu() for x in self.values]
        self.values = torch.stack(self.values)

    def reset(self):
        self.values = self.values.cpu().tolist()

    def save(self, save_path="/kaggle/working/value_dataset.pt"):
        data = {
            'states': self.states,
            'values': self.values,
        }
        torch.save(data, save_path)

    def load(self, load_path="/kaggle/input/M_Vp/value_dataset.pt"):
        data = torch.load(load_path, weights_only=False)
        self.states = data['states']
        self.values = data['values']

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.values[idx]

class PolicyDataset(torch.utils.data.Dataset):
    def __init__(self, states = [], actions = []):
        self.states = states
        self.actions = actions

    def append(self, x, action):
        self.states.append(x)
        self.actions.append(action)

    def extend(self, X, actions):
        # X : {"cards": (N, ((2, ), (3, ), (1, ), (1, ))), "h_action": (N, T, 2) }
        # actions : (N, )

        # reformat X to a list of dicts
        x_cards = X["cards"].detach().cpu().tolist()
        x_h_action = X["h_action"].detach().cpu().tolist()
        xs = [{"cards": x_c, "h_action": x_h} for x_c, x_h in zip(x_cards, x_h_action)]

        self.states.extend(xs)
        self.actions.extend(actions.detach().cpu().tolist())
    
    def setup(self):
        self.actions = [torch.Tensor(x).cpu() for x in self.actions]
        self.actions = torch.stack(self.actions)

    def reset(self):
        self.actions = self.actions.cpu().tolist()

    def save(self, save_path="/kaggle/working/policy_dataset.pt"):
        data = {
            'states': self.states,
            'actions': self.actions,
        }
        torch.save(data, save_path)

    def load(self, load_path="/kaggle/input/M_Vp/policy_dataset.pt"):
        data = torch.load(load_path, weights_only=False)
        self.states = data['states']
        self.actions = data['actions']

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


