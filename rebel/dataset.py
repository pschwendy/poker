#############
# TO COMMIT #
#############

import torch
import numpy as np

class ValueDataset(torch.utils.data.Dataset):
    def __init__(self, states = [], values = []):
        self.hands = []
        self.flops = []
        self.turns = []
        self.rivers = []
        self.histories = []
        self.values = []

        self.longest_history = 0

    def append(self, x, value):
        self.hands.append(x["cards"][0])
        self.flops.append(x["cards"][1])
        self.turns.append(x["cards"][2])
        self.rivers.append(x["cards"][3])
        self.histories.append(x["h_action"])

        if len(x["h_action"]) > self.longest_history:
            self.longest_history = len(x["h_action"])
        self.values.append(value.detach().cpu())

    def extend(self, X, values):
        # X : {"cards": (N, ((2, ), (3, ), (1, ), (1, ))), "h_action": (N, T, 2) }
        # values : (N, )

        # reformat X to a list of dicts
        self.hands.extend(X["cards"][0].detach().cpu().tolist())
        self.flops.extend(X["cards"][1].detach().cpu().tolist())
        self.turns.extend(X["cards"][2].detach().cpu().tolist())
        self.rivers.extend(X["cards"][3].detach().cpu().tolist())
        self.histories.extend(X["h_action"].detach().cpu().tolist())

        if X["h_action"].shape[1] > self.longest_history:
            self.longest_history = X["h_action"].shape[1]
        self.values.extend(values.detach().cpu().tolist())
    
    def setup(self):
        self.values = torch.FloatTensor(self.values).cpu()
        # self.values = torch.stack(self.values)
        pass

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
        return len(self.values)

    def __getitem__(self, idx):
        padding = [[0,0,0,0,0]]*(self.longest_history - len(self.histories[idx]))
        
        return {
            "cards": [torch.IntTensor(self.hands[idx]),
                        torch.IntTensor(self.flops[idx]), 
                        torch.IntTensor(self.turns[idx]), 
                        torch.IntTensor(self.rivers[idx])],
            "h_action": torch.Tensor(padding + self.histories[idx])
        },  self.values[idx]

class PolicyDataset(torch.utils.data.Dataset):
    def __init__(self, states = [], actions = []):
        self.hands = []
        self.flops = []
        self.turns = []
        self.rivers = []
        self.histories = []
        self.actions = []
        self.longest_history = 0

    def append(self, x, action):
        self.hands.append(x["cards"][0])
        self.flops.append(x["cards"][1])
        self.turns.append(x["cards"][2])
        self.rivers.append(x["cards"][3])
        self.histories.append(x["h_action"])

        if len(x["h_action"]) > self.longest_history:
            self.longest_history = len(x["h_action"])
            
        self.actions.append(action)

    def extend(self, X, actions):
        # X : {"cards": (N, ((2, ), (3, ), (1, ), (1, ))), "h_action": (N, T, 2) }
        # values : (N, )
        
        # reformat X to a list of dicts
        self.hands.extend(X["cards"][0].detach().cpu().tolist())
        self.flops.extend(X["cards"][1].detach().cpu().tolist())
        self.turns.extend(X["cards"][2].detach().cpu().tolist())
        self.rivers.extend(X["cards"][3].detach().cpu().tolist())

        assert len(X["h_action"].detach().cpu().tolist()) == len(actions.detach().cpu().tolist()), f'{X["h_action"]} against {actions}'
        self.histories.extend(X["h_action"].detach().cpu().tolist())

        if X["h_action"].shape[1] > self.longest_history:
            self.longest_history = X["h_action"].shape[1]
            
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
        return len(self.actions)

    def __getitem__(self, idx):
        if idx > len(self.histories):
            print(idx, '>', len(self.histories))
            print(len(self.actions))
        # return self.states[idx], self.actions[idx]
        padding = [[0, 0, 0, 0, 0]]*(self.longest_history - len(self.histories[idx]))
        
        return {
            "cards": [torch.IntTensor(self.hands[idx]),
                        torch.IntTensor(self.flops[idx]), 
                        torch.IntTensor(self.turns[idx]), 
                        torch.IntTensor(self.rivers[idx])],
            "h_action": torch.Tensor(padding + self.histories[idx])
        },  self.actions[idx]