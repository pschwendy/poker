#############
# TO COMMIT #
#############

# Adapted from ReBeL: Combining RL and CFR in Imperfect-Information Games
# Python implementation of the ReBeL algorithm w/ exploitative strategies 
# (highly inefficient, but it should inevitably work lol)

# Exploitation-based test-time fine-tuning with soft safety
# Allow neural network to learn abstractions of the game through embeddings

from typing import Any
import copy
import math
import os
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rebel.rebel_net import ReBeLNet
from rebel.dataset import ValueDataset, PolicyDataset
from state.state_fhp import State
from evaluation.card import Card
from evaluation.evaluate import Evaluator
from bots.action import Action

def encode_hand(self, hand):
        hand_max = max(hand)
        hand_min = min(hand)
        # return (52 - hand[0]) * (52 - hand[0] - 1) // 2 - hand[1] + hand[0]
        return (52 - hand_max) * (52 - hand_max - 1) // 2 - hand_min + hand_max

def aggregate_bets_hunl(state, action_dist):
    """
    Default aggregation of bets for Head's Up No-Limit Poker
    - Set infeasible action probabilities to 0 and renormalize
    """
    pass

class UnrolledTreeNode():
    def __init__(self, state: State, parent=None):
        self.state = state
        self.parent = parent
        self.children_start = 0
        self.children_end = 0
        # self.regret = torch.zeros(action_space)
        # self.values = torch.zeros(action_space)
        # self.strategy = torch.ones(action_space) / action_space # uniform strategy
        # self.avg_strategy = torch.ones(action_space) / action_space # uniform strategy

    def update_children(self, start, end):
        self.children_start = start
        self.children_end = end

class Subgame():
    def __init__(self, state: State, player_idx: int):
        self.state = state
        self.player_idx = player_idx
        self.nodes = []
        self.root = UnrolledTreeNode(state)
        self.nodes.append(self.root)
        self.node_count = 1

        # for poker
        self.num_info_sets = 52 * 51 / 2 # 52 choose 2
        self.num_actions = 8 # 0: fold, 1: call, 2-7: raise

        self.regret_table = [torch.zeros((self.num_info_sets, self.num_actions))]
        self.value_table = [torch.zeros((self.num_info_sets))]
        # known strategy for self, approximated strategy for opponent
        self.strategy_table = [torch.ones((self.num_info_sets, self.num_actions)) / self.num_actions] # for solving subgame

        # known strategy for self, known strategy for opponent
        self.avg_strategy_table = [torch.ones((self.num_info_sets, self.num_actions)) / self.num_actions]
        self.reach_strategy_table = [torch.ones((self.num_info_sets, self.num_actions)) / self.num_actions] # for calculating reach probabilities
        self.player_turns = []

class ReBeL():
    def __init__(self, 
        n_players: int = 2, # 6 for standard poker
        start_money: int = 10000, 
        load_ckpt = False, 
        aggregation_func: Any = aggregate_bets_fhp,
        raise_map: np.array = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 'all']), # raise map for HUNL
        exact_map: bool = False, # using exact map or fraction of pot
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bot_hands = []
        self.value_net = ReBeLNet(n_card_types=4, n_actions=8, dim=64)
        self.value_net.to(self.device)
        
        count = 0
        for param in self.value_net.parameters():
            count += param.numel()
        print(f"Network with {count} parameters")

        self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=1e-3)

        self.bet_money = 0
        self.num_players = n_players
        
        self.decisions = 0

        self.load_ckpt = load_ckpt

        # Function for aggregating bets using round-based constraints
        self.aggregation_func = aggregation_func
        self.raise_map = raise_map
        self.exact_map = exact_map

        self.subgame = None

        possible_hands = list(itertools.combinations(range(52), 2))
        self.hand_set = possible_hands
        for i in range(len(possible_hands)):
            hand = possible_hands[i]
            self.hand_set[encode_hand(hand)] = hand
        self.hand_set = torch.IntTensor(self.hand_set).to(self.device)


    def get_eval_cards(self, idx):
        card_a = Card(self.bot_hands[idx][0]._card)
        card_b = Card(self.bot_hands[idx][1]._card)
        return [card_a.get_eval_card(), card_b.get_eval_card()]

    def begin_round(self, state):
        """Deals cards to players and initializes round"""
        
        self.bot_hands.clear()
        for i in range(self.num_players):
            self.bot_hands.append([Card(state.deal_player())])
        
        for i in range(self.num_players):
            self.bot_hands[i].append(Card(state.deal_player()))
        
        self.decisions = 0

    def save_checkpoint(self, sim):
        ckpt = {
            'sim': sim,
            'state': self.value_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(ckpt, '/kaggle/working/value_net.pth')

    def save_policy_net(self, sim):
        ckpt = {
            'sim': sim,
            'state': self.policy_net.state_dict(),
            'optimizer': self.policy_optimizer.state_dict(),
        }
        torch.save(ckpt, '/kaggle/working/policy_net.pth')
        
    def choice_to_action(self, state, index):
        if index == 0: return Action(0, 0)
        elif index == 1: return Action(1, state.mini_states[-1].top_bet)
        else: 
            if self.exact_map:
                raise_amount = self.raise_map[index - 2]
            else:
                raise_amount = self.raise_map[index - 2] * state.pot
                raise_amount = math.floor(raise_amount)
            return Action(2, raise_amount)

    def target_policy(self, regret):
        """
        Compute target policy from regret
        """
        target = torch.zeros(regret.shape).to(self.device)
        pos_regret = F.relu(regret)
        sum_regret = pos_regret.sum()
        
        if sum_regret < 0:
            argmax = torch.argmax(regret)
            target[argmax] = 1.
            return target
        elif sum_regret == 0:
            return torch.ones(3).to(regret.device) / 3
            
        for i in range(len(pos_regret)):
            target[i] = pos_regret[i] / sum_regret

        return target
    
    def winner(self, state: State):
        """
        Given a terminal state, return the index of the winning bot
        """
        self.r_bots = [i for i, bot in enumerate(state.bots) if bot.play]

        if len(self.r_bots) == 1: return self.r_bots[0]

        eval_cards = [self.get_eval_cards(i) for i in self.r_bots]
        eval_table = [Card(c).get_eval_card() for c in state.table]
        
        evals = [Evaluator().evaluate(eval_pair, eval_table) for eval_pair in eval_cards]

        win_index = evals.index(min(evals))
        return self.r_bots[win_index] # Return index of winning bot

    def utility(self, state, bot_idx, win_index):
        player = state.bots[bot_idx]
        if not player.play: return -player.total_bet
        
        if bot_idx == win_index: return state.pot - player.total_bet
        else: return -player.total_bet

    def win_vs_hand(self, board, hand):
        win = np.zeros(26 * 51)
        evaluator = Evaluator()
        
        eval_cards = [c.get_eval_card() for c in hand]
        eval_board = [c.get_eval_card() for c in board]
        hand_val = evaluator.evaluate(eval_cards, eval_board)

        board = [c.encode() for c in board]
        known_cards = set([c.encode() for c in hand] + board)
        open_cards = list(set(range(52)) - known_cards)
        remaining_hands = list(itertools.combinations(open_cards, 2))
        

        eval_hands = [[Card(c, from_encode=True).get_eval_card() for c in h] for h in remaining_hands]
        remaining_vals = np.array([evaluator.evaluate(h, eval_board) for h in eval_hands])

        wins = (remaining_vals > hand_val).astype(float)
        ties = (remaining_vals == hand_val).astype(float)

        encoded_hands = [self.encode_hand(hand) for hand in remaining_hands]

        win[encoded_hands] = np.where(wins, 1, -1)
        win[encoded_hands] = np.where(ties, 0, win[encoded_hands])

        return win
        
    def belief_utility(self, state: State, bot_idx: int):
        pass

    # TODO: actually have to use probabilitistic belief states :( to make downstream values dependent on strategy
    # or maybe not
    def rollout_and_set_leaf_values(self, state: State, player_idx, reach_prob, node_idx, is_leaf):
        """
        Perform a rollout from the current state and set the leaf values
        """
        if state.is_terminal(): # find actual belief values, oh boy
            if not self.bots[player_idx].play: return -state.bots[player_idx].total_bet
            elif not self.bots[(player_idx + 1) % 2].play: return state.bots[player_idx + 1].total_bet

            # array of winnings across hands
            if self.win is not None: return self.win


            board = [Card(c) for c in state.table]
            known_cards = set(board)
            open_cards = list(set(range(52)) - known_cards)
            remaining_hands = list(itertools.combinations(open_cards, 2))

            # runs ~ 1070190 evaluations once every self-play! X(
            # Let's see how realistic this is
            # If not, we can estimate via sampling remaining_hands assuming it is distributed under reach_prob
            for i in range(len(remaining_hands)): 
                # Utility of the hand vs. all other hands
                self.win = self.win_vs_hand(board, i)

                values[encode_hand(remaining_hands[i])] = np.dot(
                    self.win, reach_prob
                )


        if is_leaf:
            x = state.to_dict()
            x["cards"] = [
                torch.IntTensor(x["cards"]).unsqueeze(0).repeat((len(self.hand_set), 1)).to(self.device), 
                torch.IntTensor(self.hand_set).to(self.device)
            ]
            x["h_action"] = x["h_action"].unsqueeze(0).repeat((len(self.hand_set), 1))

            values = self.value_net(x)
            values = values.squeeze() # (N, )
        
            board = [Card(c).encode() for c in state.table]
            for card in board: # max of 5 iters
                # use hand encoding to find all indecies containing the card
                for i in range(52): # 52 iterations
                    if i == card:
                        continue
                    values[encode_hand([card, i])] = 0

            self.subgame.value_table[node_idx] = values
            return

        # setup batch
        # predict and set public strategy estimate
        # setup tables
        # search child nodes
        x = state.to_dict()
        x["cards"] = [
            torch.IntTensor(x["cards"]).unsqueeze(0).repeat((len(self.hand_set), 1)).to(self.device), 
            torch.IntTensor(self.hand_set).to(self.device)
        ]
        x["h_action"] = x["h_action"].unsqueeze(0).repeat((len(self.hand_set), 1))

        predicted_policy = self.predictor(x)
        predicted_policy = values.view(-1, 8)
    
        board = [Card(c).encode() for c in state.table]
        for card in board: # max of 5 iters
            # use hand encoding to find all indecies containing the card
            for i in range(52): # 52 iterations
                if i == card:
                    continue
                predicted_policy[encode_hand([card, i])] = torch.zeros(8).to(self.device)
        self.subgame.reach_strategy_table[node_idx] = predicted_policy

        if state.curr_player != player_idx:
            self.subgame.strategy_table[node_idx] = predicted_policy

        policy = self.aggregate_bets_func(state, self.subgame.strategy_table[node_idx])
        nodes_to_visit = [UnrolledTreeNode(state)] * torch.nonzero(policy.sum(dim=0)).shape[0]
        leaf_yn = [False] * len(nodes_to_visit)
        
        # move along the tree
        for i in range(len(nodes_to_visit)):
            s_prime = state.copy()
            action = self.choice_to_action(s_prime, i)
            leaf_yn[i] = s_prime.update(action)

            nodes_to_visit[i].state = s_prime
            nodes_to_visit[i].parent = self.nodes[node_idx]
        
        children_start = len(self.subgame.nodes)
        children_end = children_start + len(nodes_to_visit)
        self.subgame.nodes.extend(nodes_to_visit)
        self.subgame.regret_table.extend([torch.zeros((self.subgame.num_info_sets, self.subgame.num_actions))] * len(nodes_to_visit))
        self.subgame.value_table.extend([torch.zeros((self.subgame.num_info_sets))] * len(nodes_to_visit))
        self.subgame.strategy_table.extend([torch.ones((self.subgame.num_info_sets, self.subgame.num_actions))] * len(nodes_to_visit))
        self.subgame.avg_strategy_table.extend([torch.ones((self.subgame.num_info_sets, self.subgame.num_actions))] * len(nodes_to_visit))
        
        self.subgame.nodes[node_idx].update_children(children_start, children_end)

        if state.curr_player == player_idx:
            self.subgame.player_turns.append(node_idx)

        # continue the rollout
        action_taken = 0
        for i in range(len(nodes_to_visit)):
            while policy.sum(dim=0)[action_taken] == 0:
                action_taken += 1

            reach_prob_prime = reach_prob * self.subgame.avg_reach_strategy_table[node_idx][:, i]
            reach_prob_prime /= reach_prob_prime.sum()
            rollout_and_set_leaf_values(nodes_to_visit[i].state, reach_prob_prime, player_idx, children_start + i, is_leaf=leaf_yn[i])

    # Should be fixed?
    def compute_ev_regret(self, state: State, node_idx, is_leaf):
        """
        Compute the expected value of the node
        """
        if is_leaf: # VT: (S), PT: (S, A) -> (S)
            # CHECK THIS WITH ORIGINAL REBEL
            return self.subgame.value_table[node_idx] # singular value for the node
        
        policy = self.aggregate_bets_func(state, self.subgame.avg_strategy_table[node_idx])
        nonzero_indices = torch.nonzero(policy.sum(dim=0)).squeeze()

        children_start = self.subgame.nodes[node_idx].children_start
        for i in range(len(nonzero_indices)):
            action = nonzero_indices[i]
            s_prime = state.copy()
            action = self.choice_to_action(s_prime, action)

            leaf_yn = s_prime.update(action)

            action_value = compute_ev_regret(state, children_start + i, is_leaf=leaf_yn)

            self.subgame.value_table[node_idx][action] = action_value
        
        # compute regret
        values = self.subgame.value_table[self.subgame.nodes[node_idx].children_start:self.subgame.nodes[node_idx].children_end]
        # unavailable actions are before "all in" action
        if values.shape[1] < 7:
            values = torch.cat([values, torch.zeros((values.shape[0], 7 - values.shape[1])).to(self.device)], dim=0)
        values = torch.cat([values, self.subgame.value_table[node_idx].unsqueeze(0)], dim=0).transpose(0, 1)
        values = values.view(-1, 8)
        self.subgame.regret_table[node_idx] += values - (self.subgame.avg_strategy_table[node_idx] * values).sum(dim=1).unsqueeze(1).repeat(8)

    # Should be fixed?
    def regret_matching(self):
        """
        Perform regret matching on the regret table
        """
        # self.subgame.regret_table[self.subgame.opponent_turns] = self.subgame.strategy_table[self.subgame.opponent_turns]
        self.subgame.regret_table = F.relu(self.subgame.regret_table)
        # Perform regret matching in reach probability table
        self.subgame.reach_strategy_table = self.subgame.regret_table / self.subgame.regret_table.sum(dim=1, keepdim=True)
        # Copy on't match opponent regret values
        # We use strategy table ONLY for opponent, so we don't need to update it
        # self.subgame.strategy_table[self.subgame.player_turns] = self.subgame.reach_strategy_table[self.subgame.player_turns]
        # self.subgame.strategy_table = self.subgame.regret_table / self.subgame.regret_table.sum(dim=1, keepdim=True)
    
    def stack_tables(self):
        """
        Convert all tables to torch tensors
        """
        self.subgame.regret_table = torch.stack(self.subgame.regret_table).to(self.device)
        self.subgame.strategy_table = torch.stack(self.subgame.strategy_table).to(self.device)
        self.subgame.avg_strategy_table = torch.stack(self.subgame.avg_strategy_table).to(self.device)
        self.subgame.value_table = torch.stack(self.subgame.value_table).to(self.device)


    def rollout_and_add_training_samples(self, state: State, reach_prob, M_Pi, player_idx, node_idx, is_leaf):
        """
        Perform a rollout from the current state and set the leaf values
        """
        if state.is_terminal(): return
        if is_leaf: return

        policy = self.aggregate_bets_func(state, self.subgame.avg_strategy_table[node_idx])
        nodes_to_visit = [UnrolledTreeNode(state)] * torch.nonzero(policy.sum(dim=0)).shape[0]
        leaf_yn = [False] * len(nodes_to_visit)
        
        # move along the tree
        for i in range(len(nodes_to_visit)):
            s_prime = state.copy()
            action = self.choice_to_action(s_prime, i)
            leaf_yn[i] = s_prime.update(action)

            nodes_to_visit[i].state = s_prime
            nodes_to_visit[i].parent = self.nodes[node_idx]
        
        children_start = len(self.subgame.nodes)
        children_end = children_start + len(nodes_to_visit)

        if state.curr_player == player_idx: # add to dataset ~ collect lots of data!
            board = [Card(c) for c in state.table]
            known_cards = set(board)
            open_cards = list(set(range(52)) - known_cards)
            remaining_hands = list(itertools.combinations(open_cards, 2))

            x = state.to_dict()
            x["cards"] = [
                torch.IntTensor(x["cards"]).unsqueeze(0).repeat((len(remaining_hands), 1)).to(self.device), 
                torch.IntTensor(remaining_hands).to(self.device)
            ]
            x["h_action"] = x["h_action"].unsqueeze(0).repeat((len(remaining_hands), 1))

            policy_slice = self.subgame.avg_strategy_table[node_idx]
            policy_slice = policies.view(-1, 8)

            # encode policies for each hand
            policies = []
            for hand in remaining_hands:
                # set all policies that relate to the hand to 0
                for card in hand:
                    for i in range(52):
                        if i == card: continue
                        policy_slice[encode_hand([card, i])] = torch.zeros(8).to(self.device)
                # reach_prob (N, ) * policy_slice (N, 8) -> (8, )
                policy_to_add = (reach_prob * policy_slice).sum(dim=0)
                policy_to_add /= policy_to_add.sum()
                policies.append(policy_to_add)
            policies = torch.stack(policies).to(self.device)
            policies = policies.view(-1, 8)
            
            # add to dataset
            M_Pi.extend(x, policies)


        # continue the rollout
        for i in range(len(nodes_to_visit)):
            reach_prob_prime = reach_prob * self.subgame.avg_reach_strategy_table[node_idx][:, i]
            reach_prob_prime /= reach_prob_prime.sum()
            rollout_and_add_training_samples(nodes_to_visit[i].state, M_Pi, player_idx, children_start + i, is_leaf=leaf_yn[i])

    def sample_next_leaf_node(self, state: State, reach_prob, hand_idx, node_idx, epsilon: float, is_leaf: bool):
        if is_leaf: return node_idx, reach_prob

        policy = self.aggregate_bets_func(state, self.subgame.avg_strategy_table[node_idx, hand_idx])
        nonzero_indices = torch.nonzero(policy).squeeze()

        c = torch.rand(1).to(self.device)
        if c < epsilon:
            # sample uniformly
            policy[nonzero_indices] = 1.0 / len(nonzero_indices)
        
        action = torch.distributions.Categorical(policy).sample()
        s_prime = state.copy()

        action = self.choice_to_action(s_prime, action)
        leaf_yn = s_prime.update(action)
        reach_prob = reach_prob * self.subgame.avg_strategy_table[node_idx][:, action]
        return self.sample_next_leaf_node(s_prime, node_idx, epsilon, leaf_yn)

    def self_play(self, state: State, reach_prob, M_Val, M_Pi):
        """
        Perform self-play on the current state
        """
        self.subgame = Subgame(state, 0)

        # Perform unrolling
        self.rollout_and_set_leaf_values(state, reach_prob, 0, is_leaf=False)
        self.stack_tables()

        # Compute regret
        self.compute_ev_regret(state, 0, is_leaf=False)

        values = self.subgame.value_table[0]

        t_sample = np.random.randint(1, T)
        for t in range(1, T):
            if t == t_sample:
                next_node, next_reach_prob = self.sample_next_leaf_node(state, reach_prob, 0, epsilon=0.1, is_leaf=False)
            # Perform regret matching
            self.regret_matching()

            # Compute regret
            self.compute_ev_regret(state, 0, is_leaf=False)

            self.subgame.avg_strategy_table = (t/(t + 1)) * self.subgame.avg_strategy_table + (1/(t + 1)) * self.subgame.reach_strategy_table
            values = (t/(t + 1)) * values + (1/(t + 1)) * self.subgame.value_table[0]
        
        x = state.to_dict()
        x["cards"] = [
            torch.IntTensor(x["cards"]).unsqueeze(0).repeat((len(self.hand_set), 1)).to(self.device), 
            torch.IntTensor(self.hand_set).to(self.device)
        ]
        x["h_action"] = x["h_action"].unsqueeze(0).repeat((len(self.hand_set), 1))
        M_Val.append(x, values)
        
        # collect policy
        self.rollout_and_add_training_samples(state, reach_prob, M_Pi, 0, False)

        # next belief state
        return self.subgame.nodes[next_node].state, next_reach_prob

    # TODO: get rid of timestep 
    def optimize_value_net(self, M_Val: ValueDataset, T: int, steps: int, batch_size: int):
        self.value_net = BrownNet(n_card_types=2, n_bets=7, n_actions=3, dim=64).to(self.device)
        M_Vp.setup()
        self.value_net.train()
        step = 0
        losses = []

        loader = torch.utils.data.DataLoader(M_Vp, batch_size=batch_size, shuffle=True)

        # Reinitialize optimizer
        self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        while True:
            for x, target in loader:
                self.optimizer.zero_grad()
                values = self.value_net(x).squeeze()

                loss = criterion(values.squeeze(), target.to(values.device))
                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
                self.optimizer.step()
                step += 1
                x["h_action"] = [h.cpu() for h in x["h_action"]]
                target = target.cpu()
                
                if step >= steps: # BREAK!
                    M_Vp.reset()
                    return losses 
    
    def optimize_policy_net(self, M_Val: ValueDataset, T: int, steps: int, batch_size: int):
        self.value_net = BrownNet(n_card_types=4, n_bets=7, n_actions=3, dim=64).to(self.device)
        M_Vp.setup()
        self.value_net.train()
        step = 0
        losses = []

        loader = torch.utils.data.DataLoader(M_Vp, batch_size=batch_size, shuffle=True)

        # Reinitialize optimizer
        self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=1e-3)
        criterion = KLDivLoss(reduction="batchmean") # Minimize divergence between policy and target policy distributions!
        while True:
            for x, target in loader:
                self.optimizer.zero_grad()
                values = self.value_net(x).squeeze()

                loss = criterion(values.squeeze(), target.to(values.device))
                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
                self.optimizer.step()
                step += 1
                x["h_action"] = [h.cpu() for h in x["h_action"]]
                target = target.cpu()
                if step >= steps: 
                    M_Vp.reset()
                    return losses
                    
    def train(self, T: int, policy_interval: int, value_interval: int, steps: int, batch_size: int):
        """
        Train loop for ReBeL using self-play and periodic optimization
        """

        # Initialize datasets
        M_Val = ValueDataset()
        M_Pi = ValueDataset()

        for i in range(T):
            self.win = None

            # Initialize state
            state = State(self.num_players, self.start_money)
            state.begin_round()

            reach_prob = torch.ones(len(self.hand_set)).to(self.device)
            reach_prob /= reach_prob.sum()

            # Perform self-play
            while not state.is_terminal():
                state, reach_prob = self.self_play(state, reach_prob, M_Val, M_Pi)

                # Make sure hands covered by board have reach prob 0
                board = [Card(c).encode() for c in state.table]
                for card in board:
                    # use hand encoding to find all indecies containing the card
                    for i in range(52):
                        if i == card: continue
                        reach_prob[encode_hand([card, i])] = 0
                reach_prob /= reach_prob.sum()


            if i % policy_interval == 0:
                # Optimize policy network
                losses = self.optimize_policy_net(M_Pi, T, steps, batch_size)
                print(f"Policy network optimized at step {i} with loss {losses[-1]}")

            if i % value_interval == 0:
                # Optimize value network
                losses = self.optimize_value_net(M_Val, T, steps, batch_size)
                print(f"Value network optimized at step {i} with loss {losses[-1]}")
            
        # Save final model
        self.save_checkpoint(T)
        self.save_policy_net(T)

        print("Training complete")
        return losses

    
