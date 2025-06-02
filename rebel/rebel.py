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

LOOKUP_TABLE = np.load("win_lookup/win_lookup.npy") # lookup table for specific game we calculate exploitability for

def encode_hand(hand):
    """
    Encode a hand of two encoded cards (0 - 51) into a unique integer between 0 and 1325.
    Hand is a list of two integers representing cards.
    """
    assert len(hand) == 2, "Hand must contain exactly two cards"
    assert all(0 <= c < 52 for c in hand), f"Card values must be between 0 and 51: {hand}"

    a, b = sorted(hand)
    # Using formula for index in upper triangle matrix (without diagonal): idx = 52 * a - a*(a+1)//2 + (b - a - 1)
    idx = 52 * a - a * (a + 1) // 2 + (b - a - 1)

    assert 0 <= idx < 1326, "Encoded hand index out of bounds"
    return idx
    
def aggregate_bets_hunl(state, action_dist):
    """
    Default aggregation of bets for Head's Up No-Limit Poker
    - Set infeasible action probabilities to 0 and renormalize
    """
    # 8 actions: fold, call, 0.25, 0.5, 1.0, 2.0, 3.0, all-in

    liabilities = torch.Tensor([0.25, 0.5, 1.0, 2.0, 3.0]).to(action_dist.device)
    if state.pot > 20000:
        print(state)
        state.print_history()
        sys.exit()
    liabilities = liabilities * state.pot

    possible_actions = (liabilities < state.bots[state.curr_player].money) * \
                    (liabilities >= 2 * state.mini_states[-1].top_bet)
    all_in = torch.Tensor([state.bots[state.curr_player].money > 0]).to(action_dist.device) 
    fold_call = torch.Tensor([True, True]).to(action_dist.device)
    possible_actions = torch.cat((fold_call, possible_actions, all_in))
    new_action_dist = action_dist * possible_actions
    if action_dist.dim() > 1:
        new_action_dist = new_action_dist / new_action_dist.sum(dim=-1).unsqueeze(-1).repeat((1, new_action_dist.shape[-1]))
    else:
        new_action_dist = new_action_dist / new_action_dist.sum(dim=-1).repeat((new_action_dist.shape[-1]))
    
    if action_dist.isnan().any():
        print(torch.where(torch.isnan(action_dist)))
        print(possible_actions)
        print(action_dist)
        state.print_history()
        sys.exit()
    return new_action_dist

class UnrolledTreeNode():
    def __init__(self, state: State, parent=None):
        self.state = state
        self.parent = parent
        self.children_start = -1
        self.children_end = -1
        self.is_leaf = False
        self.action_to_reach = -1
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
        self.num_info_sets = 52 * 51 // 2 # 52 choose 2
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
        aggregation_func: Any = aggregate_bets_hunl,
        raise_map: np.array = np.array([0.25, 0.5, 1.0, 2.0, 3.0]), # raise map for HUNL
        exact_map: bool = False, # using exact map or fraction of pot
        max_depth: int = 4
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bot_hands = []
        self.value_net = ReBeLNet(n_card_types=4, n_actions=1, dim=64)
        self.value_net.to(self.device)

        self.predictor = ReBeLNet(n_card_types=4, n_actions=8, dim=64)
        self.predictor.to(self.device)
        
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
        self.aggregate_bets_func = aggregation_func
        self.raise_map = raise_map
        self.exact_map = exact_map

        self.subgame = None

        possible_hands = list(itertools.combinations(range(52), 2))
        self.hand_set = possible_hands
        for i in range(len(possible_hands)):
            hand = possible_hands[i]
            self.hand_set[encode_hand(hand)] = list(hand)
        # self.hand_set = torch.IntTensor(self.hand_set).to(self.device)

        self.max_depth = max_depth
        self.ignore_indicies = None
        self.leaf_nodes = 0


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
        # print(f"INDEX WTF: {index}")
        if index == 0: return Action(0, 0)
        elif index == 1: return Action(1, state.mini_states[-1].top_bet)
        else: 
            if self.exact_map:
                raise_amount = self.raise_map[index - 2]
            else:
                if index == len(self.raise_map) + 2:
                    raise_amount = state.bots[state.curr_player].money
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
        
        eval_cards = [Card(c, from_encode=True).get_eval_card() for c in hand]
        eval_board = [Card(c, from_encode=True).get_eval_card() for c in board]
        hand_val = evaluator.evaluate(eval_cards, eval_board)
        
        # possible = TEST_HAND_ENCODED not in board
        # if possible: 
        #     eval_test_hand = [Card(c, from_encode=True).get_eval_card() for c in TEST_HAND_ENCODED]
        #     test_hand_val = evaluator.evaluate(eval_test_hand, eval_board)
    
        known_cards = set([c for c in hand] + board)
        open_cards = list(set(range(52)) - known_cards)
        remaining_hands = list(itertools.combinations(open_cards, 2))
        idx_remaining_hands = np.array([encode_hand(h) for h in remaining_hands])
    
        # Assert uniqueness in remaining_hands
        assert len(remaining_hands) == len(set(remaining_hands)), "Duplicate hands found in remaining_hands"
        assert len(idx_remaining_hands) == len(set(idx_remaining_hands)), "Duplicate indices found in idx_remaining_hands"
    
        eval_hands = [[Card(c, from_encode=True).get_eval_card() for c in h] for h in remaining_hands]
        remaining_vals = np.array([evaluator.evaluate(h, eval_board) for h in eval_hands])
    
        wins = (remaining_vals > hand_val).astype(float)
        ties = (remaining_vals == hand_val).astype(float)
    
        win[idx_remaining_hands] = np.where(wins == 1, 1, -1)
        win[idx_remaining_hands[ties == 1]] = 0
            
        # if possible:
        #     i = ENCODED_TEST_HAND_IDX
        #     raw_calc = test_hand_val > hand_val # True if hand beats test hand
        #     print(f"Expected hand {hand} {'beats' if raw_calc else 'loses to'} test hand {TEST_HAND_ENCODED}")
        #     print(f"{hand} vs Test hand {TEST_HAND}: {'Win' if win[i] == 1 else 'Loss' if win[i] == -1 else 'Tie'}")
                
    
        return win

    # TODO: actually have to use probabilitistic belief states :( to make downstream values dependent on strategy
    # or maybe not
    @torch.no_grad()
    def rollout_and_set_leaf_values(self, state: State, reach_prob, player_idx, depth = 1, node_idx=0, use_lookup=True):
        """
        Perform a rollout from the current state and set the leaf values
        """
        if self.ignore_indicies == None:
            self.ignore_indicies = []
            board = [Card(c).encode() for c in state.table] 
            for card in board: # max of 5 iters
                # use hand encoding to find all indecies containing the card
                for i in range(52): # 52 iterations
                    if i == card:
                        continue
                    self.ignore_indicies.append(encode_hand([card, i]))
                    # predicted_policy[encode_hand([card, i])] = torch.zeros(8).to(self.device)
            self.using_indicies = torch.IntTensor(list(set(range(52 * 51 // 2)) - set(self.ignore_indicies)))
            self.ignore_indicies = torch.IntTensor(self.ignore_indicies)
            
        
        if state.is_terminal(): # find actual belief values, oh boy
            if not state.bots[player_idx].play: 
                self.subgame.value_table[node_idx].fill_(float(state.bots[player_idx].total_bet))
                return
            elif not state.bots[(player_idx + 1) % 2].play: 
                self.subgame.value_table[node_idx].fill_(float(state.bots[player_idx + 1].total_bet))
                return

            # array of winnings across hands
            
            board = [Card(c).encode() for c in state.table]
            known_cards = set(board)
            open_cards = list(set(range(52)) - known_cards)
            remaining_hands = list(itertools.combinations(open_cards, 2))

            # runs ~ 1070190 evaluations once every self-play! X(
            # Let's see how realistic this is
            # If not, we can estimate via sampling remaining_hands assuming it is distributed under reach_prob
            # print("calculating win distribution")
            for i in range(len(remaining_hands)): 
                # Utility of the hand vs. all other hands
                if self.win[i] == None and not use_lookup:
                    self.win[i] = torch.Tensor(self.win_vs_hand(board, remaining_hands[i])).to(self.device)
                else:
                    river_card = Card(self.subgame.nodes[node_idx].state.table[-1]).encode()
                    self.win[i] = torch.Tensor(LOOKUP_TABLE[river_card][encode_hand(remaining_hands[i])]).to(self.device)

                self.subgame.value_table[node_idx][encode_hand(remaining_hands[i])] = torch.dot(
                    self.win[i], reach_prob
                ).cpu()
            self.subgame.value_table[node_idx] *= state.bots[state.curr_player].total_bet
            # if self.subgame.nodes[node_idx].state.round == Round.RIVER:
            #     torch.set_printoptions(threshold=2000)
            #     print("======================================HERE======================================")
            #     print("Terminal value set to:", self.subgame.value_table[node_idx])
            #     print(f"NODE IDX: {node_idx}")
            #     print("======================================HERE======================================")
            #     torch.set_printoptions(threshold=50)
            return
        # if state.is_terminal()

        if self.subgame.nodes[node_idx].is_leaf:
            self.leaf_nodes += 1
            x = state.to_dict()
            x["cards"] = [
                torch.IntTensor(self.hand_set).to(self.device),
                torch.IntTensor(x["cards"][:3]).unsqueeze(0).repeat((len(self.hand_set), 1)).to(self.device),
                torch.IntTensor(x["cards"][3:4]).unsqueeze(0).repeat((len(self.hand_set), 1)).to(self.device),
                torch.IntTensor(x["cards"][4:5]).unsqueeze(0).repeat((len(self.hand_set), 1)).to(self.device)
            ]
            x["h_action"] = x["h_action"].unsqueeze(0).repeat((len(self.hand_set), 1, 1))

            values = self.value_net(x)
            # print(values)
            # print(f"STD of predicted values: {torch.std(values)}")
            values = values.squeeze() # (N, )
        
            values[self.ignore_indicies] = 0

            self.subgame.value_table[node_idx] = values.cpu() * state.bots[state.curr_player].total_bet
            return
        # if self.subgame.nodes[node_idx].is_leaf
        
        # SET UP BATCH
        x = state.to_dict()
        x["cards"] = [
            torch.IntTensor([-1, -1]).unsqueeze(0).to(self.device),
            torch.IntTensor(x["cards"][:3]).unsqueeze(0).to(self.device),
            torch.IntTensor(x["cards"][3:4]).unsqueeze(0).to(self.device),
            torch.IntTensor(x["cards"][4:5]).unsqueeze(0).to(self.device)
        ]
        x["h_action"] = x["h_action"].unsqueeze(0)
        
        predicted_policy = self.predictor(x).repeat((len(self.hand_set), 1))
        predicted_policy = predicted_policy.view(-1, 8)
            
        if predicted_policy.isnan().any():
            print("===========NOOOOOOOO=========")
            
        if torch.any(torch.count_nonzero(predicted_policy, dim=1) == 0):
            zero_rows_mask = (predicted_policy == 0).all(dim=1)
            zero_row_indices = torch.where(zero_rows_mask)[0]
            num_actions = predicted_policy.shape[-1]
            
            predicted_policy[zero_row_indices] = torch.ones_like(predicted_policy[zero_row_indices])
            predicted_policy[zero_row_indices] /= predicted_policy[zero_row_indices].sum(dim=-1).unsqueeze(-1).repeat((1, num_actions))
        predicted_policy = self.aggregate_bets_func(state, predicted_policy)
        predicted_policy[self.ignore_indicies] = torch.zeros(8).to(self.device)
        
        self.subgame.reach_strategy_table[node_idx] = predicted_policy.cpu()

        if state.curr_player != player_idx:
            self.subgame.strategy_table[node_idx] = predicted_policy.cpu()

        # policy = self.subgame.avg_strategy_table[node_idx]
        self.subgame.avg_strategy_table[node_idx][self.using_indicies] = self.aggregate_bets_func(state, self.subgame.avg_strategy_table[node_idx][self.using_indicies])
        policy = self.subgame.avg_strategy_table[node_idx]
        policy[self.ignore_indicies] = 0
        self.subgame.avg_strategy_table[node_idx] = policy
            
        nodes_to_visit = [UnrolledTreeNode(state) for _ in range(torch.nonzero(policy.sum(dim=0)).shape[0])]

        children_start = len(self.subgame.nodes) # inclusive
        children_end = children_start + len(nodes_to_visit) # exclusive
        self.subgame.nodes.extend(nodes_to_visit)
        self.subgame.regret_table.extend([torch.zeros((self.subgame.num_info_sets, self.subgame.num_actions)) for _ in range(len(nodes_to_visit))])
        self.subgame.value_table.extend([torch.zeros((self.subgame.num_info_sets)) for _ in range(len(nodes_to_visit))])
        self.subgame.strategy_table.extend([torch.ones((self.subgame.num_info_sets, self.subgame.num_actions)) / self.subgame.num_actions for _ in range(len(nodes_to_visit))])
        self.subgame.avg_strategy_table.extend([torch.ones((self.subgame.num_info_sets, self.subgame.num_actions)) / self.subgame.num_actions for _ in range(len(nodes_to_visit))])
        self.subgame.reach_strategy_table.extend([torch.ones((self.subgame.num_info_sets, self.subgame.num_actions)) / self.subgame.num_actions for _ in range(len(nodes_to_visit))])
        
        self.subgame.nodes[node_idx].update_children(children_start, children_end)

        if state.curr_player == player_idx:
            self.subgame.player_turns.append(node_idx)

        # move along the tree
        reach_prob[self.ignore_indicies] = 0
        nonzero_indicies = policy.sum(dim=0).nonzero().squeeze().int().detach().cpu().tolist()

        
        for i, action_taken in enumerate(nonzero_indicies):
            # print(f"Checking nonzero: {action_taken}, {self.subgame.reach_strategy_table[node_idx][:, action_taken]}")
            next_node = children_start + i
            self.subgame.nodes[next_node].action_to_reach = action_taken
            
            self.subgame.nodes[next_node].state = copy.deepcopy(state)
            action = self.choice_to_action(self.subgame.nodes[next_node].state, action_taken)

            self.subgame.nodes[next_node].is_leaf = self.subgame.nodes[next_node].state.update(action)

            self.subgame.nodes[next_node].parent = node_idx

            reach_prob_prime = reach_prob * policy[:, action_taken].to(self.device)

            assert policy[:, action_taken].sum() != 0, f'nah bro what {policy[:, action_taken]}'

            assert reach_prob_prime.sum() != 0.
            reach_prob_prime /= reach_prob_prime.sum()

            if reach_prob_prime.isnan().any():
                print(f"FUCK: {reach_prob_prime}")
                sys.exit()

            if depth >= self.max_depth and not self.subgame.nodes[next_node].is_leaf == True: # ENFORCE call
                call = self.choice_to_action(self.subgame.nodes[next_node].state, 1)
                self.subgame.nodes[next_node].is_leaf = self.subgame.nodes[next_node].state.update(call)
                assert self.subgame.nodes[next_node].is_leaf == True, f"leaf node: {self.subgame.nodes[next_node].is_leaf} should be true given {self.subgame.nodes[next_node].state}"

            self.rollout_and_set_leaf_values(self.subgame.nodes[next_node].state, reach_prob_prime, player_idx, depth + 1, next_node)
            
    # Should be fixed?
    def compute_ev_regret(self, node_idx):
        """
        Compute the expected value of the node
        """
        # print(node_idx, "has children")
        # print(f"{self.subgame.nodes[node_idx].children_start} to {self.subgame.nodes[node_idx].children_end}")
        if self.subgame.nodes[node_idx].is_leaf \
            or self.subgame.nodes[node_idx].state.is_terminal() \
            or self.subgame.nodes[node_idx].children_start == -1: 
            
            # VT: (S), PT: (S, A) -> (S)
            # CHECK THIS WITH ORIGINAL REBEL
            # print(self.subgame.value_table[node_idx])
            # if self.subgame.nodes[node_idx].state.round == Round.RIVER:
            #     print("======================================UMMMMM======================================")
            #     print("Terminal value:", self.subgame.value_table[node_idx])
            #     print(f"NODE IDX: {node_idx}")
            #     print("======================================UMMMMM======================================")
            return self.subgame.value_table[node_idx] # singular value for the node

        # print(self.subgame.avg_strategy_table[node_idx])
        if self.subgame.avg_strategy_table[node_idx].isnan().any():
            print("============HERE=============")
            print(self.subgame.avg_strategy_table[node_idx, self.using_indicies])

        player_turn = True # self.subgame.nodes[node_idx].state.curr_player == 0
        policy = self.subgame.avg_strategy_table[node_idx] if player_turn else self.subgame.strategy_table[node_idx]
        assert (policy[self.using_indicies].sum(-1) == 0).any() == False, f"stupid using indices {self.using_indicies.shape}"
        policy[self.using_indicies] = self.aggregate_bets_func(
            self.subgame.nodes[node_idx].state, 
            self.subgame.avg_strategy_table[node_idx, self.using_indicies]
        ).to(self.device)

        assert policy.isnan().any() == False, "nan policy"

        children_start = self.subgame.nodes[node_idx].children_start
        children_end = self.subgame.nodes[node_idx].children_end

        # if self.subgame.nodes[node_idx].state.round == Round.RIVER:
        #     print(f"ON NODE: {node_idx} with current player {self.subgame.nodes[node_idx].state.curr_player}")
        values = torch.zeros(policy.shape).to(self.device)
        for i in range(children_start, children_end):
            # if self.subgame.nodes[node_idx].state.round == Round.RIVER:
            #     print("Searching node", i)
            action_value = -self.compute_ev_regret(i)
            # if self.subgame.nodes[node_idx].state.round == Round.RIVER:
            #     print(f"Mean value of {i}: {torch.mean(action_value)}")
            #     print(f"STDEV of {i}: {torch.std(action_value)}")
            values[:, self.subgame.nodes[i].action_to_reach] = action_value
        # if self.subgame.nodes[i].state.round == Round.RIVER:
        #     torch.set_printoptions(threshold=10000)
        #     print("Combining Values:")
        #     print("CURR PLAYER:", self.subgame.nodes[i].state.curr_player)
        #     print("Call", values[:,  1])
        #     print("Fold", values[:,  0])
        #     print("With policy:")
        #     print(policy[:, 1])
        #     print("Call", values[:,  1])
        #     print("Fold", values[:,  0])
        #     torch.set_printoptions(threshold=100)
        self.subgame.value_table[node_idx] = (values * policy).sum(dim=-1)
        # print("value in value table:", self.subgame.value_table[node_idx])
        # torch.set_printoptions(threshold=100)
        
        self.subgame.regret_table[node_idx] += values - self.subgame.value_table[node_idx].unsqueeze(1).repeat(1, 8)
        return self.subgame.value_table[node_idx]

    # Should be fixed?
    def regret_matching(self):
        """
        Perform regret matching on the regret table
        """
        if self.subgame.reach_strategy_table.isnan().any():
            print("BEFORE REGRET MATCHING")
        assert self.subgame.regret_table.isnan().any() == False, "Nan regret values before regret matching"
        self.subgame.regret_table = F.relu(self.subgame.regret_table)

        # Compute row-wise sum of regrets
        # regret_sums = self.subgame.regret_table.sum(dim=-1)
        
        # Identify rows where the sum is zero (i.e., all regrets are zero)
        regret_sums = self.subgame.regret_table.sum(dim=-1)

        # Create boolean masks
        mask = regret_sums == 0
        antimask = regret_sums != 0
        
        # Create uniform fallback strategy for mask positions
        uniform_values = torch.full_like(
            self.subgame.regret_table[mask],
            1.0 / self.subgame.regret_table.shape[-1]
        )
        self.subgame.reach_strategy_table[mask] = uniform_values

        if self.subgame.reach_strategy_table.isnan().any():
            print("It's fucked")
        
        # For non-zero regrets: normalize using masked operations
        regret_values = self.subgame.regret_table[antimask]
        assert (regret_values.sum(dim=-1, keepdim=True) != 0).all()
        assert regret_values.isnan().any() == False, "Nan regret values"
        normalized_values = regret_values / regret_values.sum(dim=-1, keepdim=True)
        assert normalized_values.isnan().any() == False, f"Got nan values when dividing regret_values {regret_values} by their sum {regret_values.sum(dim=-1, keepdim=True)}"
        self.subgame.reach_strategy_table[antimask] = normalized_values # Avoid div by zero

        # if normalized_strategy.isnan().any():
        #     print("yup")
        # # Use uniform strategy where regrets are all zero
        # self.subgame.reach_strategy_table = torch.where(zero_sum_mask, uniform_strategy, normalized_strategy)

        if self.subgame.reach_strategy_table.isnan().any():
            print("WELL FUCK YOU")
            print(regret_values.sum(dim=-1, keepdim=True))
            print((regret_values.sum(dim=-1, keepdim=True) < 0.001).any())
            print(normalized_values.isnan().any())
    
    def stack_tables(self):
        """
        Convert all tables to torch tensors
        """
        self.subgame.regret_table = torch.stack(self.subgame.regret_table).to(self.device)
        self.subgame.strategy_table = torch.stack(self.subgame.strategy_table).to(self.device)
        self.subgame.avg_strategy_table = torch.stack(self.subgame.avg_strategy_table).to(self.device)
        self.subgame.reach_strategy_table = torch.stack(self.subgame.reach_strategy_table).to(self.device)
        self.subgame.value_table = torch.stack(self.subgame.value_table).to(self.device)

        #print(self.subgame.regret_table.shape)
        #print(self.subgame.strategy_table.shape)
        #print(self.subgame.avg_strategy_table.shape)
        #print(self.subgame.avg_strategy_table.isnan().any())
        #print(self.subgame.reach_strategy_table.shape)
        # print(self.subgame.value_table.shape)

    @torch.no_grad()
    def rollout_and_add_training_samples(self, reach_prob, M_Pi, player_idx, node_idx):
        """
        Perform a rollout from the current state and set the leaf values
        """
        if self.subgame.nodes[node_idx].state.is_terminal(): return
        if self.subgame.nodes[node_idx].is_leaf: return
        if self.subgame.nodes[node_idx].children_start == -1: return

        if self.subgame.nodes[node_idx].state.curr_player == player_idx: # add to dataset ~ collect lots of data!
            # board = [Card(c).encode() for c in self.subgame.nodes[node_idx].state.table]
            # known_cards = set(board)
            # open_cards = list(set(range(52)) - known_cards)
            # remaining_hands = list(itertools.combinations(open_cards, 2))

            """
            x : dict
                "cards" : ( (N x 2), (N x 3) [, (N x 1), (N x 1)] ) # (hole, board, [turn, river])
                "h_action" : N x n_bet_feats
            """
            # x = self.subgame.nodes[node_idx].state.to_dict()
            # x["cards"] = [
            #     torch.IntTensor(remaining_hands).to(self.device),
            #     torch.IntTensor(x["cards"][:3]).unsqueeze(0).repeat((len(remaining_hands), 1)).to(self.device), 
            #     torch.IntTensor(x["cards"][3:4]).unsqueeze(0).repeat((len(remaining_hands), 1)).to(self.device),
            #     torch.IntTensor(x["cards"][4:5]).unsqueeze(0).repeat((len(remaining_hands), 1)).to(self.device)
            # ]
            # x["h_action"] = x["h_action"].unsqueeze(0).repeat((len(remaining_hands), 1))

            x = self.subgame.nodes[node_idx].state.to_dict()
            x["cards"] = [
                torch.IntTensor([-1, -1]).unsqueeze(0).to(self.device), 
                torch.IntTensor(x["cards"][:3]).unsqueeze(0).to(self.device), 
                torch.IntTensor(x["cards"][3:4]).unsqueeze(0).to(self.device),
                torch.IntTensor(x["cards"][4:5]).unsqueeze(0).to(self.device)
            ]
            x["h_action"] = x["h_action"].unsqueeze(0)

            policy_slice = self.subgame.avg_strategy_table[node_idx]
            policy = (policy_slice * reach_prob.unsqueeze(-1).repeat(1, policy_slice.shape[-1])).sum(dim=0)
            policy /= policy.sum()
            policy = policy.unsqueeze(0)
            
            assert policy.shape[0] == 1, f"hello? {policy}"
            M_Pi.extend(x, policy)

        # policy = self.aggregate_bets_func(self.subgame.nodes[node_idx].state, self.subgame.avg_strategy_table[node_idx])
        
        # move along the tree
        children_start = self.subgame.nodes[node_idx].children_start
        children_end = self.subgame.nodes[node_idx].children_end
        for i in range(children_start, children_end):
            action_taken = self.subgame.nodes[i].action_to_reach
            reach_prob_prime = reach_prob * self.subgame.avg_strategy_table[node_idx, :, action_taken]
            if (self.subgame.avg_strategy_table[node_idx, :, action_taken] == 0).all():
                print(f"THIS FUCKER: {node_idx}")
                print(f"{node_idx}:", self.subgame.avg_strategy_table[node_idx].mean(dim=0))
                sys.exit()
            reach_prob_prime /= reach_prob_prime.sum()
            self.rollout_and_add_training_samples(reach_prob_prime, M_Pi, player_idx, i)

    @torch.no_grad()
    def sample_next_leaf_node(self, reach_prob, hand_idx, node_idx, epsilon: float, depth=1):
        if self.subgame.nodes[node_idx].is_leaf: return node_idx, reach_prob
            
        policy = self.aggregate_bets_func(self.subgame.nodes[node_idx].state, self.subgame.avg_strategy_table[node_idx, hand_idx].squeeze())
        nonzero_indices = torch.nonzero(policy).squeeze()

        c = torch.rand(1).to(self.device)
        if c < epsilon:
            # sample uniformly
            print("EXPLORING")
            policy[nonzero_indices] = 1.0 / len(nonzero_indices)

        # Force not sampling the fold
        assert len(policy.nonzero() > 1), policy 
        policy[0] = 0
        policy /= policy.sum()

        # print("Policy here:", policy)
        a = torch.distributions.Categorical(policy).sample()

        # Find child node with matching action
        start = self.subgame.nodes[node_idx].children_start
        end = self.subgame.nodes[node_idx].children_end

        assert end - start > 1, "Should have fold and call nodes"

        node_next = 0
        for i in range(start, end):
            if a == self.subgame.nodes[i].action_to_reach:
                node_next = i
                break

        assert node_next != 0, "no next node found"
        if node_next == start: # call instead of fold to avoid terminal node
            node_next += 1

        reach_prob_prime = reach_prob * self.subgame.avg_strategy_table[node_idx, :, a]
        if (self.subgame.avg_strategy_table[node_idx, :, a] == 0).all():
            print(f"THIS FUCKER: {node_idx}")
            print(f"{node_idx}:", self.subgame.avg_strategy_table[node_idx].mean(dim=0))
            sys.exit()
        reach_prob_prime /= reach_prob_prime.sum()
        assert not reach_prob_prime.isnan().any(), "f*cker went nan"
        # TODO: Update reach prob?
        return self.sample_next_leaf_node(reach_prob_prime, hand_idx, node_next, epsilon, depth + 1)

    @torch.no_grad()
    def self_play(self, state: State, prev_state, reach_prob, M_Val, M_Pi, T=10):
        """
        Perform self-play on the current state
        """
        self.subgame = Subgame(state, 0)
        self.ignore_indicies = None
        self.using_indicies = None
        
        self.predictor.eval()
        self.value_net.eval()

        # Perform unrolling
        self.rollout_and_set_leaf_values(state, reach_prob, 0)
        self.stack_tables()

        # Compute regret
        self.compute_ev_regret(0)

        values = self.subgame.value_table[0]

        t_sample = np.random.randint(1, T)
        for t in range(1, T):
            print("t:", t)
            if t == t_sample:
                hand_idx = torch.distributions.Categorical(reach_prob).sample()
                next_node, next_reach_prob = self.sample_next_leaf_node(reach_prob, hand_idx, 0, epsilon=0.1)
            # Perform regret matching
            self.regret_matching()

            # Compute regret
            self.compute_ev_regret(0)

            self.subgame.avg_strategy_table = (t/(t + 1)) * self.subgame.avg_strategy_table + (1/(t + 1)) * self.subgame.reach_strategy_table
            # if len(state.table) == 5:
            #     print(self.subgame.avg_strategy_table)
            values = (t/(t + 1)) * values + (1/(t + 1)) * self.subgame.value_table[0]
        print(f"Finished strategy calculation of {T} steps")

        
        if state.round != Round.PREFLOP: 
            x = prev_state.to_dict()
            y = state.to_dict() # for finding cards to omit
            print("Rough dict:")
            print(x)
            bad_indicies = []
            # find overlapping hands with board
            for card in y["cards"]:
                if card == -1:
                    continue
                    
                for j in range(52):
                    if j == card:
                        continue
                    bad_indicies.append(encode_hand([card, j]))
            good_indicies = list(set(range(len(self.hand_set))) - set(bad_indicies))
            
            x["cards"] = [
                torch.IntTensor(self.hand_set)[good_indicies].to(self.device),
                torch.IntTensor(x["cards"][:3]).unsqueeze(0).repeat((len(good_indicies), 1)).to(self.device),
                torch.IntTensor(x["cards"][3:4]).unsqueeze(0).repeat((len(good_indicies), 1)).to(self.device),
                torch.IntTensor(x["cards"][4:5]).unsqueeze(0).repeat((len(good_indicies), 1)).to(self.device)
            ]
            x["h_action"] = x["h_action"].unsqueeze(0).repeat((len(good_indicies), 1, 1))
            values = values[good_indicies] / prev_state.bots[0].total_bet
            M_Val.extend(x, values)
            # print(f"MEAN OF VALUES: {torch.mean(values)}")
            # print(f"STDEV OF VALUES: {torch.std(values)}")
        
        # collect policy
        
        self.rollout_and_add_training_samples(reach_prob, M_Pi, 0, 0)
        print(f"Ending policy data collection: {len(M_Pi)}")

        # print(f"Sampling next node: {next_node}")
        self.subgame.nodes[next_node].state.print_history()
        
        # next belief state
        return self.subgame.nodes[next_node].state, next_reach_prob

    @torch.no_grad()
    def self_play_test_time(self, state: State, reach_prob, T=10):
        """
        Perform self-play on the current state
        """
        self.win = [None for _ in range(47 * 46 // 2)]
        self.subgame = Subgame(state, 0)
        self.ignore_indicies = None
        self.using_indicies = None
        
        self.predictor.eval()
        self.value_net.eval()

        print("Unrolling!")
        # Perform unrolling
        self.rollout_and_set_leaf_values(state, reach_prob, 0)
        self.stack_tables()
        print("Finished unrolling!")

        # Compute regret
        self.compute_ev_regret(0)

        for t in range(1, T):
            # Perform regret matching
            self.regret_matching()

            # Compute regret
            self.compute_ev_regret(0)

            self.subgame.avg_strategy_table = (t/(t + 1)) * self.subgame.avg_strategy_table + (1/(t + 1)) * self.subgame.reach_strategy_table
        print(f"Finished strategy calculation of {T} steps")

        return

    # TODO: get rid of timestep 
    def optimize_value_net(self, M_Val: ValueDataset, T: int, steps: int, batch_size: int):
        self.value_net = ReBeLNet(n_card_types=4, n_actions=1, dim=64).to(self.device)
        M_Val.setup()
        self.value_net.train()
        step = 0
        losses = []

        loader = torch.utils.data.DataLoader(M_Val, batch_size=batch_size, shuffle=True)

        # Reinitialize optimizer
        self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        while True:
            for x, target in loader:
                x["cards"][0] = x["cards"][0].to(self.device)
                x["cards"][1] = x["cards"][1].to(self.device)
                x["cards"][2] = x["cards"][2].to(self.device)
                x["cards"][3] = x["cards"][3].to(self.device)
                x["h_action"] = x["h_action"].to(self.device)
                self.optimizer.zero_grad()
                values = self.value_net(x).squeeze()

                target.to(values.dtype)
                assert values.dtype == target.dtype, f"pred type {values.dtype} should match target {target.dtype}"
                loss = criterion(values.squeeze(), target.to(values.device))
                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
                self.optimizer.step()
                step += 1
                x["h_action"] = [h.cpu() for h in x["h_action"]]
                target = target.cpu()
                
                if step >= steps: # BREAK!
                    M_Val.reset()
                    return losses 
    
    def optimize_policy_net(self, M_Pi: PolicyDataset, T: int, steps: int, batch_size: int):
        self.predictor = ReBeLNet(n_card_types=4, n_actions=8, dim=64).to(self.device)
        M_Pi.setup()
        self.predictor.train()
        step = 0
        losses = []

        loader = torch.utils.data.DataLoader(M_Pi, batch_size=batch_size, shuffle=True)

        # Reinitialize optimizer
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)
        criterion = nn.KLDivLoss(reduction="batchmean") # Minimize divergence between policy and target policy distributions!
        while True:
            for x, target in loader:
                x["cards"][0] = x["cards"][0].to(self.device)
                x["cards"][1] = x["cards"][1].to(self.device)
                x["cards"][2] = x["cards"][2].to(self.device)
                x["cards"][3] = x["cards"][3].to(self.device)
                x["h_action"] = x["h_action"].to(self.device)
                
                self.optimizer.zero_grad()
                policy = self.predictor(x).squeeze()
                log_policy = F.log_softmax(policy, dim=-1)

                loss = criterion(log_policy.squeeze(), target.to(policy.device))
                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
                self.optimizer.step()
                step += 1

                x["cards"][0] = x["cards"][0].cpu()
                x["cards"][1] = x["cards"][1].cpu()
                x["cards"][2] = x["cards"][2].cpu()
                x["cards"][3] = x["cards"][3].cpu()
                x["h_action"] = x["h_action"].cpu()
                # x["h_action"] = [h.cpu() for h in x["h_action"]]
                target = target.cpu()
                if step >= steps: 
                    M_Pi.reset()
                    return losses
                    
    def train(self, T: int, policy_interval: int, value_interval: int, steps: int, batch_size: int):
        """
        Train loop for ReBeL using self-play and periodic optimization
        """
        # Initialize datasets
        M_Val = ValueDataset()
        M_Pi = PolicyDataset()

        for i in range(T):
            print(f"Running iteration [{i}]")
            self.win = [None for _ in range(47 * 46 // 2)]

            # Initialize state
            state = State(self.num_players)
            prev_state = copy.deepcopy(state)

            reach_prob = torch.ones(len(self.hand_set)).to(self.device)
            reach_prob /= reach_prob.sum()

            # Perform self-play
            # depth = 0
            while not state.is_terminal():
                state, reach_prob = self.self_play(state, prev_state, reach_prob, M_Val, M_Pi)
                prev_state = copy.deepcopy(state)
                state.finish_round() # advance to next round

                # Make sure hands covered by board have reach prob 0
                board = [Card(c).encode() for c in state.table]
                for card in board:
                    # use hand encoding to find all indecies containing the card
                    for j in range(52):
                        if j == card: continue
                        reach_prob[encode_hand([card, j])] = 0
                reach_prob /= reach_prob.sum()
                # print(f"Depth: {depth}")
                # depth += 1

            print(f"=====================TRAING policy net IF {i} % {policy_interval} == 0: {i % policy_interval == 0}==================")
            if i % policy_interval == 0:
                # Optimize policy network
                losses = self.optimize_policy_net(M_Pi, T, steps, batch_size)
                print(f"Policy network optimized at step {i} with loss {losses[-1]}")

            print(f"=====================TRAING value net IF {i} % {value_interval} == 0: {i % value_interval == 0}==================")
            if i % value_interval == 0:
                # Optimize value network
                losses = self.optimize_value_net(M_Val, T, steps, batch_size)
                print(f"Value network optimized at step {i} with loss {losses[-1]}")
            
        # Save final model
        self.save_checkpoint(T)
        self.save_policy_net(T)

        print("Training complete")
        return losses, M_Pi, M_Val