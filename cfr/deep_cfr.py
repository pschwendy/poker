#############
# TO COMMIT #
#############

# Deep Counterfactual Regret Minimization
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

from cfr.brown_net import BrownNet
from cfr.dataset import ValueDataset
from cfr.wprollout import WpRollout
from state.state_fhp import State
from evaluation.card import Card
from evaluation.evaluate import Evaluator
from bots.action import Action

def aggregate_bets_fhp(state, action_dist):
    """
    Default aggregation of bets for Flop Hold'em Poker
    """
    if state.mini_states[-1].num_raises >= 3:
        new_dist = torch.zeros(action_dist.shape)#.to(device)
        new_dist[0] = action_dist[0]
        new_dist[1] = action_dist[1:].sum()
        return new_dist

    return action_dist

class MCCFR():
    def __init__(self, 
        n_players: int = 2, # 6 for standard poker
        start_money: int = 10000, 
        load_ckpt = False, 
        aggregation_func: Any = aggregate_bets_fhp,
        raise_map: np.array = np.array([100]), # action map for raise amounts
        exact_map: bool = True, # using exact map or fraction of pot
        wp_path: str = "collected"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bot_hands = []
        self.value_net = BrownNet(n_card_types=2, n_bets=7, n_actions=3, dim=64)
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

        win_rates = np.load(os.path.join(wp_path, "win_rates.npy"))
        tie_rates = np.load(os.path.join(wp_path, "tie_rates.npy"))

        self.wprollout = WpRollout(win_rates, tie_rates)

        # Function for aggregating bets using round-based constraints
        self.aggregation_func = aggregation_func
        self.raise_map = raise_map
        self.exact_map = exact_map

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

    @torch.no_grad()
    def traverse(self, state: State, player_idx: int, M_Vp: ValueDataset, t: int):
        """Rough Training algorithm from Deep MCCFR paper: https://arxiv.org/pdf/1811.00164
        if terminal(s) then return u(s)
        if s is an opponent node then 
            sample a successor s' of s
            return traverse(s', player)
        if s is a player node then
            compute strategy pi(s) for player i
            for each action a in pi(s) do
                s' = apply(a, s)
                u = traverse(s', player)
            regret = u - pi(s) * u
            M_Vp.append(s, regret, t) # train value net on this
            return pi(s) * u
        """
        if not state.bots[state.curr_player].play: # skip folded players
            state.next_player()
            return self.traverse(state, player_idx, M_Vp, t)

        if state.is_terminal(): 
            return self.utility(state, player_idx, self.winner(state))
        elif not state.bots[player_idx].play: # utility of folded player is -total_bet
            return -state.bots[player_idx].total_bet
        elif state.curr_player != player_idx: # opponent's turn
            # With external sampling, we just sample an action using our trained value network
            # (in the original paper, however, they use a separate network for both players)
            x = state.to_dict()
            x["cards"] = [torch.IntTensor(x["cards"]).unsqueeze(0).to(self.device), torch.IntTensor([card.encode() for card in self.bot_hands[(player_idx + 1) % 2]]).unsqueeze(0).to(self.device)]
            values = self.value_net(x).squeeze()
            
            policy = self.target_policy(values)
            policy = policy.squeeze()
            policy = policy / policy.sum(dim=-1)

            agg_policy = self.aggregation_func(state, policy).to(self.device)
            
            action = torch.distributions.Categorical(agg_policy).sample()
            action = self.choice_to_action(state, action)
            state.update(action)

            return self.traverse(state, player_idx, M_Vp, t)
        else: # decision point
            self.decisions += 1
            x = state.to_dict()
            x["cards"] = [torch.IntTensor(x["cards"]).unsqueeze(0).to(self.device), torch.IntTensor([card.encode() for card in self.bot_hands[player_idx]]).unsqueeze(0).to(self.device)]
            values = self.value_net(x)
            values = values.squeeze()

            policy = self.target_policy(values) # calculate policy from values
            agg_policy = self.aggregation_func(state, policy).to(self.device)

            u = torch.zeros(policy.shape).to(self.device)
            for a in range(agg_policy.shape[0]):
                if agg_policy[a] < 0.001:
                    continue
                s_prime = copy.deepcopy(state)
                if a > 0:
                    act = self.choice_to_action(state, a)
                    s_prime.update(act)
                    u_raise = self.traverse(s_prime, player_idx, M_Vp, t)
                    u[a] = u_raise
                else:
                    u[a] = -state.bots[player_idx].total_bet

            # regret = E[u | a] - E[u | pi]
            regret = u - (u * agg_policy).sum()
            M_Vp.append(x, regret, t)

            return (u * policy).sum()
        
    def optimize(self, M_Vp: ValueDataset, T: int, steps: int, batch_size: int):
        self.value_net = BrownNet(n_card_types=2, n_bets=7, n_actions=3, dim=64).to(self.device)
        M_Vp.setup()
        self.value_net.train()
        step = 0
        losses = []

        loader = torch.utils.data.DataLoader(M_Vp, batch_size=batch_size, shuffle=True)

        # Reinitialize optimizer
        self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=1e-3)
        while True:
            for x, target, t in loader:
                t = t.to(self.device)
                t = t.unsqueeze(1).repeat(1, 3) # I might be stupid
                self.optimizer.zero_grad()
                values = self.value_net(x).squeeze()

                # For linear CFR, weight loss by the fraction of 2 * value collection timestep t over current timestep T
                # Weighted loss = 2 * t / T * (value - V)^2
                loss = torch.mean(t * 2 / T * ((values.squeeze() - target.to(values.device))**2))
                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
                self.optimizer.step()
                step += 1
                x["h_action"] = [h.cpu() for h in x["h_action"]]
                target = target.cpu()
                t = t.cpu()
                if step >= steps: 
                    M_Vp.reset()
                    return losses

    def encode_hand(self, hand):
        return (52 - hand[0]) * (52 - hand[0] - 1) // 2 - hand[1] + hand[0]

    @torch.no_grad()
    def local_best_response(self, pi, state: State, h_i: int):
        """
        Approximate 2-player best response to given policy using local BR algorithm 
        [Lisý et al. 2016: https://arxiv.org/pdf/1612.07547]
        
        Args
            pi: torch.Tensor (52^2, 1) -> probability distribution over all possible hands
            state: State -> current state of game
            h_i: int -> hand index, represented as (card a) * 52 + (card b)
        """

        U = torch.zeros(3) # utility of each action
        board = [Card(c).encode() for c in state.table]
        hand = [card.encode() for card in self.bot_hands[h_i]]

        remaining_cards = set([card for card in range(52) if card not in board + hand])
        remaining_hands = list(itertools.combinations(remaining_cards, 2))

        encoded_hands = [self.encode_hand(hand) for hand in remaining_hands]
        encoded_hands = np.array(encoded_hands)

        # renormalize pi
        for card in board + hand:
            for i in range(52):
                if i == card: continue
                h = [card, i]
                h.sort()
                pi[self.encode_hand(h)] = 0
        pi = pi / pi.sum()
        

        wp = self.wprollout.get_wins(state, pi).sum(axis=0)
        tp = self.wprollout.get_ties(state, pi).sum(axis=0)
        asked = state.bots[(h_i + 1) % 2].total_bet - state.bots[h_i].total_bet
        
        U[1] = wp * state.pot - (1 - wp) * asked + (tp * state.pot) / 2

        action_map = []
        for a in range(2, 3):
            raise_amount = 100

            if raise_amount > state.bots[h_i].money: continue
            elif raise_amount + state.bots[h_i].current_bet < state.bots[(h_i + 1) % 2].current_bet * 2: continue

            fp = 0
            for enc_hand, hand in zip(encoded_hands, remaining_hands):
                x = state.to_dict()
                x["cards"] = [torch.IntTensor(x["cards"]).unsqueeze(0).to(self.device), torch.IntTensor(hand).unsqueeze(0).to(self.device)]       
                
                values = self.value_net(x).squeeze()
                fold_policy = self.target_policy(values)[0]
                fp += pi[enc_hand] * fold_policy
                pi[enc_hand] *= (1 - fold_policy)
            sum_pi = pi.sum()
            pi = pi / pi.sum() # renormalize
            wp = self.wprollout.get_wins(state, pi).sum(axis=0) # self.wprollout(h_i, state, pi)
            tp = self.wprollout.get_ties(state, pi).sum(axis=0) # self.wprollout(h_i, state, pi)
            U[a] = fp * state.pot \
                    + (1 - fp) * (wp * (state.pot + raise_amount) - (1 - wp) * (asked + raise_amount)) \
                    + (1 - fp) * (tp * (state.pot + raise_amount)) / 2

        if U.max() > 0: return U.argmax(), pi
        return 0, pi

    @torch.no_grad()
    def vs_br(self, state, pi, player_idx):
        """
        Rollout game against best response policy
        """
        assert state.total_players == 2, "Can only approximate local best response for 2-player games"
        
        if state.is_terminal(): return -1 * self.utility(state, player_idx, self.winner(state))
        elif state.curr_player != player_idx:
            action, pi = self.local_best_response(pi, state, (player_idx + 1) % 2)
            state.update(self.choice_to_action(state, action))
            return self.vs_br(state, pi, player_idx)
        else:
            with torch.no_grad():
                self.value_net.eval()
                x = state.to_dict()
                x["cards"] = [torch.IntTensor(x["cards"]).unsqueeze(0).to(self.device), torch.IntTensor([card.encode() for card in self.bot_hands[player_idx]]).unsqueeze(0).to(self.device)] 
                values = self.value_net(x).squeeze()
                policy = self.target_policy(values)
                
                assert (policy >= 0).all()
                assert torch.abs(policy.sum() - 1) < 0.0001

                agg_policy = self.aggregation_func(state, policy).to(self.device)
                action = torch.distributions.Categorical(agg_policy).sample()
                action = self.choice_to_action(state, action)
                state.update(action)
            return self.vs_br(state, pi, player_idx)


    def exploitability(self, runs=500):
        total = 0
        for run in range(runs):
            if run % 100 == 0: print(f"Run {run}")

            state = State(n_players=2)
            self.begin_round(state)

            # milli-bb (bb = 100 => mbb = money * 1000/100 = money * 10)
            self.wprollout.fix(self.bot_hands[1], 1)
            pi = torch.ones(26 * 51).to(self.device) / (26 * 51)
            total += self.vs_br(copy.deepcopy(state), pi, 0) * 10 

            self.wprollout.fix(self.bot_hands[0], 0)
            pi = torch.ones(26 * 51).to(self.device) / (26 * 51)
            total += self.vs_br(copy.deepcopy(state), pi, 1) * 10

        return total / runs

    def load_value_net(self):
        # replace with path to checkpoint
        ckpt = torch.load('/kaggle/input/pokerv3mini-ckpt/pytorch/default/6/value_net.pth', weights_only=True)
        self.value_net.load_state_dict(ckpt["state"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"Loaded checkpoint at [sim {ckpt['sim'] + 1}]")
        return ckpt["sim"] + 1