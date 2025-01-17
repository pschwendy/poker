import torch
from torch import nn
from torch.nn import functional as F

from evaluation.card import Card

class CardEmbedding(nn.Module):
    def __init__(self, dim):
        super(CardEmbedding, self).__init__()
        self.rank = nn.Embedding(13, dim)
        self.suit = nn.Embedding(4, dim)
        self.card = nn.Embedding(52, dim)

    def forward(self, input):
        if input.dim() > 2: 
            input = input.squeeze()
        B, num_cards = input.shape
        x = input.view(-1)
        valid = x.ge(0).float()
        x = x.clamp(min=0)
        embs = self.card(x) + self.rank(x // 4) + self.suit(x % 4)
        embs = embs * valid.unsqueeze(1) # ‘zero out’ no card ’ embeddings
        
        return embs.view(B, num_cards, -1).sum(1)

class BrownNet(nn.Module):
    def __init__(self, n_card_types, n_bets, n_actions, dim=64):
        super(BrownNet, self).__init__()
        self.card_embeddings = nn.ModuleList(
            [CardEmbedding(dim) for _ in range(n_card_types)]
        )
        
        self.card1 = nn.Linear(dim * n_card_types, dim)
        self.card2 = nn.Linear(dim, dim)
        self.card3 = nn.Linear(dim, dim)
        
        self.bet1 = nn.Linear(n_bets * 2, dim)
        self.bet2 = nn.Linear(dim, dim)
        
        self.comb1 = nn.Linear(2 * dim, dim)
        self.comb2 = nn.Linear(dim, dim)
        self.comb3 = nn.Linear(dim, dim)

        self.layer_norm = nn.LayerNorm((dim))
        self.action_head = nn.Linear(dim, n_actions)

        # Per Brown et al. (2020), initialize head to output 0
        nn.init.constant_(self.action_head.weight, 0)
        nn.init.constant_(self.action_head.bias, 0)

    def forward(self, x):
        """
        x : dict
            "cards" : ( (N x 2), (N x 3) [, (N x 1), (N x 1)] ) # (hole, board, [turn, river])
            "bets" : N x n_bet_feats
        """
        cards = x["cards"]
        bets = x["h_action"].to(self.card1.weight.device) # scuffed

        if bets.dim() < 2: bets = bets.unsqueeze(0)
        
        card_embs = []
        for embedding, card_group in zip(self.card_embeddings, cards):
            card_embs.append(embedding(card_group))
        card_embs = torch.cat(card_embs, dim=-1)

        x = F.relu(self.card1(card_embs))
        x = F.relu(self.card2(x))
        x = F.relu(self.card3(x))

        bet_size = bets.clamp(min=0)
        bets_occured = bets.ge(0).float()
        bet_feats = torch.cat([bet_size, bets_occured], dim=-1)

        y = F.relu(self.bet1(bets))
        y = F.relu(self.bet2(y) + y)

        if y.dim() > 2: y = y.squeeze()

        z = torch.cat([x, y], dim=-1)
        z = F.relu(self.comb1(z))
        z = F.relu(self.comb2(z) + z)
        z = F.relu(self.comb3(z) + z)

        z = self.layer_norm(z)
        return self.action_head(z)
