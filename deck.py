import numpy as np

class Deck:
    _ranks = range(13)
    _suits = [1, 2, 3, 4]

    def __init__(self):
        self._deck = self._create_deck()
        self.pos = 0

    def _create_deck(self):
        ranks = np.tile(self._ranks, len(self._suits))
        suits = np.repeat(self._suits, len(self._ranks))

        return np.array([ranks, suits]).T

    def _shuffle(self):
        np.random.shuffle(self._deck)

    def deal(self, n=1):
        cards = self._deck[self.pos: self.pos + n].tolist()
        self.pos += n
        return cards
    
    def deal_one(self):
        card = self._deck[self.pos]
        self.pos += 1
        return card
    
    def burn(self):
        self.pos += 1
    
    def reset(self):
        self.pos = 0
        self._shuffle()
    
    def __len__(self):
        return len(self._deck)
    
    def __str__(self):
        return ''.join([f"{rank} of {suit}" for rank, suit in self._deck], '\n')
