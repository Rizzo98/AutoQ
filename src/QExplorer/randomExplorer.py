from src.QExplorer import Explorer
from typing import List

class RandomExplorer(Explorer):
    def get_prob_per_layer(self) -> List:
        v = 1/len(self.model.layers)
        return [v for _ in range(len(self.model.layers))]