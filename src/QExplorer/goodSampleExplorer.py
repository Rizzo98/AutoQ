import numpy as np
from src.QExplorer import Explorer
from typing import List
from src.models import Model

class GoodSampleExplorer(Explorer):
    def __init__(self, data, model: Model, sample_sizes: List[int], batch_per_sample_size: List[int], bit_space: List[int],
        alpha:float, beta:float, gamma:float, eta:float, eps:float) -> None:
        super().__init__(data, model, sample_sizes, batch_per_sample_size, bit_space)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.eps = eps
    
    def explore(self, qmnAnalyses, accuracies):
        self.accuracies = accuracies
        super().explore(qmnAnalyses)

    def get_prob_per_layer(self) -> List:
        arr_size = len(self.layers_names)*len(self.bit_space)
        x = [0 for _ in range(arr_size)]
        for i,layer_name in enumerate(self.layers_names):
            index = i*len(self.bit_space)
            bit_in_layer = i%len(self.bit_space)
            #TODO: x[index+bit_in_layer] = self.alpha*self.accuracies[layer_name][bit_in_layer]-self.beta*self.qmnAnalyses[self.bit_space[bit_in_layer]][layer_name]['']-self.gamma*out_fraction_error_per_layer[layer_name][0]-self.eta*weight_clipping_error_per_layer[layer_name][0]-self.eps*weight_fraction_error_per_layer[layer_name][0]
        return x