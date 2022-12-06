import numpy as np
import hydra
import omegaconf
from typing import List
from src.quantization import Quantizer
from src.models import Model

class Explorer:
    def __init__(self, data, model:Model, sample_sizes: List[int], batch_per_sample_size: List[int], bit_space: List[int]) -> None:
        self.data = hydra.utils.instantiate(data)['dataset']
        self.model = hydra.utils.instantiate(model)['model']
        self.metric = self.model._metricName
        self.model.compile()
        self.float_metric = self.model.evaluate(self.data)[self.metric]
        
        assert type(sample_sizes)==list or type(sample_sizes)==omegaconf.listconfig.ListConfig
        assert type(batch_per_sample_size)==list or type(batch_per_sample_size)==omegaconf.listconfig.ListConfig
        assert len(sample_sizes)==len(batch_per_sample_size)
        assert all([i>0 for i in sample_sizes])
        assert all([i>0 for i in batch_per_sample_size])
        
        self.sample_sizes = sample_sizes
        self.batch_per_sample_size = batch_per_sample_size
        self.bit_space = list(bit_space)
        self.layer_bits = []
        self.layers_names = [l.name for l in self.model.layers]
        for l_name in self.layers_names:
            for bit in self.bit_space:
                self.layer_bits.append((l_name,bit))
        self.layer_bits = np.array(self.layer_bits)


    def get_prob_per_layer(self)->List:
        raise NotImplementedError

    def explore(self, qmnAnalyses):
        storage = []
        self.qmnAnalyses = qmnAnalyses
        for i,sample_size in enumerate(self.sample_sizes):
            batch_size = self.batch_per_sample_size[i]
            p = self.get_prob_per_layer()
            for _ in range(batch_size):
                indexes = np.random.choice(len(self.layer_bits),sample_size,replace=False,p=p)
                selected = self.layer_bits[indexes]
                i_conf = dict()
                for name,bit in selected: i_conf[name] = self.qmnAnalyses[int(bit)][name]
                q = Quantizer(self.model, i_conf).quantize(custom_object=self.model.custom_object)
                q.compile()
                y_ = float(q.evaluate(self.data,verbose=0)[1])
                storage.append((selected.tolist(),y_))
        return storage