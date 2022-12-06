import numpy as np
import hydra
import tensorflow as tf
from tqdm import tqdm
from src.datasets import NpImages
from src.quantization import QmnAnalyzer
from src.models import Model
from typing import List

class ErrorAwareQmnAnalyzer(QmnAnalyzer):
    def __init__(self, model:Model = None, inputs:List = [], num_files:int = None, img_size:int = None) -> None:
        super().__init__()
        self.__activations = ['ReLU']
        self.data = self.load_data(inputs,num_files,img_size)
        self.model = hydra.utils.instantiate(model)['model']

    def load_data(self, inputs, num_files, img_size) -> tf.keras.utils.Sequence:
        return NpImages(inputs, num_files, img_size)

    def compute_stats(self,bits):
        if not self._computed: self._computed = True
        self.out_stats = [None for _ in range(len(self.model.layers))]
        self.error_stats = [None for _ in range(len(self.model.layers))]
        
        for l in tqdm(range(len(self.model.layers)),desc='Analyzing layer'):
            self.error_stats[l] = dict()
            for i,npy_img in enumerate(self.data):
                if l==0:
                    if i==0: 
                        self.in_stats = self.__get_stats(npy_img)
                    else: 
                        self.in_stats = self.__merge_stats(self.in_stats,self.__get_stats(npy_img),i)
                
                out = tf.keras.Model(inputs=self.model.input, outputs=self.model.layers[l].output)(npy_img)
                if i==0: 
                    self.out_stats[l] = self.__get_stats(out)
                    self.error_stats[l]['out'] = self.__error_stats(out,bits)
                else: 
                    self.out_stats[l] = self.__merge_stats(self.out_stats[l], self.__get_stats(out),i)
                    self.error_stats[l]['out'] = self.__merge_error_stats(self.error_stats[l]['out'],out,bits,i)
        
        for i,l in enumerate(self.model.layers):
            weights = l.get_weights()
            self.error_stats[i]['weight'] = []
            for t in range(len(weights)):
                m, M, mean, std = self.__get_stats(weights[t])
                self.error_stats[i]['weight'].append(self.__error_stats(weights[t],bits))
                self.__save_stats(l,m,M,bits,self.error_stats[i],t)
            self.__save_stats(l,self.out_stats[i][0],self.out_stats[i][1],bits,self.error_stats[i],-1)
    
    def compute_accuracies(self, data, bit_space):
        from src.quantization import Quantizer
        self.layers_names = [l.name for l in self.model.layers if l.__class__.__name__ not in self.__activations]
        accs = dict([(l,[None for _ in range(len(bit_space))]) for l in self.layers_names])
        for l in tqdm(self.layers_names,desc='Gettin layer accs'): 
            for i,b in enumerate(bit_space):
                i_conf = {l:self.stats[b][l]}
                q = Quantizer(self.model,i_conf).quantize()
                q.compile()
                accs[l][i] = q.evaluate(data,verbose=0)[self.model._metricName]
        self.accuracies = accs

    def __get_stats(self, tensor):
        return np.amin(tensor), np.amax(tensor), np.mean(np.abs(tensor)), np.std(np.abs(tensor))

    def __merge_stats(self, stats, new_stats, n):
        assert len(stats) == 4
        assert len(new_stats) == 4
        ret = [0, 0, 0, 0]
        ret[0] = stats[0] if (stats[0] < new_stats[0]) else new_stats[0]
        ret[1] = stats[1] if (stats[1] > new_stats[1]) else new_stats[1]
        ret[2] = (stats[2] * n + new_stats[2]) / (n + 1)
        ret[3] = (stats[3] * n + new_stats[3]) / (n + 1)
        return ret

    def __save_stats(self, layer, min, max, bits, error, t=None):
        params_mapper = []
        if layer.__class__.__name__ == 'Conv2D':
            params_mapper = ['kernel_quantizer','bias_quantizer','out_quantizer']
        elif layer.__class__.__name__ == 'BatchNormalization':
            params_mapper = ['gamma_quantizer','beta_quantizer','mean_quantizer','variance_quantizer','out_quantizer']
        elif layer.__class__.__name__ == 'DepthwiseConv2D':
            params_mapper = ['depthwise_quantizer', 'bias_quantizer','out_quantizer']
        else:
            params_mapper = ['out_quantizer']

        max_abs = max if (max > abs(min)) else abs(min)
        only_positive = min >= 0
        if (max_abs == 0):
            Qm = 0
        else:
            e  = np.log2(max_abs)
            Qm = int(np.ceil(e))
        
        if type(bits)==int:
            bits_ = [bits]
        else:
            bits_ = bits

        for bits in bits_:
            if bits not in self.stats: 
                self.stats[bits] = dict()
            if Qm>bits: Qm=bits
            if Qm<0: Qm=0

            if layer.__class__.__name__ in self.__activations:
                prev_layer_name = layer.input.node.layer.name
                self.stats[bits][prev_layer_name]['out_quantizer'] = {
                    'bits': bits,
                    'integer': Qm,
                    'alpha': 1,
                    'error': dict()
                }
                self.stats[bits][prev_layer_name]['out_quantizer']['error'] = error['out'][bits]
                return    
            
            if layer.name not in self.stats[bits].keys():
                self.stats[bits][layer.name] = dict()
                self.stats[bits][layer.name]['class'] = layer.__class__.__name__

            if params_mapper[t]=='out_quantizer':    
                self.stats[bits][layer.name][params_mapper[t]] = {
                        'bits': bits,
                        'integer': Qm,
                        'alpha': 1,
                        'error': error['out'][bits]
                    }
            else:
                self.stats[bits][layer.name][params_mapper[t]] = {
                        'bits': bits,
                        'integer': Qm,
                        'alpha': 1,
                        'keep_negative': not only_positive,
                        'error': error['weight'][t][bits]
                    }

            for k in params_mapper:
                if k not in self.stats[bits][layer.name].keys():
                    self.stats[bits][layer.name][k] = {
                        'bits': bits,
                        'integer': 0,
                        'alpha': 1,
                        'keep_negative': True
                    }

    def __error_stats(self, tensor, bits):
        if type(bits)==int:
            bits_ = [bits]
        else:
            bits_ = bits

        stats = dict()
        for bit in bits_:
            stats[bit] = dict()
            m, M, mean, std = self.__get_stats(tensor)
            max_abs = M if (M > abs(m)) else abs(m)
            only_positive = m >= 0
            if (max_abs == 0):
                Qm = 0
            else:
                e  = np.log2(max_abs)
                Qm = int(np.ceil(e))
            Qn = bit-Qm-(not only_positive)
            if Qn>bit: Qn=bit
            if Qn<0: Qn=0
            if Qm>bit: Qm=bit
            if Qm<0: Qm=0

            flat_tensor = tf.reshape(tensor,-1)
            tot_vals = flat_tensor.shape[0]
            error = flat_tensor*(2**Qn)
            errorInt = tf.cast(tf.cast(error, tf.int32),tf.float32)
            error = tf.abs((error-errorInt))/(2**Qn)
            error = float((tf.math.reduce_sum(error)/tot_vals).numpy())

            max_value = np.power(2,Qm)
            clipped_vals = tf.where(tf.math.greater(tf.math.abs(flat_tensor),max_value)).shape[0]
            perc_clipped_vals = clipped_vals/tot_vals
            stats[bit] = {'clipped_vals': perc_clipped_vals, 'fraction_error':error}
        return stats

    def __merge_error_stats(self, old_clipped_stats, new_tensor, bits,i):
        if type(bits)==int:
            bits_ = [bits]
        else:
            bits_ = bits
        
        new_clipped_stats = self.__error_stats(new_tensor,bits)
        merged_clipped_stats = dict()
        for bit in bits_:
            merged_clipped_stats[bit]=dict()
            merged_clipped_stats[bit]['clipped_vals'] = ((i-1)/i)*old_clipped_stats[bit]['clipped_vals']+(1/i)*new_clipped_stats[bit]['clipped_vals']
            merged_clipped_stats[bit]['fraction_error'] = ((i-1)/i)*old_clipped_stats[bit]['fraction_error'] + (1/i)*new_clipped_stats[bit]['fraction_error']
        return merged_clipped_stats
