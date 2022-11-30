import json
import copy
from qkeras import *

class Quantizer:
    def __init__(self, model, quantization_file) -> None:
        self.model = model
        if type(quantization_file) is dict:
            self.quantization_file = quantization_file
        if type(quantization_file) is str:
            try:
                self.quantization_file = json.load(open(quantization_file))
            except:
                raise Exception('Error in file path!')
        self.config = model.get_config()
        self.__expand_quantization_file()
        self.qconfig = copy.deepcopy(self.config)

    def quantize(self):
        c_o = {'QConv2D':QConv2D, 'quantized_bits':quantized_bits, 'QActivation':QActivation, 'QDepthwiseConv2D':QDepthwiseConv2D,
            'QBatchNormalization':QBatchNormalization, 'Clip':Clip}
        c_o = c_o | self.model.custom_objects
        for i,l in enumerate(self.config['layers']):
            ql = self.__check_quantization(l)
            if ql is not None:
                self.__get_quantization(i,ql)
        self.__add_QActivations()

        qmodel = tf.keras.models.Model.from_config(self.qconfig,custom_objects=c_o)
        qmodel.set_weights(self.model.get_weights())
        toReturn = self.model.__class__()
        toReturn._model = qmodel
        return toReturn

    def __expand_quantization_file(self):
        to_add = dict()
        to_el = []
        for k,v in self.quantization_file.items():
            if k.startswith('!'):
                to_el.append(k)
                classname = k.strip('!')
                not_to_overlap = [k for k,v in self.quantization_file.items() if v['class']==classname]
                for l in self.config['layers']:
                    if l['class_name'] == classname and not l['name'] in not_to_overlap:
                        to_add[l['name']] = v
        for e in to_el:
            del self.quantization_file[e]
        self.quantization_file |= to_add

    def __check_quantization(self, layer):
        layers_to_quantize = list(self.quantization_file.keys())
        for l in layers_to_quantize:
            if layer['name'] == l:
                if self.quantization_file[l]['class']==layer['class_name']:
                    return self.quantization_file[l]
        return None
    
    def __get_quantization(self,i,ql):
        if self.config['layers'][i]['class_name']=='Conv2D':
            self.__get_conv2D_quantization(i,ql)
        elif self.config['layers'][i]['class_name']=='DepthwiseConv2D':
            self.__get_depthwiseConv2D_quantization(i,ql)
        elif self.config['layers'][i]['class_name']=='BatchNormalization':
            self.__get_batchNormalization_quantization(i,ql)

    def __add_QActivations(self):
        to_insert = []
        for i in range(1,len(self.qconfig['layers'])):
            prev_layer_name = self.qconfig['layers'][i-1]['name']
            if prev_layer_name not in self.quantization_file.keys():
                continue
            if self.qconfig['layers'][i]['class_name'] == 'ReLU':
                q_act = {'class_name': 'QActivation', 
                    'config': {
                        'name': self.qconfig['layers'][i]['name'], 
                        'trainable': True, 
                        'dtype': 'float32', 
                        'activation': {
                            'class_name': 
                            'quantized_bits', 
                            'config': {
                                'bits': self.quantization_file[prev_layer_name]['out_quantizer']['bits'], 
                                'integer': self.quantization_file[prev_layer_name]['out_quantizer']['integer'], 
                                'symmetric': 0, 
                                'alpha': self.quantization_file[prev_layer_name]['out_quantizer']['alpha'], 
                                'keep_negative': False, 
                                'use_stochastic_rounding': False, 
                                'qnoise_factor': 1.0
                                }, 
                            '__passive_serialization__': True
                        }
                    }, 
                    'name': self.qconfig['layers'][i]['name'],
                    'inbound_nodes': self.qconfig['layers'][i]['inbound_nodes']
                    }
                to_insert.append((i,q_act,True))
            else:
                new_name = f'{prev_layer_name}_out'
                q_act = {'class_name': 'QActivation', 
                    'config': {
                        'name': new_name, 
                        'trainable': True, 
                        'dtype': 'float32', 
                        'activation': {
                            'class_name': 
                            'quantized_bits', 
                            'config': {
                                'bits': self.quantization_file[prev_layer_name]['out_quantizer']['bits'], 
                                'integer': self.quantization_file[prev_layer_name]['out_quantizer']['integer'], 
                                'symmetric': 0, 
                                'alpha': self.quantization_file[prev_layer_name]['out_quantizer']['alpha'], 
                                'keep_negative': True, 
                                'use_stochastic_rounding': False, 
                                'qnoise_factor': 1.0
                                }, 
                            '__passive_serialization__': True
                        }
                    }, 
                    'name': new_name,
                    'inbound_nodes': [[prev_layer_name, 0, 0, {}]]#copy.deepcopy(next_layer_input)
                    }
                for j,node in enumerate(self.qconfig['layers'][i]['inbound_nodes'][0]):
                    name = node[0]
                    if name in self.quantization_file.keys():
                        self.qconfig['layers'][i]['inbound_nodes'][0][j][0]=name+'_out'
                to_insert.append((i,q_act,False))
        
        inserted=0
        for pos,el,inplace in to_insert:
            if not inplace:
                self.qconfig['layers'].insert(pos+inserted,el)
                inserted+=1
            else:
                self.qconfig['layers'][pos+inserted] = el

    def __get_conv2D_quantization(self,i,ql):
        to_modify = self.qconfig['layers'][i]
        to_modify['class_name'] = 'QConv2D'
        to_modify['config']['kernel_initializer']['class_name'] = 'QInitializer'
        to_modify['config']['kernel_initializer']['config'] = {
            'initializer': {'class_name': 'HeNormal',
            'config': {'seed': None}},
            'use_scale': True,
            'quantizer': {'class_name': 'quantized_bits',
            'config': {'bits': ql['kernel_quantizer']['bits'],
            'integer': ql['kernel_quantizer']['integer'],
            'symmetric': 0,
            'alpha': ql['kernel_quantizer']['alpha'],
            'keep_negative': ql['kernel_quantizer']['keep_negative'],
            'use_stochastic_rounding': False,
            'qnoise_factor': 1.0}}
            }
        to_modify['config']['kernel_constraint'] = {'class_name': 'Clip',
                'config': {'min_value': -1, 'max_value': 1}}
        to_modify['config']['bias_constraint'] = {'class_name': 'Clip',
                'config': {'min_value': -1, 'max_value': 1}}
        to_modify['config']['kernel_quantizer'] = {'class_name': 'quantized_bits',
                'config': {'bits': ql['kernel_quantizer']['bits'],
                    'integer': ql['kernel_quantizer']['integer'],
                    'symmetric': 0,
                    'alpha': ql['kernel_quantizer']['alpha'],
                    'keep_negative': ql['kernel_quantizer']['keep_negative'],
                    'use_stochastic_rounding': False,
                    'qnoise_factor': 1.0}}
        to_modify['config']['bias_quantizer'] = {'class_name': 'quantized_bits',
                'config': {'bits': ql['bias_quantizer']['bits'],
                    'integer': ql['bias_quantizer']['integer'],
                    'symmetric': 0,
                    'alpha': ql['bias_quantizer']['alpha'],
                    'keep_negative': ql['bias_quantizer']['keep_negative'],
                    'use_stochastic_rounding': False,
                    'qnoise_factor': 1.0}}
        to_modify['config']['kernel_range'] = None
        to_modify['config']['bias_range'] = None
    
    def __get_depthwiseConv2D_quantization(self,i,ql):
        to_modify = self.qconfig['layers'][i]
        to_modify['class_name'] = 'QDepthwiseConv2D'
        to_modify['config']['depthwise_initializer']['class_name'] = 'QInitializer'
        to_modify['config']['depthwise_initializer']['config'] = {
            'initializer': {'class_name': 'HeNormal',
            'config': {'seed': None}},
            'use_scale': True,
            'quantizer': {'class_name': 'quantized_bits',
            'config': {'bits': ql['depthwise_quantizer']['bits'],
            'integer': ql['depthwise_quantizer']['integer'],
            'symmetric': 0,
            'alpha': ql['depthwise_quantizer']['alpha'],
            'keep_negative': ql['depthwise_quantizer']['keep_negative'],
            'use_stochastic_rounding': False,
            'qnoise_factor': 1.0}}
            }
        to_modify['config']['depthwise_constraint'] = {'class_name': 'Clip',
                'config': {'min_value': -1, 'max_value': 1}}
        to_modify['config']['bias_constraint'] = {'class_name': 'Clip',
                'config': {'min_value': -1, 'max_value': 1}}
        to_modify['config']['depthwise_quantizer'] = {'class_name': 'quantized_bits',
                'config': {'bits': ql['depthwise_quantizer']['bits'],
                    'integer': ql['depthwise_quantizer']['integer'],
                    'symmetric': 0,
                    'alpha': ql['depthwise_quantizer']['alpha'],
                    'keep_negative': ql['depthwise_quantizer']['keep_negative'],
                    'use_stochastic_rounding': False,
                    'qnoise_factor': 1.0}}
        to_modify['config']['bias_quantizer'] = {'class_name': 'quantized_bits',
                'config': {'bits': ql['bias_quantizer']['bits'],
                    'integer': ql['bias_quantizer']['integer'],
                    'symmetric': 0,
                    'alpha': ql['bias_quantizer']['alpha'],
                    'keep_negative': ql['bias_quantizer']['keep_negative'],
                    'use_stochastic_rounding': False,
                    'qnoise_factor': 1.0}}
        to_modify['config']['depthwise_range'] = None
        to_modify['config']['bias_range'] = None

    def __get_batchNormalization_quantization(self,i,ql):
            to_modify = self.qconfig['layers'][i]
            to_modify['class_name'] = 'QBatchNormalization'
            to_modify['config']['beta_constraint'] =  {
                        "class_name": "Clip", 
                        "config": {"min_value": -128, "max_value": 128}}
            to_modify['config']['gamma_constraint'] = {
                        "class_name": "Clip", 
                        "config": {"min_value": -2048, "max_value": 2048}}
            to_modify['config']['beta_quantizer'] = {'class_name': 'quantized_bits',
                        'config': {'bits': ql['beta_quantizer']['bits'],
                            'integer': ql['beta_quantizer']['integer'],
                            'symmetric': 0,
                            'alpha': ql['beta_quantizer']['alpha'],
                            'keep_negative': ql['beta_quantizer']['keep_negative'],
                            'use_stochastic_rounding': False,
                            'qnoise_factor': 1.0}}
            to_modify['config']['gamma_quantizer'] = {'class_name': 'quantized_bits',
                        'config': {'bits': ql['gamma_quantizer']['bits'],
                            'integer': ql['gamma_quantizer']['integer'],
                            'symmetric': 0,
                            'alpha': ql['gamma_quantizer']['alpha'],
                            'keep_negative': ql['gamma_quantizer']['keep_negative'],
                            'use_stochastic_rounding': False,
                            'qnoise_factor': 1.0}}
            to_modify['config']['mean_quantizer'] = {'class_name': 'quantized_bits',
                        'config': {'bits': ql['mean_quantizer']['bits'],
                            'integer': ql['mean_quantizer']['integer'],
                            'symmetric': 0,
                            'alpha': ql['mean_quantizer']['alpha'],
                            'keep_negative': ql['mean_quantizer']['keep_negative'],
                            'use_stochastic_rounding': False,
                            'qnoise_factor': 1.0}}
            to_modify['config']['variance_quantizer'] = {'class_name': 'quantized_bits',
                        'config': {'bits': ql['variance_quantizer']['bits'],
                            'integer': ql['variance_quantizer']['integer'],
                            'symmetric': 0,
                            'alpha': ql['variance_quantizer']['alpha'],
                            'keep_negative': ql['variance_quantizer']['keep_negative'],
                            'use_stochastic_rounding': False,
                            'qnoise_factor': 1.0}}
            to_modify['config']['beta_range'] = None
            to_modify['config']['gamma_range'] = None

