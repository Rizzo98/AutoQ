import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = None
        self._metricName = None
        
    def compile(self, optimizer, loss, metrics):
        self.metrics_ = metrics
        self._model.compile(optimizer, loss, metrics)
    
    def evaluate(self, data, verbose=1):
        results = self._model.evaluate(data, verbose=verbose)
        m_names = ['loss']+[m._name for m in self.metrics_]
        return dict(zip(m_names,results))
    
    def get_config(self):
        return self._model.get_config()

    @property
    def layers(self):
        return self._model.layers
    
    @property
    def input(self):
        return self._model.input
    
    @property
    def custom_objects(self):
        return None