import tensorflow as tf
from utils.initializers import BilinearInitializer
from utils.metrics import crossentropy, pixelacc, MyMeanIoU
from src.models.model import Model

class MobileFCN(Model):
    def __init__(self, model_path:str=None):
        super().__init__()
        self._metricName = 'meanIoU'
        if model_path!=None:
            self._model = tf.keras.models.load_model(model_path,
                custom_objects={'BilinearInitializer':BilinearInitializer, 'pixelacc':pixelacc, 'MyMeanIoU':MyMeanIoU,'crossentropy':crossentropy})
        
    def compile(self):
        super().compile(
            tf.keras.optimizers.Adam(learning_rate=1e-4), loss=crossentropy, 
            metrics=[MyMeanIoU(num_classes=21, name='meanIoU')]
            )

    @property
    def custom_objects(self):
        return {
            'BilinearInitializer':BilinearInitializer, 'pixelacc':pixelacc, 
            'MyMeanIoU':MyMeanIoU,'crossentropy':crossentropy
            }