import tensorflow as tf
import numpy as np

def crossentropy(y_true, y_pred_onehot):
    '''Custom cross-entropy to handle borders (class = -1).'''
    n_valid = tf.math.reduce_sum(tf.cast(y_true != 255, tf.float32))
    y_true_onehot = tf.cast(np.arange(21) == y_true, tf.float32)
    return tf.reduce_sum(-y_true_onehot * tf.math.log(y_pred_onehot + 1e-7)) / n_valid

def pixelacc(y_true, y_pred_onehot):
    '''Custom pixel accuracy to handle borders (class = -1).'''
    n_valid = tf.math.reduce_sum(tf.cast(y_true != 255, tf.float32))
    y_true = tf.cast(y_true, tf.int32)[..., 0]
    y_pred = tf.argmax(y_pred_onehot, axis=-1, output_type=tf.int32)
    return tf.reduce_sum(tf.cast(y_true == y_pred, tf.float32)) / n_valid

class MyMeanIoU(tf.keras.metrics.MeanIoU):
    '''Custom meanIoU to handle borders (class = -1).'''
    def update_state(self, y_true, y_pred_onehot, sample_weight=None):
        y_pred = tf.argmax(y_pred_onehot, axis=-1)
        ## add 1 so boundary class=0
        y_true = tf.cast(y_true+1, self._dtype)
        y_pred = tf.cast(y_pred+1, self._dtype)
        ## Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])
        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])
        ## calculate confusion matrix with one extra class
        current_cm = tf.math.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes+1,
            weights=sample_weight,
            dtype=self._dtype)
        return self.total_cm.assign_add(current_cm[1:, 1:]) # remove boundary