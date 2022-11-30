import tensorflow as tf
import os
import numpy as np
from PIL import Image

class NpImages:
    def __init__(self, input_path, num_files:None, size=None) -> None:
        self.IMAGES_FORMAT = ['png', 'jpg']
        self.__size = size
        assert os.path.isfile(input_path) or os.path.isdir(input_path), 'inputs must be a file or a folder!'
        if os.path.isfile(input_path):
            self.__images = [input_path]
        if os.path.isdir(input_path):
            self.__images = [os.path.join(input_path,i) for i in os.listdir(input_path) if i.split('.')[-1] in self.IMAGES_FORMAT]
            assert len(self.__images)>0, 'No images in inputs folder!'
        if num_files is not None:
            assert isinstance(num_files,int), 'num_files must be an integer!'
            assert num_files>0 and num_files<len(self.__images), f'num_files must be between 0 and {len(self.__images)}!'
            self.__images = self.__images[:num_files]
        
        self.__npy_images = [self.__img2npy(i) for i in range(len(self.__images))]

    def __img2npy(self, idx):
        if self.__size is None:
            return np.expand_dims(np.array(Image.open(self.__images[idx]),dtype=np.float32),axis=0)
        else:
            return np.expand_dims(tf.image.resize_with_pad(
                np.array(Image.open(self.__images[idx]),dtype=np.float32), self.__size, self.__size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
                axis=0)

    def __len__(self) -> None:
        return len(self.__images)

    def __iter__(self):
        return iter(self.__npy_images)