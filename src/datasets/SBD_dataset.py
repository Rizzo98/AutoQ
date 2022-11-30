import tensorflow as tf
import os
import math
import numpy as np
from PIL import Image
from scipy.io import loadmat

class SBD(tf.keras.utils.Sequence):
    def __init__(self, path_X, path_Y, batch_size, numImages = 'all', image_size = 512, gray_scale = False) -> None:
      self.batch_size = batch_size
      self.image_size = image_size
      self.gray_scale = gray_scale
      if numImages=='all':
          file_names = [i.split('.')[0] for i in os.listdir(path_X)]
          self.pathListX = [os.path.join(path_X, f'{i}.jpg') for i in file_names]
          self.pathListY = [os.path.join(path_Y, f'{i}.mat') for i in file_names]
      else:
          file_names = [i.split('.')[0] for i in os.listdir(path_X)][:numImages]
          self.pathListX = [os.path.join(path_X, f'{i}.jpg') for i in file_names]
          self.pathListY = [os.path.join(path_Y, f'{i}.mat') for i in file_names]

    def __len__(self):
        return math.ceil(len(self.pathListX)/self.batch_size)

    def __getitem__(self, index):
      self.X = []
      self.Y = []
      start = index*self.batch_size
      end = (index+1)*self.batch_size
      for (img_,label_) in zip(self.pathListX[start:end],self.pathListY[start:end]):
        img,label = self.__resize_with_pad(self.__get_image(img_),self.__get_label_mat(label_),size=self.image_size)
        if self.gray_scale:
            img = tf.image.rgb_to_grayscale(img)
        self.X.append(img)
        self.Y.append(label)
      self.X = tf.stack(self.X)
      self.Y = tf.stack(self.Y)

      return self.X, self.Y

    def __resize_with_pad(self, image, label, size=512):
        '''Resize a square while keeping the original aspect ratio, padding with black for the image and boundary 
        for the label.
        
        Args:
        image (array<np.uint8>): RGB values for each pixel. Shape=(height, width, 3)
        label (array<np.uint8>): Class labels for each pixel. Shape=(height, width, 1)
        size (int): length of square
            
        Returns:
        (array<np.uint8>): Resized image. Shape=(size, size, 3)
        (array<np.uint8>): Resized label. Shape=(size, size, 1)
        '''
        image = tf.image.resize_with_pad(image, size, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        ## since `resize_with_pad` pads with zeros, use fact that boundary class is -1 to pad with -1 instead.
        label = tf.image.resize_with_pad(label+1, size, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)-1
        return image, label

    def __get_image(self,path,convert2RGB=True):
        '''Retrieve image as array of RGB values from .jpg file.
        
        Args:
        path (string): Path to .jpg file
            
        Returns:
        (array<np.uint8>): RGB values for each pixel. Shape=(height, width, 3)
        '''
        if convert2RGB:
            jpg = Image.open(path).convert('RGB')
        else:
            jpg = Image.open(path)
        return np.array(jpg)
    
    def __get_label_mat(self,path):
        '''Retrieve class labels for each pixel from Berkeley SBD .mat file.
        
        Args:
        path (string): Path to .mat file
        
        Returns:
        (array<np.uint8>): Class as an integer in [0, 20] for each pixel. Shape=(height, width, 1)
        '''
        mat = loadmat(path)
        arr = mat['GTcls']['Segmentation'].item(0,0) # this is how segmentation is stored
        return arr[..., None]