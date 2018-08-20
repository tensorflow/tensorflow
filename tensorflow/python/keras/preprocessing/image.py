# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
# pylint: disable=g-import-not-at-top
"""Set of tools for real-time data augmentation on image data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_preprocessing import image
try:
  from scipy import linalg  # pylint: disable=unused-import
  from scipy import ndimage  # pylint: disable=unused-import
except ImportError:
  pass

from tensorflow.python.util.tf_export import tf_export

random_rotation = image.random_rotation
random_shift = image.random_shift
random_shear = image.random_shear
random_zoom = image.random_zoom
apply_channel_shift = image.apply_channel_shift
random_channel_shift = image.random_channel_shift
apply_brightness_shift = image.apply_brightness_shift
random_brightness = image.random_brightness
apply_affine_transform = image.apply_affine_transform
array_to_img = image.array_to_img
img_to_array = image.img_to_array
save_img = image.save_img
load_img = image.load_img
ImageDataGenerator = image.ImageDataGenerator
Iterator = image.Iterator
NumpyArrayIterator = image.NumpyArrayIterator
DirectoryIterator = image.DirectoryIterator

tf_export('keras.preprocessing.image.random_rotation')(random_rotation)
tf_export('keras.preprocessing.image.random_shift')(random_shift)
tf_export('keras.preprocessing.image.random_shear')(random_shear)
tf_export('keras.preprocessing.image.random_zoom')(random_zoom)
tf_export('keras.preprocessing.image.apply_channel_shift')(apply_channel_shift)
tf_export(
    'keras.preprocessing.image.random_channel_shift')(random_channel_shift)
tf_export(
    'keras.preprocessing.image.apply_brightness_shift')(apply_brightness_shift)
tf_export('keras.preprocessing.image.random_brightness')(random_brightness)
tf_export(
    'keras.preprocessing.image.apply_affine_transform')(apply_affine_transform)
tf_export('keras.preprocessing.image.array_to_img')(array_to_img)
tf_export('keras.preprocessing.image.img_to_array')(img_to_array)
tf_export('keras.preprocessing.image.save_img')(save_img)
tf_export('keras.preprocessing.image.load_img')(load_img)
tf_export('keras.preprocessing.image.ImageDataGenerator')(ImageDataGenerator)
tf_export('keras.preprocessing.image.Iterator')(Iterator)
tf_export('keras.preprocessing.image.NumpyArrayIterator')(NumpyArrayIterator)
tf_export('keras.preprocessing.image.DirectoryIterator')(DirectoryIterator)
