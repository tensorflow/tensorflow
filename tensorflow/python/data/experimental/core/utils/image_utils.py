# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

"""Utilities to manipulate images.

Note: these functions are not meant to be used inside of a TF graph.
"""

import numpy as np

import tensorflow.compat.v2 as tf
from tensorflow.data.experimental.core.utils import py_utils
from tensorflow.data.experimental.core.utils import tf_utils


@py_utils.memoize()
def _get_runner():
  return tf_utils.TFGraphRunner()


def decode_image(image_bytes: bytes) -> np.ndarray:
  """Returns np.array corresponding to encoded image."""
  runner = _get_runner()
  return runner.run(tf.image.decode_image, image_bytes)


def png_to_jpeg(image_bytes: bytes, quality: int = 100) -> np.ndarray:
  """Converts PNG image (bytes or str) to JPEG (bytes)."""
  runner = _get_runner()
  decode_fn = lambda img: tf.image.decode_png(img, channels=3)
  image = runner.run(decode_fn, image_bytes)
  fn = lambda img: tf.image.encode_jpeg(img, format='rgb', quality=quality)
  return runner.run(fn, image)


def jpeg_cmyk_to_rgb(image_bytes: bytes, quality: int = 100) -> np.ndarray:
  """Converts JPEG CMYK image (bytes) to RGB JPEG (bytes)."""
  runner = _get_runner()
  image = runner.run(tf.image.decode_jpeg, image_bytes)
  fn = lambda img: tf.image.encode_jpeg(img, format='rgb', quality=quality)
  return runner.run(fn, image)
