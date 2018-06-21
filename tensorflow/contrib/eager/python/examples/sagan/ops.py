# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Self-attention generative adversarial with eager execution.

Auxiliary operations.

Reference [Self-Attention Generative Adversarial
Networks](https://arxiv.org/pdf/1805.08318.pdf)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def flatten_hw(x, data_format="channels_first"):
  """Flatten the input tensor across height and width dimensions."""
  if data_format == "channels_last":
    x = tf.transpose(x, perm=[0, 3, 1, 2])  # Convert to `channels_first`

  old_shape = tf.shape(x)
  new_shape = [old_shape[0], old_shape[2] * old_shape[3], old_shape[1]]

  return tf.reshape(x, new_shape)


def broaden_hw(x, h, w, c, data_format="channels_first"):
  """Broaden dimension so that output has height and width."""
  if data_format == "channels_first":
    shape = [-1, c, h, w]
  else:
    shape = [-1, h, w, c]

  return tf.reshape(x, shape)


class BroadenHW(tf.keras.layers.Layer):
  """Wrapper class so that `broaden_hw` can be used in `tf.keras.Sequential`."""

  def __init__(self, h, w, c, data_format="channels_first"):
    super(BroadenHW, self).__init__()
    self.h = h
    self.w = w
    self.c = c
    self.data_format = data_format

  def call(self, x):
    return broaden_hw(
        x, h=self.h, w=self.w, c=self.c, data_format=self.data_format)

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape).as_list()
    if self.data_format == "channels_first":
      output_shape = (input_shape[0], self.c, self.h, self.w)
    else:
      output_shape = (input_shape[0], self.h, self.w, self.c)

    return tf.TensorShape(output_shape)
