# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""TensorFlow ops for Batch Normalization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging

from tensorflow.contrib.layers import batch_norm


def batch_normalize(tensor_in,
                    epsilon=1e-5,
                    convnet=False,
                    decay=0.9,
                    scale_after_normalization=True):
  """Batch normalization. Note this is deprecated. 
  Instead, please use contrib.layers.batch_norm. You can get is_training
  via `tf.python.framework.ops.get_collection("IS_TRAINING")`.

  Args:
    tensor_in: input `Tensor`, 4D shape: [batch, in_height, in_width, in_depth].
    epsilon : A float number to avoid being divided by 0.
    convnet: Whether this is for convolutional net use. If `True`, moments
        will sum across axis `[0, 1, 2]`. Otherwise, only `[0]`.
    decay: Decay rate for exponential moving average.
    scale_after_normalization: Whether to scale after normalization.

  Returns:
    A batch-normalized `Tensor`.
  """
  logging.warning("learn.ops.batch_normalize is deprecated, \
    please use contrib.layers.batch_norm.")
  is_training = ops.get_collection("IS_TRAINING")
  return batch_norm(tensor_in,
    is_training=is_training,
    epsilon=epsilon,
    decay=decay,
    scale=scale_after_normalization)
