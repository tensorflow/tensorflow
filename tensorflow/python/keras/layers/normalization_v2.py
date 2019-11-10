# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""The V2 implementation of Normalization layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.layers import normalization
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.BatchNormalization', v1=[])  # pylint: disable=missing-docstring
class BatchNormalization(normalization.BatchNormalizationBase):

  __doc__ = normalization.replace_in_base_docstring([
      ('{{TRAINABLE_ATTRIBUTE_NOTE}}',
       '''
  **About setting `layer.trainable = False` on a `BatchNormalization layer:**

  The meaning of setting `layer.trainable = False` is to freeze the layer,
  i.e. its internal state will not change during training:
  its trainable weights will not be updated
  during `fit()` or `train_on_batch()`, and its state updates will not be run.

  Usually, this does not necessarily mean that the layer is run in inference
  mode (which is normally controlled by the `training` argument that can
  be passed when calling a layer). "Frozen state" and "inference mode"
  are two separate concepts.

  However, in the case of the `BatchNormalization` layer, **setting
  `trainable = False` on the layer means that the layer will be
  subsequently run in inference mode** (meaning that it will use
  the moving mean and the moving variance to normalize the current batch,
  rather than using the mean and variance of the current batch).

  This behavior has been introduced in TensorFlow 2.0, in order
  to enable `layer.trainable = False` to produce the most commonly
  expected behavior in the convnet fine-tuning use case.

  Note that:
    - This behavior only occurs as of TensorFlow 2.0. In 1.*,
      setting `layer.trainable = False` would freeze the layer but would
      not switch it to inference mode.
    - Setting `trainable` on an model containing other layers will
      recursively set the `trainable` value of all inner layers.
    - If the value of the `trainable`
      attribute is changed after calling `compile()` on a model,
      the new value doesn't take effect for this model
      until `compile()` is called again.
      ''')])

  _USE_V2_BEHAVIOR = True
