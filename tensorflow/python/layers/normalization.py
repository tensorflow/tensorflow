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
# =============================================================================

"""Contains the normalization layer classes and their functional aliases.
"""

from tensorflow.python.util import lazy_loader

normalization = lazy_loader.LazyLoader(
    'normalization', globals(),
    'tf_keras.legacy_tf_layers.normalization')


# pylint: disable=invalid-name
# lazy load all the attributes until they are accessed for the first time
def __getattr__(name):
  if name in ['BatchNormalization', 'BatchNorm']:
    return normalization.BatchNormalization
  elif name in ['batch_normalization', 'batch_norm']:
    return normalization.batch_normalization
  else:
    raise AttributeError(f'module {__name__} doesn\'t have attribute {name}')
