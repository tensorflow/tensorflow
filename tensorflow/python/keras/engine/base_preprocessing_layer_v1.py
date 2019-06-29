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
"""Contains the base ProcessingLayer and a subclass that uses Combiners."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.ops import state_ops


class CombinerPreprocessingLayer(
    base_preprocessing_layer.CombinerPreprocessingLayer):
  """V1-compatible CombinerPreprocessingLayer.

  This class overrides several methods of the CombinerPreprocessingLayer to
  make it compatible with V1 execution. End users should not need to worry about
  the implementation details here; Keras will export the appropriate class under
  the 'CombinerPreprocessingLayer' symbol. (Users should not directly
  instantiate engine.base_preprocessing_layer/_v1.CombinerPreprocessingLayer).

  When creating a subclass of PreprocessingLayer, you can create a V1-compatible
  subclass as follows:

  class MyProcLayer(MyProcLayer,
                    base_preprocessing_layer_v1.CombinerPreprocessingLayer):
    pass

  Note that the same classname is required for serialization purposes.

  This is only necessary for internal classes, since any class that inherits
  from tf.keras.[...].CombinerPreprocessingLayer will get the right symbol.
  """

  def _restore_updates(self):
    """Recreates a dict of updates from the layer's weights."""
    data_dict = {}
    for name, var in self.state_variables.items():
      data_dict[name] = K.get_session().run(var)
    return data_dict

  def _dataset_is_infinite(self, dataset):
    """True if the passed dataset is infinite."""
    dataset_size = K.get_session().run(cardinality.cardinality(dataset))
    return dataset_size == cardinality.INFINITE

  def _get_dataset_iterator(self, dataset):
    """Gets an iterator from a tf.data.Dataset."""
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    session = K.get_session()
    next_element = iterator.get_next()
    return lambda: session.run(next_element)

  def _set_state_variables(self, updates):
    """Directly update the internal state of this Layer. V1 compatible."""
    # TODO(momernick): Do we need to do any more input sanitization?
    if not self.built:
      raise RuntimeError('_set_state_variables() must be called after build().')

    assignments = []
    for var_name, value in updates.items():
      assignments.append(
          state_ops.assign(self.state_variables[var_name], value))
    K.get_session().run(assignments)
