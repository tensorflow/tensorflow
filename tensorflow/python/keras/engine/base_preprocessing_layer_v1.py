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

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import type_spec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.keras.engine import training_generator_v1
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import state_ops
from tensorflow.python.ops.ragged import ragged_tensor


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

  def __init__(self, combiner, **kwargs):
    super(CombinerPreprocessingLayer, self).__init__(combiner, **kwargs)
    self._previously_updated = False

  def _restore_updates(self):
    """Recreates a dict of updates from the layer's weights."""
    data_dict = {}
    for name, var in self.state_variables.items():
      data_dict[name] = K.get_session().run(var)
    return data_dict

  def _get_dataset_iterator(self, dataset):
    """Gets an iterator from a tf.data.Dataset."""
    iterator = dataset_ops.make_initializable_iterator(dataset)
    session = K.get_session()
    session.run(iterator.initializer)
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

  def adapt(self, data, reset_state=True):
    """Fits the state of the preprocessing layer to the data being passed.

    Args:
      data: The data to train on. It can be passed either as a tf.data Dataset,
        or as a numpy array.
      reset_state: Optional argument specifying whether to clear the state of
        the layer at the start of the call to `adapt`, or whether to start from
        the existing state. Subclasses may choose to throw if reset_state is set
        to 'False'.
    """
    if reset_state:
      accumulator = None
    else:
      accumulator = self._combiner.restore(self._restore_updates())
    if isinstance(data, (list, tuple)):
      data = ops.convert_to_tensor_v2_with_dispatch(data)
    if not isinstance(data, (dataset_ops.DatasetV2, np.ndarray, ops.Tensor,
                             ragged_tensor.RaggedTensor)):
      raise ValueError('`adapt()` requires a batched Dataset, a Tensor, '
                       'or a Numpy array as input, '
                       'got {}'.format(type(data)))

    if isinstance(data, dataset_ops.DatasetV2):
      # Validate that the dataset only contains single-tensor elements.
      if not isinstance(data.element_spec, type_spec.TypeSpec):
        raise TypeError(
            'The dataset should yield single-Tensor elements. Use `dataset.map`'
            'to select the element of interest.\n'
            'Got dataset.element_spec=' + str(data.element_spec))
      # Validate the datasets to try and ensure we haven't been passed one with
      # infinite size. That would cause an infinite loop here.
      if tf_utils.dataset_is_infinite(data):
        raise ValueError(
            'The dataset passed to `adapt()` has an infinite number of '
            'elements. Please use `dataset.take(...)` to make the number '
            'of elements finite.')
      next_data = self._get_dataset_iterator(data)
      # TODO(fchollet): consider checking if the dataset is already batched
      # and otherwise batching it.
    elif isinstance(data, (ops.Tensor, ragged_tensor.RaggedTensor)):
      next_data = self._get_dataset_iterator(
          dataset_ops.Dataset.from_tensor_slices(data).batch(512))
    else:
      generator, _ = training_generator_v1.convert_to_generator_like(
          data, batch_size=512)
      # If the data is not a dataset, we can iterate over it using next(foo);
      # here, we wrap that into a callable.
      next_data = lambda: next(generator)

    # TODO(momernick): Some sort of status bar?
    # TODO(momernick): Implement parallel processing here?
    try:
      data_element = next_data()

      # First, see if the layer is built or not. If it is not, then we must
      # build it.
      if not self.built:
        try:
          # If this is a Numpy array or tensor, we can get shape from .shape.
          # If not, an attribute error will be thrown.
          data_shape = data_element.shape
          data_shape_nones = tuple([None] * len(data_element.shape))
        except AttributeError:
          # The input has an unknown number of dimensions.
          data_shape = None
          data_shape_nones = None

        # TODO (b/159261555): move this to base layer build.
        batch_input_shape = getattr(self, '_batch_input_shape', None)
        if batch_input_shape is None:
          # Set the number of dimensions.
          self._batch_input_shape = data_shape_nones

        self.build(data_shape)

      # Once we have built the Layer, we can process the input data. We do so
      # until we've gotten an exception indicating that we have no more data.
      while True:
        accumulator = self._combiner.compute(data_element, accumulator)
        data_element = next_data()
    # Note that this belongs to the outer indentation of 'try' - we need to
    # catch exceptions resulting from the first 'next_data()' invocation as
    # well.
    except (StopIteration, errors.OutOfRangeError):
      pass

    updates = self._combiner.extract(accumulator)
    self._set_state_variables(updates)
