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

import abc
import collections

import numpy as np

from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import training_generator
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.experimental.preprocessing.PreprocessingLayer')
class PreprocessingLayer(Layer):
  """Base class for PreprocessingLayers."""
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def adapt(self, data, reset_state=True):
    # TODO(momernick): Add examples.
    """Fits the state of the preprocessing layer to the data being passed.

    Arguments:
        data: The data to train on. It can be passed either as a tf.data
          Dataset, or as a numpy array.
        reset_state: Optional argument specifying whether to clear the state of
          the layer at the start of the call to `adapt`, or whether to start
          from the existing state. This argument may not be relevant to all
          preprocessing layers: a subclass of PreprocessingLayer may choose to
            throw if 'reset_state' is set to False.
    """
    pass


class CombinerPreprocessingLayer(PreprocessingLayer):
  """Base class for PreprocessingLayers that do computation using a Combiner.

  This class provides several helper methods to make creating a
  PreprocessingLayer easier. It assumes that the core of your computation will
  be done via a Combiner object. Subclassing this class to create a
  PreprocessingLayer allows your layer to be compatible with distributed
  computation.

  This class is compatible with Tensorflow 2.0+.
  """

  def __init__(self, combiner, **kwargs):
    super(CombinerPreprocessingLayer, self).__init__(**kwargs)
    self._combiner = combiner
    self._previously_updated = False
    self.state_variables = collections.OrderedDict()

  def _add_state_variable(self,
                          name,
                          shape,
                          dtype,
                          initializer=None,
                          partitioner=None,
                          use_resource=None,
                          **kwargs):
    """Add a variable that can hold state which is updated during adapt().

    Args:
      name: Variable name.
      shape: Variable shape. Defaults to scalar if unspecified.
      dtype: The type of the variable. Defaults to `self.dtype` or `float32`.
      initializer: initializer instance (callable).
      partitioner: Partitioner to be passed to the `Trackable` API.
      use_resource: Whether to use `ResourceVariable`
      **kwargs: Additional keyword arguments. Accepted values are `getter` and
        `collections`.

    Returns:
      The created variable.
    """
    weight = self.add_weight(
        name=name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        regularizer=None,
        trainable=False,
        constraint=None,
        partitioner=partitioner,
        use_resource=use_resource,
        **kwargs)
    # TODO(momernick): Do not allow collisions here.
    self.state_variables[name] = weight
    return weight

  def _restore_updates(self):
    """Recreates a dict of updates from the layer's weights."""
    data_dict = {}
    for name, var in self.state_variables.items():
      data_dict[name] = var.numpy()
    return data_dict

  def _dataset_is_infinite(self, dataset):
    """True if the passed dataset is infinite."""
    return math_ops.equal(
        cardinality.cardinality(dataset), cardinality.INFINITE)

  def _get_dataset_iterator(self, dataset):
    """Gets an iterator from a tf.data.Dataset."""
    return dataset_ops.make_one_shot_iterator(dataset).get_next

  def adapt(self, data, reset_state=True):
    """Fits the state of the preprocessing layer to the data being passed.

    Arguments:
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

    if not isinstance(data, (dataset_ops.DatasetV2, np.ndarray)):
      raise ValueError(
          'adapt() requires a Dataset or a Numpy array as input, got {}'.format(
              type(data)))

    if isinstance(data, dataset_ops.DatasetV2):
      # Validate the datasets to try and ensure we haven't been passed one with
      # infinite size. That would cause an infinite loop here.
      if self._dataset_is_infinite(data):
        raise ValueError(
            'The dataset passed to "adapt()" has an infinite number of '
            'elements. Please use dataset.take(...) to make the number '
            'of elements finite.')
      next_data = self._get_dataset_iterator(data)
    else:
      generator, _ = training_generator.convert_to_generator_like(
          data, batch_size=len(data))
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
          # If not, an attribute error will be thrown (and we can assume the
          # input data is a scalar with shape None.
          shape = data_element.shape
        except AttributeError:
          shape = None
        self.build(shape)

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

  def _set_state_variables(self, updates):
    """Directly update the internal state of this Layer.

    This method expects a string-keyed dict of {state_variable_name: state}. The
    precise nature of the state, and the names associated, are describe by
    the subclasses of CombinerPreprocessingLayer.

    Args:
      updates: A string keyed dict of weights to update.

    Raises:
      RuntimeError: if 'build()' was not called before 'set_processing_state'.
    """
    # TODO(momernick): Do we need to do any more input sanitization?
    if not self.built:
      raise RuntimeError('_set_state_variables() must be called after build().')

    with ops.init_scope():
      for var_name, value in updates.items():
        self.state_variables[var_name].assign(value)


def convert_to_list(values, sparse_default_value=-1):
  """Convert a TensorLike, CompositeTensor, or ndarray into a Python list."""
  if ragged_tensor.is_ragged(values):
    # There is a corner case when dealing with ragged tensors: if you get an
    # actual RaggedTensor (not a RaggedTensorValue) passed in non-eager mode,
    # you can't call to_list() on it without evaluating it first. However,
    # because we don't yet fully support composite tensors across Keras,
    # K.get_value() won't evaluate the tensor.
    # TODO(momernick): Get Keras to recognize composite tensors as Tensors
    # and then replace this with a call to K.get_value.
    if (isinstance(values, ragged_tensor.RaggedTensor) and
        not context.executing_eagerly()):
      values = K.get_session(values).run(values)
    values = values.to_list()

  # TODO(momernick): Add a sparse_tensor.is_sparse() method to replace this
  # check.
  if isinstance(values,
                (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)):
    dense_tensor = sparse_ops.sparse_tensor_to_dense(
        values, default_value=sparse_default_value)
    values = K.get_value(dense_tensor)

  if isinstance(values, (ops.EagerTensor, ops.Tensor)):
    values = K.get_value(values)

  # We may get passed a ndarray or the code above may give us a ndarray.
  # In either case, we want to force it into a standard python list.
  if isinstance(values, np.ndarray):
    values = values.tolist()

  return values


class Combiner(object):
  """Functional object that defines a shardable computation.

  This object defines functions required to create and manipulate data objects.
  These data objects, referred to below as 'accumulators', are computation-
  specific and may be implemented alongside concrete subclasses of Combiner
  (if necessary - some computations may be simple enough that standard Python
  types can be used as accumulators).

  The intent for this class is that by describing computations in this way, we
  can arbitrarily shard a dataset, perform computations on a subset, and then
  merge the computation into a final result. This enables distributed
  computation.

  The combiner itself does not own any state - all computational state is owned
  by the accumulator objects. This is so that we can have an arbitrary number of
  Combiners (thus sharding the computation N ways) without risking any change
  to the underlying computation. These accumulator objects are uniquely
  associated with each Combiner; a Combiner defines what the accumulator object
  should be and will only work with accumulators of that type.
  """
  __metaclass__ = abc.ABCMeta

  def __repr__(self):
    return '<{}>'.format(self.__class__.__name__)

  @abc.abstractmethod
  def compute(self, batch_values, accumulator=None):
    """Compute a step in this computation, returning a new accumulator.

    This method computes a step of the computation described by this Combiner.
    If an accumulator is passed, the data in that accumulator is also used; so
    compute(batch_values) results in f(batch_values), while
    compute(batch_values, accumulator) results in
    merge(f(batch_values), accumulator).

    Args:
      batch_values: A list of ndarrays representing the values of the inputs for
        this step of the computation.
      accumulator: the current accumulator. Can be None.

    Returns:
      An accumulator that includes the passed batch of inputs.
    """
    pass

  @abc.abstractmethod
  def merge(self, accumulators):
    """Merge several accumulators to a single accumulator.

    This method takes the partial values in several accumulators and combines
    them into a single accumulator. This computation must not be order-specific
    (that is, merge([a, b]) must return the same result as merge([b, a]).

    Args:
      accumulators: the accumulators to merge, as a list.

    Returns:
      A merged accumulator.
    """
    pass

  @abc.abstractmethod
  def extract(self, accumulator):
    """Convert an accumulator into a dict of output values.

    Args:
      accumulator: The accumulator to convert.

    Returns:
      A dict of ndarrays representing the data in this accumulator.
    """
    pass

  @abc.abstractmethod
  def restore(self, output):
    """Create an accumulator based on 'output'.

    This method creates a new accumulator with identical internal state to the
    one used to create the data in 'output'. This means that if you do

    output_data = combiner.extract(accumulator_1)
    accumulator_2 = combiner.restore(output_data)

    then accumulator_1 and accumulator_2 will have identical internal state, and
    computations using either of them will be equivalent.

    Args:
      output: The data output from a previous computation. Should be in the same
        form as provided by 'extract_output'.

    Returns:
      A new accumulator.
    """
    pass

  @abc.abstractmethod
  def serialize(self, accumulator):
    """Serialize an accumulator for a remote call.

    This function serializes an accumulator to be sent to a remote process.

    Args:
      accumulator: The accumulator to serialize.

    Returns:
      A byte string representing the passed accumulator.
    """
    pass

  @abc.abstractmethod
  def deserialize(self, encoded_accumulator):
    """Deserialize an accumulator received from 'serialize()'.

    This function deserializes an accumulator serialized by 'serialize()'.

    Args:
      encoded_accumulator: A byte string representing an accumulator.

    Returns:
      The accumulator represented by the passed byte_string.
    """
    pass
