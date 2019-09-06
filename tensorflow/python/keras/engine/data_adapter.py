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
"""Adapter module that convert different input data objects into tf.dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import itertools
import math

import numpy as np
import six

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework.ops import composite_tensor
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect


@six.add_metaclass(abc.ABCMeta)
class DataAdapter(object):
  """Base class for input data adapter.

  In TF 2.0, tf.data is the preferred API for user to feed in data. In order
  to simplify the training code path, all the input data object will be
  converted to `tf.data.Dataset` if possible.

  Note that since this class is mainly targeted for TF 2.0, it might have a lot
  of assumptions under the hood, eg eager context by default, distribution
  strategy, etc. In the meantime, some legacy feature support might be dropped,
  eg, Iterator from dataset API in v1, etc.

  The sample usage of this class is like:

  ```
  x = tf.data.Dataset.range(100)
  adapter_cls = [NumpyArrayDataAdapter, ..., DatasetAdapter]
  applicable_adapters = [cls for cls in adapter_cls if cls.can_handle(x)]
  if len(applicable_adapters) != 1:
    raise ValueError("Expect only one adapter class to handle the input")

  dataset = applicable_adapters[0](x).get_dataset()
  for data in dataset:
    # training
  ```
  """

  @staticmethod
  def can_handle(x, y=None):
    """Whether the current DataAdapter could handle the input x and y.

    Structure wise, x and y can be single object, or list of objects if there
    multiple input/output, or dictionary of objects when the intput/output are
    named.

    Args:
      x: input features.
      y: target labels. Note that y could be None in the case of prediction.

    Returns:
      boolean
    """
    raise NotImplementedError

  @abc.abstractmethod
  def __init__(self, x, y=None, **kwargs):
    """Create a DataAdapter based on data inputs.

    The caller must make sure to call `can_handle()` first before invoking this
    method. Provide unsupported data type will result into unexpected behavior.

    Args:
      x: input features.
      y: target labels. Note that y could be None in the case of prediction.
      **kwargs: Other keyword arguments for DataAdapter during the construction
        of the tf.dataset.Dataset. For example:
        - Numpy data might have `sample_weights` which will be used for
          weighting the loss function during training.
        - Numpy data might need to have `batch_size` parameter when constructing
          the dataset and iterator.
        - Certain input might need to be distribution strategy aware. When
          `distribution_strategy` is passed, the created dataset need to respect
          the strategy.
        DataAdapter might choose to ignore any keyword argument if it doesn't
        use it, or raise exception if any required argument is not provide.
    """
    if not self.can_handle(x, y):
      raise ValueError("{} Cannot handle input {}".format(self.__class__, x))

  @abc.abstractmethod
  def get_dataset(self):
    """Get a dataset instance for the current DataAdapter.

    Note that the dataset returned does not repeat for epoch, so caller might
    need to create new iterator for the same dataset at the beginning of the
    epoch. This behavior might change in future.

    Returns:
      An tf.dataset.Dataset. Caller might use the dataset in different
      context, eg iter(dataset) in eager to get the value directly, or in graph
      mode, provide the iterator tensor to Keras model function.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def get_size(self):
    """Return the size (number of batches) for the dataset created.

    For certain type of the data input, the number of batches is known, eg for
    Numpy data, the size is same as (number_of_element / batch_size). Whereas
    for dataset or python generator, the size is unknown since it may or may not
    have a end state.

    Returns:
      int, the number of batches for the dataset, or None if it is unknown. The
      caller could use this to control the loop of training, show progress bar,
      or handle unexpected StopIteration error.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def batch_size(self):
    """Return the batch size of the dataset created.

    For certain type of the data input, the batch size is known, and even
    required, like numpy array. Where as for dataset, the batch is unknown
    unless we take a peek.

    Returns:
      int, the batch size of the dataset, or None if it is unknown.
    """
    raise NotImplementedError

  def representative_batch_size(self):
    """Return a representative size for batches in the dataset.

    This is not guaranteed to be the batch size for all batches in the
    dataset. It just needs to be a rough approximation for batch sizes in
    the dataset.

    Returns:
      int, a representative size for batches found in the dataset,
      or None if it is unknown.
    """
    return self.batch_size()

  @abc.abstractmethod
  def has_partial_batch(self):
    """Whether the dataset has partial batch at the end."""
    raise NotImplementedError

  @abc.abstractmethod
  def partial_batch_size(self):
    """The size of the final partial batch for dataset.

    Will return None if has_partial_batch is False or batch_size is None.
    """
    raise NotImplementedError

  def should_recreate_iterator(self, steps_per_epoch):
    """Returns whether a new iterator should be created every epoch."""
    # Only recreate iterator when the data has a fixed length, which will be
    # fully consumed every epoch, or has a unknown length (dataset, generator)
    # and will be fully consumed (steps_per_epoch is None)
    return self.get_size() is not None or steps_per_epoch is None


class TensorLikeDataAdapter(DataAdapter):
  """Adapter that handles Tensor-like objects, e.g. EagerTensor and NumPy."""

  @staticmethod
  def can_handle(x, y=None):
    # TODO(kaftan): Check performance implications of using a flatten
    #  here for other types of inputs.
    flat_inputs = nest.flatten(x)
    if y is not None:
      flat_inputs += nest.flatten(y)

    def _is_tensor(v):
      if isinstance(v, (ops.Tensor, np.ndarray)):
        return True
      return False

    return all(_is_tensor(v) for v in flat_inputs)

  def __init__(self,
               x,
               y=None,
               sample_weights=None,
               batch_size=None,
               epochs=1,
               steps=None,
               shuffle=False,
               **kwargs):
    super(TensorLikeDataAdapter, self).__init__(x, y, **kwargs)
    x = _process_numpy_inputs(x)
    y = _process_numpy_inputs(y)
    sample_weights = _process_numpy_inputs(sample_weights)

    # If sample_weights are not specified for an output use 1.0 as weights.
    if sample_weights is not None and any(w is None for w in sample_weights):
      weight = next(s for s in sample_weights if s is not None)
      sample_weights = training_utils.list_to_tuple([
          array_ops.ones((weight.shape[0],)) if sw is None else sw
          for sw in sample_weights
      ])

    if y is not None and sample_weights is not None:
      inputs = (x, y, sample_weights)
    elif y is not None:
      # Sample weight is only needed for training, so if y is None, then
      # sample_weight is ignored.
      inputs = (x, y)
    else:
      inputs = (x,)

    num_samples = set(int(i.shape[0]) for i in nest.flatten(inputs))
    if len(num_samples) > 1:
      msg = "Data cardinality is ambiguous:\n"
      for label, data in zip(["x", "y", "sample_weight"], inputs):
        msg += "  {} sizes: {}\n".format(
            label, ", ".join([str(i.shape[0]) for i in nest.flatten(data)]))
      msg += "Please provide data which shares the same first dimension."
      raise ValueError(msg)
    num_samples = num_samples.pop()

    # If batch_size is not passed but steps is, calculate from the input data.
    if steps and not batch_size:
      batch_size = int(math.ceil(num_samples / steps))

    if not batch_size:
      raise ValueError(
          "`batch_size` or `steps` is required for `Tensor` or `NumPy`"
          " input data.")

    self._size = int(math.ceil(num_samples / batch_size))
    self._batch_size = batch_size

    num_full_batches = int(num_samples // batch_size)
    self._partial_batch_size = num_samples % batch_size

    # Vectorized version of shuffle.
    # This is a performance improvement over using `from_tensor_slices`.
    # The indices of the data are shuffled and batched, and these indices
    # are then zipped with the data and used to extract a batch of the data
    # at each step. The performance improvements here come from:
    # 1. vectorized batch using gather
    # 2. parallelized map
    # 3. pipelined permutation generation
    # 4. optimized permutation batching
    # 5. disabled static optimizations

    indices_dataset = dataset_ops.DatasetV2.range(1).repeat()
    def permutation(_):
      # It turns out to be more performant to make a new set of indices rather
      # than reusing the same range Tensor. (presumably because of buffer
      # forwarding.)
      indices = math_ops.range(num_samples, dtype=dtypes.int64)
      if shuffle:
        indices = random_ops.random_shuffle(indices)
      return indices

    # We prefetch a single element. Computing large permutations can take quite
    # a while so we don't want to wait for prefetching over an epoch boundary to
    # trigger the next permutation. On the other hand, too many simultaneous
    # shuffles can contend on a hardware level and degrade all performance.
    indices_dataset = indices_dataset.map(permutation).prefetch(1)

    def slice_batch_indices(indices):
      """Convert a Tensor of indices into a dataset of batched indices.

      This step can be accomplished in several ways. The most natural is to
      slice the Tensor in a Dataset map. (With a condition on the upper index to
      handle the partial batch.) However it turns out that coercing the Tensor
      into a shape which is divisible by the batch size (and handling the last
      partial batch separately) allows for a much more favorable memory access
      pattern and improved performance.

      Args:
        indices: Tensor which determines the data order for an entire epoch.

      Returns:
        A Dataset of batched indices.
      """
      num_in_full_batch = num_full_batches * batch_size
      first_k_indices = array_ops.slice(indices, [0], [num_in_full_batch])
      first_k_indices = array_ops.reshape(
          first_k_indices, [num_full_batches, batch_size])

      flat_dataset = dataset_ops.DatasetV2.from_tensor_slices(first_k_indices)
      if self._partial_batch_size:
        index_remainder = dataset_ops.DatasetV2.from_tensors(array_ops.slice(
            indices, [num_in_full_batch], [self._partial_batch_size]))
        flat_dataset = flat_dataset.concatenate(index_remainder)
      return flat_dataset

    indices_dataset = indices_dataset.flat_map(slice_batch_indices)
    dataset = dataset_ops.DatasetV2.zip((
        indices_dataset,
        dataset_ops.DatasetV2.from_tensors(inputs).repeat()
    ))

    def grab_batch(i, data):
      return nest.map_structure(lambda d: array_ops.gather(d, i, axis=0), data)

    dataset = dataset.map(
        grab_batch, num_parallel_calls=dataset_ops.AUTOTUNE)

    # Default optimizations are disabled to avoid the overhead of (unnecessary)
    # input pipeline graph serialization and deserialization
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    self._dataset = dataset

  def get_dataset(self):
    return self._dataset

  def get_size(self):
    return self._size

  def batch_size(self):
    return self._batch_size

  def has_partial_batch(self):
    return self._partial_batch_size > 0

  def partial_batch_size(self):
    return self._partial_batch_size or None

  def should_recreate_iterator(self, _):
    # An infinite dataset is always created here.
    return False


class CompositeTensorDataAdapter(DataAdapter):
  """Adapter that handles Tensor-like objects, e.g. EagerTensor and NumPy."""

  @staticmethod
  def can_handle(x, y=None):
    flat_inputs = nest.flatten(x)
    if y is not None:
      flat_inputs += nest.flatten(y)

    def _is_composite(v):
      # Dataset inherits from CompositeTensor but shouldn't be handled here.
      if (isinstance(v, composite_tensor.CompositeTensor) and
          not isinstance(v, dataset_ops.DatasetV2)):
        return True
      return False

    def _is_tensor_or_composite(v):
      if isinstance(v, (ops.Tensor, np.ndarray)):
        return True
      return _is_composite(v)

    return (any(_is_composite(v) for v in flat_inputs) and
            all(_is_tensor_or_composite(v) for v in flat_inputs))

  def __init__(self, x, y=None, sample_weights=None, batch_size=None,
               steps=None, shuffle=False, **kwargs):
    super(CompositeTensorDataAdapter, self).__init__(x, y, **kwargs)
    x = _process_numpy_inputs(x)
    y = _process_numpy_inputs(y)
    sample_weights = _process_numpy_inputs(sample_weights)

    # If sample_weights are not specified for an output use 1.0 as weights.
    if (sample_weights is not None and
        any([sw is None for sw in sample_weights])):
      weight = next(s for s in sample_weights if s is not None)
      sample_weights = training_utils.list_to_tuple([
          array_ops.ones((weight.shape[0],)) if sw is None else sw
          for sw in sample_weights
      ])

    if y is not None and sample_weights is not None:
      inputs = (x, y, sample_weights)
    elif y is not None:
      # Sample weight is only needed for training, so if y is None, then
      # sample_weight is ignored.
      inputs = (x, y)
    else:
      inputs = (x,)

    dataset = dataset_ops.DatasetV2.from_tensor_slices(inputs)
    num_samples = int(nest.flatten(x)[0].shape[0])
    if shuffle:
      dataset = dataset.shuffle(num_samples)

    # If batch_size is not passed but steps is, calculate from the input data.
    if steps and not batch_size:
      batch_size = int(math.ceil(num_samples/steps))

    if not batch_size:
      raise ValueError(
          "`batch_size` or `steps` is required for `Tensor` or `NumPy`"
          " input data.")

    dataset = dataset.batch(batch_size)
    self._size = int(math.ceil(num_samples / batch_size))
    self._batch_size = batch_size
    self._has_partial_batch = (self._size != (num_samples // batch_size))

    self._partial_batch_size = None
    if self._has_partial_batch:
      self._partial_batch_size = (
          num_samples - (self._size - 1) * self._batch_size)

    self._dataset = dataset

  def get_dataset(self):
    return self._dataset

  def get_size(self):
    return self._size

  def batch_size(self):
    return self._batch_size

  def has_partial_batch(self):
    return self._has_partial_batch

  def partial_batch_size(self):
    return self._partial_batch_size


class ListsOfScalarsDataAdapter(DataAdapter):
  """Adapter that handles lists of scalars and lists of lists of scalars."""

  @staticmethod
  def can_handle(x, y=None):
    handles_x = ListsOfScalarsDataAdapter._is_list_of_scalars(x)
    handles_y = True
    if y is not None:
      handles_y = ListsOfScalarsDataAdapter._is_list_of_scalars(y)
    return handles_x and handles_y

  @staticmethod
  def _is_list_of_scalars(inp):
    if isinstance(inp, (float, int, str)):
      return True
    if isinstance(inp, (list, tuple)):
      return ListsOfScalarsDataAdapter._is_list_of_scalars(inp[0])
    return False

  def __init__(
      self, x, y=None, sample_weights=None, batch_size=None,
      shuffle=False, **kwargs):
    super(ListsOfScalarsDataAdapter, self).__init__(x, y, **kwargs)
    x = np.asarray(x)
    if y is not None:
      y = np.asarray(y)
    if sample_weights is not None:
      sample_weights = np.asarray(sample_weights)

    self._internal_adapter = TensorLikeDataAdapter(
        x, y=y, sample_weights=sample_weights,
        batch_size=batch_size, shuffle=shuffle, **kwargs)

  def get_dataset(self):
    return self._internal_adapter.get_dataset()

  def get_size(self):
    return self._internal_adapter.get_size()

  def batch_size(self):
    return self._internal_adapter.batch_size()

  def has_partial_batch(self):
    return self._internal_adapter.has_partial_batch()

  def partial_batch_size(self):
    return self._internal_adapter.partial_batch_size()


class DatasetAdapter(DataAdapter):
  """Adapter that handles `tf.data.Dataset`."""

  @staticmethod
  def can_handle(x, y=None):
    return isinstance(x, (dataset_ops.DatasetV1, dataset_ops.DatasetV2))

  def __init__(self, x, y=None, sample_weights=None, **kwargs):
    super(DatasetAdapter, self).__init__(x, y, **kwargs)
    if not is_none_or_empty(y):
      raise ValueError("`y` argument is not supported when using "
                       "dataset as input.")
    if not is_none_or_empty(sample_weights):
      raise ValueError("`sample_weight` argument is not supported when using "
                       "dataset as input.")
    # Note that the dataset instance is immutable, its fine to reusing the user
    # provided dataset.
    self._dataset = x

  def get_dataset(self):
    return self._dataset

  def get_size(self):
    # The size of dataset is unknown, unless its fully consumed.
    return None

  def batch_size(self):
    return None

  def has_partial_batch(self):
    return False

  def partial_batch_size(self):
    return None


class GeneratorDataAdapter(DataAdapter):
  """Adapter that handles python generator."""

  @staticmethod
  def can_handle(x, y=None):
    return tf_inspect.isgenerator(x)

  def __init__(self, x, y=None, sample_weights=None, **kwargs):
    super(GeneratorDataAdapter, self).__init__(x, y, **kwargs)
    if not is_none_or_empty(y):
      raise ValueError("`y` argument is not supported when using "
                       "python generator as input.")
    if not is_none_or_empty(sample_weights):
      raise ValueError("`sample_weight` argument is not supported when using "
                       "python generator as input.")

    # Since we have to know the dtype of the python generator when we build the
    # dataset, we have to take a peek for the python generator first. Since the
    # peeked data cannot be push back to generator, we create a new generator by
    # adding the peeked data at head.
    def dynamic_shape_like(t):
      return tuple(None for _ in t.shape)

    peek = next(x)
    nested_dtypes = nest.map_structure(lambda t: t.dtype, peek)
    nested_shape = nest.map_structure(dynamic_shape_like, peek)
    # Note that dataset API takes a callable that creates a generator object,
    # rather than generator itself, which is why we define a function here.
    def reassemble():
      return itertools.chain([peek], x)

    self._first_batch_size = int(nest.flatten(peek)[0].shape[0])
    self._dataset = dataset_ops.DatasetV2.from_generator(
        reassemble, nested_dtypes, output_shapes=nested_shape)

  def get_dataset(self):
    return self._dataset

  def get_size(self):
    return None

  def batch_size(self):
    return None

  def representative_batch_size(self):
    return self._first_batch_size

  def has_partial_batch(self):
    return False

  def partial_batch_size(self):
    return None


class KerasSequenceAdapter(DataAdapter):
  """Adapter that handles `keras.utils.Sequence`."""

  @staticmethod
  def can_handle(x, y=None):
    return isinstance(x, data_utils.Sequence)

  def __init__(self, x, y=None, sample_weights=None, shuffle=False, **kwargs):
    super(KerasSequenceAdapter, self).__init__(x, y, **kwargs)
    if not is_none_or_empty(y):
      raise ValueError("`y` argument is not supported when using "
                       "`keras.utils.Sequence` as input.")
    if not is_none_or_empty(sample_weights):
      raise ValueError("`sample_weight` argument is not supported when using "
                       "`keras.utils.Sequence` as input.")
    def dynamic_shape_like(t):
      return tuple(None for _ in t.shape)

    peek = x[0]
    nested_dtypes = nest.map_structure(lambda t: t.dtype, peek)
    nested_shape = nest.map_structure(dynamic_shape_like, peek)

    def generator():
      for i in range(len(x)):
        yield x[i]
    dataset = dataset_ops.DatasetV2.from_generator(generator, nested_dtypes,
                                                   output_shapes=nested_shape)
    if shuffle:
      dataset = dataset.shuffle(len(x))
    self._dataset = dataset
    self._size = len(x)
    self._first_batch_size = int(nest.flatten(peek)[0].shape[0])

  def get_dataset(self):
    return self._dataset

  def get_size(self):
    return self._size

  def batch_size(self):
    return None

  def representative_batch_size(self):
    return self._first_batch_size

  def has_partial_batch(self):
    return False

  def partial_batch_size(self):
    return None


ALL_ADAPTER_CLS = [
    ListsOfScalarsDataAdapter, TensorLikeDataAdapter, DatasetAdapter,
    GeneratorDataAdapter, KerasSequenceAdapter, CompositeTensorDataAdapter
]


def select_data_adapter(x, y):
  """Selects a data adapter than can handle a given x and y."""
  adapter_cls = [cls for cls in ALL_ADAPTER_CLS if cls.can_handle(x, y)]
  if not adapter_cls:
    # TODO(scottzhu): This should be a less implementation-specific error.
    raise ValueError(
        "Failed to find data adapter that can handle "
        "input: {}, {}".format(
            _type_name(x), _type_name(y)))
  elif len(adapter_cls) > 1:
    raise RuntimeError(
        "Data adapters should be mutually exclusive for "
        "handling inputs. Found multiple adapters {} to handle "
        "input: {}, {}".format(
            adapter_cls, _type_name(x), _type_name(y)))
  return adapter_cls[0]


def _type_name(x):
  """Generates a description of the type of an object."""
  if isinstance(x, dict):
    key_types = set(_type_name(key) for key in x.keys())
    val_types = set(_type_name(key) for key in x.values())
    return "({} containing {} keys and {} values)".format(
        type(x), key_types, val_types)
  if isinstance(x, (list, tuple)):
    types = set(_type_name(val) for val in x)
    return "({} containing values of types {})".format(
        type(x), types)
  return str(type(x))


def _process_numpy_inputs(inputs):
  """Process numpy array inputs.

  For numpy inputs, it is possible to be single numpy array, or list/dict of
  them. They could also be preprocessed by other lib to match with the order
  of position for the model. The result here should be something that can be
  used to build dataset.

  Args:
    inputs: single or list/tuple/dict of numpy array.
  Returns:
    numpy arrays can be used to build dataset.
  """
  if is_none_or_empty(inputs):
    return None
  flat_inputs = nest.flatten(inputs)
  if len(flat_inputs) == 1:
    return flat_inputs[0]

  def _convert_non_tensor(x):
    # Don't call `ops.convert_to_tensor` on all `inputs` because
    # `SparseTensors` can't be converted to `Tensor`.
    if isinstance(x, np.ndarray):
      return ops.convert_to_tensor(x)
    return x

  inputs = nest.map_structure(_convert_non_tensor, inputs)
  # For more complicated structure, we only convert the out most list to tuple
  # since dataset will stack the list, but treat elements in the tuple as
  # individual element.
  return training_utils.list_to_tuple(inputs)


def is_none_or_empty(inputs):
  # util method to check if the input is a None or a empty list.
  # the python "not" check will raise an error like below if the input is a
  # numpy array
  # "The truth value of an array with more than one element is ambiguous.
  # Use a.any() or a.all()"
  return inputs is None or not nest.flatten(inputs)
