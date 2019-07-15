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
from tensorflow.python.framework import ops
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils import data_utils
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

  @abc.abstractmethod
  def has_partial_batch(self):
    """Whether the dataset has partial batch at the end."""
    raise NotImplementedError


class NumpyArrayDataAdapter(DataAdapter):
  """Adapter that handles the Numpy array."""

  @staticmethod
  def can_handle(x, y=None):
    flat_inputs = nest.flatten(x)
    if y is not None:
      flat_inputs += nest.flatten(y)

    return all(isinstance(v, np.ndarray) for v in flat_inputs)

  def __init__(self, x, y=None, sample_weights=None, batch_size=None,
               shuffle=False, distribution_strategy=None, **kwargs):
    super(NumpyArrayDataAdapter, self).__init__(x, y, **kwargs)
    x = _process_numpy_inputs(x)
    y = _process_numpy_inputs(y)
    sample_weights = _process_numpy_inputs(sample_weights)
    if y is not None and sample_weights is not None:
      inputs = (x, y, sample_weights)
    elif y is not None:
      # Sample weight is only needed for training, so if y is None, then
      # sample_weight is ignored.
      inputs = (x, y)
    else:
      inputs = (x,)

    if not batch_size:
      raise ValueError("batch size is required for Numpy input data.")

    if distribution_strategy is not None:
      dataset = distribution_strategy.experimental_make_numpy_dataset(inputs)
    else:
      dataset = dataset_ops.DatasetV2.from_tensor_slices(inputs)

    num_samples = int(nest.flatten(x)[0].shape[0])
    if shuffle:
      # Note that we use the full input data length as buffer window, which
      # might have memory consumption consequence. This is on the radar of
      # tf.data team and they will address it.
      dataset = dataset.shuffle(num_samples)
    self._dataset = dataset.batch(batch_size)
    self._size = int(math.ceil(num_samples / batch_size))
    self._batch_size = batch_size
    self._has_partial_batch = (self._size != (num_samples // batch_size))

  def get_dataset(self):
    return self._dataset

  def get_size(self):
    return self._size

  def batch_size(self):
    return self._batch_size

  def has_partial_batch(self):
    return self._has_partial_batch


# TODO(scottzhu): Eventually the numpy array and eager tensor should be treated
# in the same way. Merge this class with NumpyArrayDataAdapter.
class TensorDataAdapter(DataAdapter):
  """Adapter that handles Tensorflow eager tensors."""

  @staticmethod
  def can_handle(x, y=None):
    flat_inputs = nest.flatten(x)
    if y is not None:
      flat_inputs += nest.flatten(y)

    return all(isinstance(v, ops.Tensor) for v in flat_inputs)

  def __init__(self, x, y=None, sample_weights=None, batch_size=None,
               shuffle=False, **kwargs):
    super(TensorDataAdapter, self).__init__(x, y, **kwargs)
    x = _process_numpy_inputs(x)
    y = _process_numpy_inputs(y)
    sample_weights = _process_numpy_inputs(sample_weights)
    if y is not None and sample_weights is not None:
      inputs = (x, y, sample_weights)
    elif y is not None:
      # Sample weight is only needed for training, so if y is None, then
      # sample_weight is ignored.
      inputs = (x, y)
    else:
      inputs = (x,)

    # TODO(scottzhu): We should treat data tensor same as numpy array, make
    # the batch_size a required param.
    # if not batch_size:
    #   raise ValueError("batch size is required for tensor input data.")
    dataset = dataset_ops.DatasetV2.from_tensor_slices(inputs)
    num_samples = int(nest.flatten(x)[0].shape[0])
    if shuffle:
      dataset = dataset.shuffle(num_samples)
    if batch_size:
      dataset = dataset.batch(batch_size)
      self._size = int(math.ceil(num_samples / batch_size))
      self._batch_size = batch_size
      self._has_partial_batch = (self._size != (num_samples // batch_size))
    else:
      self._size = 1
      self._batch_size = num_samples
      self._has_partial_batch = False
    self._dataset = dataset

  def get_dataset(self):
    return self._dataset

  def get_size(self):
    return self._size

  def batch_size(self):
    return self._batch_size

  def has_partial_batch(self):
    return self._has_partial_batch


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
    peek = next(x)
    nested_dtypes = nest.map_structure(lambda t: t.dtype, peek)
    nested_shape = nest.map_structure(lambda t: t.shape, peek)
    # Note that dataset API takes a callable that creates a generator object,
    # rather than generator itself, which is why we define a function here.
    def reassemble():
      return itertools.chain([peek], x)

    self._batch_size = int(nest.flatten(peek)[0].shape[0])
    self._dataset = dataset_ops.DatasetV2.from_generator(
        reassemble, nested_dtypes, output_shapes=nested_shape)

  def get_dataset(self):
    return self._dataset

  def get_size(self):
    return None

  def batch_size(self):
    return self._batch_size

  def has_partial_batch(self):
    return False


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
    peek = x[0]
    nested_dtypes = nest.map_structure(lambda t: t.dtype, peek)
    nested_shape = nest.map_structure(lambda t: t.shape, peek)

    def generator():
      for i in range(len(x)):
        yield x[i]
    dataset = dataset_ops.DatasetV2.from_generator(generator, nested_dtypes,
                                                   output_shapes=nested_shape)
    if shuffle:
      dataset = dataset.shuffle(len(x))
    self._dataset = dataset
    self._size = len(x)
    self._batch_size = int(nest.flatten(peek)[0].shape[0])

  def get_dataset(self):
    return self._dataset

  def get_size(self):
    return self._size

  def batch_size(self):
    return self._batch_size

  def has_partial_batch(self):
    return False


ALL_ADAPTER_CLS = [NumpyArrayDataAdapter, TensorDataAdapter, DatasetAdapter,
                   GeneratorDataAdapter, KerasSequenceAdapter]


def select_data_adapter(x, y):
  adapter_cls = [cls for cls in ALL_ADAPTER_CLS if cls.can_handle(x, y)]
  if not adapter_cls:
    raise ValueError("Failed to find data adapter that can handle "
                     "input: {}, {}".format(type(x), type(y)))
  elif len(adapter_cls) > 1:
    raise RuntimeError("Data adapter should be mutually exclusive for "
                       "handling inputs. Found multiple adapter {} to handle "
                       "input: {}, {}".format(adapter_cls, type(x), type(y)))
  return adapter_cls[0]


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
