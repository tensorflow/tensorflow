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

import numpy as np
import six

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops
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

  dataset = applicable_adapters[0]().get_dataset(x)
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
  def get_dataset(self, x, y=None, **kwargs):
    """Convert the input x and y into dataset.

    The caller must make sure to call `can_handle()` first before invoking this
    method. Provide unsupported data type will result into unexpected behavior.

    Args:
      x: input features.
      y: target labels. Note that y could be None in the case of prediction.
      **kwargs: Other keyword arguments for DataAdapter during the construction
        of the tf.dataset.Dataset. For example:
        - Numpy data might need to have `batch_size` parameter when constructing
          the dataset and iterator.
        - Numpy data might have "evaluation_split" which will split the input
          data into training and validation set.
        - Numpy data might have `sample_weights` which will be used for
          weighting the loss function during training.
        DataAdapter might choose to ignore any keyword argument if it doesn't
        use it, or raise exception if any required argument is not provide.

    Returns:
      An tf.dataset.Dataset. Caller might use the dataset in different
      context, eg iter(dataset) in eager to get the value directly, or in graph
      mode, provide the iterator tensor to Keras model function.
    """
    raise NotImplementedError


class NumpyArrayDataAdapter(DataAdapter):
  """Adapter that handles the Numpy array."""

  @staticmethod
  def can_handle(x, y=None):
    if y is not None and type(x) is not type(y):
      raise ValueError("input feature and target should have same type, got "
                       "x: {}, y: {}".format(type(x), type(y)))
    return isinstance(x, np.ndarray)

  def get_dataset(self, x, y=None, sample_weights=None, batch_size=None,
                  shuffle=False, **kwargs):
    # TODO(scottzhu): Handle validation_split
    if y is not None and sample_weights is not None:
      inputs = (x, y, sample_weights)
    elif y is not None:
      # Sample weight is only needed for training, so if y is None, then
      # sample_weight is ignored.
      inputs = (x, y)
    else:
      inputs = x

    if not batch_size:
      raise ValueError("batch size is required for Numpy input data.")

    # TODO(scottzhu): might need to check large data input (> 2G).
    dataset = dataset_ops.DatasetV2.from_tensor_slices(inputs)
    if shuffle:
      num_samples = int(nest.flatten(x)[0].shape[0])
      dataset = dataset.shuffle(num_samples)
    return dataset.batch(batch_size)


class TensorDataAdapter(DataAdapter):
  """Adapter that handles Tensorflow tensors."""

  @staticmethod
  def can_handle(x, y=None):
    return isinstance(x, ops.Tensor)

  def get_dataset(self, x, y=None, batch_size=None, shuffle=False, **kwargs):
    inputs = x if y is None else (x, y)

    # Do we need batch_size for data tensor?
    if not batch_size:
      raise ValueError("batch size is required for tensor input data.")
    dataset = dataset_ops.DatasetV2.from_tensor_slices(inputs)
    if shuffle:
      num_samples = int(nest.flatten(x)[0].shape[0])
      dataset = dataset.shuffle(num_samples)
    return dataset.batch(batch_size)


class DatasetAdapter(DataAdapter):
  """Adapter that handles `tf.data.Dataset`."""

  @staticmethod
  def can_handle(x, y=None):
    return isinstance(x, (dataset_ops.DatasetV1, dataset_ops.DatasetV2))

  def get_dataset(self, x, y=None, **kwargs):
    # TODO(scottzhu): throw error when sample_weights, etc is provided.
    if y is not None:
      raise ValueError("target input is expected to be None when using "
                       "dataset as input.")
    return x


class GeneratorDataAdapter(DataAdapter):
  """Adapter that handles python generator."""

  @staticmethod
  def can_handle(x, y=None):
    return tf_inspect.isgenerator(x)

  def get_dataset(self, x, y=None, **kwargs):
    # TODO(scottzhu): throw error when sample_weights, etc is provided.
    if y is not None:
      raise ValueError("target input is expected to be None when using "
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

    return dataset_ops.DatasetV2.from_generator(reassemble, nested_dtypes,
                                                output_shapes=nested_shape)


class KerasSequenceAdapter(DataAdapter):
  """Adapter that handles `keras.utils.Sequence`."""

  @staticmethod
  def can_handle(x, y=None):
    return isinstance(x, data_utils.Sequence)

  def get_dataset(self, x, y=None, shuffle=False, **kwargs):
    # TODO(scottzhu): throw error when sample_weights, etc is provided.
    if y is not None:
      raise ValueError("target input is expected to be None when using "
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
    return dataset
