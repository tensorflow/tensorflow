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

import abc
import contextlib
import functools
import itertools
import math
import random

import numpy as np

from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import input_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import dataset_creator
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export


class DataAdapter(object, metaclass=abc.ABCMeta):
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
      raise ValueError("{} Cannot handle input {}, {}".format(
          self.__class__, x, y))

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

  @abc.abstractmethod
  def should_recreate_iterator(self):
    """Returns whether a new iterator should be created every epoch."""
    raise NotImplementedError

  def get_samples(self):
    """Returns number of samples in the data, or `None`."""
    if not self.get_size() or not self.batch_size():
      return None
    total_sample = self.get_size() * self.batch_size()
    if self.has_partial_batch():
      total_sample -= (self.batch_size() - self.partial_batch_size())
    return total_sample

  def on_epoch_end(self):
    """A hook called after each epoch."""
    pass


class TensorLikeDataAdapter(DataAdapter):
  """Adapter that handles Tensor-like objects, e.g. EagerTensor and NumPy."""

  @staticmethod
  def can_handle(x, y=None):
    # TODO(kaftan): Check performance implications of using a flatten
    #  here for other types of inputs.
    flat_inputs = nest.flatten(x)
    if y is not None:
      flat_inputs += nest.flatten(y)

    tensor_types = _get_tensor_types()

    def _is_tensor(v):
      if isinstance(v, tensor_types):
        return True
      return False

    return all(_is_tensor(v) for v in flat_inputs)

  def __init__(self,
               x,
               y=None,
               sample_weights=None,
               sample_weight_modes=None,
               batch_size=None,
               epochs=1,
               steps=None,
               shuffle=False,
               **kwargs):
    super(TensorLikeDataAdapter, self).__init__(x, y, **kwargs)
    x, y, sample_weights = _process_tensorlike((x, y, sample_weights))
    sample_weight_modes = broadcast_sample_weight_modes(
        sample_weights, sample_weight_modes)

    # If sample_weights are not specified for an output use 1.0 as weights.
    (sample_weights, _, _) = training_utils.handle_partial_sample_weights(
        y, sample_weights, sample_weight_modes, check_all_flat=True)

    inputs = pack_x_y_sample_weight(x, y, sample_weights)

    num_samples = set(int(i.shape[0]) for i in nest.flatten(inputs)).pop()
    _check_data_cardinality(inputs)

    # If batch_size is not passed but steps is, calculate from the input data.
    # Default to 32 for backwards compat.
    if not batch_size:
      batch_size = int(math.ceil(num_samples / steps)) if steps else 32

    self._size = int(math.ceil(num_samples / batch_size))
    self._batch_size = batch_size

    num_full_batches = int(num_samples // batch_size)
    self._partial_batch_size = num_samples % batch_size

    if isinstance(shuffle, str):
      shuffle = shuffle.lower()

    self._shuffle = shuffle
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

    indices_dataset = dataset_ops.DatasetV2.range(1)
    if shuffle != "batch":
      indices_dataset = indices_dataset.repeat(epochs)

    def permutation(_):
      # It turns out to be more performant to make a new set of indices rather
      # than reusing the same range Tensor. (presumably because of buffer
      # forwarding.)
      indices = math_ops.range(num_samples, dtype=dtypes.int64)
      if shuffle and shuffle != "batch":
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

      if shuffle == "batch":
        # 1024 is a magic constant that has not been properly evaluated
        flat_dataset = flat_dataset.shuffle(1024).repeat(epochs)
      return flat_dataset

    indices_dataset = indices_dataset.flat_map(slice_batch_indices)

    dataset = self.slice_inputs(indices_dataset, inputs)

    if shuffle == "batch":
      def shuffle_batch(*batch):
        return nest.map_structure(random_ops.random_shuffle, batch)
      dataset = dataset.map(shuffle_batch)

    self._dataset = dataset

  def slice_inputs(self, indices_dataset, inputs):
    """Slice inputs into a Dataset of batches.

    Given a Dataset of batch indices and the unsliced inputs,
    this step slices the inputs in a parallelized fashion
    and produces a dataset of input batches.

    Args:
      indices_dataset: A Dataset of batched indices
      inputs: A python data structure that contains the inputs, targets,
        and possibly sample weights.

    Returns:
      A Dataset of input batches matching the batch indices.
    """
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
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    if self._shuffle:
      # See b/141490660 for more details.
      options.experimental_external_state_policy = (
          options_lib.ExternalStatePolicy.IGNORE)
    dataset = dataset.with_options(options)
    return dataset

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

  def should_recreate_iterator(self):
    # An infinite dataset is always created here.
    return False


class GenericArrayLikeDataAdapter(TensorLikeDataAdapter):
  """Adapter that handles array-like data without forcing it into memory.

  This adapter handles array-like datasets that may be too big to fully
  fit into memory.

  Specifically, this adapter handles any Python class which implements:
  `__get_item__`, `__len__`, `shape`, and `dtype` with the same meanings
  as Numpy, but it ignores any case where all the inputs are Tensors or Numpy
  arrays (because that case is handled by the base TensorLikeDataAdapter).

  It ignores scipy sparse matrices and Composite Tensors because those are
  handled by the CompositeTensorDataAdapter.

  It also does not handle lists/tuples of scalars, because those are handled
  by the ListsOfScalarsDataAdapter.
  """

  @staticmethod
  def can_handle(x, y=None):
    flat_inputs = nest.flatten(x)
    if y is not None:
      flat_inputs += nest.flatten(y)

    def _is_array_like(v):
      """Return True if v is a Tensor, array, or is array-like."""
      return (
          hasattr(v, "__getitem__") and
          hasattr(v, "shape") and
          hasattr(v, "dtype") and
          hasattr(v, "__len__")
      )

    if (not TensorLikeDataAdapter.can_handle(x, y) and
        not CompositeTensorDataAdapter.can_handle(x, y)):
      return all(_is_array_like(v) for v in flat_inputs)
    else:
      return False

  def __init__(self, *args, **kwargs):
    logging.warning(
        "Keras is training/fitting/evaluating on array-like data. Keras may "
        "not be optimized for this format, so if your input data format is "
        "supported by TensorFlow I/O (https://github.com/tensorflow/io) we "
        "recommend using that to load a Dataset instead.")

    super(GenericArrayLikeDataAdapter, self).__init__(*args, **kwargs)

  def slice_inputs(self, indices_dataset, inputs):
    """Slice inputs into a Dataset of batches.

    Given a Dataset of batch indices and the unsliced inputs,
    this step slices the inputs in a parallelized fashion
    and produces a dataset of input batches.

    Args:
      indices_dataset: A Dataset of batched indices
      inputs: A python data structure that contains the inputs, targets,
        and possibly sample weights.

    Returns:
      A Dataset of input batches matching the batch indices.
    """
    flat_inputs = nest.flatten(inputs)
    def dynamic_shape_like(t):
      shape = list(t.shape)
      shape[0] = None
      return tuple(shape)

    flat_dtypes = [inp.dtype for inp in flat_inputs]
    contiguous = True
    if self._shuffle and self._shuffle != "batch":
      contiguous = False

    def grab_batch(indices):
      """Grab a batch of data from the inputs."""
      # This uses a py_function to avoid converting the array-like
      # into a Tensor before slicing it, because converting the array-like
      # to a Tensor may force it into memory..
      def py_method(ind):
        def slice_array(data):
          return training_utils.slice_arrays(data, ind.numpy(),
                                             contiguous=contiguous)
        return [slice_array(inp) for inp in flat_inputs]

      flat_out = script_ops.eager_py_func(py_method, [indices], flat_dtypes)
      for v, original_inp in zip(flat_out, flat_inputs):
        v.set_shape(dynamic_shape_like(original_inp))
      return nest.pack_sequence_as(inputs, flat_out)

    dataset = indices_dataset.map(
        grab_batch, num_parallel_calls=dataset_ops.AUTOTUNE)

    return dataset


class DatasetCreatorAdapter(DataAdapter):
  """Adapter that handles dataset functions."""

  def __init__(self, x, y, steps=None, distribution_strategy=None, **kwargs):
    super(DatasetCreatorAdapter, self).__init__(x, **kwargs)

    if not isinstance(x, dataset_creator.DatasetCreator):
      raise TypeError("The input of a `DatasetCreatorAdapter` should be a "
                      "`DatasetCreator` but it received type {}.".format(
                          type(x)))
    if steps is None:
      raise ValueError("When using a "
                       "`tf.keras.utils.experimental.DatasetCreator`, "
                       "`steps_per_epoch`, `validation_steps` or `steps` "
                       "argument must be provided in `Model.fit`, "
                       "`Model.evaluate`, or `Model.predict`.")
    self.dataset_creator = x
    self.steps = steps
    self.strategy = distribution_strategy

  @staticmethod
  def can_handle(x, y=None):
    if isinstance(x, dataset_creator.DatasetCreator):
      assert y is None
      return True

  def should_recreate_iterator(self):
    # We expect users to shuffle the dataset in their `dataset_fn` supplied to
    # `DatasetCreator`. Since that is a buffered shuffle, we intend to not reset
    # the dataset so the batches that are not shuffled can still be pulled.
    return False

  def get_size(self):
    return None  # To be inferred by `DataHandler`.

  def get_dataset(self):
    return self.strategy.distribute_datasets_from_function(
        self.dataset_creator, options=self.dataset_creator.input_options)

  def batch_size(self):
    raise NotImplementedError()

  def has_partial_batch(self):
    raise NotImplementedError()

  def partial_batch_size(self):
    raise NotImplementedError()


class CompositeTensorDataAdapter(DataAdapter):
  """Adapter that handles composite tensor."""

  @staticmethod
  def can_handle(x, y=None):
    flat_inputs = nest.flatten(x)
    if y is not None:
      flat_inputs += nest.flatten(y)

    def _is_composite(v):
      # Dataset/iterator/DistributedDataset inherits from CompositeTensor but
      # should be handled by DatasetAdapter and GeneratorAdapter.
      if (tf_utils.is_extension_type(v) and
          not isinstance(v,
                         (dataset_ops.DatasetV2, iterator_ops.IteratorBase)) and
          not _is_distributed_dataset(v)):
        return True
      # Support Scipy sparse tensors if scipy is installed
      return _is_scipy_sparse(v)

    def _is_tensor_or_composite(v):
      if isinstance(v, (ops.Tensor, np.ndarray)):
        return True
      return _is_composite(v)

    return (any(_is_composite(v) for v in flat_inputs) and
            all(_is_tensor_or_composite(v) for v in flat_inputs))

  def __init__(self,
               x,
               y=None,
               sample_weights=None,
               sample_weight_modes=None,
               batch_size=None,
               steps=None,
               shuffle=False,
               **kwargs):
    super(CompositeTensorDataAdapter, self).__init__(x, y, **kwargs)
    x, y, sample_weights = _process_tensorlike((x, y, sample_weights))
    sample_weight_modes = broadcast_sample_weight_modes(
        sample_weights, sample_weight_modes)

    # If sample_weights are not specified for an output use 1.0 as weights.
    (sample_weights, _, _) = training_utils.handle_partial_sample_weights(
        y, sample_weights, sample_weight_modes, check_all_flat=True)

    inputs = pack_x_y_sample_weight(x, y, sample_weights)

    dataset = dataset_ops.DatasetV2.from_tensor_slices(inputs)
    num_samples = int(nest.flatten(x)[0].shape[0])
    if shuffle:
      dataset = dataset.shuffle(num_samples)

    # If batch_size is not passed but steps is, calculate from the input data.
    # Default to 32 for backwards compat.
    if not batch_size:
      batch_size = int(math.ceil(num_samples / steps)) if steps else 32

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

  def should_recreate_iterator(self):
    return True


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
    if isinstance(inp, (float, int, str, bytes, bytearray)):
      return True
    if isinstance(inp, (list, tuple)) and inp:
      return ListsOfScalarsDataAdapter._is_list_of_scalars(inp[0])
    return False

  def __init__(self,
               x,
               y=None,
               sample_weights=None,
               sample_weight_modes=None,
               batch_size=None,
               shuffle=False,
               **kwargs):
    super(ListsOfScalarsDataAdapter, self).__init__(x, y, **kwargs)
    x = np.asarray(x)
    if y is not None:
      y = np.asarray(y)
    if sample_weights is not None:
      sample_weights = np.asarray(sample_weights)
    sample_weight_modes = broadcast_sample_weight_modes(
        sample_weights, sample_weight_modes)

    self._internal_adapter = TensorLikeDataAdapter(
        x,
        y=y,
        sample_weights=sample_weights,
        sample_weight_modes=sample_weight_modes,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs)

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

  def should_recreate_iterator(self):
    return True


class DatasetAdapter(DataAdapter):
  """Adapter that handles `tf.data.Dataset`."""

  @staticmethod
  def can_handle(x, y=None):
    return (isinstance(x, (data_types.DatasetV1, data_types.DatasetV2)) or
            _is_distributed_dataset(x))

  def __init__(self,
               x,
               y=None,
               sample_weights=None,
               steps=None,
               **kwargs):
    super(DatasetAdapter, self).__init__(x, y, **kwargs)
    # Note that the dataset instance is immutable, its fine to reuse the user
    # provided dataset.
    self._dataset = x

    # The user-provided steps.
    self._user_steps = steps

    self._validate_args(y, sample_weights, steps)

  def get_dataset(self):
    return self._dataset

  def get_size(self):
    return  # Inferred in `DataHandler`.

  def batch_size(self):
    return None

  def has_partial_batch(self):
    return False

  def partial_batch_size(self):
    return None

  def should_recreate_iterator(self):
    # Since DistributedDatasets have no cardinality, the user must provide
    # all steps that need to be run, calling `.repeat()` as needed.
    if _is_distributed_dataset(self._dataset):
      return False

    # If user doesn't supply `steps`, or if they supply `steps` that
    # exactly equals the size of the `Dataset`, create a new iterator
    # each epoch.
    return (self._user_steps is None or
            cardinality.cardinality(self._dataset).numpy() == self._user_steps)

  def _validate_args(self, y, sample_weights, steps):
    """Validates `__init__` arguments."""
    # Arguments that shouldn't be passed.
    if not is_none_or_empty(y):
      raise ValueError("`y` argument is not supported when using "
                       "dataset as input.")
    if not is_none_or_empty(sample_weights):
      raise ValueError("`sample_weight` argument is not supported when using "
                       "dataset as input.")

    if steps is None:
      if _is_distributed_dataset(self._dataset):
        raise ValueError("When providing a distributed dataset, you must "
                         "specify the number of steps to run.")

      size = cardinality.cardinality(self._dataset).numpy()
      if size == cardinality.INFINITE and steps is None:
        raise ValueError(
            "When providing an infinite dataset, you must specify "
            "the number of steps to run (if you did not intend to "
            "create an infinite dataset, make sure to not call "
            "`repeat()` on the dataset).")


class GeneratorDataAdapter(DataAdapter):
  """Adapter that handles python generators and iterators."""

  @staticmethod
  def can_handle(x, y=None):
    return ((hasattr(x, "__next__") or hasattr(x, "next"))
            and hasattr(x, "__iter__")
            and not isinstance(x, data_utils.Sequence))

  def __init__(self,
               x,
               y=None,
               sample_weights=None,
               workers=1,
               use_multiprocessing=False,
               max_queue_size=10,
               model=None,
               **kwargs):
    # Generators should never shuffle as exhausting the generator in order to
    # shuffle the batches is inefficient.
    kwargs.pop("shuffle", None)

    if not is_none_or_empty(y):
      raise ValueError("`y` argument is not supported when using "
                       "python generator as input.")
    if not is_none_or_empty(sample_weights):
      raise ValueError("`sample_weight` argument is not supported when using "
                       "python generator as input.")

    super(GeneratorDataAdapter, self).__init__(x, y, **kwargs)

    # Since we have to know the dtype of the python generator when we build the
    # dataset, we have to look at a batch to infer the structure.
    peek, x = self._peek_and_restore(x)
    peek = self._standardize_batch(peek)
    peek = _process_tensorlike(peek)

    # Need to build the Model on concrete input shapes.
    if model is not None and not model.built:
      concrete_x, _, _ = unpack_x_y_sample_weight(peek)
      model.distribute_strategy.run(
          lambda x: model(x, training=False), args=(concrete_x,))

    self._first_batch_size = int(nest.flatten(peek)[0].shape[0])

    def _get_dynamic_shape(t):
      shape = t.shape
      # Unknown number of dimensions, `as_list` cannot be called.
      if shape.rank is None:
        return shape
      return tensor_shape.TensorShape([None for _ in shape.as_list()])

    output_shapes = nest.map_structure(_get_dynamic_shape, peek)
    output_types = nest.map_structure(lambda t: t.dtype, peek)

    # Note that dataset API takes a callable that creates a generator object,
    # rather than generator itself, which is why we define a function here.
    generator_fn = self._handle_multiprocessing(x, workers, use_multiprocessing,
                                                max_queue_size)

    def wrapped_generator():
      for data in generator_fn():
        yield self._standardize_batch(data)

    dataset = dataset_ops.DatasetV2.from_generator(
        wrapped_generator, output_types, output_shapes=output_shapes)

    if workers == 1 and not use_multiprocessing:
      dataset = dataset.prefetch(1)

    self._dataset = dataset

  def _standardize_batch(self, data):
    """Standardizes a batch output by a generator."""
    # Removes `None`s.
    x, y, sample_weight = unpack_x_y_sample_weight(data)
    data = pack_x_y_sample_weight(x, y, sample_weight)

    data = nest.list_to_tuple(data)

    def _convert_dtype(t):
      if (isinstance(t, np.ndarray) and issubclass(t.dtype.type, np.floating)):
        return np.array(t, dtype=backend.floatx())
      return t

    data = nest.map_structure(_convert_dtype, data)
    return data

  @staticmethod
  def _peek_and_restore(x):
    peek = next(x)
    return peek, itertools.chain([peek], x)

  def _handle_multiprocessing(self, x, workers, use_multiprocessing,
                              max_queue_size):
    """Create a callable, possibly including an Enqueuer."""
    if workers > 1 or (workers > 0 and use_multiprocessing):
      def generator_fn():
        enqueuer = data_utils.GeneratorEnqueuer(
            x, use_multiprocessing=use_multiprocessing)
        enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        return enqueuer.get()
    else:
      generator_fn = lambda: x
    return generator_fn

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
    return

  def should_recreate_iterator(self):
    return False


class KerasSequenceAdapter(GeneratorDataAdapter):
  """Adapter that handles `keras.utils.Sequence`."""

  @staticmethod
  def can_handle(x, y=None):
    return isinstance(x, data_utils.Sequence)

  def __init__(self,
               x,
               y=None,
               sample_weights=None,
               shuffle=False,
               workers=1,
               use_multiprocessing=False,
               max_queue_size=10,
               model=None,
               **kwargs):
    if not is_none_or_empty(y):
      raise ValueError("`y` argument is not supported when using "
                       "`keras.utils.Sequence` as input.")
    if not is_none_or_empty(sample_weights):
      raise ValueError("`sample_weight` argument is not supported when using "
                       "`keras.utils.Sequence` as input.")

    self._size = len(x)
    self._shuffle_sequence = shuffle
    self._keras_sequence = x
    self._enqueuer = None
    super(KerasSequenceAdapter, self).__init__(
        x,
        shuffle=False,  # Shuffle is handed in the _make_callable override.
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=max_queue_size,
        model=model,
        **kwargs)

  @staticmethod
  def _peek_and_restore(x):
    return x[0], x

  def _handle_multiprocessing(self, x, workers, use_multiprocessing,
                              max_queue_size):
    if workers > 1 or (workers > 0 and use_multiprocessing):
      def generator_fn():
        self._enqueuer = data_utils.OrderedEnqueuer(
            x, use_multiprocessing=use_multiprocessing,
            shuffle=self._shuffle_sequence)
        self._enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        return self._enqueuer.get()
    else:
      def generator_fn():
        order = range(len(x))
        if self._shuffle_sequence:
          # Match the shuffle convention in OrderedEnqueuer.
          order = list(order)
          random.shuffle(order)

        for i in order:
          yield x[i]

    return generator_fn

  def get_size(self):
    return self._size

  def should_recreate_iterator(self):
    return True

  def on_epoch_end(self):
    if self._enqueuer:
      self._enqueuer.stop()
    self._keras_sequence.on_epoch_end()


ALL_ADAPTER_CLS = [
    ListsOfScalarsDataAdapter, TensorLikeDataAdapter,
    GenericArrayLikeDataAdapter, DatasetAdapter, GeneratorDataAdapter,
    KerasSequenceAdapter, CompositeTensorDataAdapter, DatasetCreatorAdapter
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


def _process_tensorlike(inputs):
  """Process tensor-like inputs.

  This function:

  (1) Converts `Numpy` arrays to `Tensor`s.
  (2) Converts `Scipy` sparse matrices to `SparseTensor`s.
  (2) Converts `list`s to `tuple`s (for `tf.data` support).

  Args:
    inputs: Structure of `Tensor`s, `NumPy` arrays, or tensor-like.

  Returns:
    Structure of `Tensor`s or tensor-like.
  """

  def _convert_numpy_and_scipy(x):
    if isinstance(x, np.ndarray):
      dtype = None
      if issubclass(x.dtype.type, np.floating):
        dtype = backend.floatx()
      return tensor_conversion.convert_to_tensor_v2_with_dispatch(
          x, dtype=dtype
      )
    elif _is_scipy_sparse(x):
      return _scipy_sparse_to_sparse_tensor(x)
    return x

  inputs = nest.map_structure(_convert_numpy_and_scipy, inputs)
  return nest.list_to_tuple(inputs)


def is_none_or_empty(inputs):
  # util method to check if the input is a None or a empty list.
  # the python "not" check will raise an error like below if the input is a
  # numpy array
  # "The truth value of an array with more than one element is ambiguous.
  # Use a.any() or a.all()"
  return inputs is None or not nest.flatten(inputs)


def broadcast_sample_weight_modes(target_structure, sample_weight_modes):
  """Match sample_weight_modes structure with output structure."""
  if target_structure is None or not nest.flatten(target_structure):
    return sample_weight_modes

  if isinstance(sample_weight_modes, str):
    if isinstance(target_structure, dict):
      return {key: sample_weight_modes for key in target_structure.keys()}
    return [sample_weight_modes for _ in target_structure]

  if sample_weight_modes:
    try:
      nest.assert_same_structure(
          training_utils.list_to_tuple(target_structure),
          training_utils.list_to_tuple(sample_weight_modes))
    except (ValueError, TypeError):
      target_str = str(nest.map_structure(lambda _: "...", target_structure))
      mode_str = str(nest.map_structure(lambda _: "...", sample_weight_modes))

      # Attempt to coerce sample_weight_modes to the target structure. This
      # implicitly depends on the fact that Model flattens outputs for its
      # internal representation.
      try:
        sample_weight_modes = nest.pack_sequence_as(
            target_structure, nest.flatten(sample_weight_modes))
        logging.warning(
            "sample_weight modes were coerced from\n  {}\n    to  \n  {}"
            .format(target_str, mode_str))
      except (ValueError, TypeError):
        raise ValueError(
            "Unable to match target structure and sample_weight_modes "
            "structure:\n  {}\n    to  \n  {}".format(target_str, mode_str))

  return sample_weight_modes


class DataHandler(object):
  """Handles iterating over epoch-level `tf.data.Iterator` objects."""

  def __init__(self,
               x,
               y=None,
               sample_weight=None,
               batch_size=None,
               steps_per_epoch=None,
               initial_epoch=0,
               epochs=1,
               shuffle=False,
               class_weight=None,
               max_queue_size=10,
               workers=1,
               use_multiprocessing=False,
               model=None,
               steps_per_execution=None,
               distribute=True):
    """Initializes a `DataHandler`.

    Arguments:
      x: See `Model.fit`.
      y: See `Model.fit`.
      sample_weight: See `Model.fit`.
      batch_size: See `Model.fit`.
      steps_per_epoch: See `Model.fit`.
      initial_epoch: See `Model.fit`.
      epochs: See `Model.fit`.
      shuffle: See `Model.fit`.
      class_weight: See `Model.fit`.
      max_queue_size: See `Model.fit`.
      workers: See `Model.fit`.
      use_multiprocessing: See `Model.fit`.
      model: The `Model` instance. Needed in order to correctly `build` the
        `Model` using generator-like inputs (see `GeneratorDataAdapter`).
      steps_per_execution: See `Model.compile`.
      distribute: Whether to distribute the `tf.dataset`.
        `PreprocessingLayer.adapt` does not support distributed datasets,
        `Model` should always set this to `True`.
    """

    self._initial_epoch = initial_epoch
    self._epochs = epochs
    self._insufficient_data = False
    self._model = model

    # `steps_per_execution_value` is the cached initial value.
    # `steps_per_execution` is mutable and may be changed by the DataAdapter
    # to handle partial executions.
    if steps_per_execution is None:
      self._steps_per_execution = 1
      self._steps_per_execution_value = 1
    else:
      self._steps_per_execution = steps_per_execution
      self._steps_per_execution_value = steps_per_execution.numpy().item()

    adapter_cls = select_data_adapter(x, y)
    self._adapter = adapter_cls(
        x,
        y,
        batch_size=batch_size,
        steps=steps_per_epoch,
        epochs=epochs - initial_epoch,
        sample_weights=sample_weight,
        shuffle=shuffle,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        distribution_strategy=ds_context.get_strategy(),
        model=model)

    strategy = ds_context.get_strategy()

    self._current_step = 0
    self._step_increment = self._steps_per_execution_value - 1
    self._insufficient_data = False

    self._configure_dataset_and_inferred_steps(strategy, x, steps_per_epoch,
                                               class_weight, distribute)

  def _configure_dataset_and_inferred_steps(self, strategy, x, steps_per_epoch,
                                            class_weight, distribute):
    """Configure the `_dataset` and `_inferred_steps` attributes."""
    del x
    dataset = self._adapter.get_dataset()
    if class_weight:
      dataset = dataset.map(_make_class_weight_map_fn(class_weight))
    self._inferred_steps = self._infer_steps(steps_per_epoch, dataset)

    # `PreprocessingLayer.adapt` does not currently support distributed
    # datasets, so we pass `distribute=False` there.
    if distribute and not _is_distributed_dataset(dataset):
      dataset = strategy.experimental_distribute_dataset(dataset)
    self._dataset = dataset
    self._validate_data_handler()

  def enumerate_epochs(self):
    """Yields `(epoch, tf.data.Iterator)`."""
    with self._truncate_execution_to_epoch():
      data_iterator = iter(self._dataset)
      for epoch in range(self._initial_epoch, self._epochs):
        if self._insufficient_data:  # Set by `catch_stop_iteration`.
          break
        if self._adapter.should_recreate_iterator():
          data_iterator = iter(self._dataset)
        yield epoch, data_iterator
        self._adapter.on_epoch_end()

  @contextlib.contextmanager
  def _truncate_execution_to_epoch(self):
    """Truncates steps per execution to at most one epoch."""
    should_truncate = (
        self._inferred_steps is not None and
        self._steps_per_execution_value > self._inferred_steps)
    original_value = self._steps_per_execution_value
    try:
      if should_truncate:
        self._steps_per_execution.assign(self._inferred_steps)
        self._steps_per_execution_value = self._inferred_steps
      yield
    finally:
      if should_truncate:
        self._steps_per_execution.assign(original_value)
        self._steps_per_execution_value = original_value

  def sync(self):
    context.async_wait()

  @contextlib.contextmanager
  def catch_stop_iteration(self):
    """Catches errors when an iterator runs out of data."""
    try:
      yield
      self.sync()
    except (StopIteration, errors.OutOfRangeError):
      if self._inferred_steps is None:
        self._inferred_steps = self._current_step
      else:
        self._insufficient_data = True
        total_epochs = self._epochs - self._initial_epoch
        logging.warning(
            "Your input ran out of data; interrupting training. "
            "Make sure that your dataset or generator can generate at "
            "least `steps_per_epoch * epochs` batches (in this case, "
            "{} batches). You may need to use the repeat() function "
            "when building your dataset.".format(total_epochs *
                                                 self._inferred_steps))

  def steps(self):
    """Yields steps for the current epoch."""
    self._current_step = 0
    # `self._inferred_steps` can be changed by `catch_stop_iteration`.
    while (self._inferred_steps is None or
           self._current_step < self._inferred_steps):
      if self._insufficient_data:  # Set by `catch_stop_iteration`.
        break

      can_run_full_execution = (
          self._steps_per_execution_value == 1 or
          self._inferred_steps is None or
          self._inferred_steps - self._current_step >=
          self._steps_per_execution_value)

      if can_run_full_execution:
        self._step_increment = self._steps_per_execution_value - 1
        yield self._current_step
        self._current_step += self._steps_per_execution_value
      else:
        # Last partial execution.
        steps_remaining = self._inferred_steps - self._current_step
        self._steps_per_execution.assign(steps_remaining)
        self._step_increment = steps_remaining - 1
        yield self._current_step
        self._current_step += steps_remaining
        self._steps_per_execution.assign(self._steps_per_execution_value)

  @property
  def step_increment(self):
    """The number to increment the step for `on_batch_end` methods."""
    return self._step_increment

  @property
  def inferred_steps(self):
    """The inferred steps per epoch of the created `Dataset`.

    This will be `None` in the case where:

    (1) A `Dataset` of unknown cardinality was passed to the `DataHandler`, and
    (2) `steps_per_epoch` was not provided, and
    (3) The first epoch of iteration has not yet completed.

    Returns:
      The inferred steps per epoch of the created `Dataset`.
    """
    return self._inferred_steps

  @property
  def should_sync(self):
    # Catch OutOfRangeError for Datasets of unknown size.
    # This blocks until the batch has finished executing.
    # TODO(b/150292341): Allow multiple async steps here.
    return self._inferred_steps is None

  def _log_indefinite_training_warning(self):
    logging.warning("The training loop will run indefinitely since you have "
                    "set `steps_per_epoch=-1`. Please use batch-level "
                    "callbacks to save checkpoints or log training progress, "
                    "etc")

  def _infer_steps(self, steps, dataset):
    """Infers steps_per_epoch needed to loop through a dataset."""
    if steps == -1:
      self._log_indefinite_training_warning()
      return None

    if steps is not None:
      return steps

    adapter_steps = self._adapter.get_size()
    if adapter_steps is not None:
      return adapter_steps

    size = cardinality.cardinality(dataset)
    if size == cardinality.INFINITE and steps is None:
      raise ValueError(
          "When passing an infinitely repeating dataset, please specify a "
          "`steps_per_epoch` value so that epoch level "
          "callbacks continue to work. The value can be arbitrary, or a number "
          "that you think correctly defines the size of an epoch. "
          "Epoch-level callbacks will then be called at this interval.")
    if size >= 0:
      return size.numpy().item()
    return None

  @property
  def _samples(self):
    return self._adapter.get_samples()

  def _validate_data_handler(self):
    # TODO(b/152094471): Support this with DistIter.get_next_as_optional.
    if self._steps_per_execution_value > 1 and self._inferred_steps is None:
      raise ValueError(
          "Could not infer the size of the data. With "
          "`steps_per_execution > 1`, you must specify the number of steps "
          "to run.")


class _ClusterCoordinatorDataHandler(DataHandler):
  """A `DataHandler` that is compatible with `ClusterCoordinator`."""

  def __init__(self, x, y=None, **kwargs):
    if not isinstance(x, dataset_creator.DatasetCreator):
      x = self._convert_to_dataset_creator(x, y, **kwargs)

    super().__init__(x=x, **kwargs)

  def _convert_to_dataset_creator(self, x, y, **kwargs):
    """Converts non-tf.data.Dataset to `DatasetCreator` instances."""

    def _dataset_fn(input_context):
      del input_context
      data_adapter_cls = select_data_adapter(x, y)
      return data_adapter_cls(x=x, y=y, **kwargs).get_dataset()

    # This check is needed because types like `tf.data.Dataset` don't work with
    # PSS yet. So only apply this logic to the types we can support.
    if (isinstance(x, _get_tensor_types()) and
        isinstance(y, _get_tensor_types())):
      return dataset_creator.DatasetCreator(_dataset_fn)
    else:
      raise NotImplementedError(
          "Only `tf.keras.utils.experimental.DatasetCreator`, `tf.Tensor`, "
          "numpy arrays and pandas dataframes are supported types at this "
          "time.")

  def _configure_dataset_and_inferred_steps(self, strategy, x, steps_per_epoch,
                                            class_weight, distribute):
    if not isinstance(x, dataset_creator.DatasetCreator):
      raise TypeError("When using `ParameterServerStrategy`, `x` must be a "
                      "`DatasetCreator`.")

    def per_worker_dataset_fn():

      return strategy.distribute_datasets_from_function(
          x, options=x.input_options)

    self._dataset = self._model._cluster_coordinator.create_per_worker_dataset(  # pylint: disable=protected-access
        per_worker_dataset_fn)

    if steps_per_epoch == -1:
      self._inferred_steps = None
      self._log_indefinite_training_warning()
    else:
      self._inferred_steps = steps_per_epoch

  def sync(self):
    self._model._cluster_coordinator.join()  # pylint: disable=protected-access


def get_data_handler(*args, **kwargs):
  if getattr(kwargs["model"], "_cluster_coordinator", None):
    return _ClusterCoordinatorDataHandler(*args, **kwargs)
  return DataHandler(*args, **kwargs)


def _make_class_weight_map_fn(class_weight):
  """Applies class weighting to a `Dataset`.

  The `Dataset` is assumed to be in format `(x, y)` or `(x, y, sw)`, where
  `y` must be a single `Tensor`.

  Args:
    class_weight: A map where the keys are integer class ids and values are
      the class weights, e.g. `{0: 0.2, 1: 0.6, 2: 0.3}`

  Returns:
    A function that can be used with `tf.data.Dataset.map` to apply class
    weighting.
  """
  class_ids = list(sorted(class_weight.keys()))
  expected_class_ids = list(range(len(class_ids)))
  if class_ids != expected_class_ids:
    error_msg = (
        "Expected `class_weight` to be a dict with keys from 0 to one less "
        "than the number of classes, found {}").format(class_weight)
    raise ValueError(error_msg)

  class_weight_tensor = tensor_conversion.convert_to_tensor_v2_with_dispatch(
      [class_weight[int(c)] for c in class_ids]
  )

  def _class_weights_map_fn(*data):
    """Convert `class_weight` to `sample_weight`."""
    x, y, sw = unpack_x_y_sample_weight(data)

    if nest.is_nested(y):
      raise ValueError(
          "`class_weight` is only supported for Models with a single output.")

    if y.shape.rank > 2:
      raise ValueError("`class_weight` not supported for "
                       "3+ dimensional targets.")

    y_classes = smart_cond.smart_cond(
        y.shape.rank == 2 and backend.shape(y)[1] > 1,
        lambda: backend.argmax(y, axis=1),
        lambda: math_ops.cast(backend.reshape(y, (-1,)), dtypes.int64))

    cw = array_ops.gather_v2(class_weight_tensor, y_classes)
    if sw is not None:
      cw = math_ops.cast(cw, sw.dtype)
      sw, cw = expand_1d((sw, cw))
      # `class_weight` and `sample_weight` are multiplicative.
      sw = sw * cw
    else:
      sw = cw

    return x, y, sw

  return _class_weights_map_fn


def expand_1d(data):
  """Expands 1-dimensional `Tensor`s into 2-dimensional `Tensor`s."""

  def _expand_single_1d_tensor(t):
    # Leaves `CompositeTensor`s as-is.
    if (isinstance(t, ops.Tensor) and
        isinstance(t.shape, tensor_shape.TensorShape) and t.shape.rank == 1):
      return array_ops.expand_dims_v2(t, axis=-1)
    return t

  return nest.map_structure(_expand_single_1d_tensor, data)


def train_validation_split(arrays, validation_split):
  """Split arrays into train and validation subsets in deterministic order.

  The last part of data will become validation data.

  Args:
    arrays: Tensors to split. Allowed inputs are arbitrarily nested structures
      of Tensors and NumPy arrays.
    validation_split: Float between 0 and 1. The proportion of the dataset to
      include in the validation split. The rest of the dataset will be included
      in the training split.
  Returns:
    `(train_arrays, validation_arrays)`
  """

  def _can_split(t):
    tensor_types = _get_tensor_types()
    return isinstance(t, tensor_types) or t is None

  flat_arrays = nest.flatten(arrays)
  unsplitable = [type(t) for t in flat_arrays if not _can_split(t)]
  if unsplitable:
    raise ValueError(
        "`validation_split` is only supported for Tensors or NumPy "
        "arrays, found following types in the input: {}".format(unsplitable))

  if all(t is None for t in flat_arrays):
    return arrays, arrays

  first_non_none = None
  for t in flat_arrays:
    if t is not None:
      first_non_none = t
      break

  # Assumes all arrays have the same batch shape or are `None`.
  batch_dim = int(first_non_none.shape[0])
  split_at = int(math.floor(batch_dim * (1. - validation_split)))

  if split_at == 0 or split_at == batch_dim:
    raise ValueError(
        "Training data contains {batch_dim} samples, which is not sufficient "
        "to split it into a validation and training set as specified by "
        "`validation_split={validation_split}`. Either provide more data, or a "
        "different value for the `validation_split` argument." .format(
            batch_dim=batch_dim, validation_split=validation_split))

  def _split(t, start, end):
    if t is None:
      return t
    return t[start:end]

  train_arrays = nest.map_structure(
      functools.partial(_split, start=0, end=split_at), arrays)
  val_arrays = nest.map_structure(
      functools.partial(_split, start=split_at, end=batch_dim), arrays)

  return train_arrays, val_arrays


@keras_export("keras.utils.unpack_x_y_sample_weight", v1=[])
def unpack_x_y_sample_weight(data):
  """Unpacks user-provided data tuple.

  This is a convenience utility to be used when overriding
  `Model.train_step`, `Model.test_step`, or `Model.predict_step`.
  This utility makes it easy to support data of the form `(x,)`,
  `(x, y)`, or `(x, y, sample_weight)`.

  Standalone usage:

  >>> features_batch = tf.ones((10, 5))
  >>> labels_batch = tf.zeros((10, 5))
  >>> data = (features_batch, labels_batch)
  >>> # `y` and `sample_weight` will default to `None` if not provided.
  >>> x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
  >>> sample_weight is None
  True

  Example in overridden `Model.train_step`:

  ```python
  class MyModel(tf.keras.Model):

    def train_step(self, data):
      # If `sample_weight` is not provided, all samples will be weighted
      # equally.
      x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

      with tf.GradientTape() as tape:
        y_pred = self(x, training=True)
        loss = self.compiled_loss(
          y, y_pred, sample_weight, regularization_losses=self.losses)
        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

      self.compiled_metrics.update_state(y, y_pred, sample_weight)
      return {m.name: m.result() for m in self.metrics}
  ```

  Args:
    data: A tuple of the form `(x,)`, `(x, y)`, or `(x, y, sample_weight)`.

  Returns:
    The unpacked tuple, with `None`s for `y` and `sample_weight` if they are not
    provided.
  """
  if not isinstance(data, tuple):
    return (data, None, None)
  elif len(data) == 1:
    return (data[0], None, None)
  elif len(data) == 2:
    return (data[0], data[1], None)
  elif len(data) == 3:
    return (data[0], data[1], data[2])
  else:
    error_msg = ("Data is expected to be in format `x`, `(x,)`, `(x, y)`, "
                 "or `(x, y, sample_weight)`, found: {}").format(data)
    raise ValueError(error_msg)


@keras_export("keras.utils.pack_x_y_sample_weight", v1=[])
def pack_x_y_sample_weight(x, y=None, sample_weight=None):
  """Packs user-provided data into a tuple.

  This is a convenience utility for packing data into the tuple formats
  that `Model.fit` uses.

  Standalone usage:

  >>> x = tf.ones((10, 1))
  >>> data = tf.keras.utils.pack_x_y_sample_weight(x)
  >>> isinstance(data, tf.Tensor)
  True
  >>> y = tf.ones((10, 1))
  >>> data = tf.keras.utils.pack_x_y_sample_weight(x, y)
  >>> isinstance(data, tuple)
  True
  >>> x, y = data

  Args:
    x: Features to pass to `Model`.
    y: Ground-truth targets to pass to `Model`.
    sample_weight: Sample weight for each element.

  Returns:
    Tuple in the format used in `Model.fit`.
  """
  if y is None:
    # For single x-input, we do no tuple wrapping since in this case
    # there is no ambiguity. This also makes NumPy and Dataset
    # consistent in that the user does not have to wrap their Dataset
    # data in an unecessary tuple
    if not nest.is_nested(x):
      return x
    else:
      return (x,)
  elif sample_weight is None:
    return (x, y)
  else:
    return (x, y, sample_weight)


def single_batch_iterator(strategy,
                          x,
                          y=None,
                          sample_weight=None,
                          class_weight=None):
  """Creates a single-batch dataset."""
  x, y, sample_weight = _process_tensorlike((x, y, sample_weight))
  if y is None:
    data = (x,)
  elif sample_weight is None:
    data = (x, y)
  else:
    data = (x, y, sample_weight)

  _check_data_cardinality(data)
  dataset = dataset_ops.DatasetV2.from_tensors(data)
  if class_weight:
    dataset = dataset.map(_make_class_weight_map_fn(class_weight))
  dataset = strategy.experimental_distribute_dataset(dataset)
  return iter(dataset)


def _check_data_cardinality(data):
  num_samples = set(int(i.shape[0]) for i in nest.flatten(data))
  if len(num_samples) > 1:
    msg = "Data cardinality is ambiguous:\n"
    for label, single_data in zip(["x", "y", "sample_weight"], data):
      msg += "  {} sizes: {}\n".format(
          label, ", ".join(str(i.shape[0]) for i in nest.flatten(single_data)))
    msg += "Make sure all arrays contain the same number of samples."
    raise ValueError(msg)


def _get_tensor_types():
  try:
    import pandas as pd  # pylint: disable=g-import-not-at-top

    return (ops.Tensor, np.ndarray, pd.Series, pd.DataFrame)
  except ImportError:
    return (ops.Tensor, np.ndarray)


def _is_scipy_sparse(x):
  try:
    from scipy.sparse import issparse  # pylint: disable=g-import-not-at-top

    return issparse(x)
  except ImportError:
    return False


def _scipy_sparse_to_sparse_tensor(t):
  """Converts a SciPy sparse matrix to a SparseTensor."""
  sparse_coo = t.tocoo()
  row, col = sparse_coo.row, sparse_coo.col
  data, shape = sparse_coo.data, sparse_coo.shape
  if issubclass(data.dtype.type, np.floating):
    data = data.astype(backend.floatx())
  indices = np.concatenate(
      (np.expand_dims(row, axis=1), np.expand_dims(col, axis=1)), axis=1)
  return sparse_tensor.SparseTensor(indices, data, shape)


def _is_distributed_dataset(ds):
  return isinstance(ds, input_lib.DistributedDatasetInterface)
