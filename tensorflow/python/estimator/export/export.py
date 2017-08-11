# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Configuration and utilities for receiving inputs at serving time."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time

import six

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat


_SINGLE_FEATURE_DEFAULT_NAME = 'feature'
_SINGLE_RECEIVER_DEFAULT_NAME = 'input'


class ServingInputReceiver(collections.namedtuple('ServingInputReceiver',
                                                  ['features',
                                                   'receiver_tensors'])):
  """A return type for a serving_input_receiver_fn.

  The expected return values are:
    features: A dict of string to `Tensor` or `SparseTensor`, specifying the
      features to be passed to the model.
    receiver_tensors: a `Tensor`, or a dict of string to `Tensor`, specifying
      input nodes where this receiver expects to be fed.  Typically, this is a
      single placeholder expecting serialized `tf.Example` protos.
  """
  # TODO(soergel): add receiver_alternatives when supported in serving.

  def __new__(cls, features, receiver_tensors):
    if features is None:
      raise ValueError('features must be defined.')
    if not isinstance(features, dict):
      features = {_SINGLE_FEATURE_DEFAULT_NAME: features}
    for name, tensor in features.items():
      if not isinstance(name, six.string_types):
        raise ValueError('feature keys must be strings: {}.'.format(name))
      if not (isinstance(tensor, ops.Tensor)
              or isinstance(tensor, sparse_tensor.SparseTensor)):
        raise ValueError(
            'feature {} must be a Tensor or SparseTensor.'.format(name))

    if receiver_tensors is None:
      raise ValueError('receiver_tensors must be defined.')
    if not isinstance(receiver_tensors, dict):
      receiver_tensors = {_SINGLE_RECEIVER_DEFAULT_NAME: receiver_tensors}
    for name, tensor in receiver_tensors.items():
      if not isinstance(name, six.string_types):
        raise ValueError(
            'receiver_tensors keys must be strings: {}.'.format(name))
      if not isinstance(tensor, ops.Tensor):
        raise ValueError(
            'receiver_tensor {} must be a Tensor.'.format(name))

    return super(ServingInputReceiver, cls).__new__(
        cls, features=features, receiver_tensors=receiver_tensors)


def build_parsing_serving_input_receiver_fn(feature_spec,
                                            default_batch_size=None):
  """Build a serving_input_receiver_fn expecting fed tf.Examples.

  Creates a serving_input_receiver_fn that expects a serialized tf.Example fed
  into a string placeholder.  The function parses the tf.Example according to
  the provided feature_spec, and returns all parsed Tensors as features.

  Args:
    feature_spec: a dict of string to `VarLenFeature`/`FixedLenFeature`.
    default_batch_size: the number of query examples expected per batch.
        Leave unset for variable batch size (recommended).

  Returns:
    A serving_input_receiver_fn suitable for use in serving.
  """
  def serving_input_receiver_fn():
    """An input_fn that expects a serialized tf.Example."""
    serialized_tf_example = array_ops.placeholder(dtype=dtypes.string,
                                                  shape=[default_batch_size],
                                                  name='input_example_tensor')
    receiver_tensors = {'examples': serialized_tf_example}
    features = parsing_ops.parse_example(serialized_tf_example, feature_spec)
    return ServingInputReceiver(features, receiver_tensors)

  return serving_input_receiver_fn


def build_raw_serving_input_receiver_fn(features, default_batch_size=None):
  """Build a serving_input_receiver_fn expecting feature Tensors.

  Creates an serving_input_receiver_fn that expects all features to be fed
  directly.

  Args:
    features: a dict of string to `Tensor`.
    default_batch_size: the number of query examples expected per batch.
        Leave unset for variable batch size (recommended).

  Returns:
    A serving_input_receiver_fn.
  """
  def serving_input_receiver_fn():
    """A serving_input_receiver_fn that expects features to be fed directly."""
    receiver_tensors = {}
    for name, t in features.items():
      shape_list = t.get_shape().as_list()
      shape_list[0] = default_batch_size
      shape = tensor_shape.TensorShape(shape_list)

      # Reuse the feature tensor name for the placeholder, excluding the index
      placeholder_name = t.name.split(':')[0]
      receiver_tensors[name] = array_ops.placeholder(dtype=t.dtype,
                                                     shape=shape,
                                                     name=placeholder_name)
    # TODO(b/34885899): remove the unnecessary copy
    # The features provided are simply the placeholders, but we defensively copy
    # the dict because it may be mutated.
    return ServingInputReceiver(receiver_tensors, receiver_tensors.copy())

  return serving_input_receiver_fn


### Below utilities are specific to SavedModel exports.


def build_all_signature_defs(receiver_tensors, export_outputs):
  """Build `SignatureDef`s for all export outputs."""
  if not isinstance(receiver_tensors, dict):
    receiver_tensors = {'receiver': receiver_tensors}
  if export_outputs is None or not isinstance(export_outputs, dict):
    raise ValueError('export_outputs must be a dict.')

  signature_def_map = {
      '{}'.format(output_key or 'None'):
      export_output.as_signature_def(receiver_tensors)
      for output_key, export_output in export_outputs.items()}

  return signature_def_map


# When we create a timestamped directory, there is a small chance that the
# directory already exists because another worker is also writing exports.
# In this case we just wait one second to get a new timestamp and try again.
# If this fails several times in a row, then something is seriously wrong.
MAX_DIRECTORY_CREATION_ATTEMPTS = 10


def get_timestamped_export_dir(export_dir_base):
  """Builds a path to a new subdirectory within the base directory.

  Each export is written into a new subdirectory named using the
  current time.  This guarantees monotonically increasing version
  numbers even across multiple runs of the pipeline.
  The timestamp used is the number of seconds since epoch UTC.

  Args:
    export_dir_base: A string containing a directory to write the exported
        graph and checkpoints.
  Returns:
    The full path of the new subdirectory (which is not actually created yet).

  Raises:
    RuntimeError: if repeated attempts fail to obtain a unique timestamped
      directory name.
  """
  attempts = 0
  while attempts < MAX_DIRECTORY_CREATION_ATTEMPTS:
    export_timestamp = int(time.time())

    export_dir = os.path.join(
        compat.as_bytes(export_dir_base),
        compat.as_bytes(str(export_timestamp)))
    if not gfile.Exists(export_dir):
      # Collisions are still possible (though extremely unlikely): this
      # directory is not actually created yet, but it will be almost
      # instantly on return from this function.
      return export_dir
    time.sleep(1)
    attempts += 1
    logging.warn(
        'Export directory {} already exists; retrying (attempt {}/{})'.format(
            export_dir, attempts, MAX_DIRECTORY_CREATION_ATTEMPTS))
  raise RuntimeError('Failed to obtain a unique export directory name after '
                     '{} attempts.'.format(MAX_DIRECTORY_CREATION_ATTEMPTS))


def get_temp_export_dir(timestamped_export_dir):
  """Builds a directory name based on the argument but starting with 'temp-'.

  This relies on the fact that TensorFlow Serving ignores subdirectories of
  the base directory that can't be parsed as integers.

  Args:
    timestamped_export_dir: the name of the eventual export directory, e.g.
      /foo/bar/<timestamp>

  Returns:
    A sister directory prefixed with 'temp-', e.g. /foo/bar/temp-<timestamp>.
  """
  (dirname, basename) = os.path.split(timestamped_export_dir)
  temp_export_dir = os.path.join(
      compat.as_bytes(dirname),
      compat.as_bytes('temp-{}'.format(basename)))
  return temp_export_dir

