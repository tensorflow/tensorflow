# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Tools to work with checkpoints."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import gfile
from tensorflow.python.training import saver
from tensorflow.python.training import training as train


def _get_checkpoint_filename(filepattern):
  """Returns checkpoint filename given directory or specific filepattern."""
  if gfile.IsDirectory(filepattern):
    return saver.latest_checkpoint(filepattern)
  return filepattern


def _load_checkpoint(filepattern):
  """Returns CheckpointReader for latest checkpoint.

  Args:
    filepattern: Directory with checkpoints file or path to checkpoint.

  Returns:
    `CheckpointReader` object.

  Raises:
    ValueError: if checkpoint_dir doesn't have 'checkpoint' file or checkpoints.
  """
  filename = _get_checkpoint_filename(filepattern)
  if filename is None:
    raise ValueError("Couldn't find 'checkpoint' file or checkpoints in "
                     "given directory %s" % filepattern)
  return train.NewCheckpointReader(filename)


def load_variable(checkpoint_dir, name):
  """Returns a Tensor with the contents of the given variable in the checkpoint.

  Args:
    checkpoint_dir: Directory with checkpoints file or path to checkpoint.
    name: Name of the tensor to return.

  Returns:
    `Tensor` object.
  """
  # TODO(b/29227106): Fix this in the right place and remove this.
  if name.endswith(":0"):
    name = name[:-2]
  reader = _load_checkpoint(checkpoint_dir)
  return reader.get_tensor(name)


def list_variables(checkpoint_dir):
  """Returns list of all variables in the latest checkpoint.

  Args:
    checkpoint_dir: Directory with checkpoints file or path to checkpoint.

  Returns:
    List of tuples `(name, shape)`.
  """
  reader = _load_checkpoint(checkpoint_dir)
  variable_map = reader.get_variable_to_shape_map()
  names = sorted(variable_map.keys())
  result = []
  for name in names:
    result.append((name, variable_map[name]))
  return result
