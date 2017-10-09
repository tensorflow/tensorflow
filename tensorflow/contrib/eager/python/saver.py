"""Saver for eager mode TensorFlow."""
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import saver as _saver


def _init_from_checkpoint(self, *args, **kwargs):
  """Overrides default init by loading value from checkpoint."""
  self.old_init(*args, **kwargs)
  # pylint: disable=protected-access
  if self._shared_name not in self.ckpt_var_cache:
    raise errors.NotFoundError(None, None,
                               "%s not found in checkpoint" % self._shared_name)

  val = self.ckpt_var_cache[self._shared_name]
  if val is not None:
    self.assign(self.ckpt_var_cache[self._shared_name])
    # Avoid assigning for the second time.
    self.ckpt_var_cache[self._shared_name] = None
  # pylint: enable=protected-access


@contextlib.contextmanager
def restore_variables_on_create(save_path):
  """ContextManager that restores variables on creation.

    When save_path is None (e.g. No checkpoint), does nothing.
    Otherwise, it preloads all values from checkpoint. When the
    corresponding variable is first created, it assigns the checkpoint
    value to the variable.

    ```python
    with restore_variables_on_create(
        tf.train.latest_checkpoint(checkpoint_dir)):
    ```

  Args:
    save_path: The checkpoint file prefix.

  Yields:
    Nothing.

  Raises:
    NotFoundError: If the variable is not found in checkpoint.
  """
  if save_path:
    ckpt_var_cache = dict()
    reader = checkpoint_utils.load_checkpoint(save_path)
    for k, _ in checkpoint_utils.list_variables(save_path):
      ckpt_var_cache[k] = reader.get_tensor(k)

    old_init = getattr(
        resource_variable_ops.ResourceVariable, "_init_from_args", None)
    assert old_init, "ResourceVariable misses _init_from_args method."
    setattr(resource_variable_ops.ResourceVariable, "_init_from_args",
            _init_from_checkpoint)
    setattr(resource_variable_ops.ResourceVariable, "old_init", old_init)
    setattr(resource_variable_ops.ResourceVariable, "ckpt_var_cache",
            ckpt_var_cache)
  try:
    yield
  except Exception as e:
    raise e
  finally:
    if save_path:
      setattr(resource_variable_ops.ResourceVariable, "_init_from_args",
              old_init)
      setattr(resource_variable_ops.ResourceVariable, "old_init", None)
      setattr(resource_variable_ops.ResourceVariable, "ckpt_var_cache", None)


class Saver(object):
  """A simple tf.train.Saver adapter for eager mode.

    save and restore API are similar to the tf.train.Saver, except that
    session is not needed.

  Args:
    var_list: A list of variables.
  """

  def __init__(self, var_list):
    self._saver = _saver.Saver(var_list=var_list)

  def save(self, save_path, global_step=None):
    """Saves variables.

    Args:
      save_path: See save method in tf.train.Saver.
      global_step: See save method in tf.train.Saver.

    Returns:
      See save method in tf.train.Saver.
    """
    with ops.device("/device:CPU:0"):
      return self._saver.save(None, save_path, write_meta_graph=False,
                              global_step=global_step)

  def restore(self, save_path):
    """Restores previously saved variables.

    Args:
      save_path: See restore method in tf.train.Saver.
    """
    with ops.device("/device:CPU:0"):
      self._saver.restore(None, save_path)

