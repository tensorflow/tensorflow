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

from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import saver as _saver


def _init_from_checkpoint(self, *args, **kwargs):
  """Overrides default init by loading value from checkpoint."""
  # pylint: disable=protected-access
  self._old_init(*args, **kwargs)
  ckpt_name = self._map_func(self._shared_name)
  if ckpt_name not in self._ckpt_var_cache:
    raise errors.NotFoundError(None, None,
                               "%s not found in checkpoint" % ckpt_name)

  val = self._ckpt_var_cache.get(ckpt_name, None)
  if val is not None:
    self.assign(val)
    # Avoid assigning for the second time.
    self._ckpt_var_cache[ckpt_name] = None
  # pylint: enable=protected-access


@contextlib.contextmanager
def restore_variables_on_create(save_path, map_func=None):
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
    map_func: A function that given the variable name as argument
        and returns a variable name in checkpoint for restore. If
        None, use the variable with the same name in checkpoint to restore.
        It's an error that the mapped variable name doesn't exist in
        checkpoint.

  Yields:
    Nothing.

  Raises:
    NotFoundError: If the variable is not found in checkpoint.
    ValueError: If not used in eager mode or map_func is not callable.
  """
  if not context.executing_eagerly():
    raise ValueError(
        "Currently, restore_variables_on_create can only be used with "
        "eager execution enabled.")
  if save_path:
    if map_func is None:
      map_func_wrapper = lambda self, x: x
    else:
      if not callable(map_func):
        raise ValueError("map_func must be callable.")
      map_func_wrapper = lambda self, x: map_func(x)

    ckpt_var_cache = {}
    reader = checkpoint_utils.load_checkpoint(save_path)
    for k, _ in checkpoint_utils.list_variables(save_path):
      ckpt_var_cache[k] = reader.get_tensor(k)

    old_init = getattr(resource_variable_ops.ResourceVariable,
                       "_init_from_args", None)
    assert old_init, "ResourceVariable misses _init_from_args method."
    setattr(resource_variable_ops.ResourceVariable, "_init_from_args",
            _init_from_checkpoint)
    setattr(resource_variable_ops.ResourceVariable, "_old_init", old_init)
    setattr(resource_variable_ops.ResourceVariable, "_map_func",
            map_func_wrapper)
    setattr(resource_variable_ops.ResourceVariable, "_ckpt_var_cache",
            ckpt_var_cache)
  try:
    yield
  except Exception as e:
    raise e
  finally:
    if save_path:
      setattr(resource_variable_ops.ResourceVariable, "_init_from_args",
              old_init)
      setattr(resource_variable_ops.ResourceVariable, "_old_init", None)
      setattr(resource_variable_ops.ResourceVariable, "_map_func", None)
      setattr(resource_variable_ops.ResourceVariable, "_ckpt_var_cache", None)


class Saver(object):
  """A tf.train.Saver adapter for use when eager execution is enabled.

  `Saver`'s name-based checkpointing strategy is fragile. Please switch to
  `tf.train.Checkpoint` or `tf.keras.Model.save_weights`, which perform a more
  robust object-based saving. These APIs will load checkpoints written by
  `Saver`.
  """

  def __init__(self, var_list):
    """A  tf.train.Saver adapter for use when eager execution is enabled.

      The API, and on-disk format, mimic tf.train.Saver except that no
      Session is needed.

    Args:
      var_list: The list of variables that will be saved and restored. Either a
        list of `tf.Variable` objects, or a dictionary mapping names to
        `tf.Variable` objects.

    Raises:
      RuntimeError: if invoked when eager execution has not been enabled.
    """
    if not context.executing_eagerly():
      raise RuntimeError("tfe.Saver can only be used when eager "
                         "execution is enabled. Use tf.train.Saver when "
                         "building graphs.")
    self._saver = _saver.Saver(var_list=var_list)

  def save(self, file_prefix, global_step=None):
    """Saves variables.

    Args:
      file_prefix: Path prefix of files created for the checkpoint.
      global_step: If provided the global step number is appended to file_prefix
        to create the checkpoint filename. The optional argument can be a
        Tensor, a Variable, or an integer.

    Returns:
      A string: prefix of filenames created for the checkpoint. This may be
       an extension of file_prefix that is suitable to pass as an argument
       to a subsequent call to `restore()`.
    """
    with ops.device("/device:CPU:0"):
      return self._saver.save(
          None, file_prefix, write_meta_graph=False, global_step=global_step)

  def restore(self, file_prefix):
    """Restores previously saved variables.

    Args:
      file_prefix: Path prefix where parameters were previously saved.
        Typically obtained from a previous `save()` call, or from
        `tf.train.latest_checkpoint`.
    """
    with ops.device("/device:CPU:0"):
      self._saver.restore(None, file_prefix)


def get_optimizer_variables(optimizer):
  """Returns a list of variables for the given `tf.train.Optimizer`.

  Equivalent to `optimizer.variables()`.

  Args:
    optimizer: An instance of `tf.train.Optimizer` which has created variables
      (typically after a call to `Optimizer.minimize`).
  Returns:
    A list of variables which have been created by the `Optimizer`.
  """
  return optimizer.variables()
