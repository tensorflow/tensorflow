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
"""Experimental API for TensorFlow's "Eager" mode of execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import threading

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.platform import app
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib

GRAPH_MODE = 0
EAGER_MODE = 1

# Default execution mode.
_default_mode = GRAPH_MODE


# TODO(agarwal): better name ?
class _EagerContext(threading.local):
  """Thread local eager context."""

  def __init__(self):
    super(_EagerContext, self).__init__()
    self.device_index = -1
    self.mode = _default_mode
    self.scope_name = ""
    self.recording_summaries = False


# TODO(agarwal): rename to EagerContext / EagerRuntime ?
# TODO(agarwal): consider keeping the corresponding Graph here.
class Context(object):
  """Environment in which eager operations execute."""

  def __init__(self):
    self._eager_context = _EagerContext()
    # Create a handle
    opts = pywrap_tensorflow.TF_NewSessionOptions(
        target=compat.as_bytes(""), config=None)
    with errors.raise_exception_on_not_ok_status() as status:
      self._handle = pywrap_tensorflow.TFE_NewContext(opts, status)
      pywrap_tensorflow.TF_DeleteSessionOptions(opts)
    # Store list of devices
    self._devices = []
    with errors.raise_exception_on_not_ok_status() as status:
      device_list = pywrap_tensorflow.TFE_ContextListDevices(
          self._handle, status)
    try:
      for i in range(pywrap_tensorflow.TF_DeviceListCount(device_list)):
        with errors.raise_exception_on_not_ok_status() as status:
          dev_name = pywrap_tensorflow.TF_DeviceListName(device_list, i, status)
        self._devices.append(pydev.canonical_name(dev_name))
    finally:
      pywrap_tensorflow.TF_DeleteDeviceList(device_list)

    self._summary_writer_resource = None

  def __del__(self):
    try:
      if self._handle is not None:
        with errors.raise_exception_on_not_ok_status() as status:
          pywrap_tensorflow.TFE_DeleteContext(self._handle, status)
    except (AttributeError, TypeError):
      # Sometimes deletion during program shutdown throws exception as other
      # modules are no longer available.
      pass

  def __str__(self):
    lines = [
        "Eager TensorFlow environment with %d devices" % (len(self._devices))
    ]
    for i, d in enumerate(self._devices):
      lines.append("   Device %d: %s" % (i, d))
    return "\n".join(lines)

  @tf_contextlib.contextmanager
  def _mode(self, mode):
    ctx = self._eager_context
    old_mode = ctx.mode
    ctx.mode = mode
    try:
      yield
    finally:
      ctx.mode = old_mode

  def in_graph_mode(self):
    """Returns True if current thread is in GRAPH mode."""
    return self._eager_context.mode == GRAPH_MODE

  def in_eager_mode(self):
    """Returns True if current thread is in EAGER mode."""
    return self._eager_context.mode == EAGER_MODE

  @property
  def scope_name(self):
    """Returns scope name for the current thread."""
    return self._eager_context.scope_name

  @scope_name.setter
  def scope_name(self, s):
    """Sets scope name for the current thread."""
    self._eager_context.scope_name = s

  @property
  def summary_writer_resource(self):
    """Returns summary writer resource."""
    return self._summary_writer_resource

  @summary_writer_resource.setter
  def summary_writer_resource(self, resource):
    """Sets summary writer resource."""
    self._summary_writer_resource = resource

  @property
  def recording_summaries(self):
    """Returns True if recording summaries is enabled in current thread.."""
    return self._eager_context.recording_summaries

  @recording_summaries.setter
  def recording_summaries(self, val):
    """Enables recording summaries is enabled in current thread.."""
    self._eager_context.recording_summaries = val

  # TODO(agarwal): remove?
  @property
  def _device_index(self):
    return self._eager_context.device_index

  # TODO(agarwal): remove?
  @_device_index.setter
  def _device_index(self, val):
    self._eager_context.device_index = val

  @property
  def device_name(self):
    """Returns the device name for the current thread."""
    index = self._device_index
    return "CPU:0" if index < 0 else self._devices[index]

  def devices(self):
    """List of the names of devices available to execute operations."""
    return self._devices

  def num_gpus(self):
    """The number of GPUs available to execute operations."""
    # TODO(ashankar): Use TF_DeviceListType to count GPU devices.
    return len(self._devices) - 1

  def as_default(self):
    """Returns a context manager to make self the default for this thread."""
    return _default_context_stack.get_controller(self)


# TODO(agarwal): make this public and move into its own file.
class _DefaultStack(threading.local):
  """A thread-local stack of objects for providing implicit defaults."""

  def __init__(self):
    super(_DefaultStack, self).__init__()
    self._enforce_nesting = True
    self.stack = []

  def get_default(self):
    return self.stack[-1] if len(self.stack) >= 1 else None

  def reset(self):
    self.stack = []

  def is_cleared(self):
    return not self.stack

  @property
  def enforce_nesting(self):
    return self._enforce_nesting

  @enforce_nesting.setter
  def enforce_nesting(self, value):
    self._enforce_nesting = value

  @tf_contextlib.contextmanager
  def get_controller(self, default):
    """A context manager for manipulating a default stack."""
    try:
      self.stack.append(default)
      yield default
    finally:
      if self._enforce_nesting:
        if self.stack[-1] is not default:
          raise AssertionError(
              "Nesting violated for default stack of %s objects" %
              type(default))
        self.stack.pop()
      else:
        self.stack.remove(default)


class _DefaultContextStack(_DefaultStack):  # pylint: disable=protected-access
  """A thread-local stack of Context objects."""

  def __init__(self):
    super(_DefaultContextStack, self).__init__()
    self._global_default_context = None

  def get_default(self):
    """Returns a thread local object if present, else a global default."""
    return (super(_DefaultContextStack, self).get_default() or
            self.global_default_context)

  @property
  def global_default_context(self):
    if self._global_default_context is None:
      self._global_default_context = Context()
    return self._global_default_context

  def reset(self):
    super(_DefaultContextStack, self).reset()
    self._global_default_context = None


_default_context_stack = _DefaultContextStack()


def get_default_context():
  """Returns a default Context object."""
  return _default_context_stack.get_default()


# TODO(agarwal): switch users to get_default_context and get rid of this
# function.
def context():
  return get_default_context()


def in_graph_mode():
  """Returns True if current thread is in GRAPH mode for default context."""
  return get_default_context().in_graph_mode()


def in_eager_mode():
  """Returns True if current thread is in EAGER mode for default context."""
  return get_default_context().in_eager_mode()


def graph_mode():
  """Context-manager to enable GRAPH mode for current thread."""
  return get_default_context()._mode(GRAPH_MODE)  # pylint: disable=protected-access


def eager_mode():
  """Context-manager to enable EAGER mode for current thread."""
  return get_default_context()._mode(EAGER_MODE)  # pylint: disable=protected-access


# TODO(agarwal): get rid of this and use ops.name_scope instead.
@contextlib.contextmanager
def namescope(name):
  """ContextManager for creating hierarchical name scopes."""
  ctx = get_default_context()
  old_name = ctx.scope_name
  ctx.scope_name = "%s/%s" % (old_name, name) if old_name else name
  try:
    yield
  finally:
    ctx.scope_name = old_name


def scope_name():
  """Name of the current scope."""
  return get_default_context().scope_name


@tf_contextlib.contextmanager
def device(name):
  """Context-manager to force placement of operations and Tensors on a device.

  For example:
  ```python
  with tfe.device('gpu:0'):
    with tfe.device('cpu:0'):
      shape = tfe.Tensor([], dtype=tf.int32)
    x = ops.truncated_normal(shape, tf.float32)
  ```
  will ensure that the `shape` Tensor is on CPU but the `truncated_normal`
  operation
  runs on GPU 0.

  Args:
    name: Name of the device (see get_default_context().devices()), or None to
      enable automatic placement.

  Yields:
    Nothing.

  Raises:
    ValueError: If name does not correspond to a valid device.
  """
  device_index = -1
  ctx = get_default_context()
  if name is not None:
    name = pydev.canonical_name(name)
    all_devices = ctx.devices()
    for i, d in enumerate(all_devices):
      # TODO(ashankar): This will change when we have distributed support.
      # At that point, should not look for a string suffix but be able to
      # do a full string comparison.
      if d.endswith(name):
        device_index = i
        break
    if device_index < 0:
      raise ValueError("device {} does not match the available devices ({})".
                       format(name, all_devices))
  old_device_index = ctx._device_index  # pylint: disable=protected-access
  try:
    ctx._device_index = device_index  # pylint: disable=protected-access
    yield
  finally:
    ctx._device_index = old_device_index  # pylint: disable=protected-access


@contextlib.contextmanager
def record_summaries():
  """Context-manager to enable recording of summaries."""
  ctx = get_default_context()
  old = ctx.recording_summaries
  ctx.recording_summaries = True
  try:
    yield
  finally:
    ctx.recording_summaries = old


def should_record_summary():
  """True if a summary should be recorded now."""
  c = get_default_context()
  return c.recording_summaries and c.summary_writer_resource is not None


def run(main=None, argv=None):
  """Runs the program with an optional 'main' function and 'argv' list.

  The program will run with eager execution enabled.

  Args:
    main: the main function to run
    argv: the arguments to pass to it
  """
  enable_eager_execution()
  app.run(main, argv)


# TODO(apassos): This should not be a part of the public API.
def enable_eager_execution():
  """Enables, for the rest of the lifetime of this program, eager execution.

  If not called immediately on startup risks creating breakage and bugs.
  """
  global _default_mode
  assert _default_mode == GRAPH_MODE
  _default_mode = EAGER_MODE
