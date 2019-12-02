# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Utility functions shared between SavedModel saving/loading implementations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import types

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.training.tracking import layer_utils as trackable_layer_utils
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.lazy_loader import LazyLoader


# pylint:disable=g-inconsistent-quotes
training_lib = LazyLoader(
    "training_lib", globals(),
    "tensorflow.python.keras.engine.training")
# pylint:enable=g-inconsistent-quotes


def use_wrapped_call(layer, call_fn, default_training_value=None,
                     return_method=False):
  """Creates fn that adds the losses returned by call_fn & returns the outputs.

  Args:
    layer: A Keras layer object
    call_fn: tf.function that takes layer inputs (and possibly a training arg),
      and returns a tuple of (outputs, list of losses).
    default_training_value: Default value of the training kwarg. If `None`, the
      default is `K.learning_phase()`.
    return_method: Whether to return a method bound to the layer.

  Returns:
    function that calls call_fn and returns the outputs. Losses returned by
    call_fn are added to the layer losses.
  """
  expects_training_arg = layer_uses_training_bool(layer)
  if hasattr(call_fn, 'original_call'):  # call_fn is a LayerCall object
    original_call = call_fn.original_call
    # In Python 3, callable objects are not compatible with inspect.getargspec
    call_fn = call_fn.__call__
  else:
    original_call = call_fn
  fn, arg_spec = maybe_add_training_arg(
      original_call, call_fn, expects_training_arg, default_training_value)

  def return_outputs_and_add_losses(*args, **kwargs):
    """Returns the outputs from the call_fn, and adds the losses."""
    inputs_arg_index = 1 if return_method else 0
    inputs = args[inputs_arg_index]
    args = args[inputs_arg_index + 1:]
    outputs, losses = fn(inputs, *args, **kwargs)
    layer.add_loss(losses, inputs)
    return outputs

  decorated = tf_decorator.make_decorator(
      target=call_fn,
      decorator_func=return_outputs_and_add_losses,
      decorator_argspec=arg_spec)

  if return_method:
    return types.MethodType(decorated, layer)
  else:
    return decorated


def layer_uses_training_bool(layer):
  """Returns whether this layer or any of its children uses the training arg."""
  if layer._expects_training_arg:  # pylint: disable=protected-access
    return True
  visited = {layer}
  to_visit = list_all_layers(layer)
  while to_visit:
    layer = to_visit.pop()
    if layer in visited:
      continue
    if layer._expects_training_arg:  # pylint: disable=protected-access
      return True
    visited.add(layer)
    to_visit.extend(list_all_layers(layer))
  return False


def list_all_layers(obj):
  if isinstance(obj, training_lib.Model):
    return obj.layers
  else:
    return list(
        trackable_layer_utils.filter_empty_layer_containers(obj._layers))  # pylint: disable=protected-access


def list_all_layers_and_sublayers(obj):
  s = set([obj])
  s.update(itertools.chain.from_iterable(
      list_all_layers_and_sublayers(layer) for layer in list_all_layers(obj)))
  return s


def maybe_add_training_arg(
    original_call, wrapped_call, expects_training_arg, default_training_value):
  """Decorate call and optionally adds training argument.

  If a layer expects a training argument, this function ensures that 'training'
  is present in the layer args or kwonly args, with the default training value.

  Args:
    original_call: Original call function.
    wrapped_call: Wrapped call function.
    expects_training_arg: Whether to include 'training' argument.
    default_training_value: Default value of the training kwarg to include in
      the arg spec. If `None`, the default is `K.learning_phase()`.

  Returns:
    Tuple of (
      function that calls `wrapped_call` and sets the training arg,
      Argspec of returned function or `None` if the argspec is unchanged)
  """
  if not expects_training_arg:
    return wrapped_call, None

  def wrap_with_training_arg(*args, **kwargs):
    """Wrap the `wrapped_call` function, and set training argument."""
    training_arg_index = get_training_arg_index(original_call)
    training = get_training_arg(training_arg_index, args, kwargs)
    if training is None:
      training = default_training_value or K.learning_phase()

    args = list(args)
    kwargs = kwargs.copy()

    def replace_training_and_call(training):
      set_training_arg(training, training_arg_index, args, kwargs)
      return wrapped_call(*args, **kwargs)

    return tf_utils.smart_cond(
        training,
        lambda: replace_training_and_call(True),
        lambda: replace_training_and_call(False))

  # Create arg spec for decorated function. If 'training' is not defined in the
  # args of the original arg spec, then add it to kwonlyargs.
  arg_spec = tf_inspect.getfullargspec(original_call)
  defaults = list(arg_spec.defaults) if arg_spec.defaults is not None else []

  kwonlyargs = arg_spec.kwonlyargs
  kwonlydefaults = arg_spec.kwonlydefaults or {}
  # Add training arg if it does not exist, or set the default training value.
  if 'training' not in arg_spec.args:
    kwonlyargs.append('training')
    kwonlydefaults['training'] = default_training_value
  else:
    index = arg_spec.args.index('training')
    training_default_index = len(arg_spec.args) - index
    if (arg_spec.defaults and
        len(arg_spec.defaults) >= training_default_index and
        defaults[-training_default_index] is None):
      defaults[-training_default_index] = default_training_value

  decorator_argspec = tf_inspect.FullArgSpec(
      args=arg_spec.args,
      varargs=arg_spec.varargs,
      varkw=arg_spec.varkw,
      defaults=defaults,
      kwonlyargs=kwonlyargs,
      kwonlydefaults=kwonlydefaults,
      annotations=arg_spec.annotations)
  return wrap_with_training_arg, decorator_argspec


def get_training_arg_index(call_fn):
  """Returns the index of 'training' in the layer call function arguments.

  Args:
    call_fn: Call function.

  Returns:
    - n: index of 'training' in the call function arguments.
    - -1: if 'training' is not found in the arguments, but layer.call accepts
          variable keyword arguments
    - None: if layer doesn't expect a training argument.
  """
  arg_list = tf_inspect.getfullargspec(call_fn).args
  if tf_inspect.ismethod(call_fn):
    arg_list = arg_list[1:]
  if 'training' in arg_list:
    return arg_list.index('training')
  else:
    return -1


def set_training_arg(training, index, args, kwargs):
  if index is None:
    pass
  elif index >= 0 and len(args) > index:
    args[index] = training
  else:
    kwargs['training'] = training
  return args, kwargs


def get_training_arg(index, args, kwargs):
  if index is None:
    return None
  elif index >= 0 and len(args) > index:
    return args[index]
  else:
    return kwargs.get('training', None)


def remove_training_arg(index, args, kwargs):
  if index is None:
    pass
  elif index >= 0 and len(args) > index:
    args.pop(index)
  else:
    kwargs.pop('training', None)
