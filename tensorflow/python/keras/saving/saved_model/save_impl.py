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
"""Keras SavedModel serialization.

TODO (kathywu): Move to layer_serialization.py. Some model-specific logic should
go to model_serialization.py.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import weakref

from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import load as keras_load
from tensorflow.python.keras.saving.saved_model import serialized_attributes
from tensorflow.python.keras.saving.saved_model import utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.lazy_loader import LazyLoader

# To avoid circular dependencies between keras/engine and keras/saving,
# code in keras/saving must delay imports.

# TODO(b/134426265): Switch back to single-quotes to match the rest of the file
# once the issue with copybara is fixed.
# pylint:disable=g-inconsistent-quotes
base_layer = LazyLoader(
    "base_layer", globals(),
    "tensorflow.python.keras.engine.base_layer")
metrics = LazyLoader("metrics", globals(),
                     "tensorflow.python.keras.metrics")
input_layer = LazyLoader(
    "input_layer", globals(),
    "tensorflow.python.keras.engine.input_layer")
training_lib = LazyLoader(
    "training_lib", globals(),
    "tensorflow.python.keras.engine.training")
sequential_lib = LazyLoader(
    "sequential_lib", globals(),
    "tensorflow.python.keras.engine.sequential")
# pylint:enable=g-inconsistent-quotes


def should_skip_serialization(layer):
  """Skip serializing extra objects and functions if layer inputs aren't set."""
  saved_model_input_spec_set = (isinstance(layer, training_lib.Model) and
                                layer._saved_model_inputs_spec is not None)  # pylint: disable=protected-access
  if not layer.built and not saved_model_input_spec_set:
    logging.warning('Skipping full serialization of Keras layer {}, because '
                    'it is not built.'.format(layer))
    return True
  return False


def wrap_layer_objects(layer, serialization_cache):
  """Returns extra trackable objects to attach to the serialized layer.

  Args:
    layer: Keras Layer object.
    serialization_cache: Dictionary shared between all objects during
      serialization.

  Returns:
    A dictionary containing all checkpointable objects from a
    SerializedAttributes object. See LayerAttributes and ModelAttributes for
    entire list of objects
  """
  # Wrap all regularization losses as tf.functions.
  # First, generate list of all regularization losses in this layer and
  # sublayers.
  all_losses = layer._callable_losses[:]  # pylint: disable=protected-access
  for child_layer in utils.list_all_layers(layer):
    all_losses.extend(child_layer._callable_losses)  # pylint: disable=protected-access
  # Next, wrap all loss functions as tf.functions. Use the serialization cache
  # to store already-wrapped functions.
  keras_loss_cache = serialization_cache.setdefault('keras_losses', {})
  wrapped_loss_functions = []
  for loss_fn in all_losses:
    if loss_fn in keras_loss_cache:
      wrapped_loss_functions.append(keras_loss_cache[loss_fn])
    else:
      wrapped_loss = _wrap_unconditional_loss(loss_fn, len(keras_loss_cache))
      keras_loss_cache[loss_fn] = wrapped_loss
      wrapped_loss_functions.append(wrapped_loss)
  wrapped_layer_losses = [keras_loss_cache[fn]
                          for fn in layer._callable_losses[:]]  # pylint: disable=protected-access

  layer_metrics = data_structures._DictWrapper(  # pylint: disable=protected-access
      {m.name: m for m in layer._metrics})  # pylint: disable=protected-access
  return dict(
      variables=data_structures.ListWrapper(layer.variables),
      trainable_variables=data_structures.ListWrapper(
          layer.trainable_variables),
      non_trainable_variables=data_structures.ListWrapper(
          layer.non_trainable_variables),
      layers=data_structures.ListWrapper(utils.list_all_layers(layer)),
      metrics=data_structures.ListWrapper(layer.metrics),
      regularization_losses=data_structures.ListWrapper(
          wrapped_loss_functions),
      layer_regularization_losses=data_structures.ListWrapper(
          wrapped_layer_losses),
      layer_metrics=layer_metrics)
  # pylint: disable=protected-access


def wrap_layer_functions(layer, serialization_cache):
  """Returns dict of wrapped layer call function and losses in tf.functions.

  Args:
    layer: Keras Layer object.
    serialization_cache: Dictionary shared between all objects during
      serialization.

  Returns:
    A dictionary containing all keras tf.functions to serialize. See
    LayerAttributes and ModelAttributes for the list of all attributes.
  """
  # Since Sequential models may be modified in place using model.add() or
  # model.pop(), don't use saved functions.
  if (isinstance(layer, keras_load.RevivedLayer) and
      not isinstance(layer, sequential_lib.Sequential)):
    return {fn_name: getattr(layer.keras_api, fn_name, None)
            for fn_name in serialized_attributes.LayerAttributes.all_functions}

  # Reset the losses of the layer and its children. The call function in each
  # child layer is replaced with tf.functions.
  original_fns = _replace_child_layer_functions(layer, serialization_cache)
  original_losses = _reset_layer_losses(layer)

  # Wrap all the layer call and activity regularizer functions.

  # Use LayerCallCollection to ensure that all layer call functions (__call__,
  # call with losses) are traced with the same inputs.
  call_collection = LayerCallCollection(layer)
  call_fn_with_losses = call_collection.add_function(
      _wrap_call_and_conditional_losses(layer),
      '{}_layer_call_and_return_conditional_losses'.format(layer.name))
  call_fn = call_collection.add_function(
      _extract_outputs_from_fn(layer, call_fn_with_losses),
      '{}_layer_call_fn'.format(layer.name))

  fns = {'call_and_return_conditional_losses': call_fn_with_losses,
         '__call__': call_fn}

  if layer._activity_regularizer is not None:  # pylint: disable=protected-access
    fns['activity_regularizer_fn'] = _wrap_activity_regularizer(layer)
    fns['call_and_return_all_conditional_losses'] = (
        call_collection.add_function(
            _append_activity_regularizer_loss(layer,
                                              call_fn_with_losses,
                                              fns['activity_regularizer_fn']),
            '{}_layer_call_and_return_all_conditional_losses'.format(layer.name)
            ))
  else:
    fns['activity_regularizer_fn'] = None
    fns['call_and_return_all_conditional_losses'] = call_fn_with_losses

  # Manually trigger traces before restoring the overwritten functions. The
  # functions are traced within the layer call context to ensure that layer
  # functions (e.g. add_loss) behave as though running in graph mode.
  with base_layer_utils.call_context().enter(
      layer, inputs=None, build_graph=True, training=None, saving=True):
    for fn in fns.values():
      if fn is not None and fn.input_signature is not None:
        fn.get_concrete_function()

  # Restore overwritten functions and losses
  _restore_child_layer_functions(original_fns)
  _restore_layer_losses(original_losses)

  return fns


def default_save_signature(layer):
  original_losses = _reset_layer_losses(layer)
  fn = saving_utils.trace_model_call(layer)
  fn.get_concrete_function()
  _restore_layer_losses(original_losses)
  return fn


def _replace_child_layer_functions(layer, serialization_cache):
  """Replaces functions in the children layers with wrapped tf.functions.

  This step allows functions from parent layers to reference the wrapped
  functions from their children layers instead of retracing the ops.

  This function also resets all losses stored in the layer. These are stored in
  the returned dictionary. Use `_restore_child_layer_functions` to restore
  the original attributes.

  Args:
    layer: Keras Layer object.
    serialization_cache: Dictionary shared between all objects during
      serialization.

  Returns:
    Dictionary mapping layer objects -> original functions and losses:
      { Child layer 1: {
          'losses': Original losses,
          'call': Original call function
          '_activity_regularizer': Original activity regularizer},
        Child layer 2: ...
      }
  """
  # pylint: disable=protected-access
  original_fns = {}

  def replace_layer_functions(child_layer, serialized_fns):
    """Replaces layer call and activity regularizer with wrapped functions."""
    original_fns[child_layer] = {
        'call': child_layer.call,
        '_activity_regularizer': child_layer._activity_regularizer
    }
    with trackable.no_automatic_dependency_tracking_scope(child_layer):
      try:
        child_layer._activity_regularizer = serialized_fns.get(
            'activity_regularizer_fn')
      except AttributeError:
        # Some layers have an unsettable activity regularizer.
        pass
      child_layer.call = utils.use_wrapped_call(
          child_layer,
          serialized_fns['call_and_return_conditional_losses'],
          default_training_value=False)

  def replace_metric_functions(child_layer, serialized_fns):
    """Replaces metric functions with wrapped functions."""
    original_fns[child_layer] = {
        '__call__': child_layer.__call__,
        'result': child_layer.result,
        'update_state': child_layer.update_state
    }
    with trackable.no_automatic_dependency_tracking_scope(child_layer):
      child_layer.__call__ = serialized_fns['__call__']
      child_layer.result = serialized_fns['result']
      child_layer.update_state = serialized_fns['update_state']

  for child_layer in utils.list_all_layers(layer):
    if isinstance(child_layer, input_layer.InputLayer):
      continue

    if child_layer not in serialization_cache[constants.KERAS_CACHE_KEY]:
      serialized_functions = (
          child_layer._trackable_saved_model_saver._get_serialized_attributes(
              serialization_cache).functions)
    else:
      serialized_functions = (
          serialization_cache[constants.KERAS_CACHE_KEY][child_layer].functions)
    if not serialized_functions:
      # This indicates either:
      #   - circular dependency, which means the current layer's functions
      #     should be wrapped first.
      #   - Child layer's inputs are not defined, so its functions have not been
      #     wrapped. In this case, no replacement is necessary so move on to the
      #     next child.
      continue

    if isinstance(child_layer, metrics.Metric):
      replace_metric_functions(child_layer, serialized_functions)
    else:
      replace_layer_functions(child_layer, serialized_functions)

  return original_fns
  # pylint: enable=protected-access


def _restore_child_layer_functions(original_fns):
  """Restores attributes replaced with `_replace_child_layer_functions`."""
  for child_layer, fns in original_fns.items():
    with trackable.no_automatic_dependency_tracking_scope(child_layer):
      for fn_name, fn in fns.items():
        try:
          setattr(child_layer, fn_name, fn)  # pylint: disable=protected-access
        except AttributeError:
          pass  # In the case of _activity_regularizer, setting the attribute
          # may be disallowed.


# pylint: disable=protected-access
def _reset_layer_losses(parent_layer):
  """Resets losses of layer and its sublayers, and returns original losses."""
  losses_dict = {}
  for layer in utils.list_all_layers_and_sublayers(parent_layer):
    losses_dict[layer] = {'losses': layer._losses[:],
                          'eager_losses': layer._eager_losses[:]}
    with trackable.no_automatic_dependency_tracking_scope(layer):
      layer._losses = []
      layer._eager_losses = []
  return losses_dict


def _restore_layer_losses(losses_dict):
  for layer in losses_dict:
    with trackable.no_automatic_dependency_tracking_scope(layer):
      layer._losses = losses_dict[layer]['losses']
      layer._eager_losses = losses_dict[layer]['eager_losses']
# pylint: enable=protected-access


class LayerCallCollection(object):
  """Groups wrapped layer call functions.

  This is used to ensure that all layer call functions are traced with the same
  inputs-
    - call
    - call_and_return_conditional_losses
    - call_and_return_all_conditional_losses
  """

  def __init__(self, layer):
    self.layer = layer

    self.layer_call_method = _get_layer_call_method(layer)
    self._expects_training_arg = utils.layer_uses_training_bool(layer)
    self._training_arg_index = utils.get_training_arg_index(
        self.layer_call_method)

    # If the layer call function has kwargs, then the traced function cannot
    # have an input signature.
    arg_spec = tf_inspect.getfullargspec(self.layer_call_method)
    self._has_kwargs = bool(self._expects_training_arg or
                            arg_spec.defaults or
                            arg_spec.kwonlyargs or
                            arg_spec.varkw)

    self._input_signature = self._generate_input_signature(layer)
    self._functions = weakref.WeakValueDictionary()
    # Bool indicating whether this object is currently tracing the layer call
    # functions.
    self.tracing = False

    # Get the input argument name from the args.
    args = arg_spec.args
    if tf_inspect.ismethod(self.layer_call_method):
      args = args[1:]
    self._input_arg_name = args[0] if args else 'inputs'

  def _generate_input_signature(self, layer):
    """Inspects layer object and returns the inferred input signature.

    Args:
      layer: Layer object.

    Returns:
      List of possibly nested TensorSpecs of the layer call function inputs.
      The list does not contain the `training` argument.
    """
    if (isinstance(layer.call, def_function.Function) and
        layer.call.input_signature is not None):
      return layer.call.input_signature
    else:
      if isinstance(layer, training_lib.Model):
        return saving_utils.model_input_signature(layer)
      elif layer.input_spec is not None:

        def to_tensor_spec_or_none(x):
          spec = input_spec.to_tensor_spec(x, layer._compute_dtype)  # pylint: disable=protected-access
          # If the shape is too general (e.g. multiple dimensions are allowed),
          # return None so that separate functions can be generated for each
          # inferred input signature.
          # TODO(b/134962016): currently partial signatures are not supported.
          if spec.shape == tensor_shape.TensorShape(None):
            return None
          return spec
        input_signature = [nest.map_structure(
            to_tensor_spec_or_none, layer.input_spec)]

        return input_signature
      else:
        return None

  def add_trace(self, *args, **kwargs):
    """Traces all functions with the same args and kwargs.

    Args:
      *args: Positional args passed to the original function.
      **kwargs: Keyword args passed to the original function.
    """
    args = list(args)
    kwargs = kwargs.copy()
    self.tracing = True
    for fn in self._functions.values():
      # TODO(kathywu): Replace arguments with broader shapes defined in the
      # input signature.
      if self._expects_training_arg:
        def trace_with_training(value, fn=fn):
          utils.set_training_arg(value, self._training_arg_index, args, kwargs)
          with K.learning_phase_scope(value):
            fn.get_concrete_function(*args, **kwargs)

        trace_with_training(True)
        trace_with_training(False)
      else:
        fn.get_concrete_function(*args, **kwargs)
    self.tracing = False

  @property
  def fn_input_signature(self):
    """Returns input signature for the wrapped layer call function."""
    if self._has_kwargs:
      # Input signatures may only describe tensor arguments and kwargs are not
      # supported.
      return None
    if None in nest.flatten(self._input_signature):
      # TODO(b/134962016): If input signature cannot be partially defined.
      return None
    return self._input_signature

  def training_arg_was_passed(self, args, kwargs):
    if not self.layer._expects_training_arg and self._expects_training_arg:  # pylint: disable=protected-access
      return (utils.get_training_arg(self._training_arg_index, args, kwargs)
              is not None)
    else:
      return self.layer._call_arg_was_passed(  # pylint: disable=protected-access
          'training', args, kwargs, inputs_in_args=True)

  def get_training_arg_value(self, args, kwargs):
    if not self.layer._expects_training_arg and self._expects_training_arg:  # pylint: disable=protected-access
      return utils.get_training_arg(self._training_arg_index, args, kwargs)
    else:
      return self.layer._get_call_arg_value(  # pylint: disable=protected-access
          'training', args, kwargs, inputs_in_args=True)

  def get_input_arg_value(self, args, kwargs):
    return self.layer._get_call_arg_value(  # pylint: disable=protected-access
        self._input_arg_name, args, kwargs, inputs_in_args=True)

  def _maybe_wrap_with_training_arg(self, call_fn):
    """Wraps call function with added training argument if necessary."""
    if not self.layer._expects_training_arg and self._expects_training_arg:  # pylint: disable=protected-access
      # Add training arg to wrapper function.
      arg_spec = tf_inspect.getfullargspec(call_fn)
      args = arg_spec.args + ['training']
      defaults = list(arg_spec.defaults or [])
      defaults.append(False)
      new_arg_spec = tf_inspect.FullArgSpec(
          args=args,
          varargs=arg_spec.varargs,
          varkw=arg_spec.varkw,
          defaults=defaults,
          kwonlyargs=arg_spec.kwonlyargs,
          kwonlydefaults=arg_spec.kwonlydefaults,
          annotations=arg_spec.annotations)

      # Set new training arg index
      self._training_arg_index = len(args) - 1
      if tf_inspect.ismethod(call_fn):
        self._training_arg_index -= 1

      def wrap_with_training_arg(*args, **kwargs):
        # Remove the training value, since the original call_fn does not expect
        # a training arg. Instead, the training value will be propagated using
        # the call context created in LayerCall.
        args = list(args)
        kwargs = kwargs.copy()
        utils.remove_training_arg(self._training_arg_index, args, kwargs)
        return call_fn(*args, **kwargs)

      return tf_decorator.make_decorator(
          target=call_fn,
          decorator_func=wrap_with_training_arg,
          decorator_argspec=new_arg_spec)

    return call_fn

  def add_function(self, call_fn, name):
    """Adds a layer call function to the collection."""
    self._functions[name] = fn = LayerCall(
        self, self._maybe_wrap_with_training_arg(call_fn), name,
        input_signature=self.fn_input_signature)

    if (None not in nest.flatten(self._input_signature) and
        self._has_kwargs):
      # Manually add traces for layers that have keyword arguments and have
      # a fully defined input signature.
      self.add_trace(*self._input_signature)
    return fn


def layer_call_wrapper(call_collection, method):
  """Ensures layer losses are kept the same, and runs method in call context."""
  def wrapper(*args, **kwargs):
    """Calls method within call context."""
    layer = call_collection.layer
    training = None
    inputs = call_collection.get_input_arg_value(args, kwargs)
    # pylint: disable=protected-access
    if (args or kwargs) and call_collection.training_arg_was_passed(
        args, kwargs):
      training = call_collection.get_training_arg_value(args, kwargs)
    # pylint: enable=protected-access
    original_losses = _reset_layer_losses(layer)
    with base_layer_utils.call_context().enter(
        layer, inputs=inputs, build_graph=False, training=training,
        saving=True):
      with ops.enable_auto_cast_variables(layer._compute_dtype_object):  # pylint: disable=protected-access
        ret = method(*args, **kwargs)
    _restore_layer_losses(original_losses)
    return ret
  return tf_decorator.make_decorator(target=method, decorator_func=wrapper)


class LayerCall(def_function.Function):
  """Function that triggers traces of other functions in the same collection."""

  def __init__(self, call_collection, python_function, *args, **kwargs):
    self.call_collection = call_collection
    self.original_call = call_collection.layer_call_method
    python_function = layer_call_wrapper(call_collection, python_function)
    super(LayerCall, self).__init__(python_function, *args, **kwargs)

  def __call__(self, *args, **kwargs):
    if not self.call_collection.tracing:
      self.call_collection.add_trace(*args, **kwargs)
    return super(LayerCall, self).__call__(*args, **kwargs)

  def get_concrete_function(self, *args, **kwargs):
    if not self.call_collection.tracing:
      self.call_collection.add_trace(*args, **kwargs)
    return super(LayerCall, self).get_concrete_function(*args, **kwargs)


def _wrap_call_and_conditional_losses(layer):
  """Wraps call function that returns a tuple of (outputs, losses).

  The losses returned are conditional on the inputs passed to the call function.
  Unconditional losses (e.g. weight regularizeration) are wrapped separately.

  Args:
    layer: a Keras layer object

  Returns:
    python call function that returns outputs and conditional losses -- excludes
    activity regularizer
  """
  # Create function that generates both outputs and losses
  layer_call = _get_layer_call_method(layer)
  def call_and_return_conditional_losses(inputs, *args, **kwargs):
    """Returns layer (call_output, conditional losses) tuple."""
    call_output = layer_call(inputs, *args, **kwargs)
    if version_utils.is_v1_layer_or_model(layer):
      conditional_losses = layer.get_losses_for(inputs)
    else:
      conditional_losses = [
          l for l in layer.losses if not hasattr(l, '_unconditional_loss')
      ]
    return call_output, conditional_losses

  return _create_call_fn_decorator(layer, call_and_return_conditional_losses)


def _extract_outputs_from_fn(layer, call_and_return_conditional_losses):
  """Returns a function that returns only call function outputs."""
  if isinstance(layer, keras_load.RevivedLayer):
    return layer.keras_api.__call__  # pylint: disable=protected-access
  def call(inputs, *args, **kwargs):
    return call_and_return_conditional_losses(inputs, *args, **kwargs)[0]
  return _create_call_fn_decorator(layer, call)


def _append_activity_regularizer_loss(
    layer, call_fn_with_losses, activity_regularizer_fn):
  """Appends activity regularizer loss to losses returned by the wrapped fn."""
  def fn(inputs, *args, **kwargs):
    outputs, losses = call_fn_with_losses(inputs, *args, **kwargs)
    losses.append(activity_regularizer_fn(outputs))
    return outputs, losses
  return _create_call_fn_decorator(layer, fn)


def _create_call_fn_decorator(layer, wrapped_call):
  call_fn = _get_layer_call_method(layer)
  fn, arg_spec = utils.maybe_add_training_arg(
      call_fn, wrapped_call, layer._expects_training_arg,  # pylint: disable=protected-access
      default_training_value=False)
  return tf_decorator.make_decorator(
      target=call_fn,
      decorator_func=fn,
      decorator_argspec=arg_spec)


def _wrap_unconditional_loss(loss_fn, index):
  """Wraps callable/unconditional loss, returning a serializable function."""
  # Extract original loss function from partial function
  fn = loss_fn.args[0] if isinstance(loss_fn, functools.partial) else loss_fn
  if isinstance(fn, def_function.Function):
    return fn
  else:
    return def_function.Function(
        fn, 'loss_fn_{}'.format(index), input_signature=[])


def _wrap_activity_regularizer(layer):
  """Wraps the activity regularizer."""
  # pylint: disable=protected-access
  if isinstance(layer._activity_regularizer, def_function.Function):
    return layer._activity_regularizer
  return def_function.Function(
      layer._activity_regularizer,
      '{}_activity_regularizer'.format(layer.name),
      input_signature=[tensor_spec.TensorSpec(None, layer.dtype or K.floatx())])
  # pylint: enable=protected-access


def _get_layer_call_method(layer):
  if isinstance(layer.call, (def_function.Function, function.ConcreteFunction)):
    return layer.call.python_function
  return layer.call
