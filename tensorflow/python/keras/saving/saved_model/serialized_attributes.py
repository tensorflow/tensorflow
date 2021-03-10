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
"""Helper classes that list&validate all attributes to serialize to SavedModel.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import def_function
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import save_impl
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking.tracking import AutoTrackable

# TODO(b/134426265): Switch back to single-quotes to match the rest of the file
# once the issue with copybara is fixed.
# pylint:disable=g-inconsistent-quotes
base_layer = LazyLoader(
    "base_layer", globals(),
    "tensorflow.python.keras.engine.base_layer")
training_lib = LazyLoader(
    "training_lib", globals(),
    "tensorflow.python.keras.engine.training")
metrics = LazyLoader("metrics", globals(),
                     "tensorflow.python.keras.metrics")
recurrent = LazyLoader(
    "recurrent", globals(),
    "tensorflow.python.keras.layers.recurrent")
# pylint:enable=g-inconsistent-quotes


class SerializedAttributes(object):
  """Class that tracks and validates all serialization attributes.

  Keras models contain many Python-defined components. For example, the
  trainable_variable property lists the model's trainable variables by
  recursively retrieving the trainable variables from each of the child layers.
  Another example is model.call, a python function that calls child layers and
  adds ops to the backend graph.

  Only Tensorflow checkpointable objects and functions can be serialized to
  SavedModel. Serializing a Keras model as-is results in a checkpointable object
  that does not resemble a Keras model at all. Thus, extra checkpointable
  objects and functions must be created during serialization.

  **Defining new serialized attributes**
  Child classes should be defined using:
    SerializedAttributes.with_attributes(
        'name', checkpointable_objects=[...], functions=[...], copy_from=[...])
  This class is used to cache generated checkpointable objects and functions,
  ensuring that new objects and functions are generated a single time.

  **Usage during serialization**
  Each Layer/Model object should have a corresponding instance of
  SerializedAttributes. Create a new instance by calling
  `SerializedAttributes.new(obj)`. Objects and functions may be saved using
  `.set_and_validate_checkpointable_objects`/`.set_and_and_validate_functions`.
  The properties `.checkpointable_objects` and `.functions` returns the cached
  values.

  **Adding/changing attributes to save to SavedModel**
  1. Change the call to `SerializedAttributes.with_attributes` in the correct
     class:
     - CommonEndpoints: Base attributes to be added during serialization. If
       these attributes are present in a Trackable object, it can be
       deserialized to a Keras Model.
     - LayerAttributes: Attributes to serialize for Layer objects.
     - ModelAttributes: Attributes to serialize for Model objects.
  2. Update class docstring
  3. Update arguments to any calls to `set_and_validate_*`. For example, if
     `call_raw_tensors` is added to the ModelAttributes function list, then
     a `call_raw_tensors` function should be passed to
     `set_and_validate_functions`.

  **Common endpoints vs other attributes**
  Only common endpoints are attached directly to the root object. Keras-specific
  attributes are saved to a separate trackable object with the name "keras_api".
  The number of objects attached to the root is limited because any naming
  conflicts will cause user code to break.

  Another reason is that this will only affect users who call
  `tf.saved_model.load` instead of `tf.keras.models.load_model`. These are
  advanced users who are likely to have defined their own tf.functions and
  trackable objects. The added Keras-specific attributes are kept out of the way
  in the "keras_api" namespace.

  Properties defined in this class may be used to filter out keras-specific
  attributes:
  - `functions_to_serialize`: Returns dict of functions to attach to the root
      object.
  - `checkpointable_objects_to_serialize`: Returns dict of objects to attach to
      the root object (including separate trackable object containing
      keras-specific attributes)

  All changes to the serialized attributes must be backwards-compatible, so
  attributes should not be removed or modified without sufficient justification.
  """

  @staticmethod
  def with_attributes(
      name, checkpointable_objects=None, functions=None, copy_from=None):
    """Creates a subclass with all attributes as specified in the arguments.

    Args:
      name: Name of subclass
      checkpointable_objects: List of checkpointable objects to be serialized
        in the SavedModel.
      functions: List of functions to be serialized in the SavedModel.
      copy_from: List of other SerializedAttributes subclasses. The returned
        class will copy checkpoint objects/functions from each subclass.

    Returns:
      Child class with attributes as defined in the `checkpointable_objects`
      and `functions` lists.
    """
    checkpointable_objects = checkpointable_objects or []
    functions = functions or []

    if copy_from is not None:
      for cls in copy_from:
        checkpointable_objects.extend(cls.all_checkpointable_objects)
        functions.extend(cls.all_functions)

    classdict = {
        'all_checkpointable_objects': set(checkpointable_objects),
        'all_functions': set(functions)}
    return type(name, (SerializedAttributes,), classdict)

  @staticmethod
  def new(obj):
    """Returns a new SerializedAttribute object."""
    if isinstance(obj, training_lib.Model):
      return ModelAttributes()
    elif isinstance(obj, metrics.Metric):
      return MetricAttributes()
    elif isinstance(obj, recurrent.RNN):
      return RNNAttributes()
    elif isinstance(obj, base_layer.Layer):
      return LayerAttributes()
    else:
      raise TypeError('Internal error during serialization: Expected Keras '
                      'Layer object, got {} of type {}'.format(obj, type(obj)))

  def __init__(self):
    self._object_dict = {}
    self._function_dict = {}
    self._keras_trackable = AutoTrackable()

  @property
  def functions(self):
    """Returns dictionary of all functions."""
    return {key: value for key, value in self._function_dict.items()
            if value is not None}

  @property
  def checkpointable_objects(self):
    """Returns dictionary of all checkpointable objects."""
    return {key: value for key, value in self._object_dict.items()
            if value is not None}

  @property
  def functions_to_serialize(self):
    """Returns functions to attach to the root object during serialization."""
    functions = {}
    for key, v in self.functions.items():
      if key in CommonEndpoints.all_functions:
        functions[key] = (v.wrapped_call if isinstance(v, save_impl.LayerCall)
                          else v)
    return functions

  @property
  def objects_to_serialize(self):
    """Returns objects to attach to the root object during serialization."""
    objects = {key: value for key, value in self.checkpointable_objects.items()
               if key in CommonEndpoints.all_checkpointable_objects}
    objects[constants.KERAS_ATTR] = self._keras_trackable
    return objects

  def set_and_validate_functions(self, function_dict):
    """Saves function dictionary, and validates dictionary values."""
    for key in self.all_functions:
      if key in function_dict:
        if (function_dict[key] is not None and  # Not all functions are required
            not isinstance(function_dict[key],
                           (def_function.Function, save_impl.LayerCall))):
          raise ValueError(
              'Function dictionary contained a non-function object: {} (for key'
              ' {})'.format(function_dict[key], key))
        fn = function_dict[key]
        self._function_dict[key] = fn

        # Extract TensorFlow `Function` from LayerCall.
        tf_fn = fn.wrapped_call if isinstance(fn, save_impl.LayerCall) else fn
        setattr(self._keras_trackable, key, tf_fn)
      else:
        raise ValueError('Function {} missing from serialized function dict.'
                         .format(key))
    return self.functions

  def set_and_validate_objects(self, object_dict):
    """Saves objects to a dictionary, and validates the values."""
    for key in self.all_checkpointable_objects:
      if key in object_dict:
        if not isinstance(object_dict[key], trackable.Trackable):
          raise ValueError(
              'Object dictionary contained a non-trackable object: {} (for key'
              ' {})'.format(object_dict[key], key))
        self._object_dict[key] = object_dict[key]
        setattr(self._keras_trackable, key, object_dict[key])
      else:
        raise ValueError(
            'Object {} missing from serialized object dict.'.format(key))
    return self.checkpointable_objects


class CommonEndpoints(SerializedAttributes.with_attributes(
    'CommonEndpoints',
    checkpointable_objects=['variables', 'trainable_variables',
                            'regularization_losses'],
    functions=['__call__', 'call_and_return_all_conditional_losses',
               '_default_save_signature'])):
  """Common endpoints shared by all models loadable by Keras.

  List of all attributes:
    variables: List of all variables in the model and its sublayers.
    trainable_variables: List of all trainable variables in the model and its
      sublayers.
    regularization_losses: List of all unconditional losses (losses not
      dependent on the inputs) in the model and its sublayers.
    __call__: Function that takes inputs and returns the outputs of the model
      call function.
    call_and_return_all_conditional_losses: Function that returns a tuple of
      (call function outputs, list of all losses that depend on the inputs).
    _default_save_signature: Traced model call function. This is only included
      if the top level exported object is a Keras model.
  """


class LayerAttributes(SerializedAttributes.with_attributes(
    'LayerAttributes',
    checkpointable_objects=['non_trainable_variables', 'layers', 'metrics',
                            'layer_regularization_losses', 'layer_metrics'],
    functions=['call_and_return_conditional_losses', 'activity_regularizer_fn'],
    copy_from=[CommonEndpoints]
    )):
  """Layer checkpointable objects + functions that are saved to the SavedModel.

  List of all attributes:
    All attributes from CommonEndpoints
    non_trainable_variables: List of non-trainable variables in the layer and
      its sublayers.
    layers: List of all sublayers.
    metrics: List of all metrics in the layer and its sublayers.
    call_and_return_conditional_losses: Function that takes inputs and returns a
      tuple of (outputs of the call function, list of input-dependent losses).
      The list of losses excludes the activity regularizer function, which is
      separate to allow the deserialized Layer object to define a different
      activity regularizer.
    activity_regularizer_fn: Callable that returns the activity regularizer loss
    layer_regularization_losses: List of losses owned only by this layer.
    layer_metrics: List of metrics owned by this layer.
  """


class ModelAttributes(SerializedAttributes.with_attributes(
    'ModelAttributes',
    copy_from=[LayerAttributes])):
  """Model checkpointable objects + functions that are saved to the SavedModel.

  List of all attributes:
    All attributes from LayerAttributes (including CommonEndpoints)
  """
  # TODO(kathywu): Add attributes `compile_losses` and `compile_metrics`, which
  #  list all losses and metrics defined by `model.compile`.


class MetricAttributes(
    SerializedAttributes.with_attributes(
        'MetricAttributes',
        checkpointable_objects=['variables'],
        functions=[],
    )):
  """Attributes that are added to Metric objects when saved to SavedModel.

  List of all attributes:
    variables: list of all variables
  """
  pass


class RNNAttributes(SerializedAttributes.with_attributes(
    'RNNAttributes',
    checkpointable_objects=['states'],
    copy_from=[LayerAttributes])):
  """RNN checkpointable objects + functions that are saved to the SavedModel.

  List of all attributes:
    All attributes from LayerAttributes (including CommonEndpoints)
    states: List of state variables
  """

