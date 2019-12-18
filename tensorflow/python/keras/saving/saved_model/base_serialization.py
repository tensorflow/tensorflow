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
"""Helper classes that list&validate all attributes to serialize to SavedModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import json

import six

from tensorflow.python.training.tracking import tracking
from tensorflow.python.util import serialization


@six.add_metaclass(abc.ABCMeta)
class SavedModelSaver(object):
  """Saver defining the methods and properties used to serialize Keras objects.
  """

  def __init__(self, obj):
    self.obj = obj

  @abc.abstractproperty
  def object_identifier(self):
    """String stored in object identifier field in the SavedModel proto.

    Returns:
      A string with the object identifier, which is used at load time.
    """
    raise NotImplementedError

  @property
  def tracking_metadata(self):
    """String stored in metadata field in the SavedModel proto.

    Returns:
      A serialized JSON storing information necessary for recreating this layer.
    """
    # TODO(kathywu): check that serialized JSON can be loaded (e.g., if an
    # object is in the python property)
    return json.dumps(
        self.python_properties,
        default=serialization.get_json_type)

  def list_extra_dependencies_for_serialization(self, serialization_cache):
    """Lists extra dependencies to serialize to SavedModel.

    By overriding this method, extra dependencies can be attached to the
    serialized Layer. For example, this is used to save the list of `variables`
    and `trainable_variables`, which are python properties in a Layer object,
    but are represented as a static list in the SavedModel.

    Args:
      serialization_cache: A dictionary shared between all objects in the same
        object graph. This object is passed to both
        `_list_extra_dependencies_for_serialization` and
        `_list_functions_for_serialization`.

    Returns:
      A dictionary mapping attribute names to trackable objects. The entire list
      of attributes are listed in the `saved_model._LayerAttributes` class.
    """
    return self.objects_to_serialize(serialization_cache)

  def list_functions_for_serialization(self, serialization_cache):
    """Lists extra functions to serialize to the SavedModel.

    Args:
      serialization_cache: Dictionary passed to all objects in the same object
        graph during serialization.

    Returns:
        A dictionary mapping attribute names to `Function` or
        `ConcreteFunction`.
    """
    fns = self.functions_to_serialize(serialization_cache)

    # The parent AutoTrackable class saves all user-defined tf.functions, and
    # returns them in _list_functions_for_serialization(). Add these functions
    # to the dict.
    fns.update(
        tracking.AutoTrackable._list_functions_for_serialization(  # pylint:disable=protected-access
            self.obj, serialization_cache))
    return fns

  @abc.abstractproperty
  def python_properties(self):
    """Returns dictionary of python properties to save in the metadata.

    This dictionary must be serializable and deserializable to/from JSON.

    When loading, the items in this dict are used to initialize the object and
    define attributes in the revived object.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def objects_to_serialize(self, serialization_cache):
    """Returns dictionary of extra checkpointable objects to serialize.

    See `functions_to_serialize` for an explanation of this function's
    effects.

    Args:
      serialization_cache: Dictionary passed to all objects in the same object
        graph during serialization.

    Returns:
        A dictionary mapping attribute names to checkpointable objects.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def functions_to_serialize(self, serialization_cache):
    """Returns extra functions to include when serializing a Keras object.

    Normally, when calling exporting an object to SavedModel, only the
    functions and objects defined by the user are saved. For example:

    ```
    obj = tf.Module()
    obj.v = tf.Variable(1.)

    @tf.function
    def foo(...): ...

    obj.foo = foo

    w = tf.Variable(1.)

    tf.saved_model.save(obj, 'path/to/saved/model')
    loaded = tf.saved_model.load('path/to/saved/model')

    loaded.v  # Variable with the same value as obj.v
    loaded.foo  # Equivalent to obj.foo
    loaded.w  # AttributeError
    ```

    Assigning trackable objects to attributes creates a graph, which is used for
    both checkpointing and SavedModel serialization.

    When the graph generated from attribute tracking is insufficient, extra
    objects and functions may be added at serialization time. For example,
    most models do not have their call function wrapped with a @tf.function
    decorator. This results in `model.call` not being saved. Since Keras objects
    should be revivable from the SavedModel format, the call function is added
    as an extra function to serialize.

    This function and `objects_to_serialize` is called multiple times when
    exporting to SavedModel. Please use the cache to avoid generating new
    functions and objects. A fresh cache is created for each SavedModel export.

    Args:
      serialization_cache: Dictionary passed to all objects in the same object
        graph during serialization.

    Returns:
        A dictionary mapping attribute names to `Function` or
        `ConcreteFunction`.
    """
    raise NotImplementedError
