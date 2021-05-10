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
"""FeatureColumn serialization, deserialization logic."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.feature_column import feature_column_v2 as fc_lib
from tensorflow.python.feature_column import sequence_feature_column as sfc_lib
from tensorflow.python.ops import init_ops
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export


_FEATURE_COLUMNS = [
    fc_lib.BucketizedColumn, fc_lib.CrossedColumn, fc_lib.EmbeddingColumn,
    fc_lib.HashedCategoricalColumn, fc_lib.IdentityCategoricalColumn,
    fc_lib.IndicatorColumn, fc_lib.NumericColumn,
    fc_lib.SequenceCategoricalColumn, fc_lib.SequenceDenseColumn,
    fc_lib.SharedEmbeddingColumn, fc_lib.VocabularyFileCategoricalColumn,
    fc_lib.VocabularyListCategoricalColumn, fc_lib.WeightedCategoricalColumn,
    init_ops.TruncatedNormal, sfc_lib.SequenceNumericColumn
]


@tf_export('__internal__.feature_column.serialize_feature_column', v1=[])
def serialize_feature_column(fc):
  """Serializes a FeatureColumn or a raw string key.

  This method should only be used to serialize parent FeatureColumns when
  implementing FeatureColumn.get_config(), else serialize_feature_columns()
  is preferable.

  This serialization also keeps information of the FeatureColumn class, so
  deserialization is possible without knowing the class type. For example:

  a = numeric_column('x')
  a.get_config() gives:
  {
      'key': 'price',
      'shape': (1,),
      'default_value': None,
      'dtype': 'float32',
      'normalizer_fn': None
  }
  While serialize_feature_column(a) gives:
  {
      'class_name': 'NumericColumn',
      'config': {
          'key': 'price',
          'shape': (1,),
          'default_value': None,
          'dtype': 'float32',
          'normalizer_fn': None
      }
  }

  Args:
    fc: A FeatureColumn or raw feature key string.

  Returns:
    Keras serialization for FeatureColumns, leaves string keys unaffected.

  Raises:
    ValueError if called with input that is not string or FeatureColumn.
  """
  if isinstance(fc, six.string_types):
    return fc
  elif isinstance(fc, fc_lib.FeatureColumn):
    return {'class_name': fc.__class__.__name__, 'config': fc.get_config()}
  else:
    raise ValueError('Instance: {} is not a FeatureColumn'.format(fc))


@tf_export('__internal__.feature_column.deserialize_feature_column', v1=[])
def deserialize_feature_column(config,
                               custom_objects=None,
                               columns_by_name=None):
  """Deserializes a `config` generated with `serialize_feature_column`.

  This method should only be used to deserialize parent FeatureColumns when
  implementing FeatureColumn.from_config(), else deserialize_feature_columns()
  is preferable. Returns a FeatureColumn for this config.

  Args:
    config: A Dict with the serialization of feature columns acquired by
      `serialize_feature_column`, or a string representing a raw column.
    custom_objects: A Dict from custom_object name to the associated keras
      serializable objects (FeatureColumns, classes or functions).
    columns_by_name: A Dict[String, FeatureColumn] of existing columns in order
      to avoid duplication.

  Raises:
    ValueError if `config` has invalid format (e.g: expected keys missing,
    or refers to unknown classes).

  Returns:
    A FeatureColumn corresponding to the input `config`.
  """
  # TODO(b/118939620): Simplify code if Keras utils support object deduping.
  if isinstance(config, six.string_types):
    return config
  # A dict from class_name to class for all FeatureColumns in this module.
  # FeatureColumns not part of the module can be passed as custom_objects.
  module_feature_column_classes = {
      cls.__name__: cls for cls in _FEATURE_COLUMNS}
  if columns_by_name is None:
    columns_by_name = {}

  (cls,
   cls_config) = _class_and_config_for_serialized_keras_object(
       config,
       module_objects=module_feature_column_classes,
       custom_objects=custom_objects,
       printable_module_name='feature_column_v2')

  if not issubclass(cls, fc_lib.FeatureColumn):
    raise ValueError(
        'Expected FeatureColumn class, instead found: {}'.format(cls))

  # Always deserialize the FeatureColumn, in order to get the name.
  new_instance = cls.from_config(  # pylint: disable=protected-access
      cls_config,
      custom_objects=custom_objects,
      columns_by_name=columns_by_name)

  # If the name already exists, re-use the column from columns_by_name,
  # (new_instance remains unused).
  return columns_by_name.setdefault(
      _column_name_with_class_name(new_instance), new_instance)


def serialize_feature_columns(feature_columns):
  """Serializes a list of FeatureColumns.

  Returns a list of Keras-style config dicts that represent the input
  FeatureColumns and can be used with `deserialize_feature_columns` for
  reconstructing the original columns.

  Args:
    feature_columns: A list of FeatureColumns.

  Returns:
    Keras serialization for the list of FeatureColumns.

  Raises:
    ValueError if called with input that is not a list of FeatureColumns.
  """
  return [serialize_feature_column(fc) for fc in feature_columns]


def deserialize_feature_columns(configs, custom_objects=None):
  """Deserializes a list of FeatureColumns configs.

  Returns a list of FeatureColumns given a list of config dicts acquired by
  `serialize_feature_columns`.

  Args:
    configs: A list of Dicts with the serialization of feature columns acquired
      by `serialize_feature_columns`.
    custom_objects: A Dict from custom_object name to the associated keras
      serializable objects (FeatureColumns, classes or functions).

  Returns:
    FeatureColumn objects corresponding to the input configs.

  Raises:
    ValueError if called with input that is not a list of FeatureColumns.
  """
  columns_by_name = {}
  return [
      deserialize_feature_column(c, custom_objects, columns_by_name)
      for c in configs
  ]


def _column_name_with_class_name(fc):
  """Returns a unique name for the feature column used during deduping.

  Without this two FeatureColumns that have the same name and where
  one wraps the other, such as an IndicatorColumn wrapping a
  SequenceCategoricalColumn, will fail to deserialize because they will have the
  same name in columns_by_name, causing the wrong column to be returned.

  Args:
    fc: A FeatureColumn.

  Returns:
    A unique name as a string.
  """
  return fc.__class__.__name__ + ':' + fc.name


def _serialize_keras_object(instance):
  """Serialize a Keras object into a JSON-compatible representation."""
  _, instance = tf_decorator.unwrap(instance)
  if instance is None:
    return None

  if hasattr(instance, 'get_config'):
    name = instance.__class__.__name__
    config = instance.get_config()
    serialization_config = {}
    for key, item in config.items():
      if isinstance(item, six.string_types):
        serialization_config[key] = item
        continue

      # Any object of a different type needs to be converted to string or dict
      # for serialization (e.g. custom functions, custom classes)
      try:
        serialized_item = _serialize_keras_object(item)
        if isinstance(serialized_item, dict) and not isinstance(item, dict):
          serialized_item['__passive_serialization__'] = True
        serialization_config[key] = serialized_item
      except ValueError:
        serialization_config[key] = item

    return {'class_name': name, 'config': serialization_config}
  if hasattr(instance, '__name__'):
    return instance.__name__
  raise ValueError('Cannot serialize', instance)


def _deserialize_keras_object(identifier,
                              module_objects=None,
                              custom_objects=None,
                              printable_module_name='object'):
  """Turns the serialized form of a Keras object back into an actual object."""
  if identifier is None:
    return None

  if isinstance(identifier, dict):
    # In this case we are dealing with a Keras config dictionary.
    config = identifier
    (cls, cls_config) = _class_and_config_for_serialized_keras_object(
        config, module_objects, custom_objects, printable_module_name)

    if hasattr(cls, 'from_config'):
      arg_spec = tf_inspect.getfullargspec(cls.from_config)
      custom_objects = custom_objects or {}

      if 'custom_objects' in arg_spec.args:
        return cls.from_config(
            cls_config,
            custom_objects=dict(
                list(custom_objects.items())))
      return cls.from_config(cls_config)
    else:
      # Then `cls` may be a function returning a class.
      # in this case by convention `config` holds
      # the kwargs of the function.
      custom_objects = custom_objects or {}
      return cls(**cls_config)
  elif isinstance(identifier, six.string_types):
    object_name = identifier
    if custom_objects and object_name in custom_objects:
      obj = custom_objects.get(object_name)
    else:
      obj = module_objects.get(object_name)
      if obj is None:
        raise ValueError(
            'Unknown ' + printable_module_name + ': ' + object_name)
    # Classes passed by name are instantiated with no args, functions are
    # returned as-is.
    if tf_inspect.isclass(obj):
      return obj()
    return obj
  elif tf_inspect.isfunction(identifier):
    # If a function has already been deserialized, return as is.
    return identifier
  else:
    raise ValueError('Could not interpret serialized %s: %s' %
                     (printable_module_name, identifier))


def _class_and_config_for_serialized_keras_object(
    config,
    module_objects=None,
    custom_objects=None,
    printable_module_name='object'):
  """Returns the class name and config for a serialized keras object."""
  if (not isinstance(config, dict) or 'class_name' not in config or
      'config' not in config):
    raise ValueError('Improper config format: ' + str(config))

  class_name = config['class_name']
  cls = _get_registered_object(class_name, custom_objects=custom_objects,
                               module_objects=module_objects)
  if cls is None:
    raise ValueError('Unknown ' + printable_module_name + ': ' + class_name)

  cls_config = config['config']

  deserialized_objects = {}
  for key, item in cls_config.items():
    if isinstance(item, dict) and '__passive_serialization__' in item:
      deserialized_objects[key] = _deserialize_keras_object(
          item,
          module_objects=module_objects,
          custom_objects=custom_objects,
          printable_module_name='config_item')
    elif (isinstance(item, six.string_types) and
          tf_inspect.isfunction(_get_registered_object(item, custom_objects))):
      # Handle custom functions here. When saving functions, we only save the
      # function's name as a string. If we find a matching string in the custom
      # objects during deserialization, we convert the string back to the
      # original function.
      # Note that a potential issue is that a string field could have a naming
      # conflict with a custom function name, but this should be a rare case.
      # This issue does not occur if a string field has a naming conflict with
      # a custom object, since the config of an object will always be a dict.
      deserialized_objects[key] = _get_registered_object(item, custom_objects)
  for key, item in deserialized_objects.items():
    cls_config[key] = deserialized_objects[key]

  return (cls, cls_config)


def _get_registered_object(name, custom_objects=None, module_objects=None):
  if custom_objects and name in custom_objects:
    return custom_objects[name]
  elif module_objects and name in module_objects:
    return module_objects[name]
  return None

