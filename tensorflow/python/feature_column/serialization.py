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
from tensorflow.python.util.lazy_loader import LazyLoader

# Prevent circular dependencies with Keras serialization.
generic_utils = LazyLoader(
    'generic_utils', globals(),
    'tensorflow.python.keras.utils.generic_utils')

_FEATURE_COLUMNS = [
    fc_lib.BucketizedColumn, fc_lib.CrossedColumn, fc_lib.EmbeddingColumn,
    fc_lib.HashedCategoricalColumn, fc_lib.IdentityCategoricalColumn,
    fc_lib.IndicatorColumn, fc_lib.NumericColumn,
    fc_lib.SequenceCategoricalColumn, fc_lib.SequenceDenseColumn,
    fc_lib.SharedEmbeddingColumn, fc_lib.VocabularyFileCategoricalColumn,
    fc_lib.VocabularyListCategoricalColumn, fc_lib.WeightedCategoricalColumn,
    init_ops.TruncatedNormal, sfc_lib.SequenceNumericColumn
]


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
    return generic_utils.serialize_keras_class_and_config(
        fc.__class__.__name__, fc.get_config())  # pylint: disable=protected-access
  else:
    raise ValueError('Instance: {} is not a FeatureColumn'.format(fc))


def deserialize_feature_column(config,
                               custom_objects=None,
                               columns_by_name=None):
  """Deserializes a `config` generated with `serialize_feature_column`.

  This method should only be used to deserialize parent FeatureColumns when
  implementing FeatureColumn.from_config(), else deserialize_feature_columns()
  is preferable. Returns a FeatureColumn for this config.
  TODO(b/118939620): Simplify code if Keras utils support object deduping.

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
  if isinstance(config, six.string_types):
    return config
  # A dict from class_name to class for all FeatureColumns in this module.
  # FeatureColumns not part of the module can be passed as custom_objects.
  module_feature_column_classes = {
      cls.__name__: cls for cls in _FEATURE_COLUMNS}
  if columns_by_name is None:
    columns_by_name = {}

  (cls,
   cls_config) = generic_utils.class_and_config_for_serialized_keras_object(
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
  same name in colums_by_name, causing the wrong column to be returned.

  Args:
    fc: A FeatureColumn.

  Returns:
    A unique name as a string.
  """
  return fc.__class__.__name__ + ':' + fc.name
