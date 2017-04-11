# pylint: disable=g-bad-file-header
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
"""Utility functions relating DataFrames to Estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.layers import feature_column
from tensorflow.contrib.learn.python.learn.dataframe import series as ss
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import parsing_ops


def _to_feature_spec(tensor, default_value=None):
  if isinstance(tensor, sparse_tensor.SparseTensor):
    return parsing_ops.VarLenFeature(dtype=tensor.dtype)
  else:
    return parsing_ops.FixedLenFeature(shape=tensor.get_shape(),
                                       dtype=tensor.dtype,
                                       default_value=default_value)


def _infer_feature_specs(dataframe, keys_with_defaults):
  with ops.Graph().as_default():
    tensors = dataframe.build()
    feature_specs = {
        name: _to_feature_spec(tensor, keys_with_defaults.get(name))
        for name, tensor in tensors.items()}
  return feature_specs


def _build_alternate_universe(
    dataframe, base_input_keys_with_defaults, feature_keys):
  """Create an alternate universe assuming that the base series are defined.

  The resulting graph will be used with an `input_fn` that provides exactly
  those features.

  Args:
    dataframe: the underlying `DataFrame`
    base_input_keys_with_defaults: a `dict` from the names of columns to
      considered base features to their default values.
    feature_keys: the names of columns to be used as features (including base
      features and derived features).

  Returns:
    A `dict` mapping names to rebuilt `Series`.
  """
  feature_specs = _infer_feature_specs(dataframe, base_input_keys_with_defaults)

  alternate_universe_map = {
      dataframe[name]: ss.PredefinedSeries(name, feature_specs[name])
      for name in base_input_keys_with_defaults.keys()
  }

  def _in_alternate_universe(orig_series):
    # pylint: disable=protected-access
    # Map Series in the original DataFrame to series rebuilt assuming base_keys.
    try:
      return alternate_universe_map[orig_series]
    except KeyError:
      rebuilt_inputs = []
      for i in orig_series._input_series:
        rebuilt_inputs.append(_in_alternate_universe(i))
      rebuilt_series = ss.TransformedSeries(rebuilt_inputs,
                                            orig_series._transform,
                                            orig_series._output_name)
      alternate_universe_map[orig_series] = rebuilt_series
      return rebuilt_series

  orig_feature_series_dict = {fk: dataframe[fk] for fk in feature_keys}
  new_feature_series_dict = ({name: _in_alternate_universe(x)
                              for name, x in orig_feature_series_dict.items()})
  return new_feature_series_dict, feature_specs


def to_feature_columns_and_input_fn(dataframe,
                                    base_input_keys_with_defaults,
                                    feature_keys,
                                    label_keys=None,
                                    **kwargs):
  """Build a list of FeatureColumns and an input_fn for use with Estimator.

  Args:
    dataframe: the underlying dataframe
    base_input_keys_with_defaults: a dict from the names of columns to be
      considered base features to their default values.  These columns will be
      fed via input_fn.
    feature_keys: the names of columns from which to generate FeatureColumns.
      These may include base features and/or derived features.
    label_keys: the names of columns to be used as labels.  None is
      acceptable for unsupervised learning.
    **kwargs: Additional keyword arguments, unused here.

  Returns:
    A tuple of two elements:
    * A list of `FeatureColumn`s to be used when constructing an Estimator
    * An input_fn, i.e. a function that returns a pair of dicts
      (features, labels), each mapping string names to Tensors.
      the feature dict provides mappings for all the base columns required
      by the FeatureColumns.

  Raises:
    ValueError: when the feature and label key sets are non-disjoint, or the
      base_input and label sets are non-disjoint.
  """
  if feature_keys is None or not feature_keys:
    raise ValueError("feature_keys must be specified.")

  if label_keys is None:
    label_keys = []

  base_input_keys = base_input_keys_with_defaults.keys()

  in_two = (set(feature_keys) & set(label_keys)) or (set(base_input_keys) &
                                                     set(label_keys))
  if in_two:
    raise ValueError("Columns cannot be used for both features and labels: %s"
                     % ", ".join(in_two))

  # Obtain the feature series in the alternate universe
  new_feature_series_dict, feature_specs = _build_alternate_universe(
      dataframe, base_input_keys_with_defaults, feature_keys)

  # TODO(soergel): Allow non-real, non-dense DataFrameColumns
  for key in new_feature_series_dict.keys():
    spec = feature_specs[key]
    if not (
        isinstance(spec, parsing_ops.FixedLenFeature)
        and (spec.dtype.is_integer or spec.dtype.is_floating)):
      raise ValueError("For now, only real dense columns can be passed from "
                       "DataFrame to Estimator.  %s is %s of %s" % (
                           (key, type(spec).__name__, spec.dtype)))

  # Make FeatureColumns from these
  feature_columns = [feature_column.DataFrameColumn(name, s)
                     for name, s in new_feature_series_dict.items()]

  # Make a new DataFrame with only the Series needed for input_fn.
  # This is important to avoid starting queue feeders that won't be used.
  limited_dataframe = dataframe.select_columns(
      list(base_input_keys) + list(label_keys))

  # Build an input_fn suitable for use with Estimator.
  def input_fn():
    """An input_fn() for feeding the given set of DataFrameColumns."""
    # It's important to build all the tensors together in one DataFrame.
    # If we did df.select() for both key sets and then build those, the two
    # resulting DataFrames would be shuffled independently.
    tensors = limited_dataframe.build(**kwargs)

    base_input_features = {key: tensors[key] for key in base_input_keys}
    labels = {key: tensors[key] for key in label_keys}

    # TODO(soergel): Remove this special case when b/30367437 is fixed.
    if len(labels) == 1:
      labels = list(labels.values())[0]

    return base_input_features, labels

  return feature_columns, input_fn
