# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

# Copied over from https://github.com/tensorflow/datasets/blob/234a7f1efac76ea04d1cbe3762e51525e7821486/tensorflow_datasets/core/lazy_imports_lib.py - all code belongs to original authors

import tensorflow as tf
import numpy as np
import dataclasses

from tensorflow.data.experimental.core import dataset_info
from tensorflow.data.experimental.core.utils import dataset_utils
from tensorflow.data.experimental.core.dataset_info import features
from tensorflow.data.experimental.core import lazy_imports_lib
from tensorflow.data.experimental.core.utils import py_utils
from tensorflow.data.experimental.core.utils import type_utils

try:
  import pandas  # pylint: disable=g-import-not-at-top
  import pandas.io.formats.style  # pylint: disable=g-import-not-at-top
  DataFrame = pandas.DataFrame
except ImportError:
  DataFrame = object

TreeDict = type_utils.TreeDict


@dataclasses.dataclass
class ColumnInfo:
  """`pandas.DataFrame` columns info (name, style formating,...).
  Attributes:
    name: Name of the column
    format_fn: Function applied to each column items, which returns the
      displayed string object (eventually HTML)
  """
  name: str
  format_fn: Optional[Callable[[np.ndarray], str]] = None
  # Should also add a `style.apply` function

  @classmethod
  def from_spec(
      cls,
      path: Tuple[str],
      ds_info: Optional[dataset_info.DatasetInfo],
  ) -> 'ColumnInfo':
    """Formatter which filters values hard to read and format."""
    name = '/'.join(path)

    # If ds_info is not provided, no formatting
    if not ds_info:
      # Could use the spec for a better formatting ?
      return cls(name)

    # Extract feature for formatting
    feature, sequence_rank = _get_feature(path, ds_info.features)

    # Sequence would require special formatting, not supported for now
    if sequence_rank == 0:
      repr_fn = feature.repr_html
    elif sequence_rank == 1:
      repr_fn = feature.repr_html_batch
    elif sequence_rank > 1:
      repr_fn = feature.repr_html_ragged

    def repr_fn_with_debug(val):  # Wrap repr_fn to add debug info
      try:
        return repr_fn(val)
      except Exception as e:  # pylint: disable=broad-except
        err_msg = (
            f'HTML formatting of column {name} failed:\n'
            f' * feature: {feature}\n'
            f' * input: {val!r}\n'
        )
        py_utils.reraise(e, prefix=err_msg)

    return ColumnInfo(
        name='/'.join(path),
        format_fn=repr_fn_with_debug,
    )


def _get_feature(
    path: Tuple[str, ...],
    feature: features.FeatureConnector,
) -> Tuple[features.FeatureConnector, int]:
  """Recursively extracts the feature and sequence rank (plain, ragged, ...)."""
  sequence_rank = 0

  # Collapse the nested sequences
  while isinstance(feature, features.Sequence):
    # Subclasses like `Video` shouldn't be recursed into.
    # But sequence of dict like `TranslationVariableLanguages` should.
    # Currently, there is no good way for a composed sub-feature to only
    # display a single column instead of one per sub-feature.
    # So `MyFeature({'x': tf.int32, 'y': tf.bool})` will have 2 columns `x`
    # and `y`.
    if type(feature) != features.Sequence and not path:  # pylint: disable=unidiomatic-typecheck
      break
    sequence_rank += 1
    feature = feature.feature  # Extract inner feature  # pytype: disable=attribute-error

  if path:  # Has level deeper, recurse
    feature = typing.cast(features.FeaturesDict, feature)
    feature, nested_sequence_rank = _get_feature(path[1:], feature[path[0]])  # pytype: disable=wrong-arg-types
    sequence_rank += nested_sequence_rank

  return feature, sequence_rank


class StyledDataFrame(DataFrame):
  """`pandas.DataFrame` displayed as `pandas.io.formats.style.Styler`.
  `StyledDataFrame` is a `pandas.DataFrame` with better Jupyter notebook
  representation. Contrary to regular `pandas.DataFrame`, the `style` is
  attached to the `pandas.DataFrame`.
  ```
  df = StyledDataFrame(...)
  df.current_style.apply(...)  # Configure the style
  df  # The data-frame is displayed using ` pandas.io.formats.style.Styler`
  ```
  """
  # StyledDataFrame could be improved such as the style is forwarded when
  # selecting sub-data frames.

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # Use name-mangling for forward-compatibility in case pandas
    # adds a `_styler` attribute in the future.
    self.__styler: Optional[pandas.io.formats.style.Styler] = None

  @property
  def current_style(self) -> 'pandas.io.formats.style.Styler':
    """Like `pandas.DataFrame.style`, but attach the style to the DataFrame."""
    if self.__styler is None:
      self.__styler = super().style
    return self.__styler

  def _repr_html_(self) -> str:
    # See base class for doc
    if self.__styler is None:
      return super()._repr_html_()
    return self.__styler._repr_html_()  # pylint: disable=protected-access


def _make_columns(
    specs: TreeDict[tf.TypeSpec],
    ds_info: Optional[dataset_info.DatasetInfo],
) -> List[ColumnInfo]:
  """Extract the columns info of the `panda.DataFrame`."""
  return [
      ColumnInfo.from_spec(path, ds_info)
      for path, _ in py_utils.flatten_with_path(specs)
  ]


def _make_row_dict(
    ex: TreeDict[np.ndarray],
    columns: List[ColumnInfo],
) -> Dict[str, np.ndarray]:
  """Convert a single example into a `pandas.DataFrame` row."""
  values = tf.nest.flatten(ex)
  return {column.name: v for column, v in zip(columns, values)}


def as_dataframe(
    ds: tf.data.Dataset,
    ds_info: Optional[dataset_info.DatasetInfo] = None,
) -> StyledDataFrame:
  """Convert the dataset into a pandas dataframe.
  Warning: The dataframe will be loaded entirely in memory, you may
  want to call `tfds.as_dataframe` on a subset of the data instead:
  ```
  df = tfds.as_dataframe(ds.take(10), ds_info)
  ```
  Args:
    ds: `tf.data.Dataset`. The tf.data.Dataset object to convert to panda
      dataframe. Examples should not be batched. The full dataset will be
      loaded.
    ds_info: Dataset info object. If given, helps improving the formatting.
      Available either through `tfds.load('mnist', with_info=True)` or
      `tfds.builder('mnist').info`
  Returns:
    dataframe: The `pandas.DataFrame` object
  """
  # Raise a clean error message if panda isn't installed.
  lazy_imports_lib.lazy_imports.pandas  # pylint: disable=pointless-statement

  # Pack `as_supervised=True` datasets
  if ds_info:
    ds = dataset_info.pack_as_supervised_ds(ds, ds_info)

  # Flatten the keys names, specs,... while keeping the feature key definition
  # order
  columns = _make_columns(ds.element_spec, ds_info=ds_info)
  rows = [_make_row_dict(ex, columns) for ex in dataset_utils.as_numpy(ds)]
  df = StyledDataFrame(rows)
  df.current_style.format({c.name: c.format_fn for c in columns if c.format_fn})
  return df
