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

"""Access registered datasets."""

import os
import posixpath
import re
import typing
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Type

from absl import flags
from absl import logging
import tensorflow.compat.v2 as tf

from tensorflow.data.experimental.core import constants
from tensorflow.data.experimental.core import dataset_builder
from tensorflow.data.experimental.core import decode
from tensorflow.data.experimental.core import naming
from tensorflow.data.experimental.core import read_only_builder
from tensorflow.data.experimental.core import registered
from tensorflow.data.experimental.core import splits as splits_lib
from tensorflow.data.experimental.core.features import feature as feature_lib
from tensorflow.data.experimental.core.utils import gcs_utils
from tensorflow.data.experimental.core.utils import py_utils
from tensorflow.data.experimental.core.utils import read_config as read_config_lib
from tensorflow.data.experimental.core.utils import type_utils
from tensorflow.data.experimental.core.utils import version

# Copied over from https://github.com/tensorflow/datasets/blob/234a7f1efac76ea04d1cbe3762e51525e7821486/tensorflow_datasets/core/load.py - all code belongs to original authors

# pylint: disable=logging-format-interpolation

FLAGS = flags.FLAGS

Tree = type_utils.Tree
TreeDict = type_utils.TreeDict

PredicateFn = Callable[[Type[dataset_builder.DatasetBuilder]], bool]

_NAME_STR_ERR = """\
Parsing builder name string {} failed.
The builder name string must be of the following format:
  dataset_name[/config_name][:version][/kwargs]
  Where:
    * dataset_name and config_name are string following python variable naming.
    * version is of the form x.y.z where {{x,y,z}} can be any digit or *.
    * kwargs is a comma list separated of arguments and values to pass to
      builder.
  Examples:
    my_dataset
    my_dataset:1.2.*
    my_dataset/config1
    my_dataset/config1:1.*.*
    my_dataset/config1/arg1=val1,arg2=val2
    my_dataset/config1:1.2.3/right=True,foo=bar,rate=1.2
"""

_DATASET_NOT_FOUND_ERR = """\
Check that:
    - if dataset was added recently, it may only be available
      in `tfds-nightly`
    - the dataset name is spelled correctly
    - dataset class defines all base class abstract methods
    - the module defining the dataset class is imported
"""


# Regex matching 'dataset/config:1.*.*/arg=123'
_NAME_REG = re.compile(
    r"^"
    r"(?P<dataset_name>\w+)"
    r"(/(?P<config>[\w\-\.]+))?"
    r"(:(?P<version>(\d+|\*)(\.(\d+|\*)){2}))?"
    r"(/(?P<kwargs>(\w+=\w+)(,\w+=[^,]+)*))?"
    r"$")


# Regex matching 'dataset/config/1.3.0'
_FULL_NAME_REG = re.compile(r"^{ds_name}/({config_name}/)?{version}$".format(
    ds_name=r"\w+",
    config_name=r"[\w\-\.]+",
    version=r"[0-9]+\.[0-9]+\.[0-9]+",
))


class DatasetNotFoundError(ValueError):
  """The requested Dataset was not found."""

  def __init__(self, name, is_abstract=False):
    self.is_abstract = is_abstract
    all_datasets_str = "\n\t- ".join([""] + list_builders())
    if is_abstract:
      error_string = ("Dataset %s is an abstract class so cannot be created. "
                      "Please make sure to instantiate all abstract methods.\n"
                      "%s") % (name, _DATASET_NOT_FOUND_ERR)
    else:
      error_string = ("Dataset %s not found. Available datasets:%s\n"
                      "%s") % (name, all_datasets_str, _DATASET_NOT_FOUND_ERR)
    ValueError.__init__(self, error_string)


def list_builders() -> List[str]:
  """Returns the string names of all `tfds.core.DatasetBuilder`s."""
  return sorted(list(registered._DATASET_REGISTRY))  # pylint: disable=protected-access


def builder_cls(name: str) -> Type[dataset_builder.DatasetBuilder]:
  """Fetches a `tfds.core.DatasetBuilder` class by string name.
  Args:
    name: `str`, the registered name of the `DatasetBuilder` (the class name
      as camel or snake case: `MyDataset` or `my_dataset`).
  Returns:
    A `tfds.core.DatasetBuilder` class.
  Raises:
    DatasetNotFoundError: if `name` is unrecognized.
  """
  name, kwargs = _dataset_name_and_kwargs_from_name_str(name)
  if kwargs:
    raise ValueError(
        "`builder_cls` only accept the `dataset_name` without config, "
        "version or arguments. Got: name='{}', kwargs={}".format(name, kwargs))

  # pylint: disable=protected-access
  if name in registered._ABSTRACT_DATASET_REGISTRY:
    raise DatasetNotFoundError(name, is_abstract=True)
  if name not in registered._DATASET_REGISTRY:
    raise DatasetNotFoundError(name)
  return registered._DATASET_REGISTRY[name]  # pytype: disable=bad-return-type
  # pylint: enable=protected-access


def builder(
    name: str,
    *,
    data_dir: Optional[str] = None,
    **builder_init_kwargs: Any
) -> dataset_builder.DatasetBuilder:
  """Fetches a `tfds.core.DatasetBuilder` by string name.
  Args:
    name: `str`, the registered name of the `DatasetBuilder` (the class name
      as camel or snake case: `MyDataset` or `my_dataset`).
      This can be either `'dataset_name'` or
      `'dataset_name/config_name'` for datasets with `BuilderConfig`s.
      As a convenience, this string may contain comma-separated keyword
      arguments for the builder. For example `'foo_bar/a=True,b=3'` would use
      the `FooBar` dataset passing the keyword arguments `a=True` and `b=3`
      (for builders with configs, it would be `'foo_bar/zoo/a=True,b=3'` to
      use the `'zoo'` config and pass to the builder keyword arguments `a=True`
      and `b=3`).
    data_dir: Path to the dataset(s). See `tfds.load` for more information.
    **builder_init_kwargs: `dict` of keyword arguments passed to the
      `DatasetBuilder`. These will override keyword arguments passed in `name`,
      if any.
  Returns:
    A `tfds.core.DatasetBuilder`.
  Raises:
    DatasetNotFoundError: if `name` is unrecognized.
  """
  builder_name, builder_kwargs = _dataset_name_and_kwargs_from_name_str(name)

  # Try loading the code (if it exists)
  try:
    cls = builder_cls(builder_name)
  except DatasetNotFoundError as e:
    if e.is_abstract:
      raise  # Abstract can't be instanciated neither from code nor files.
    cls = None  # Class not found
    not_found_error = e  # Save the exception to eventually reraise

  version_explicitly_given = "version" in builder_kwargs

  # Try loading from files first:
  # * If code not present.
  # * If version explicitly given (backward/forward compatibility).
  # Note: If `builder_init_kwargs` are set (e.g. version='experimental_latest',
  # custom config,...), read from generation code.
  if (not cls or version_explicitly_given) and not builder_init_kwargs:
    builder_dir = find_builder_dir(name, data_dir=data_dir)
    if builder_dir is not None:  # A generated dataset was found on disk
      return read_only_builder.builder_from_directory(builder_dir)

  # If loading from files was skipped (e.g. files not found), load from the
  # source code.
  if cls:
    with py_utils.try_reraise(prefix=f"Failed to construct dataset {name}: "):
      return cls(data_dir=data_dir, **builder_kwargs, **builder_init_kwargs)  # pytype: disable=not-instantiable

  # If neither the code nor the files are found, raise DatasetNotFoundError
  raise not_found_error


def load(
    name: str,
    *,
    split: Optional[Tree[splits_lib.Split]] = None,
    data_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    shuffle_files: bool = False,
    download: bool = True,
    as_supervised: bool = False,
    decoders: Optional[TreeDict[decode.Decoder]] = None,
    read_config: Optional[read_config_lib.ReadConfig] = None,
    with_info: bool = False,
    builder_kwargs: Optional[Dict[str, Any]] = None,
    download_and_prepare_kwargs: Optional[Dict[str, Any]] = None,
    as_dataset_kwargs: Optional[Dict[str, Any]] = None,
    try_gcs: bool = False,
):
  # pylint: disable=line-too-long
  """Loads the named dataset into a `tf.data.Dataset`.
  `tfds.load` is a convenience method that:
  1. Fetch the `tfds.core.DatasetBuilder` by name:
     ```python
     builder = tfds.builder(name, data_dir=data_dir, **builder_kwargs)
     ```
  2. Generate the data (when `download=True`):
     ```python
     builder.download_and_prepare(**download_and_prepare_kwargs)
     ```
  3. Load the `tf.data.Dataset` object:
     ```python
     ds = builder.as_dataset(
         split=split,
         as_supervised=as_supervised,
         shuffle_files=shuffle_files,
         read_config=read_config,
         decoders=decoders,
         **as_dataset_kwargs,
     )
     ```
  See: https://www.tensorflow.org/datasets/overview#load_a_dataset for more
  examples.
  If you'd like NumPy arrays instead of `tf.data.Dataset`s or `tf.Tensor`s,
  you can pass the return value to `tfds.as_numpy`.
  **Warning**: calling this function might potentially trigger the download
  of hundreds of GiB to disk. Refer to the `download` argument.
  Args:
    name: `str`, the registered name of the `DatasetBuilder` (the snake case
      version of the class name). This can be either `"dataset_name"` or
      `"dataset_name/config_name"` for datasets with `BuilderConfig`s.
      As a convenience, this string may contain comma-separated keyword
      arguments for the builder. For example `"foo_bar/a=True,b=3"` would use
      the `FooBar` dataset passing the keyword arguments `a=True` and `b=3`
      (for builders with configs, it would be `"foo_bar/zoo/a=True,b=3"` to
      use the `"zoo"` config and pass to the builder keyword arguments `a=True`
      and `b=3`).
    split: Which split of the data to load (e.g. `'train'`, `'test'`,
      `['train', 'test']`, `'train[80%:]'`,...). See our
      [split API guide](https://www.tensorflow.org/datasets/splits).
      If `None`, will return all splits in a `Dict[Split, tf.data.Dataset]`
    data_dir: `str`, directory to read/write data. Defaults to the value of
      the environment variable TFDS_DATA_DIR, if set, otherwise falls back to
      "~/tensorflow_datasets".
    batch_size: `int`, if set, add a batch dimension to examples. Note that
      variable length features will be 0-padded. If
      `batch_size=-1`, will return the full dataset as `tf.Tensor`s.
    shuffle_files: `bool`, whether to shuffle the input files.
      Defaults to `False`.
    download: `bool` (optional), whether to call
      `tfds.core.DatasetBuilder.download_and_prepare`
      before calling `tf.DatasetBuilder.as_dataset`. If `False`, data is
      expected to be in `data_dir`. If `True` and the data is already in
      `data_dir`, `download_and_prepare` is a no-op.
    as_supervised: `bool`, if `True`, the returned `tf.data.Dataset`
      will have a 2-tuple structure `(input, label)` according to
      `builder.info.supervised_keys`. If `False`, the default,
      the returned `tf.data.Dataset` will have a dictionary with all the
      features.
    decoders: Nested dict of `Decoder` objects which allow to customize the
      decoding. The structure should match the feature structure, but only
      customized feature keys need to be present. See
      [the guide](https://github.com/tensorflow/datasets/tree/master/docs/decode.md)
      for more info.
    read_config: `tfds.ReadConfig`, Additional options to configure the
      input pipeline (e.g. seed, num parallel reads,...).
    with_info: `bool`, if True, tfds.load will return the tuple
      (tf.data.Dataset, tfds.core.DatasetInfo) containing the info associated
      with the builder.
    builder_kwargs: `dict` (optional), keyword arguments to be passed to the
      `tfds.core.DatasetBuilder` constructor. `data_dir` will be passed
      through by default.
    download_and_prepare_kwargs: `dict` (optional) keyword arguments passed to
      `tfds.core.DatasetBuilder.download_and_prepare` if `download=True`. Allow
      to control where to download and extract the cached data. If not set,
      cache_dir and manual_dir will automatically be deduced from data_dir.
    as_dataset_kwargs: `dict` (optional), keyword arguments passed to
      `tfds.core.DatasetBuilder.as_dataset`.
    try_gcs: `bool`, if True, tfds.load will see if the dataset exists on
      the public GCS bucket before building it locally.
  Returns:
    ds: `tf.data.Dataset`, the dataset requested, or if `split` is None, a
      `dict<key: tfds.Split, value: tf.data.Dataset>`. If `batch_size=-1`,
      these will be full datasets as `tf.Tensor`s.
    ds_info: `tfds.core.DatasetInfo`, if `with_info` is True, then `tfds.load`
      will return a tuple `(ds, ds_info)` containing dataset information
      (version, features, splits, num_examples,...). Note that the `ds_info`
      object documents the entire dataset, regardless of the `split` requested.
      Split-specific information is available in `ds_info.splits`.
  """
  # pylint: enable=line-too-long
  if builder_kwargs is None:
    builder_kwargs = {}

  # Set data_dir
  if try_gcs and gcs_utils.is_dataset_on_gcs(name):
    data_dir = gcs_utils.gcs_path("datasets")

  dbuilder = builder(name, data_dir=data_dir, **builder_kwargs)
  if download:
    download_and_prepare_kwargs = download_and_prepare_kwargs or {}
    dbuilder.download_and_prepare(**download_and_prepare_kwargs)

  if as_dataset_kwargs is None:
    as_dataset_kwargs = {}
  as_dataset_kwargs = dict(as_dataset_kwargs)
  as_dataset_kwargs.setdefault("split", split)
  as_dataset_kwargs.setdefault("as_supervised", as_supervised)
  as_dataset_kwargs.setdefault("batch_size", batch_size)
  as_dataset_kwargs.setdefault("decoders", decoders)
  as_dataset_kwargs.setdefault("shuffle_files", shuffle_files)
  as_dataset_kwargs.setdefault("read_config", read_config)

  ds = dbuilder.as_dataset(**as_dataset_kwargs)
  if with_info:
    return ds, dbuilder.info
  return ds


def find_builder_dir(
    name: str,
    *,
    data_dir: Optional[str] = None,
) -> Optional[str]:
  """Search whether the given dataset is present on disk and return its path.
  Note:
   * If the dataset is present, but is legacy (no feature config file), None
     is returned.
   * If the config isn't specified, the function try to infer the default
     config name from the original `DatasetBuilder`.
   * The function searches in all `data_dir` registered with
     `tfds.core.add_data_dir`. If the dataset exists in multiple dirs, an error
     is raised.
  Args:
    name: Builder name (e.g. `my_ds`, `my_ds/config`, `my_ds:1.2.0`,...)
    data_dir: Path where to search for the dataset
      (e.g. `~/tensorflow_datasets`).
  Returns:
    path: The dataset path found, or None if the dataset isn't found.
  """
  # Search the dataset across all registered data_dirs
  all_builder_dirs = []
  for current_data_dir in constants.list_data_dirs(given_data_dir=data_dir):
    builder_dir = _find_builder_dir_single_dir(
        name, data_dir=current_data_dir
    )
    if builder_dir:
      all_builder_dirs.append(builder_dir)
  if not all_builder_dirs:
    return None
  elif len(all_builder_dirs) != 1:
    # Rather than raising error every time, we could potentially be smarter
    # and load the most recent version across all files, but should be
    # carefull when partial version is requested ('my_dataset:3.*.*').
    # Could add some `MultiDataDirManager` API:
    # ```
    # manager = MultiDataDirManager(given_data_dir=data_dir)
    # with manager.merge_data_dirs() as virtual_data_dir:
    #  virtual_builder_dir = _find_builder_dir(name, data_dir=virtual_data_dir)
    #  builder_dir = manager.resolve(virtual_builder_dir)
    # ```
    raise ValueError(
        f"Dataset {name} detected in multiple locations: {all_builder_dirs}. "
        "Please resolve the ambiguity by explicitly setting `data_dir=`."
    )
  else:
    return next(iter(all_builder_dirs))  # List has a single element


def _find_builder_dir_single_dir(
    name: str,
    *,
    data_dir: str,
) -> Optional[str]:
  """Same as `find_builder_dir` but require explicit dir."""
  builder_name, builder_kwargs = _dataset_name_and_kwargs_from_name_str(name)
  config_name = builder_kwargs.pop("config", None)
  version_str = builder_kwargs.pop("version", None)
  if builder_kwargs:
    # Datasets with additional kwargs require the original builder code.
    return None

  # Construct the `ds_name/config/` path
  builder_dir = os.path.join(data_dir, builder_name)
  if not config_name:
    # If the BuilderConfig is not specified:
    # * Either the dataset don't have config
    # * Either the default config should be used
    # Currently, in order to infer the default config, we are still relying on
    # the code.
    # TODO(tfds): How to avoid code dependency and automatically infer the
    # config existance and name ?
    config_name = _get_default_config_name(builder_name)

  # If has config (explicitly given or default config), append it to the path
  if config_name:
    builder_dir = os.path.join(builder_dir, config_name)

  # Extract the version
  version_str = _get_version_str(builder_dir, requested_version=version_str)

  if not version_str:  # Version not given or found
    return None

  builder_dir = os.path.join(builder_dir, version_str)

  # Check for builder dir existance
  if not tf.io.gfile.exists(builder_dir):
    return None
  # Backward compatibility, in order to be a valid ReadOnlyBuilder, the folder
  # has to contain the feature configuration.
  if not tf.io.gfile.exists(feature_lib.make_config_path(builder_dir)):
    return None
  return builder_dir


def _get_default_config_name(name: str) -> Optional[str]:
  """Returns the default config of the given dataset, None if not found."""
  # Search for the DatasetBuilder generation code
  try:
    builder_cls_ = builder_cls(name)
  except DatasetNotFoundError:
    return None

  # If code found, return the default config
  if builder_cls_.BUILDER_CONFIGS:
    return builder_cls_.BUILDER_CONFIGS[0].name
  return None


def _get_version_str(
    builder_dir: str,
    *,
    requested_version: Optional[str] = None,
) -> Optional[str]:
  """Returns the version name found in the directory.
  Args:
    builder_dir: Directory containing the versions (`builder_dir/1.0.0/`,...)
    requested_version: Optional version to search (e.g. `1.0.0`, `2.*.*`,...)
  Returns:
    version_str: The version directory name found in `builder_dir`.
  """
  all_versions = version.list_all_versions(builder_dir)
  # Version not given, using the last one.
  if not requested_version and all_versions:
    return str(all_versions[-1])
  # Version given, return the biggest version matching `requested_version`
  for v in reversed(all_versions):
    if v.match(requested_version):
      return str(v)
  # Directory don't has version, or requested_version don't match
  return None


def _dataset_name_and_kwargs_from_name_str(name_str):
  """Extract kwargs from name str."""
  res = _NAME_REG.match(name_str)
  if not res:
    raise ValueError(_NAME_STR_ERR.format(name_str))
  name = res.group("dataset_name")
  # Normalize the name to accept CamelCase
  name = naming.camelcase_to_snakecase(name)
  kwargs = _kwargs_str_to_kwargs(res.group("kwargs"))
  try:
    for attr in ["config", "version"]:
      val = res.group(attr)
      if val is None:
        continue
      if attr in kwargs:
        raise ValueError("Dataset %s: cannot pass %s twice." % (name, attr))
      kwargs[attr] = val
    return name, kwargs
  except:
    logging.error(_NAME_STR_ERR.format(name_str))   # pylint: disable=logging-format-interpolation
    raise


def _kwargs_str_to_kwargs(kwargs_str):
  if not kwargs_str:
    return {}
  kwarg_strs = kwargs_str.split(",")
  kwargs = {}
  for kwarg_str in kwarg_strs:
    kwarg_name, kwarg_val = kwarg_str.split("=")
    kwargs[kwarg_name] = _cast_to_pod(kwarg_val)
  return kwargs


def _cast_to_pod(val):
  """Try cast to int, float, bool, str, in that order."""
  bools = {"True": True, "False": False}
  if val in bools:
    return bools[val]
  try:
    return int(val)
  except ValueError:
    try:
      return float(val)
    except ValueError:
      return tf.compat.as_text(val)


def _get_all_versions(
    current_version: version.Version,
    extra_versions: Iterable[version.Version],
    current_version_only: bool,
) -> Iterable[str]:
  """Returns the list of all current versions."""
  # Merge current version with all extra versions
  version_list = [current_version]
  if not current_version_only:
    version_list.extend(extra_versions)
  # Filter datasets which do not have a version (version is `None`) as they
  # should not be instantiated directly (e.g wmt_translate)
  return {str(v) for v in version_list if v}


def _iter_single_full_names(
    builder_name: str,
    builder_cls: Type[dataset_builder.DatasetBuilder],  # pylint: disable=redefined-outer-name
    current_version_only: bool,
) -> Iterator[str]:
  """Iterate over a single builder full names."""
  if builder_cls.BUILDER_CONFIGS:
    for config in builder_cls.BUILDER_CONFIGS:
      for v in _get_all_versions(
          config.version,
          config.supported_versions,
          current_version_only=current_version_only,
      ):
        yield posixpath.join(builder_name, config.name, v)
  else:
    for v in _get_all_versions(
        builder_cls.VERSION,
        builder_cls.SUPPORTED_VERSIONS,
        current_version_only=current_version_only
    ):
      yield posixpath.join(builder_name, v)


def _iter_full_names(
    predicate_fn: Optional[PredicateFn],
    current_version_only: bool,
) -> Iterator[str]:
  """Yield all registered datasets full_names (see `list_full_names`)."""
  for builder_name, builder_cls in registered._DATASET_REGISTRY.items():  # pylint: disable=redefined-outer-name,protected-access
    builder_cls = typing.cast(Type[dataset_builder.DatasetBuilder], builder_cls)
    # Only keep requested datasets
    if predicate_fn is not None and not predicate_fn(builder_cls):
      continue
    for full_name in _iter_single_full_names(
        builder_name,
        builder_cls,
        current_version_only=current_version_only,
    ):
      yield full_name


_DEFAULT_PREDICATE_FN = None


def list_full_names(
    predicate_fn: Optional[PredicateFn] = _DEFAULT_PREDICATE_FN,
    current_version_only: bool = False,
) -> List[str]:
  """Lists all registered datasets full_names.
  Args:
    predicate_fn: `Callable[[Type[DatasetBuilder]], bool]`, if set, only
      returns the dataset names which satisfy the predicate.
    current_version_only: If True, only returns the current version.
  Returns:
    The list of all registered dataset full names.
  """
  return sorted(_iter_full_names(
      predicate_fn=predicate_fn,
      current_version_only=current_version_only,
  ))


def single_full_names(
    builder_name: str,
    current_version_only: bool = True,
) -> List[str]:
  """Returns the list `['ds/c0/v0',...]` or `['ds/v']` for a single builder."""
  return sorted(_iter_single_full_names(
      builder_name,
      registered._DATASET_REGISTRY[builder_name],  # pylint: disable=protected-access
      current_version_only=current_version_only,  # pytype: disable=wrong-arg-types
  ))


def is_full_name(full_name: str) -> bool:
  """Returns whether the string pattern match `ds/config/1.2.3` or `ds/1.2.3`.
  Args:
    full_name: String to check.
  Returns:
    `bool`.
  """
  return bool(_FULL_NAME_REG.match(full_name))
