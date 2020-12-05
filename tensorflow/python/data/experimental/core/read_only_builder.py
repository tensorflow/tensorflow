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

"""Load Datasets without reading dataset generation code."""

import os
import typing
from typing import Any, Optional, Tuple, Type

import tensorflow as tf

from tensorflow.data.experimental.core import constants
from tensorflow.data.experimental.core import dataset_builder
from tensorflow.data.experimental.core import dataset_info
from tensorflow.data.experimental.core import naming
from tensorflow.data.experimental.core import registered
from tensorflow.data.experimental.core import utils
from tensorflow.data.experimental.core.features import feature as feature_lib
from tensorflow.data.experimental.core.utils import version as version_lib


class ReadOnlyBuilder(
    dataset_builder.FileReaderBuilder, skip_registration=True
):
  """Generic DatasetBuilder loading from a directory."""

  def __init__(self, builder_dir: str):
    """Constructor.

    Args:
      builder_dir: Directory of the dataset to load (e.g.
        `~/tensorflow_datasets/mnist/3.0.0/`)

    Raises:
      FileNotFoundError: If the builder_dir does not exists.
    """
    builder_dir = os.path.expanduser(builder_dir)
    info_path = os.path.join(builder_dir, dataset_info.DATASET_INFO_FILENAME)
    if not tf.io.gfile.exists(info_path):
      raise FileNotFoundError(
          f'Could not load `ReadOnlyBuilder`: {info_path} does not exists.'
      )

    # Restore name, config, info
    info_proto = dataset_info.read_from_json(info_path)
    self.name = info_proto.name
    self.VERSION = version_lib.Version(info_proto.version)  # pylint: disable=invalid-name
    if info_proto.config_name:
      builder_config = dataset_builder.BuilderConfig(
          name=info_proto.config_name,
          description=info_proto.config_description,
          version=info_proto.version or None,
      )
    else:
      builder_config = None
    # __init__ will call _build_data_dir, _create_builder_config,
    # _pick_version to set the data_dir, config, and version
    super().__init__(
        data_dir=builder_dir,
        config=builder_config,
        version=info_proto.version,
    )
    if self.info.features is None:
      raise ValueError(
          f'Cannot restore {self.info.full_name}. It likelly mean the dataset '
          'was generated with an old TFDS version (<=3.2.1).'
      )

  def _create_builder_config(
      self, builder_config: Optional[dataset_builder.BuilderConfig]
  ) -> Optional[dataset_builder.BuilderConfig]:
    return builder_config  # BuilderConfig is created in __init__

  def _pick_version(self, version: str) -> utils.Version:
    return utils.Version(version)

  def _build_data_dir(self, data_dir: str) -> Tuple[str, str]:
    return data_dir, data_dir  # _data_dir_root, _data_dir are builder_dir.

  def _info(self) -> dataset_info.DatasetInfo:
    return dataset_info.DatasetInfo(builder=self)

  def _download_and_prepare(self, **kwargs):  # pylint: disable=arguments-differ
    # DatasetBuilder.download_and_prepare is a no-op as self.data_dir already
    # exists.
    raise AssertionError('ReadOnlyBuilder can\'t be generated.')


def builder_from_directory(builder_dir: str) -> dataset_builder.DatasetBuilder:
  """Loads a `tfds.core.DatasetBuilder` from the given generated dataset path.

  This function reconstruct the `tfds.core.DatasetBuilder` without
  requirering the original generation code.

  It will read the `<builder_dir>/features.json` in order to infer the
  structure (feature names, nested dict,...) and content (image, sequence,...)
  of the dataset. The serialization format is defined in
  `tfds.features.FeatureConnector` in `to_json()`.

  Note: This function only works for datasets generated with TFDS `4.0.0` or
  above.

  Args:
    builder_dir: `str`, path of the directory containing the dataset to read (
      e.g. `~/tensorflow_datasets/mnist/3.0.0/`).

  Returns:
    builder: `tf.core.DatasetBuilder`, builder for dataset at the given path.
  """
  return ReadOnlyBuilder(builder_dir=builder_dir)


def builder_from_files(
    name: str, **builder_kwargs: Any,
) -> dataset_builder.DatasetBuilder:
  """Loads a `tfds.core.DatasetBuilder` from files, auto-infering location.

  This function is similar to `tfds.builder` (same signature), but create
  the `tfds.core.DatasetBuilder` directly from files, without loading
  original generation source code.

  It does not supports:

   * namespaces (e.g. 'kaggle:dataset')
   * config objects (`dataset/config` valid, but not `config=MyConfig()`)
   * `version='experimental_latest'`

  Args:
    name: Dataset name.
    **builder_kwargs: `tfds.core.DatasetBuilder` kwargs.

  Returns:
    builder: The loaded dataset builder.

  Raises:
    DatasetNotFoundError: If the dataset cannot be loaded.
  """
  # Find and load dataset builder.
  builder_dir = _find_builder_dir(name, **builder_kwargs)
  if builder_dir is not None:  # A generated dataset was found on disk
    return builder_from_directory(builder_dir)
  else:
    data_dirs = constants.list_data_dirs(
        given_data_dir=builder_kwargs.get('data_dir')
    )
    raise registered.DatasetNotFoundError(
        f'Could not find dataset files for: {name}. Make sure the dataset '
        f'has been generated in: {data_dirs}.'
    )


def _find_builder_dir(name: str, **builder_kwargs: Any) -> Optional[str]:
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
    **builder_kwargs: `tfds.core.DatasetBuilder` kwargs.

  Returns:
    path: The dataset path found, or None if the dataset isn't found.
  """
  # Normalize builder kwargs
  ns_name, ds_name, builder_kwargs = naming.parse_builder_name_kwargs(
      name, **builder_kwargs
  )
  version = builder_kwargs.pop('version', None)
  config = builder_kwargs.pop('config', None)
  data_dir = builder_kwargs.pop('data_dir', None)

  # Builder cannot be found if it uses:
  # * namespace
  # * version='experimental_latest'
  # * config objects (rather than `str`)
  # * custom DatasetBuilder.__init__ kwargs
  if (
      ns_name
      or version == 'experimental_latest'
      or isinstance(config, dataset_builder.BuilderConfig)
      or builder_kwargs
  ):
    return None

  # Search the dataset across all registered data_dirs
  all_builder_dirs = []
  for current_data_dir in constants.list_data_dirs(given_data_dir=data_dir):
    builder_dir = _find_builder_dir_single_dir(
        ds_name,
        data_dir=current_data_dir,
        version_str=str(version) if version else None,
        config_name=config,
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
        f'Dataset {name} detected in multiple locations: {all_builder_dirs}. '
        'Please resolve the ambiguity by explicitly setting `data_dir=`.'
    )
  else:
    return next(iter(all_builder_dirs))  # List has a single element


def _find_builder_dir_single_dir(
    builder_name: str,
    *,
    data_dir: str,
    config_name: Optional[str] = None,
    version_str: Optional[str] = None,
) -> Optional[str]:
  """Same as `find_builder_dir` but require explicit dir."""
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
    config_name = _get_default_config_name(builder_dir, builder_name)

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


def _get_default_config_name(builder_dir: str, name: str) -> Optional[str]:
  """Returns the default config of the given dataset, None if not found."""
  # Search for the DatasetBuilder generation code
  try:
    cls = registered.imported_builder_cls(name)
    cls = typing.cast(Type[dataset_builder.DatasetBuilder], cls)
  except registered.DatasetNotFoundError:
    pass
  else:
    # If code found, return the default config
    if cls.BUILDER_CONFIGS:
      return cls.BUILDER_CONFIGS[0].name

  # Otherwise, try to load default config from common metadata
  return dataset_builder.load_default_config_name(utils.as_path(builder_dir))


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
  all_versions = version_lib.list_all_versions(builder_dir)
  # Version not given, using the last one.
  if not requested_version and all_versions:
    return str(all_versions[-1])
  # Version given, return the biggest version matching `requested_version`
  for v in reversed(all_versions):
    if v.match(requested_version):
      return str(v)
  # Directory don't has version, or requested_version don't match
  return None
