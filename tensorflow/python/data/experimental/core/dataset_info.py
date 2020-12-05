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

# Copied over from https://github.com/tensorflow/datasets/blob/234a7f1efac76ea04d1cbe3762e51525e7821486/tensorflow_datasets/core/dataset_info.py - all code belongs to original authors


"""DatasetInfo records the information we know about a dataset.
This includes things that we know about the dataset statically, i.e.:
 - schema
 - description
 - canonical location
 - does it have validation and tests splits
 - size
 - etc.
This also includes the things that can and should be computed once we've
processed the dataset as well:
 - number of examples (in each split)
 - feature statistics (in each split)
 - etc.
"""

import abc
import json
import os
import posixpath
import tempfile

from absl import logging
import six
import tensorflow.compat.v2 as tf

from tensorflow.data.experimental.core import lazy_imports_lib
from tensorflow.data.experimental.core import naming
from tensorflow.data.experimental.core import splits as splits_lib
from tensorflow.data.experimental.core import utils
from tensorflow.data.experimental.core.features import feature as feature_lib
from tensorflow.data.experimental.core.features import top_level_feature
from tensorflow.data.experimental.core.dataset_info_generanted_pb2.py import dataset_info_generated_pb2 as dataset_info_pb2
from tensorflow.data.experimental.core.utils import gcs_utils
from tensorflow.data.experimental.core.utils import type_utils

from google.protobuf import json_format


# Name of the file to output the DatasetInfo protobuf object.
DATASET_INFO_FILENAME = "dataset_info.json"
LICENSE_FILENAME = "LICENSE"

INFO_STR = """tfds.core.DatasetInfo(
    name='{name}',
    version={version},
    description='{description}',
    homepage='{homepage}',
    features={features},
    total_num_examples={total_num_examples},
    splits={splits},
    supervised_keys={supervised_keys},
    citation={citation},
    redistribution_info={redistribution_info},
)
"""


# TODO(tfds): Do we require to warn the user about the peak memory used while
# constructing the dataset?
class DatasetInfo(object):
  """Information about a dataset.
  `DatasetInfo` documents datasets, including its name, version, and features.
  See the constructor arguments and properties for a full list.
  Note: Not all fields are known on construction and may be updated later
  by `compute_dynamic_properties`. For example: the min and max values of a
  feature is typically updated during data generation (i.e. on calling
  builder.download_and_prepare()`).
  """

  def __init__(self,
               *,
               builder,
               description=None,
               features=None,
               supervised_keys=None,
               homepage=None,
               citation=None,
               metadata=None,
               redistribution_info=None):
    """Constructs DatasetInfo.
    Args:
      builder: `DatasetBuilder`, dataset builder for this info.
      description: `str`, description of this dataset.
      features: `tfds.features.FeaturesDict`, Information on the feature dict
        of the `tf.data.Dataset()` object from the `builder.as_dataset()`
        method.
      supervised_keys: `tuple` of `(input_key, target_key)`, Specifies the
        input feature and the label for supervised learning, if applicable for
        the dataset. The keys correspond to the feature names to select in
        `info.features`. When calling `tfds.core.DatasetBuilder.as_dataset()`
        with `as_supervised=True`, the `tf.data.Dataset` object will yield
        the (input, target) defined here.
      homepage: `str`, optional, the homepage for this dataset.
      citation: `str`, optional, the citation to use for this dataset.
      metadata: `tfds.core.Metadata`, additonal object which will be
        stored/restored with the dataset. This allows for storing additional
        information with the dataset.
      redistribution_info: `dict`, optional, information needed for
        redistribution, as specified in `dataset_info_pb2.RedistributionInfo`.
        The content of the `license` subfield will automatically be written to a
        LICENSE file stored with the dataset.
    """
    self._builder = builder

    if builder.builder_config:
      config_name = builder.builder_config.name
      config_description = builder.builder_config.description
    else:
      config_name = None
      config_description = None

    self._info_proto = dataset_info_pb2.DatasetInfo(
        name=builder.name,
        description=utils.dedent(description),
        version=str(builder.version),
        config_name=config_name,
        config_description=config_description,
        citation=utils.dedent(citation),
        redistribution_info=dataset_info_pb2.RedistributionInfo(
            license=utils.dedent(redistribution_info.pop("license")),
            **redistribution_info) if redistribution_info else None)

    if homepage:
      self._info_proto.location.urls[:] = [homepage]

    if features:
      if not isinstance(features, top_level_feature.TopLevelFeature):
        raise ValueError(
            "DatasetInfo.features only supports FeaturesDict or Sequence at "
            "the top-level. Got {}".format(features))
    self._features = features
    self._splits = splits_lib.SplitDict(self._builder.name)
    if supervised_keys is not None:
      assert isinstance(supervised_keys, tuple)
      assert len(supervised_keys) == 2
      self._info_proto.supervised_keys.input = supervised_keys[0]
      self._info_proto.supervised_keys.output = supervised_keys[1]

    if metadata and not isinstance(metadata, Metadata):
      raise ValueError(
          "Metadata should be a `tfds.core.Metadata` instance. Received "
          "{}".format(metadata))
    self._metadata = metadata

    # Is this object initialized with both the static and the dynamic data?
    self._fully_initialized = False

  @property
  def as_proto(self):
    return self._info_proto

  @property
  def name(self):
    return self.as_proto.name

  @property
  def full_name(self):
    """Full canonical name: (<dataset_name>/<config_name>/<version>)."""
    names = [self._builder.name]
    if self._builder.builder_config:
      names.append(self._builder.builder_config.name)
    names.append(str(self.version))
    return posixpath.join(*names)

  @property
  def description(self):
    return self.as_proto.description

  @property
  def version(self):
    return self._builder.version

  @property
  def homepage(self):
    urls = self.as_proto.location.urls
    tfds_homepage = "https://www.tensorflow.org/datasets/catalog/{}".format(
        self.name)
    return urls and urls[0] or tfds_homepage

  @property
  def citation(self):
    return self.as_proto.citation

  @property
  def data_dir(self):
    return self._builder.data_dir

  @property
  def dataset_size(self):
    """Generated dataset files size, in bytes."""
    # For old datasets, maybe empty.
    return sum(split.num_bytes for split in self.splits.values())

  @property
  def download_size(self):
    """Downloaded files size, in bytes."""
    # Fallback to deprecated `size_in_bytes` if `download_size` is empty.
    return self.as_proto.download_size or self.as_proto.size_in_bytes

  @download_size.setter
  def download_size(self, size):
    self.as_proto.download_size = size

  @property
  def features(self):
    return self._features

  @property
  def metadata(self):
    return self._metadata

  @property
  def supervised_keys(self):
    if not self.as_proto.HasField("supervised_keys"):
      return None
    supervised_keys = self.as_proto.supervised_keys
    return (supervised_keys.input, supervised_keys.output)

  @property
  def redistribution_info(self):
    return self.as_proto.redistribution_info

  @property
  def splits(self):
    return self._splits.copy()

  def update_splits_if_different(self, split_dict):
    """Overwrite the splits if they are different from the current ones.
    * If splits aren't already defined or different (ex: different number of
      shards), then the new split dict is used. This will trigger stats
      computation during download_and_prepare.
    * If splits are already defined in DatasetInfo and similar (same names and
      shards): keep the restored split which contains the statistics (restored
      from GCS or file)
    Args:
      split_dict: `tfds.core.SplitDict`, the new split
    """
    assert isinstance(split_dict, splits_lib.SplitDict)

    # If splits are already defined and identical, then we do not update
    if self._splits and splits_lib.check_splits_equals(
        self._splits, split_dict):
      return

    self._set_splits(split_dict)

  def _set_splits(self, split_dict):
    """Split setter (private method)."""
    # Update the dictionary representation.
    # Use from/to proto for a clean copy
    self._splits = split_dict.copy()

    # Update the proto
    del self.as_proto.splits[:]  # Clear previous
    for split_info in split_dict.to_proto():
      self.as_proto.splits.add().CopyFrom(split_info)

  @property
  def initialized(self):
    """Whether DatasetInfo has been fully initialized."""
    return self._fully_initialized

  def _dataset_info_path(self, dataset_info_dir):
    return os.path.join(dataset_info_dir, DATASET_INFO_FILENAME)

  def _license_path(self, dataset_info_dir):
    return os.path.join(dataset_info_dir, LICENSE_FILENAME)

  def compute_dynamic_properties(self):
    self._compute_dynamic_properties(self._builder)
    self._fully_initialized = True

  def _compute_dynamic_properties(self, builder):
    """Update from the DatasetBuilder."""
    # Fill other things by going over the dataset.
    splits = self.splits
    for split_info in utils.tqdm(
        splits.values(), desc="Computing statistics...", unit=" split"):
      try:
        split_name = split_info.name
        # Fill DatasetFeatureStatistics.
        dataset_feature_statistics, schema = get_dataset_feature_statistics(
            builder, split_name)

        # Add the statistics to this split.
        split_info.statistics.CopyFrom(dataset_feature_statistics)

        # Set the schema at the top-level since this is independent of the
        # split.
        self.as_proto.schema.CopyFrom(schema)

      except tf.errors.InvalidArgumentError:
        # This means there is no such split, even though it was specified in the
        # info, the least we can do is to log this.
        logging.error(("%s's info() property specifies split %s, but it "
                       "doesn't seem to have been generated. Please ensure "
                       "that the data was downloaded for this split and re-run "
                       "download_and_prepare."), self.name, split_name)
        raise

    # Set splits to trigger proto update in setter
    self._set_splits(splits)

  @property
  def as_json(self):
    return json_format.MessageToJson(self.as_proto, sort_keys=True)

  def write_to_directory(self, dataset_info_dir):
    """Write `DatasetInfo` as JSON to `dataset_info_dir`."""
    # Save the features structure & metadata (vocabulary, labels,...)
    if self.features:
      self.features.save_config(dataset_info_dir)

    # Save any additional metadata
    if self.metadata is not None:
      self.metadata.save_metadata(dataset_info_dir)

    if self.redistribution_info.license:
      with tf.io.gfile.GFile(self._license_path(dataset_info_dir), "w") as f:
        f.write(self.redistribution_info.license)

    with tf.io.gfile.GFile(self._dataset_info_path(dataset_info_dir), "w") as f:
      f.write(self.as_json)

  def read_from_directory(self, dataset_info_dir):
    """Update DatasetInfo from the JSON file in `dataset_info_dir`.
    This function updates all the dynamically generated fields (num_examples,
    hash, time of creation,...) of the DatasetInfo.
    This will overwrite all previous metadata.
    Args:
      dataset_info_dir: `str` The directory containing the metadata file. This
        should be the root directory of a specific dataset version.
    Raises:
      FileNotFoundError: If the file can't be found.
    """
    logging.info("Load dataset info from %s", dataset_info_dir)

    json_filename = self._dataset_info_path(dataset_info_dir)
    if not tf.io.gfile.exists(json_filename):
      raise FileNotFoundError(
          "Try to load `DatasetInfo` from a directory which does not exist or "
          "does not contain `dataset_info.json`. Please delete the directory "
          f"`{dataset_info_dir}`  if you are trying to re-generate the "
          "dataset."
      )

    # Load the metadata from disk
    parsed_proto = read_from_json(json_filename)

    # Update splits
    split_dict = splits_lib.SplitDict.from_proto(self.name, parsed_proto.splits)
    self._set_splits(split_dict)

    # Restore the feature metadata (vocabulary, labels names,...)
    if self.features:
      self.features.load_metadata(dataset_info_dir)
    # For `ReadOnlyBuilder`, reconstruct the features from the config.
    elif tf.io.gfile.exists(feature_lib.make_config_path(dataset_info_dir)):
      self._features = feature_lib.FeatureConnector.from_config(
          dataset_info_dir
      )
    if self.metadata is not None:
      self.metadata.load_metadata(dataset_info_dir)

    # Update fields which are not defined in the code. This means that
    # the code will overwrite fields which are present in
    # dataset_info.json.
    for field_name, field in self.as_proto.DESCRIPTOR.fields_by_name.items():
      field_value = getattr(self._info_proto, field_name)
      field_value_restored = getattr(parsed_proto, field_name)

      try:
        is_defined = self._info_proto.HasField(field_name)
      except ValueError:
        is_defined = bool(field_value)

      try:
        is_defined_in_restored = parsed_proto.HasField(field_name)
      except ValueError:
        is_defined_in_restored = bool(field_value_restored)

      # If field is defined in code, we ignore the value
      if is_defined:
        if field_value != field_value_restored:
          logging.info(
              "Field info.%s from disk and from code do not match. Keeping "
              "the one from code.", field_name)
        continue
      # If the field is also not defined in JSON file, we do nothing
      if not is_defined_in_restored:
        continue
      # Otherwise, we restore the dataset_info.json value
      if field.type == field.TYPE_MESSAGE:
        field_value.MergeFrom(field_value_restored)
      else:
        setattr(self._info_proto, field_name, field_value_restored)

    if self._builder._version != self.version:  # pylint: disable=protected-access
      raise AssertionError(
          "The constructed DatasetInfo instance and the restored proto version "
          "do not match. Builder version: {}. Proto version: {}".format(
              self._builder._version, self.version))  # pylint: disable=protected-access

    # Mark as fully initialized.
    self._fully_initialized = True

  def initialize_from_bucket(self):
    """Initialize DatasetInfo from GCS bucket info files."""
    # In order to support Colab, we use the HTTP GCS API to access the metadata
    # files. They are copied locally and then loaded.
    tmp_dir = tempfile.mkdtemp("tfds")
    data_files = gcs_utils.gcs_dataset_info_files(self.full_name)
    if not data_files:
      return
    logging.info("Load pre-computed DatasetInfo (eg: splits, num examples,...) "
                 "from GCS: %s", self.full_name)
    for fname in data_files:
      out_fname = os.path.join(tmp_dir, os.path.basename(fname))
      tf.io.gfile.copy(gcs_utils.gcs_path(fname), out_fname)
    self.read_from_directory(tmp_dir)

  def __repr__(self):
    splits_pprint = _indent("\n".join(["{"] + [
        "    '{}': {},".format(k, split.num_examples)
        for k, split in sorted(self.splits.items())
    ] + ["}"]))
    features_pprint = _indent(repr(self.features))
    citation_pprint = _indent('"""{}"""'.format(self.citation.strip()))
    return INFO_STR.format(
        name=self.name,
        version=self.version,
        description=self.description,
        total_num_examples=self.splits.total_num_examples,
        features=features_pprint,
        splits=splits_pprint,
        citation=citation_pprint,
        homepage=self.homepage,
        supervised_keys=self.supervised_keys,
        # Proto add a \n that we strip.
        redistribution_info=str(self.redistribution_info).strip())


def _indent(content):
  """Add indentation to all lines except the first."""
  lines = content.split("\n")
  return "\n".join([lines[0]] + ["    " + l for l in lines[1:]])


def _populate_shape(shape_or_dict, prefix, schema_features):
  """Populates shape in the schema."""
  if isinstance(shape_or_dict, (tuple, list)):
    feature_name = "/".join(prefix)
    if shape_or_dict and feature_name in schema_features:
      schema_feature = schema_features[feature_name]
      schema_feature.ClearField("shape")
      for dim in shape_or_dict:
        # We denote `None`s as -1 in the shape proto.
        schema_feature.shape.dim.add().size = -1 if dim is None else dim
    return
  for name, val in shape_or_dict.items():
    prefix.append(name)
    _populate_shape(val, prefix, schema_features)
    prefix.pop()


def get_dataset_feature_statistics(builder, split):
  """Calculate statistics for the specified split."""
  tfdv = lazy_imports_lib.lazy_imports.tensorflow_data_validation
  # TODO(epot): Avoid hardcoding file format.
  filetype_suffix = "tfrecord"
  if filetype_suffix not in ["tfrecord", "csv"]:
    raise ValueError(
        "Cannot generate statistics for filetype {}".format(filetype_suffix))
  filepattern = naming.filepattern_for_dataset_split(
      builder.name, split, builder.data_dir, filetype_suffix)
  # Avoid generating a large number of buckets in rank histogram
  # (default is 1000).
  stats_options = tfdv.StatsOptions(num_top_values=10,
                                    num_rank_histogram_buckets=10)
  if filetype_suffix == "csv":
    statistics = tfdv.generate_statistics_from_csv(
        filepattern, stats_options=stats_options)
  else:
    statistics = tfdv.generate_statistics_from_tfrecord(
        filepattern, stats_options=stats_options)
  schema = tfdv.infer_schema(statistics)
  schema_features = {feature.name: feature for feature in schema.feature}
  # Override shape in the schema.
  for feature_name, feature in builder.info.features.items():
    _populate_shape(feature.shape, [feature_name], schema_features)

  # Remove legacy field.
  if getattr(schema, "generate_legacy_feature_spec", None) is not None:
    schema.ClearField("generate_legacy_feature_spec")
  return statistics.datasets[0], schema


def read_from_json(path: type_utils.PathLike) -> dataset_info_pb2.DatasetInfo:
  """Read JSON-formatted proto into DatasetInfo proto."""
  json_str = utils.as_path(path).read_text()
  # Parse it back into a proto.
  parsed_proto = json_format.Parse(json_str, dataset_info_pb2.DatasetInfo())
  return parsed_proto


def pack_as_supervised_ds(
    ds: tf.data.Dataset,
    ds_info: DatasetInfo,
) -> tf.data.Dataset:
  """Pack `(input, label)` dataset as `{'key0': input, 'key1': label}`."""
  if (
      ds_info.supervised_keys
      and isinstance(ds.element_spec, tuple)
      and len(ds.element_spec) == 2
  ):
    x_key, y_key = ds_info.supervised_keys
    ds = ds.map(lambda x, y: {x_key: x, y_key: y})
    return ds
  else:  # If dataset isn't a supervised tuple (input, label), return as-is
    return ds


@six.add_metaclass(abc.ABCMeta)
class Metadata(dict):
  """Abstract base class for DatasetInfo metadata container.
  `builder.info.metadata` allows the dataset to expose additional general
  information about the dataset which are not specific to a feature or
  individual example.
  To implement the interface, overwrite `save_metadata` and
  `load_metadata`.
  See `tfds.core.MetadataDict` for a simple implementation that acts as a
  dict that saves data to/from a JSON file.
  """

  @abc.abstractmethod
  def save_metadata(self, data_dir):
    """Save the metadata."""
    raise NotImplementedError()

  @abc.abstractmethod
  def load_metadata(self, data_dir):
    """Restore the metadata."""
    raise NotImplementedError()


class MetadataDict(Metadata, dict):
  """A `tfds.core.Metadata` object that acts as a `dict`.
  By default, the metadata will be serialized as JSON.
  """

  def _build_filepath(self, data_dir):
    return os.path.join(data_dir, "metadata.json")

  def save_metadata(self, data_dir):
    """Save the metadata."""
    with tf.io.gfile.GFile(self._build_filepath(data_dir), "w") as f:
      json.dump(self, f)

  def load_metadata(self, data_dir):
    """Restore the metadata."""
    self.clear()
    with tf.io.gfile.GFile(self._build_filepath(data_dir), "r") as f:
      self.update(json.load(f))


class BeamMetadataDict(MetadataDict):
  """A `tfds.core.Metadata` object supporting Beam-generated datasets."""

  def __init__(self, *args, **kwargs):
    super(BeamMetadataDict, self).__init__(*args, **kwargs)
    self._tempdir = tempfile.mkdtemp("tfds_beam_metadata")

  def _temp_filepath(self, key):
    return os.path.join(self._tempdir, "%s.json" % key)

  def __setitem__(self, key, item):
    """Creates write sink for beam PValues or sets value of key in `dict`.
    If the item is a PValue, it is expected to contain exactly one element,
    which will be written out as a temporary JSON file once the beam pipeline
    runs. These outputs will be loaded and stored in a single JSON when
    `save_metadata` is called after the pipeline completes.
    Args:
      key: hashable type, the key for the item.
      item: `beam.pvalue.PValue` or other, the metadata value.
    """
    beam = lazy_imports_lib.lazy_imports.apache_beam
    if isinstance(item, beam.pvalue.PValue):
      if key in self:
        raise ValueError("Already added PValue with key: %s" % key)
      logging.info("Lazily adding metadata item with Beam: %s", key)
      def _to_json(item_list):
        if len(item_list) != 1:
          raise ValueError(
              "Each metadata PValue must contain a single element. Got %d." %
              len(item_list))
        item = item_list[0]
        return json.dumps(item)
      _ = (item
           | "metadata_%s_tolist" % key >> beam.combiners.ToList()
           | "metadata_%s_tojson" % key >> beam.Map(_to_json)
           | "metadata_%s_write" % key >> beam.io.WriteToText(
               self._temp_filepath(key),
               num_shards=1,
               shard_name_template=""))
    super(BeamMetadataDict, self).__setitem__(key, item)

  def save_metadata(self, data_dir):
    """Save the metadata inside the beam job."""
    beam = lazy_imports_lib.lazy_imports.apache_beam
    for key, item in self.items():
      if isinstance(item, beam.pvalue.PValue):
        with tf.io.gfile.GFile(self._temp_filepath(key), "r") as f:
          self[key] = json.load(f)
    tf.io.gfile.rmtree(self._tempdir)
    super(BeamMetadataDict, self).save_metadata(data_dir)
