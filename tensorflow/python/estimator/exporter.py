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
"""`Exporter` class represents different flavors of model export."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os

from tensorflow.python.estimator import gc
from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging


class Exporter(object):
  """A class representing a type of model export."""

  @abc.abstractproperty
  def name(self):
    """Directory name.

    A directory name under the export base directory where exports of
    this type are written.  Should not be `None` nor empty.
    """
    pass

  @abc.abstractmethod
  def export(self, estimator, export_path, checkpoint_path, eval_result,
             is_the_final_export):
    """Exports the given `Estimator` to a specific format.

    Args:
      estimator: the `Estimator` to export.
      export_path: A string containing a directory where to write the export.
      checkpoint_path: The checkpoint path to export.
      eval_result: The output of `Estimator.evaluate` on this checkpoint.
      is_the_final_export: This boolean is True when this is an export in the
        end of training.  It is False for the intermediate exports during
        the training.
        When passing `Exporter` to `tf.estimator.train_and_evaluate`
        `is_the_final_export` is always False if `TrainSpec.max_steps` is
        `None`.

    Returns:
      The string path to the exported directory or `None` if export is skipped.
    """
    pass


class _SavedModelExporter(Exporter):
  """This class exports the serving graph and checkpoints.

     This class provides a basic exporting functionality and serves as a
     foundation for specialized `Exporter`s.
  """

  def __init__(self,
               name,
               serving_input_receiver_fn,
               assets_extra=None,
               as_text=False,
               strip_default_attrs=True):
    """Create an `Exporter` to use with `tf.estimator.EvalSpec`.

    Args:
      name: unique name of this `Exporter` that is going to be used in the
        export path.
      serving_input_receiver_fn: a function that takes no arguments and returns
        a `ServingInputReceiver`.
      assets_extra: An optional dict specifying how to populate the assets.extra
        directory within the exported SavedModel.  Each key should give the
        destination path (including the filename) relative to the assets.extra
        directory.  The corresponding value gives the full path of the source
        file to be copied.  For example, the simple case of copying a single
        file without renaming it is specified as
        `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
      as_text: whether to write the SavedModel proto in text format. Defaults to
        `False`.
      strip_default_attrs: Boolean. If set, default attrs in the `GraphDef` will
        be stripped on write. This is the default behavior and recommended for
        better forward compatibility of the resulting `SavedModel`.

    Raises:
      ValueError: if any arguments is invalid.
    """
    self._name = name
    self._serving_input_receiver_fn = serving_input_receiver_fn
    self._assets_extra = assets_extra
    self._as_text = as_text
    self._strip_default_attrs = strip_default_attrs

  @property
  def name(self):
    return self._name

  def export(self, estimator, export_path, checkpoint_path, eval_result,
             is_the_final_export):
    del is_the_final_export

    export_result = estimator.export_savedmodel(
        export_path,
        self._serving_input_receiver_fn,
        assets_extra=self._assets_extra,
        as_text=self._as_text,
        checkpoint_path=checkpoint_path,
        strip_default_attrs=self._strip_default_attrs)

    return export_result


class FinalExporter(Exporter):
  """This class exports the serving graph and checkpoints in the end.

  This class performs a single export in the end of training.
  """

  def __init__(self,
               name,
               serving_input_receiver_fn,
               assets_extra=None,
               as_text=False):
    """Create an `Exporter` to use with `tf.estimator.EvalSpec`.

    Args:
      name: unique name of this `Exporter` that is going to be used in the
        export path.
      serving_input_receiver_fn: a function that takes no arguments and returns
        a `ServingInputReceiver`.
      assets_extra: An optional dict specifying how to populate the assets.extra
        directory within the exported SavedModel.  Each key should give the
        destination path (including the filename) relative to the assets.extra
        directory.  The corresponding value gives the full path of the source
        file to be copied.  For example, the simple case of copying a single
        file without renaming it is specified as
        `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
      as_text: whether to write the SavedModel proto in text format. Defaults to
        `False`.

    Raises:
      ValueError: if any arguments is invalid.
    """
    self._saved_model_exporter = _SavedModelExporter(name,
                                                     serving_input_receiver_fn,
                                                     assets_extra, as_text)

  @property
  def name(self):
    return self._saved_model_exporter.name

  def export(self, estimator, export_path, checkpoint_path, eval_result,
             is_the_final_export):
    if not is_the_final_export:
      return None

    tf_logging.info('Performing the final export in the end of training.')

    return self._saved_model_exporter.export(estimator, export_path,
                                             checkpoint_path, eval_result,
                                             is_the_final_export)


class LatestExporter(Exporter):
  """This class regularly exports the serving graph and checkpoints.

  In addition to exporting, this class also garbage collects stale exports.
  """

  def __init__(self,
               name,
               serving_input_receiver_fn,
               assets_extra=None,
               as_text=False,
               exports_to_keep=5):
    """Create an `Exporter` to use with `tf.estimator.EvalSpec`.

    Args:
      name: unique name of this `Exporter` that is going to be used in the
        export path.
      serving_input_receiver_fn: a function that takes no arguments and returns
        a `ServingInputReceiver`.
      assets_extra: An optional dict specifying how to populate the assets.extra
        directory within the exported SavedModel.  Each key should give the
        destination path (including the filename) relative to the assets.extra
        directory.  The corresponding value gives the full path of the source
        file to be copied.  For example, the simple case of copying a single
        file without renaming it is specified as
        `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
      as_text: whether to write the SavedModel proto in text format. Defaults to
        `False`.
      exports_to_keep: Number of exports to keep.  Older exports will be
        garbage-collected.  Defaults to 5.  Set to `None` to disable garbage
        collection.

    Raises:
      ValueError: if any arguments is invalid.
    """
    self._saved_model_exporter = _SavedModelExporter(name,
                                                     serving_input_receiver_fn,
                                                     assets_extra, as_text)
    self._exports_to_keep = exports_to_keep
    if exports_to_keep is not None and exports_to_keep <= 0:
      raise ValueError(
          '`exports_to_keep`, if provided, must be positive number')

  @property
  def name(self):
    return self._saved_model_exporter.name

  def export(self, estimator, export_path, checkpoint_path, eval_result,
             is_the_final_export):
    export_result = self._saved_model_exporter.export(
        estimator, export_path, checkpoint_path, eval_result,
        is_the_final_export)

    self._garbage_collect_exports(export_path)
    return export_result

  def _garbage_collect_exports(self, export_dir_base):
    """Deletes older exports, retaining only a given number of the most recent.

    Export subdirectories are assumed to be named with monotonically increasing
    integers; the most recent are taken to be those with the largest values.

    Args:
      export_dir_base: the base directory under which each export is in a
        versioned subdirectory.
    """
    if self._exports_to_keep is None:
      return

    def _export_version_parser(path):
      # create a simple parser that pulls the export_version from the directory.
      filename = os.path.basename(path.path)
      if not (len(filename) == 10 and filename.isdigit()):
        return None
      return path._replace(export_version=int(filename))

    # pylint: disable=protected-access
    keep_filter = gc._largest_export_versions(self._exports_to_keep)
    delete_filter = gc._negation(keep_filter)
    for p in delete_filter(
        gc._get_paths(export_dir_base, parser=_export_version_parser)):
      try:
        gfile.DeleteRecursively(p.path)
      except errors_impl.NotFoundError as e:
        tf_logging.warn('Can not delete %s recursively: %s', p.path, e)
    # pylint: enable=protected-access
