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
"""ExportStrategy class represents different flavors of model export."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from tensorflow.python.estimator import gc
from tensorflow.python.estimator import util
from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging

__all__ = ['ExportStrategy', 'make_export_strategy']


class ExportStrategy(
    collections.namedtuple('ExportStrategy', ['name', 'export_fn'])):
  """A class representing a type of model export.

  Typically constructed by a utility function specific to the exporter, such as
  `saved_model_export_utils.make_export_strategy()`.

  The fields are:
    name: The directory name under the export base directory where exports of
      this type will be written.
    export_fn: A function that writes an export, given an estimator, a
      destination path, and optionally a checkpoint path and an evaluation
      result for that checkpoint.  Note the export_fn() may choose whether or
      not to export based on the eval result or based on an internal timer or
      any other criterion, if exports are not desired for every checkpoint.

    The signature of this function must be one of:

    * `(estimator, export_path) -> export_path`
    * `(estimator, export_path, checkpoint_path) -> export_path`
    * `(estimator, export_path, checkpoint_path, eval_result) -> export_path`
  """

  def export(self,
             estimator,
             export_path,
             checkpoint_path=None,
             eval_result=None):
    """Exports the given Estimator to a specific format.

    Args:
      estimator: the Estimator to export.
      export_path: A string containing a directory where to write the export.
      checkpoint_path: The checkpoint path to export.  If None (the default),
        the strategy may locate a checkpoint (e.g. the most recent) by itself.
      eval_result: The output of Estimator.evaluate on this checkpoint.  This
        should be set only if checkpoint_path is provided (otherwise it is
        unclear which checkpoint this eval refers to).

    Returns:
      The string path to the exported directory.

    Raises:
      ValueError: if the export_fn does not have the required signature.
    """
    export_fn_args = util.fn_args(self.export_fn)
    kwargs = {}
    if 'checkpoint_path' in export_fn_args:
      kwargs['checkpoint_path'] = checkpoint_path
    if 'eval_result' in export_fn_args:
      if 'checkpoint_path' not in export_fn_args:
        raise ValueError('An export_fn accepting eval_result must also accept '
                         'checkpoint_path.')
      kwargs['eval_result'] = eval_result

    return self.export_fn(estimator, export_path, **kwargs)


def make_export_strategy(serving_input_fn,
                         assets_extra=None,
                         as_text=False,
                         exports_to_keep=5):
  """Create an ExportStrategy for use with tf.estimator.EvalSpec.

  Args:
    serving_input_fn: a function that takes no arguments and returns an
      `ServingInputReceiver`.
    assets_extra: A dict specifying how to populate the assets.extra directory
      within the exported SavedModel.  Each key should give the destination
      path (including the filename) relative to the assets.extra directory.
      The corresponding value gives the full path of the source file to be
      copied.  For example, the simple case of copying a single file without
      renaming it is specified as
      `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
    as_text: whether to write the SavedModel proto in text format.
    exports_to_keep: Number of exports to keep.  Older exports will be
      garbage-collected.  Defaults to 5.  Set to None to disable garbage
      collection.

  Returns:
    An `ExportStrategy` that can be passed to the Experiment constructor.
  """

  def export_fn(estimator, export_dir_base, checkpoint_path=None):
    """Exports the given Estimator as a SavedModel.

    Args:
      estimator: the Estimator to export.
      export_dir_base: A string containing a directory to write the exported
        graph and checkpoints.
      checkpoint_path: The checkpoint path to export.  If None (the default),
        the most recent checkpoint found within the model directory is chosen.

    Returns:
      The string path to the exported directory.

    Raises:
      ValueError: If `estimator` is a ${tf.estimator.Estimator} instance
        and `default_output_alternative_key` was specified.
    """
    export_result = estimator.export_savedmodel(
        export_dir_base,
        serving_input_fn,
        assets_extra=assets_extra,
        as_text=as_text,
        checkpoint_path=checkpoint_path)

    _garbage_collect_exports(export_dir_base, exports_to_keep)
    return export_result

  return ExportStrategy('Servo', export_fn)


def _garbage_collect_exports(export_dir_base, exports_to_keep):
  """Deletes older exports, retaining only a given number of the most recent.

  Export subdirectories are assumed to be named with monotonically increasing
  integers; the most recent are taken to be those with the largest values.

  Args:
    export_dir_base: the base directory under which each export is in a
      versioned subdirectory.
    exports_to_keep: the number of recent exports to retain.
  """
  if exports_to_keep is None:
    return

  def _export_version_parser(path):
    # create a simple parser that pulls the export_version from the directory.
    filename = os.path.basename(path.path)
    if not (len(filename) == 10 and filename.isdigit()):
      return None
    return path._replace(export_version=int(filename))

  keep_filter = gc._largest_export_versions(exports_to_keep)
  delete_filter = gc._negation(keep_filter)
  for p in delete_filter(
      gc._get_paths(export_dir_base, parser=_export_version_parser)):
    try:
      gfile.DeleteRecursively(p.path)
    except errors_impl.NotFoundError as e:
      tf_logging.warn('Can not delete %s recursively: %s', p.path, e)
