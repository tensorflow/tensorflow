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
import inspect

__all__ = ['ExportStrategy']


class ExportStrategy(collections.namedtuple('ExportStrategy',
                                            ['name', 'export_fn'])):
  """A class representing a type of model export.

  Typically constructed by a utility function specific to the exporter, such as
  `saved_model_export_utils.make_export_strategy()`.

  The fields are:
    name: The directory name under the export base directory where exports of
      this type will be written.
    export_fn: A function that writes an export, given an estimator, a
      destination path, and optionally a checkpoint path and an evaluation
      result for that checkpoint.  This export_fn() may be run repeatedly during
      continuous training, or just once at the end of fixed-length training.
      Note the export_fn() may choose whether or not to export based on the eval
      result or based on an internal timer or any other criterion, if exports
      are not desired for every checkpoint.

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
      ValueError: if the export_fn does not have the required signature
    """
    # don't break existing export_fns that don't accept checkpoint_path and
    # eval_result
    export_fn_args = inspect.getargspec(self.export_fn).args
    kwargs = {}
    if 'checkpoint_path' in export_fn_args:
      kwargs['checkpoint_path'] = checkpoint_path
    if 'eval_result' in export_fn_args:
      if 'checkpoint_path' not in export_fn_args:
        raise ValueError('An export_fn accepting eval_result must also accept '
                         'checkpoint_path.')
      kwargs['eval_result'] = eval_result

    return self.export_fn(estimator, export_path, **kwargs)
