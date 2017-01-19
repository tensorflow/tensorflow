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

__all__ = ['ExportStrategy']


class ExportStrategy(collections.namedtuple('ExportStrategy',
                                            ['name',
                                             'export_fn',
                                             'end_fn'])):
  """A class representing a type of model export.

  Typically constructed by a utility function specific to the exporter, such as
  `saved_model_export_utils.make_export_strategy()`.

  The fields are:
    name: The directory name under the export base directory where exports of
      this type will be written.
    export_fn: A function that writes an export, given an estimator and a
      destination path.  This may be run repeatedly during continuous training.
    end_fn: A function to be run at the end of training, taking a single
      argument naming the ExportStrategy-specific export directory.  This is
      typically used to take some action regarding the most recent export, such
      as copying it to another location.
  """

  def __new__(cls,
              name,
              export_fn,
              end_fn=None):

    return super(ExportStrategy, cls).__new__(cls,
                                              name=name,
                                              export_fn=export_fn,
                                              end_fn=end_fn)

  def export(self, estimator, export_path):
    return self.export_fn(estimator, export_path)

  def end(self, export_path):
    return self.end_fn(export_path)

