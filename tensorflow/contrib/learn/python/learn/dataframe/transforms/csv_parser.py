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

"""A Transform that parses lines from a CSV file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.dataframe import transform
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import parsing_ops


class CSVParser(transform.TensorFlowTransform):
  """A Transform that parses lines from a CSV file."""

  def __init__(self, column_names, default_values):
    """Initialize `CSVParser`.

    Args:
      column_names: a list of strings containing the names of columns to be
        output by the parser.
      default_values: a list containing each column.
    """
    super(CSVParser, self).__init__()
    self._column_names = tuple(column_names)
    self._default_values = default_values

  @property
  def name(self):
    return "CSVParser"

  @property
  def input_valency(self):
    return 1

  @property
  def _output_names(self):
    return self.column_names

  @transform.parameter
  def column_names(self):
    return self._column_names

  @transform.parameter
  def default_values(self):
    return self._default_values

  def _apply_transform(self, input_tensors, **kwargs):
    default_consts = [constant_op.constant(d, shape=[1])
                      for d in self._default_values]
    parsed_values = parsing_ops.decode_csv(input_tensors[0],
                                           record_defaults=default_consts)
    # pylint: disable=not-callable
    return self.return_type(*parsed_values)
