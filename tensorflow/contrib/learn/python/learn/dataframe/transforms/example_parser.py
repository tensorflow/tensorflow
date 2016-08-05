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

"""A Transform that parses serialized tensorflow.Example protos."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.contrib.learn.python.learn.dataframe import transform
from tensorflow.python.ops import parsing_ops


class ExampleParser(transform.Transform):
  """A Transform that parses serialized `tensorflow.Example` protos."""

  def __init__(self, features):
    """Initialize `ExampleParser`.

      The `features` argument must be an object that can be converted to an
      `OrderedDict`. The keys should be strings and will be used to name the
      output. Values should be either `VarLenFeature` or `FixedLenFeature`. If
      `features` is a dict, it will be sorted by key.
    Args:
      features: An object that can be converted to an `OrderedDict` mapping
      column names to feature definitions.
    """
    super(ExampleParser, self).__init__()
    if isinstance(features, dict):
      self._ordered_features = collections.OrderedDict(sorted(features.items(
      ), key=lambda f: f[0]))
    else:
      self._ordered_features = collections.OrderedDict(features)

  @property
  def name(self):
    return "ExampleParser"

  @property
  def input_valency(self):
    return 1

  @property
  def _output_names(self):
    return list(self._ordered_features.keys())

  @transform.parameter
  def feature_definitions(self):
    return self._ordered_features

  def _apply_transform(self, input_tensors, **kwargs):
    parsed_values = parsing_ops.parse_example(input_tensors[0],
                                              features=self._ordered_features)
    # pylint: disable=not-callable
    return self.return_type(**parsed_values)
