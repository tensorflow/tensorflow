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
"""Abstract base class for all predictors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Predictor(object):
  """Abstract base class for all predictors."""

  @property
  def graph(self):
    return self._graph

  @property
  def session(self):
    return self._session

  @property
  def feed_tensors(self):
    return self._feed_tensors

  @property
  def fetch_tensors(self):
    return self._fetch_tensors

  def __repr__(self):
    return '{} with feed tensors {} and fetch_tensors {}'.format(
        type(self).__name__, self._feed_tensors, self._fetch_tensors)

  def __call__(self, input_dict):
    """Returns predictions based on `input_dict`.

    Args:
      input_dict: a `dict` mapping strings to numpy arrays. These keys
        must match `self._feed_tensors.keys()`.

    Returns:
      A `dict` mapping strings to numpy arrays. The keys match
      `self.fetch_tensors.keys()`.

    Raises:
      ValueError: `input_dict` does not match `feed_tensors`.
    """
    # TODO(jamieas): make validation optional?
    input_keys = set(input_dict.keys())
    expected_keys = set(self.feed_tensors.keys())
    unexpected_keys = input_keys - expected_keys
    if unexpected_keys:
      raise ValueError(
          'Got unexpected keys in input_dict: {}\nexpected: {}'.format(
              unexpected_keys, expected_keys))

    feed_dict = {}
    for key in self.feed_tensors.keys():
      value = input_dict.get(key)
      if value is not None:
        feed_dict[self.feed_tensors[key]] = value
    return self._session.run(fetches=self.fetch_tensors, feed_dict=feed_dict)
