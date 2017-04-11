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

"""Categorical vocabulary classes to map categories to indexes.

Can be used for categorical variables, sparse variables and words.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import six


class CategoricalVocabulary(object):
  """Categorical variables vocabulary class.

  Accumulates and provides mapping from classes to indexes.
  Can be easily used for words.
  """

  def __init__(self, unknown_token="<UNK>", support_reverse=True):
    self._unknown_token = unknown_token
    self._mapping = {unknown_token: 0}
    self._support_reverse = support_reverse
    if support_reverse:
      self._reverse_mapping = [unknown_token]
    self._freq = collections.defaultdict(int)
    self._freeze = False

  def __len__(self):
    """Returns total count of mappings. Including unknown token."""
    return len(self._mapping)

  def freeze(self, freeze=True):
    """Freezes the vocabulary, after which new words return unknown token id.

    Args:
      freeze: True to freeze, False to unfreeze.
    """
    self._freeze = freeze

  def get(self, category):
    """Returns word's id in the vocabulary.

    If category is new, creates a new id for it.

    Args:
      category: string or integer to lookup in vocabulary.

    Returns:
      interger, id in the vocabulary.
    """
    if category not in self._mapping:
      if self._freeze:
        return 0
      self._mapping[category] = len(self._mapping)
      if self._support_reverse:
        self._reverse_mapping.append(category)
    return self._mapping[category]

  def add(self, category, count=1):
    """Adds count of the category to the frequency table.

    Args:
      category: string or integer, category to add frequency to.
      count: optional integer, how many to add.
    """
    category_id = self.get(category)
    if category_id <= 0:
      return
    self._freq[category] += count

  def trim(self, min_frequency, max_frequency=-1):
    """Trims vocabulary for minimum frequency.

    Remaps ids from 1..n in sort frequency order.
    where n - number of elements left.

    Args:
      min_frequency: minimum frequency to keep.
      max_frequency: optional, maximum frequency to keep.
        Useful to remove very frequent categories (like stop words).
    """
    # Sort by alphabet then reversed frequency.
    self._freq = sorted(
        sorted(
            six.iteritems(self._freq),
            key=lambda x: (isinstance(x[0], str), x[0])),
        key=lambda x: x[1],
        reverse=True)
    self._mapping = {self._unknown_token: 0}
    if self._support_reverse:
      self._reverse_mapping = [self._unknown_token]
    idx = 1
    for category, count in self._freq:
      if max_frequency > 0 and count >= max_frequency:
        continue
      if count <= min_frequency:
        break
      self._mapping[category] = idx
      idx += 1
      if self._support_reverse:
        self._reverse_mapping.append(category)
    self._freq = dict(self._freq[:idx - 1])

  def reverse(self, class_id):
    """Given class id reverse to original class name.

    Args:
      class_id: Id of the class.

    Returns:
      Class name.

    Raises:
      ValueError: if this vocabulary wasn't initalized with support_reverse.
    """
    if not self._support_reverse:
      raise ValueError("This vocabulary wasn't initalized with "
                       "support_reverse to support reverse() function.")
    return self._reverse_mapping[class_id]
