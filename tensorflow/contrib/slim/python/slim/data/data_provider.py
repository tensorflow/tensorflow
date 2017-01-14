# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains code for the DataProvider.

A DataProvider is a class which provides some predefined data types from some
source (SSTable, TFRecord, CNS directory, etc). The most basic function of a
data provider is the `Get` operation where one requests one or more types of
data, or 'items':

  provider.get(items=['image', 'sentence', 'class'])

More concretely, a data provider (a subclass of BaseDataProvider) returns a
single tensor for each requested item (data type):

  provider = MyDataProvider(...)
  image, sentence, clazz = provider.get(['image', 'sentence', 'class'])

In this example, the provider `MyDataProvider` must know how to load each item.
A data provider may be written in a way that the logic necessary to map from
each item to tensor is completely encapsulated within the data_provider itself.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class DataProvider(object):
  """Maps a list of requested data items to tensors from a data source.

  All data providers must inherit from DataProvider and implement the Get
  method which returns arbitrary types of data. No assumption is made about the
  source of the data nor the mechanism for providing it.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, items_to_tensors, num_samples):
    """Constructs the Data Provider.

    Args:
      items_to_tensors: a dictionary of names to tensors.
      num_samples: the number of samples in the dataset being provided.
    """
    self._items_to_tensors = items_to_tensors
    self._num_samples = num_samples

  def get(self, items):
    """Returns a list of tensors specified by the given list of items.

    The list of items is arbitrary different data providers satisfy different
    lists of items. For example the Pascal VOC might accept items 'image' and
    'semantics', whereas the NYUDepthV2 data provider might accept items
    'image', 'depths' and 'normals'.

    Args:
      items: a list of strings, each of which indicate a particular data type.

    Returns:
      a list of tensors, whose length matches the length of `items`, where each
      tensor corresponds to each item.

    Raises:
      ValueError: if any of the items cannot be satisfied.
    """
    self._validate_items(items)
    return [self._items_to_tensors[item] for item in items]

  def list_items(self):
    """Returns the list of item names that can be provided by the data provider.

    Returns:
      a list of item names that can be passed to Get([items]).
    """
    return self._items_to_tensors.keys()

  def num_samples(self):
    """Returns the number of data samples in the dataset.

    Returns:
      a positive whole number.
    """
    return self._num_samples

  def _validate_items(self, items):
    """Verifies that each given item is a member of the list from ListItems().

    Args:
      items: a list or tuple of strings.

    Raises:
      ValueError: if `items` is not a tuple or list or if any of the elements of
        `items` is not found in the list provided by self.ListItems().
    """
    if not isinstance(items, (list, tuple)):
      raise ValueError('items must be a list or tuple')

    valid_items = self.list_items()
    for item in items:
      if item not in valid_items:
        raise ValueError(
            'Item [%s] is invalid. Valid entries include: %s' %
            (item, valid_items))
