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

"""Implements preprocessing transformers for categorical variables (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

from tensorflow.python.util.deprecation import deprecated

# pylint: disable=g-bad-import-order
from . import categorical_vocabulary
from ..learn_io.data_feeder import setup_processor_data_feeder
# pylint: enable=g-bad-import-order


class CategoricalProcessor(object):
  """Maps documents to sequences of word ids.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  As a common convention, Nan values are handled as unknown tokens.
  Both float('nan') and np.nan are accepted.
  """

  @deprecated(None, 'Please use tensorflow/transform or tf.data for sequence '
              'processing.')
  def __init__(self, min_frequency=0, share=False, vocabularies=None):
    """Initializes a CategoricalProcessor instance.

    Args:
      min_frequency: Minimum frequency of categories in the vocabulary.
      share: Share vocabulary between variables.
      vocabularies: list of CategoricalVocabulary objects for each variable in
        the input dataset.

    Attributes:
      vocabularies_: list of CategoricalVocabulary objects.
    """
    self.min_frequency = min_frequency
    self.share = share
    self.vocabularies_ = vocabularies

  def freeze(self, freeze=True):
    """Freeze or unfreeze all vocabularies.

    Args:
      freeze: Boolean, indicate if vocabularies should be frozen.
    """
    for vocab in self.vocabularies_:
      vocab.freeze(freeze)

  def fit(self, x, unused_y=None):
    """Learn a vocabulary dictionary of all categories in `x`.

    Args:
      x: numpy matrix or iterable of lists/numpy arrays.
      unused_y: to match fit format signature of estimators.

    Returns:
      self
    """
    x = setup_processor_data_feeder(x)
    for row in x:
      # Create vocabularies if not given.
      if self.vocabularies_ is None:
        # If not share, one per column, else one shared across.
        if not self.share:
          self.vocabularies_ = [
              categorical_vocabulary.CategoricalVocabulary() for _ in row
          ]
        else:
          vocab = categorical_vocabulary.CategoricalVocabulary()
          self.vocabularies_ = [vocab for _ in row]
      for idx, value in enumerate(row):
        # Nans are handled as unknowns.
        if (isinstance(value, float) and math.isnan(value)) or value == np.nan:
          continue
        self.vocabularies_[idx].add(value)
    if self.min_frequency > 0:
      for vocab in self.vocabularies_:
        vocab.trim(self.min_frequency)
    self.freeze()
    return self

  def fit_transform(self, x, unused_y=None):
    """Learn the vocabulary dictionary and return indexies of categories.

    Args:
      x: numpy matrix or iterable of lists/numpy arrays.
      unused_y: to match fit_transform signature of estimators.

    Returns:
      x: iterable, [n_samples]. Category-id matrix.
    """
    self.fit(x)
    return self.transform(x)

  def transform(self, x):
    """Transform documents to category-id matrix.

    Converts categories to ids give fitted vocabulary from `fit` or
    one provided in the constructor.

    Args:
      x: numpy matrix or iterable of lists/numpy arrays.

    Yields:
      x: iterable, [n_samples]. Category-id matrix.
    """
    self.freeze()
    x = setup_processor_data_feeder(x)
    for row in x:
      output_row = []
      for idx, value in enumerate(row):
        # Return <UNK> when it's Nan.
        if (isinstance(value, float) and math.isnan(value)) or value == np.nan:
          output_row.append(0)
          continue
        output_row.append(self.vocabularies_[idx].get(value))
      yield np.array(output_row, dtype=np.int64)
