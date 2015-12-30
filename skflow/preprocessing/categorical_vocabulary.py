"""Categorical vocabulary classes to map categories to indexes.

Can be used for categorical variables, sparse variables and words.
"""

#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import collections
import six


class CategoricalVocabulary(object):
    """Categorical variables vocabulary class.

    Accumulates and provides mapping from classes to indexes.
    Can be easily used for words.
    """

    def __init__(self, unknown_token='<UNK>'):
        self._mapping = {unknown_token: 0}
        self._freq = collections.defaultdict(int)
        self._freeze = False

    def __len__(self):
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

        Args:
            min_frequency: minimum frequency to keep.
            max_frequency: optional, maximum frequency to keep.
                Useful to remove very frequent categories (like stop words).
        """
        for category, count in six.iteritems(self._freq):
            if count <= min_frequency and (max_frequency < 0 or
                                           count >= max_frequency):
                self._mapping.pop(category)

