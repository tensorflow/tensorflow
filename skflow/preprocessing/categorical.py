"""Implements preprocesing transformers for categorical variables."""
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

import math
import numpy as np

from skflow.preprocessing import categorical_vocabulary


class CategoricalProcessor(object):
    """Maps documents to sequences of word ids.

    As a common convention, Nan values are handled as unknown tokens.
    Both float('nan') and np.nan are accepted.

    Parameters:
        min_frequency: Minimum frequency of categories in the vocabulary.
        vocabulary: CategoricalVocabulary object.

    Attributes:
        vocabulary_: CategoricalVocabulary object.
    """

    def __init__(self, min_frequency=0, vocabulary=None):
        self.min_frequency = min_frequency
        if vocabulary:
            self.vocabulary_ = vocabulary
        else:
            self.vocabulary_ = categorical_vocabulary.CategoricalVocabulary()

    def fit(self, X, unused_y=None):
        """Learn a vocabulary dictionary of all categories in X.

        Args:
            raw_documents: numpy array or iterable.
            unused_y: to match fit format signature of estimators.

        Returns:
            self
        """
        for row in X:
            # Nans are handled as unknowns.
            if (type(row) == float and math.isnan(row)) or row == np.nan:
                continue
            self.vocabulary_.add(row)
        if self.min_frequency > 0:
            self.vocabulary_.trim(self.min_frequency)
        self.vocabulary_.freeze()
        return self

    def fit_transform(self, X, unused_y=None):
        """Learn the vocabulary dictionary and return indexies of categories.

        Args:
            X: numpy array or iterable.
            unused_y: to match fit_transform signature of estimators.

        Returns:
            X: iterable, [n_samples]. Category-id matrix.
        """
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        """Transform documents to category-id matrix.

        Converts categories to ids give fitted vocabulary from `fit` or
        one provided in the constructor.

        Args:
            X: numpy array or iterable.

        Returns:
            X: iterable, [n_samples]. Category-id matrix.
        """
        for row in X:
            # Return <UNK> when it's Nan.
            if (row is float and math.isnan(row)) or row == np.nan:
                yield 0
            yield self.vocabulary_.get(row)

