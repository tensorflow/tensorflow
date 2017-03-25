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
"""Grid search tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

from tensorflow.contrib.learn.python import learn
from tensorflow.python.platform import test

HAS_SKLEARN = os.environ.get('TENSORFLOW_SKLEARN', False)
if HAS_SKLEARN:
  try:
    # pylint: disable=g-import-not-at-top
    from sklearn import datasets
    from sklearn.grid_search import GridSearchCV
    from sklearn.metrics import accuracy_score
  except ImportError:
    HAS_SKLEARN = False


class GridSearchTest(test.TestCase):
  """Grid search tests."""

  def testIrisDNN(self):
    if HAS_SKLEARN:
      random.seed(42)
      iris = datasets.load_iris()
      feature_columns = learn.infer_real_valued_columns_from_input(iris.data)
      classifier = learn.DNNClassifier(
          feature_columns=feature_columns,
          hidden_units=[10, 20, 10],
          n_classes=3)
      grid_search = GridSearchCV(
          classifier, {'hidden_units': [[5, 5], [10, 10]]},
          scoring='accuracy',
          fit_params={'steps': [50]})
      grid_search.fit(iris.data, iris.target)
      score = accuracy_score(iris.target, grid_search.predict(iris.data))
      self.assertGreater(score, 0.5, 'Failed with score = {0}'.format(score))


if __name__ == '__main__':
  test.main()
