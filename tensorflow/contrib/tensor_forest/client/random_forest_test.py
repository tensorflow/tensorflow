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
"""Tests for TensorForestTrainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.platform import test


class TensorForestTrainerTests(test.TestCase):

  def testClassification(self):
    """Tests multi-class classification using matrix data as input."""
    hparams = tensor_forest.ForestHParams(
        num_trees=3,
        max_nodes=1000,
        num_classes=3,
        num_features=4,
        split_after_samples=20,
        inference_tree_paths=True)
    classifier = random_forest.TensorForestEstimator(hparams.fill())

    iris = base.load_iris()
    data = iris.data.astype(np.float32)
    labels = iris.target.astype(np.int32)

    classifier.fit(x=data, y=labels, steps=100, batch_size=50)
    classifier.evaluate(x=data, y=labels, steps=10)

  def testRegression(self):
    """Tests multi-class classification using matrix data as input."""

    hparams = tensor_forest.ForestHParams(
        num_trees=3,
        max_nodes=1000,
        num_classes=1,
        num_features=13,
        regression=True,
        split_after_samples=20)

    regressor = random_forest.TensorForestEstimator(hparams.fill())

    boston = base.load_boston()
    data = boston.data.astype(np.float32)
    labels = boston.target.astype(np.int32)

    regressor.fit(x=data, y=labels, steps=100, batch_size=50)
    regressor.evaluate(x=data, y=labels, steps=10)


if __name__ == "__main__":
  test.main()
