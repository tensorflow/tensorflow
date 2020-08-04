# Lint as: python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.python.distribute.combinations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest

from absl.testing import parameterized

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.framework import combinations as framework_combinations
from tensorflow.python.platform import test


class ClusterParametersTest(test.TestCase, parameterized.TestCase):
  # For this test we need to use `framework.test_combinations` because our
  # `generate` eats the cluster parameters.
  #
  # Note that we don't have a standalone combination for ClusterParameters, so
  # we should use GPUCombination which contains it.

  @framework_combinations.generate(
      framework_combinations.combine(distribution=[
          combinations.NamedDistribution(
              "HasClusterParams", lambda: None, has_chief=True, num_workers=2),
      ]),
      test_combinations=(combinations.GPUCombination(),))
  def testClusterParams(self, distribution, has_chief, num_workers):
    self.assertTrue(has_chief)
    self.assertEqual(num_workers, 2)

  @framework_combinations.generate(
      framework_combinations.combine(distribution=[
          combinations.NamedDistribution("NoClusterParams", lambda: None),
      ]),
      test_combinations=(combinations.GPUCombination(),))
  def testClusterParamsHasDefault(self, distribution, has_chief, num_workers):
    self.assertFalse(has_chief)
    self.assertEqual(num_workers, 1)

  @framework_combinations.generate(
      framework_combinations.combine(v=1),
      test_combinations=(combinations.GPUCombination(),))
  def testClusterParamsNoStrategy(self, v, has_chief, num_workers):
    self.assertFalse(has_chief)
    self.assertEqual(num_workers, 1)

  @framework_combinations.generate(
      framework_combinations.combine(distribution=[
          combinations.NamedDistribution(
              "WithClusterParams", lambda: None, has_chief=True, num_workers=2),
          combinations.NamedDistribution("WithoutClusterParams", lambda: None),
      ]),
      test_combinations=(combinations.GPUCombination(),))
  def testClusterParamsAreOptional(self, distribution):
    # If combinations library doesn't raise an exception, the test is passed.
    pass

  @framework_combinations.generate(
      framework_combinations.combine(
          ds1=combinations.NamedDistribution(
              "Strategy1", lambda: None, has_chief=True, num_workers=0),
          ds2=combinations.NamedDistribution(
              "Strategy2", lambda: None, has_chief=False, num_workers=1),
          ds3=combinations.NamedDistribution(
              "Strategy3", lambda: None, has_chief=True, num_workers=0),
      ),
      test_combinations=(combinations.GPUCombination(),))
  def testMultipleDistributionSingleWorker(self, ds1, ds2, ds3):
    # If combinations library doesn't raise an exception, the test is passed.
    pass


# unittest.expectedFailure doesn't work with parameterized test methods, so we
# have to decorate the class instead.
@unittest.expectedFailure
class ClusterParametersShouldFailTest(test.TestCase, parameterized.TestCase):

  @framework_combinations.generate(
      framework_combinations.combine(
          ds1=combinations.NamedDistribution(
              "Strategy1", lambda: None, has_chief=True, num_workers=2),
          ds2=combinations.NamedDistribution(
              "Strategy2", lambda: None, has_chief=True, num_workers=2),
      ),
      test_combinations=(combinations.GPUCombination(),))
  def testMultipleDistributionMultiWorker(self, ds1, ds2):
    # combinations library should raise an exception.
    pass

  @combinations.generate(combinations.combine(num_workers=2,))
  def testUseWithoutStrategy(self):
    # There's no perfect way to check if the test runs in a subprocess. We
    # approximate by checking the presence of TF_CONFIG, which is normally not
    # set to the main process.
    self.assertNotEqual(os.getenv("TF_CONFIG"), "")
    raise ValueError("actually run")


# Tests that we *actually* run the test method in multiple workers instead of
# just passing silently. More importantly, it verifies that the test can fail.
# Note that unittest.expectedFailure doesn't work with parameterized test
# methods, so we have to decorate the class instead.
@unittest.expectedFailure
class CombinationsExpectedFailureTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(distribution=[
          combinations.NamedDistribution(
              "OneChiefOneWorker", lambda: None, has_chief=True, num_workers=1),
          combinations.NamedDistribution(
              "TwoWorkers", lambda: None, has_chief=False, num_workers=2),
      ]))
  def testMultiWorkerCanFail(self, distribution):
    resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
    # This should fail.
    self.assertIsNone(resolver.task_id)


# Tests that we *actually* run the test method in multiple workers instead of
# just passing silently. More importantly, it verifies that the test can fail.
# Note that unittest.expectedFailure doesn't work with parameterized test
# methods, so we have to decorate the class instead.
@unittest.expectedFailure
@combinations.generate(
    combinations.combine(distribution=[
        combinations.NamedDistribution(
            "OneChiefOneWorker", lambda: None, has_chief=True, num_workers=1),
        combinations.NamedDistribution(
            "TwoWorkers", lambda: None, has_chief=False, num_workers=2),
    ]))
class CombinationsOnClassMultiWorkerExpectedFailureTest(test.TestCase,
                                                        parameterized.TestCase):

  def test(self, distribution):
    resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
    # This should fail.
    self.assertIsNone(resolver.task_id)


if __name__ == "__main__":
  combinations.main()
