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
"""A simple smoke test that runs these examples for 1 training iteraton."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import pandas as pd

from six.moves import StringIO

import tensorflow.examples.get_started.regression.imports85 as imports85

import tensorflow.examples.get_started.regression.dnn_regression as dnn_regression         # pylint: disable=g-bad-import-order,g-import-not-at-top
import tensorflow.examples.get_started.regression.linear_regression as linear_regression
import tensorflow.examples.get_started.regression.linear_regression_categorical as linear_regression_categorical

from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


def four_lines():
  # pylint: disable=line-too-long
  text = StringIO("""
      1,?,alfa-romero,gas,std,two,hatchback,rwd,front,94.50,171.20,65.50,52.40,2823,ohcv,six,152,mpfi,2.68,3.47,9.00,154,5000,19,26,16500
      2,164,audi,gas,std,four,sedan,fwd,front,99.80,176.60,66.20,54.30,2337,ohc,four,109,mpfi,3.19,3.40,10.00,102,5500,24,30,13950
      2,164,audi,gas,std,four,sedan,4wd,front,99.40,176.60,66.40,54.30,2824,ohc,five,136,mpfi,3.19,3.40,8.00,115,5500,18,22,17450
      2,?,audi,gas,std,two,sedan,fwd,front,99.80,177.30,66.30,53.10,2507,ohc,five,136,mpfi,3.19,3.40,8.50,110,5500,19,25,15250""")
  # pylint: enable=line-too-long

  return pd.read_csv(text, names=imports85.header.keys(),
                     dtype=imports85.header, na_values='?')


class RegressionTest(googletest.TestCase):
  """Test the regression examples in this directory."""

  @test.mock.patch.dict(imports85.__dict__, {'raw': four_lines})
  @test.mock.patch.dict(linear_regression.__dict__, {'STEPS': 1})
  @test.mock.patch.dict(sys.modules, {'imports85': imports85})
  def test_linear_regression(self):
    linear_regression.main([])

  @test.mock.patch.dict(imports85.__dict__, {'raw': four_lines})
  @test.mock.patch.dict(linear_regression_categorical.__dict__, {'STEPS': 1})
  @test.mock.patch.dict(sys.modules, {'imports85': imports85})
  def test_linear_regression_categorical(self):
    linear_regression_categorical.main([])

  @test.mock.patch.dict(imports85.__dict__, {'raw': four_lines})
  @test.mock.patch.dict(dnn_regression.__dict__, {'STEPS': 1})
  @test.mock.patch.dict(sys.modules, {'imports85': imports85})
  def test_dnn_regression(self):
    dnn_regression.main([])


if __name__ == '__main__':
  googletest.main()
