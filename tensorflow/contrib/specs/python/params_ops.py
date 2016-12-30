# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Operators for concise TensorFlow parameter specifications.

This module is used as an environment for evaluating expressions
in the "params" DSL.

Specifications are intended to assign simple numerical
values. Examples:

    --params "n=64; d=5" --spec "(Cr(n) | Mp([2, 2])) ** d | Fm"

The random parameter primitives are useful for running large numbers
of experiments with randomly distributed parameters:

    --params "n=Li(5,500); d=Ui(1,5)" --spec "(Cr(n) | Mp([2, 2])) ** d | Fm"

Internally, this might be implemented as follows:

    params = specs.create_params(FLAGS.params, {})
    logging.info(repr(params))
    net = specs.create_net(FLAGS.spec, inputs, params)

Note that separating the specifications into parameters and network
creation allows us to log the random parameter values easily.

The implementation of this will change soon in order to support
hyperparameter tuning with steering. Instead of returning a number,
the primitives below will return a class instance that is then
used to generate a random number by the framework.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Lint disabled because these are operators in the DSL, not regular
# Python functions.
# pylint: disable=invalid-name
# pylint: disable=wildcard-import,unused-wildcard-import,redefining-builtin
# pylint: disable=redefined-builtin,g-importing-member,no-member
# make available all math expressions
import math
from math import *
import random
# pylint: enable=wildcard-import,unused-wildcard-import,redefining-builtin
# pylint: enable=redefined-builtin,g-importing-member,no-member


def Uf(lo=0.0, hi=1.0):
  """Uniformly distributed floating number."""
  return random.uniform(lo, hi)


def Ui(lo, hi):
  """Uniformly distributed integer, inclusive limits."""
  return random.randint(lo, hi)


def Lf(lo, hi):
  """Log-uniform distributed floatint point number."""
  return math.exp(random.uniform(math.log(lo), math.log(hi)))


def Li(lo, hi):
  """Log-uniform distributed integer, inclusive limits."""
  return int(math.floor(math.exp(random.uniform(math.log(lo),
                                                math.log(hi+1-1e-5)))))


def Nt(mu, sigma, limit=3.0):
  """Normally distributed floating point number with truncation."""
  return min(max(random.gauss(mu, sigma), mu-limit*sigma), mu+limit*sigma)


# pylint: enable=invalid-name
