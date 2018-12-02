# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Core module for TensorFlow distribution objects and helpers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util import deprecation


# pylint: disable=wildcard-import,unused-import,g-import-not-at-top
with deprecation.silence():
  from tensorflow.python.ops.distributions.bernoulli import Bernoulli
  from tensorflow.python.ops.distributions.beta import Beta
  from tensorflow.python.ops.distributions.categorical import Categorical
  from tensorflow.python.ops.distributions.dirichlet import Dirichlet
  from tensorflow.python.ops.distributions.dirichlet_multinomial import DirichletMultinomial
  from tensorflow.python.ops.distributions.distribution import *
  from tensorflow.python.ops.distributions.exponential import Exponential
  from tensorflow.python.ops.distributions.gamma import Gamma
  from tensorflow.python.ops.distributions.kullback_leibler import *
  from tensorflow.python.ops.distributions.laplace import Laplace
  from tensorflow.python.ops.distributions.multinomial import Multinomial
  from tensorflow.python.ops.distributions.normal import Normal
  from tensorflow.python.ops.distributions.student_t import StudentT
  from tensorflow.python.ops.distributions.uniform import Uniform
# pylint: enable=wildcard-import,unused-import
del deprecation
