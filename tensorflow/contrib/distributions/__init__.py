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
"""Classes representing statistical distributions and ops for working with them.

## Classes for statistical distributions.

Classes that represent batches of statistical distributions.  Each class is
initialized with parameters that define the distributions.

### Base classes

@@BaseDistribution
@@ContinuousDistribution
@@DiscreteDistribution

### Univariate (scalar) distributions

@@Chi2
@@Exponential
@@Gamma
@@Normal
@@StudentT
@@Uniform

### Multivariate distributions

@@MultivariateNormal
@@DirichletMultinomial

## Posterior inference with conjugate priors.

Functions that transform conjugate prior/likelihood pairs to distributions
representing the posterior or posterior predictive.

### Normal likelihood with conjugate prior.

@@normal_conjugates_known_sigma_posterior
@@normal_congugates_known_sigma_predictive
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,wildcard-import,line-too-long

from tensorflow.contrib.distributions.python.ops.chi2 import *
from tensorflow.contrib.distributions.python.ops.dirichlet_multinomial import *
from tensorflow.contrib.distributions.python.ops.distribution import *
from tensorflow.contrib.distributions.python.ops.exponential import *
from tensorflow.contrib.distributions.python.ops.gamma import *
from tensorflow.contrib.distributions.python.ops.mvn import *
from tensorflow.contrib.distributions.python.ops.normal import *
from tensorflow.contrib.distributions.python.ops.normal_conjugate_posteriors import *
from tensorflow.contrib.distributions.python.ops.student_t import *
from tensorflow.contrib.distributions.python.ops.uniform import *
