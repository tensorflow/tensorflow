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
"""Ops for representing Bayesian computation.

## This package provides classes for Bayesian computation with TensorFlow.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,wildcard-import,line-too-long
from tensorflow.contrib.bayesflow.python.ops import entropy
from tensorflow.contrib.bayesflow.python.ops import monte_carlo
from tensorflow.contrib.bayesflow.python.ops import special_math
from tensorflow.contrib.bayesflow.python.ops import stochastic_gradient_estimators
from tensorflow.contrib.bayesflow.python.ops import stochastic_graph
from tensorflow.contrib.bayesflow.python.ops import stochastic_tensor
from tensorflow.contrib.bayesflow.python.ops import stochastic_variables
from tensorflow.contrib.bayesflow.python.ops import variational_inference
