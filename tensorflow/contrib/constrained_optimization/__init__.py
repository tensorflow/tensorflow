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
"""A library for performing constrained optimization in TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=wildcard-import
from tensorflow.contrib.constrained_optimization.python.candidates import *
from tensorflow.contrib.constrained_optimization.python.constrained_minimization_problem import *
from tensorflow.contrib.constrained_optimization.python.constrained_optimizer import *
from tensorflow.contrib.constrained_optimization.python.external_regret_optimizer import *
from tensorflow.contrib.constrained_optimization.python.swap_regret_optimizer import *
# pylint: enable=wildcard-import

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    "AdditiveExternalRegretOptimizer",
    "AdditiveSwapRegretOptimizer",
    "ConstrainedMinimizationProblem",
    "ConstrainedOptimizer",
    "find_best_candidate_distribution",
    "find_best_candidate_index",
    "MultiplicativeSwapRegretOptimizer",
]

remove_undocumented(__name__, _allowed_symbols)
