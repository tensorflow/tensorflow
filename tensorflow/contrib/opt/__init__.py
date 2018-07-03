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
"""A module containing optimization routines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=wildcard-import
from tensorflow.contrib.opt.python.training.adamax import *
from tensorflow.contrib.opt.python.training.addsign import *
from tensorflow.contrib.opt.python.training.drop_stale_gradient_optimizer import *
from tensorflow.contrib.opt.python.training.elastic_average_optimizer import *
from tensorflow.contrib.opt.python.training.external_optimizer import *
from tensorflow.contrib.opt.python.training.ggt import *
from tensorflow.contrib.opt.python.training.lazy_adam_optimizer import *
from tensorflow.contrib.opt.python.training.model_average_optimizer import *
from tensorflow.contrib.opt.python.training.moving_average_optimizer import *
from tensorflow.contrib.opt.python.training.multitask_optimizer_wrapper import *
from tensorflow.contrib.opt.python.training.nadam_optimizer import *
from tensorflow.contrib.opt.python.training.weight_decay_optimizers import *
from tensorflow.contrib.opt.python.training.powersign import *
from tensorflow.contrib.opt.python.training.variable_clipping_optimizer import *
from tensorflow.contrib.opt.python.training.weight_decay_optimizers import *
# pylint: enable=wildcard-import

from tensorflow.python.util.all_util import remove_undocumented


_allowed_symbols = [
    'AdaMaxOptimizer',
    'PowerSignOptimizer',
    'AddSignOptimizer',
    'DelayCompensatedGradientDescentOptimizer',
    'DropStaleGradientOptimizer',
    'ExternalOptimizerInterface',
    'LazyAdamOptimizer',
    'NadamOptimizer',
    'MovingAverageOptimizer',
    'MomentumWOptimizer',
    'AdamWOptimizer',
    'DecoupledWeightDecayExtension',
    'extend_with_decoupled_weight_decay',
    'ScipyOptimizerInterface',
    'VariableClippingOptimizer',
    'MultitaskOptimizerWrapper',
    'clip_gradients_by_global_norm',
    'ElasticAverageOptimizer',
    'ElasticAverageCustomGetter',
    'ModelAverageOptimizer',
    'ModelAverageCustomGetter',
    'GGTOptimizer',
]

remove_undocumented(__name__, _allowed_symbols)
