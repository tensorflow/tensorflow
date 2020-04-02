# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Various learning rate decay functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.keras.optimizer_v2 import legacy_learning_rate_decay as learning_rate_decay


exponential_decay = learning_rate_decay.exponential_decay
piecewise_constant = learning_rate_decay.piecewise_constant
polynomial_decay = learning_rate_decay.polynomial_decay
natural_exp_decay = learning_rate_decay.natural_exp_decay
inverse_time_decay = learning_rate_decay.inverse_time_decay
cosine_decay = learning_rate_decay.cosine_decay
cosine_decay_restarts = learning_rate_decay.cosine_decay_restarts
linear_cosine_decay = learning_rate_decay.linear_cosine_decay
noisy_linear_cosine_decay = learning_rate_decay.noisy_linear_cosine_decay
