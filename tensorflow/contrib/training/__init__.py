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
"""Training and input utilities.

See @{$python/contrib.training} guide.

@@batch_sequences_with_states
@@NextQueuedSequenceBatch
@@SequenceQueueingStateSaver
@@rejection_sample
@@resample_at_rate
@@stratified_sample
@@weighted_resample
@@bucket
@@bucket_by_sequence_length
@@GreedyLoadBalancingStrategy
@@byte_size_load_fn
@@FailureTolerator
@@rejection_sample
@@stratified_sample
@@resample_at_rate
@@weighted_resample
@@HParams
@@HParamDef
@@parse_values
@@python_input
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,wildcard-import
from tensorflow.contrib.training.python.training.bucket_ops import *
from tensorflow.contrib.training.python.training.device_setter import *
from tensorflow.contrib.training.python.training.evaluation import checkpoints_iterator
from tensorflow.contrib.training.python.training.evaluation import evaluate_once
from tensorflow.contrib.training.python.training.evaluation import evaluate_repeatedly
from tensorflow.contrib.training.python.training.evaluation import get_or_create_eval_step
from tensorflow.contrib.training.python.training.evaluation import StopAfterNEvalsHook
from tensorflow.contrib.training.python.training.evaluation import SummaryAtEndHook
from tensorflow.contrib.training.python.training.evaluation import wait_for_new_checkpoint
from tensorflow.contrib.training.python.training.feeding_queue_runner import FeedingQueueRunner
from tensorflow.contrib.training.python.training.hparam import *
from tensorflow.contrib.training.python.training.python_input import python_input
from tensorflow.contrib.training.python.training.resample import *
from tensorflow.contrib.training.python.training.sampling_ops import *
from tensorflow.contrib.training.python.training.sequence_queueing_state_saver import *
from tensorflow.contrib.training.python.training.training import add_gradients_summaries
from tensorflow.contrib.training.python.training.training import clip_gradient_norms
from tensorflow.contrib.training.python.training.training import create_train_op
from tensorflow.contrib.training.python.training.training import multiply_gradients
from tensorflow.contrib.training.python.training.training import train
from tensorflow.contrib.training.python.training.tuner import Tuner
# pylint: enable=unused-import,wildcard-import

from tensorflow.python.util.all_util import remove_undocumented

# Allow explicitly imported symbols. Symbols imported with * must also be
# whitelisted here or in the module docstring above.
_allowed_symbols = [
    'checkpoints_iterator', 'evaluate_once', 'evaluate_repeatedly',
    'FeedingQueueRunner', 'get_or_create_eval_step', 'StopAfterNEvalsHook',
    'SummaryAtEndHook', 'wait_for_new_checkpoint', 'add_gradients_summaries',
    'clip_gradient_norms', 'create_train_op', 'multiply_gradients', 'train']

remove_undocumented(__name__, _allowed_symbols)
