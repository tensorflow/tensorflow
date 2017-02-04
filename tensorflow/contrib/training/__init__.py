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

## Splitting sequence inputs into minibatches with state saving

Use [`SequenceQueueingStateSaver`](#SequenceQueueingStateSaver) or
its wrapper [`batch_sequences_with_states`](#batch_sequences_with_states) if
you have input data with a dynamic primary time / frame count axis which
you'd like to convert into fixed size segments during minibatching, and would
like to store state in the forward direction across segments of an example.

@@batch_sequences_with_states
@@NextQueuedSequenceBatch
@@SequenceQueueingStateSaver


## Online data resampling

To resample data with replacement on a per-example basis, use
['rejection_sample'](#rejection_sample) or
['resample_at_rate'](#resample_at_rate). For `rejection_sample`, provide
a boolean Tensor describing whether to accept or reject. Resulting batch sizes
are always the same. For `resample_at_rate`, provide the desired rate for each
example. Resulting batch sizes may vary. If you wish to specify relative
rates, rather than absolute ones, use ['weighted_resample'](#weighted_resample)
(which also returns the actual resampling rate used for each output example).

Use ['stratified_sample'](#stratified_sample) to resample without replacement
from the data to achieve a desired mix of class proportions that the Tensorflow
graph sees. For instance, if you have a binary classification dataset that is
99.9% class 1, a common approach is to resample from the data so that the data
is more balanced.

@@rejection_sample
@@resample_at_rate
@@stratified_sample
@@weighted_resample

## Bucketing

Use ['bucket'](#bucket) or
['bucket_by_sequence_length'](#bucket_by_sequence_length) to stratify
minibatches into groups ("buckets").  Use `bucket_by_sequence_length`
with the argument `dynamic_pad=True` to receive minibatches of similarly
sized sequences for efficient training via `dynamic_rnn`.

@@bucket
@@bucket_by_sequence_length
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
from tensorflow.contrib.training.python.training.failure_tolerator import *
from tensorflow.contrib.training.python.training.feeder import *
from tensorflow.contrib.training.python.training.resample import *
from tensorflow.contrib.training.python.training.sampling_ops import *
from tensorflow.contrib.training.python.training.sequence_queueing_state_saver import *
from tensorflow.contrib.training.python.training.training import add_gradients_summaries
from tensorflow.contrib.training.python.training.training import clip_gradient_norms
from tensorflow.contrib.training.python.training.training import create_train_op
from tensorflow.contrib.training.python.training.training import multiply_gradients
from tensorflow.contrib.training.python.training.training import train
from tensorflow.python.util.all_util import make_all
# pylint: enable=unused-import,wildcard-import

__all__ = make_all(__name__)
