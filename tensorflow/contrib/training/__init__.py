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
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,wildcard-import
from tensorflow.contrib.training.python.training.sequence_queueing_state_saver import *
from tensorflow.python.util.all_util import make_all

__all__ = make_all(__name__)
