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
"""Helper functions for enqueuing data from arrays and pandas `DataFrame`s."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
from tensorflow.python.estimator.inputs.queues.feeding_functions import _ArrayFeedFn
from tensorflow.python.estimator.inputs.queues.feeding_functions import _enqueue_data
from tensorflow.python.estimator.inputs.queues.feeding_functions import _GeneratorFeedFn
from tensorflow.python.estimator.inputs.queues.feeding_functions import _OrderedDictNumpyFeedFn
from tensorflow.python.estimator.inputs.queues.feeding_functions import _PandasFeedFn
# pylint: enable=unused-import

from tensorflow.python.util.deprecation import deprecated


@deprecated('2017-06-15', 'Moved to tf.contrib.training.enqueue_data.')
def enqueue_data(*args, **kwargs):
  return _enqueue_data(*args, **kwargs)
