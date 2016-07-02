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
"""##Ops for evaluation metrics and summary statistics.

### API

This module provides functions for computing streaming metrics: metrics computed
on dynamically valued `Tensors`. Each metric declaration returns a
"value_tensor", an idempotent operation that returns the current value of the
metric, and an "update_op", an operation that accumulates the information
from the current value of the `Tensors` being measured as well as returns the
value of the "value_tensor".

To use any of these metrics, one need only declare the metric, call `update_op`
repeatedly to accumulate data over the desired number of `Tensor` values (often
each one is a single batch) and finally evaluate the value_tensor. For example,
to use the `streaming_mean`:

```python
value = ...
mean_value, update_op = tf.contrib.metrics.streaming_mean(values)
sess.run(tf.initialize_local_variables())

for i in range(number_of_batches):
  print('Mean after batch %d: %f' % (i, update_op.eval())
print('Final Mean: %f' % mean_value.eval())
```

Each metric function adds nodes to the graph that hold the state necessary to
compute the value of the metric as well as a set of operations that actually
perform the computation. Every metric evaluation is composed of three steps

* Initialization: initializing the metric state.
* Aggregation: updating the values of the metric state.
* Finalization: computing the final metric value.

In the above example, calling streaming_mean creates a pair of state variables
that will contain (1) the running sum and (2) the count of the number of samples
in the sum.  Because the streaming metrics use local variables,
the Initialization stage is performed by running the op returned
by `tf.initialize_local_variables()`. It sets the sum and count variables to
zero.

Next, Aggregation is performed by examining the current state of `values`
and incrementing the state variables appropriately. This step is executed by
running the `update_op` returned by the metric.

Finally, finalization is performed by evaluating the "value_tensor"

In practice, we commonly want to evaluate across many batches and multiple
metrics. To do so, we need only run the metric computation operations multiple
times:

```python
labels = ...
predictions = ...
accuracy, update_op_acc = tf.contrib.metrics.streaming_accuracy(
    labels, predictions)
error, update_op_error = tf.contrib.metrics.streaming_mean_absolute_error(
    labels, predictions)

sess.run(tf.initialize_local_variables())
for batch in range(num_batches):
  sess.run([update_op_acc, update_op_error])

accuracy, mean_absolute_error = sess.run([accuracy, mean_absolute_error])
```

Note that when evaluating the same metric multiple times on different inputs,
one must specify the scope of each metric to avoid accumulating the results
together:

```python
labels = ...
predictions0 = ...
predictions1 = ...

accuracy0 = tf.contrib.metrics.accuracy(labels, predictions0, name='preds0')
accuracy1 = tf.contrib.metrics.accuracy(labels, predictions1, name='preds1')
```

Certain metrics, such as streaming_mean or streaming_accuracy, can be weighted
via a `weights` argument. The `weights` tensor must be the same size as the
labels and predictions tensors and results in a weighted average of the metric.

Other metrics, such as streaming_recall, streaming_precision, and streaming_auc,
are not well defined with regard to weighted samples. However, a binary
`ignore_mask` argument can be used to ignore certain values at graph executation
time.

## Metric `Ops`

@@streaming_accuracy
@@streaming_mean
@@streaming_recall
@@streaming_precision
@@streaming_auc
@@streaming_recall_at_k
@@streaming_mean_absolute_error
@@streaming_mean_relative_error
@@streaming_mean_squared_error
@@streaming_root_mean_squared_error
@@streaming_mean_cosine_distance
@@streaming_percentage_less
@@streaming_sparse_precision_at_k
@@streaming_sparse_recall_at_k
@@auc_using_histogram

@@accuracy
@@confusion_matrix

## Set `Ops`

@@set_difference
@@set_intersection
@@set_size
@@set_union

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,line-too-long,g-importing-member,wildcard-import
from tensorflow.contrib.metrics.python.metrics import *
from tensorflow.contrib.metrics.python.ops.confusion_matrix_ops import confusion_matrix
from tensorflow.contrib.metrics.python.ops.histogram_ops import auc_using_histogram
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_accuracy
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_auc
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_mean
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_mean_absolute_error
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_mean_cosine_distance
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_mean_relative_error
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_mean_squared_error
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_percentage_less
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_precision
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_precision_at_thresholds
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_recall
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_recall_at_k
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_recall_at_thresholds
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_root_mean_squared_error
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_sparse_precision_at_k
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_sparse_recall_at_k
from tensorflow.contrib.metrics.python.ops.set_ops import set_difference
from tensorflow.contrib.metrics.python.ops.set_ops import set_intersection
from tensorflow.contrib.metrics.python.ops.set_ops import set_size
from tensorflow.contrib.metrics.python.ops.set_ops import set_union
from tensorflow.python.util.all_util import make_all


__all__ = make_all(__name__)
