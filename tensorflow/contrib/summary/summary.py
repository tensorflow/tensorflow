# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""TensorFlow Summary API v2.

The operations in this package are safe to use with eager execution turned on or
off. It has a more flexible API that allows summaries to be written directly
from ops to places other than event log files, rather than propagating protos
from @{tf.summary.merge_all} to @{tf.summary.FileWriter}.

To use with eager execution enabled, write your code as follows:

```python
global_step = tf.train.get_or_create_global_step()
summary_writer = tf.contrib.summary.create_file_writer(
    train_dir, flush_millis=10000)
with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
  # model code goes here
  # and in it call
  tf.contrib.summary.scalar("loss", my_loss)
  # In this case every call to tf.contrib.summary.scalar will generate a record
  # ...
```

To use it with graph execution, write your code as follows:

```python
global_step = tf.train.get_or_create_global_step()
summary_writer = tf.contrib.summary.create_file_writer(
    train_dir, flush_millis=10000)
with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
  # model definition code goes here
  # and in it call
  tf.contrib.summary.scalar("loss", my_loss)
  # In this case every call to tf.contrib.summary.scalar will generate an op,
  # note the need to run tf.contrib.summary.all_summary_ops() to make sure these
  # ops get executed.
  # ...
  train_op = ....

with tf.Session(...) as sess:
  tf.global_variables_initializer().run()
  tf.contrib.summary.initialize(graph=tf.get_default_graph())
  # ...
  while not_done_training:
    sess.run([train_op, tf.contrib.summary.all_summary_ops()])
    # ...
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
from tensorflow.python.ops.summary_ops_v2 import all_summary_ops
from tensorflow.python.ops.summary_ops_v2 import always_record_summaries
from tensorflow.python.ops.summary_ops_v2 import audio
from tensorflow.python.ops.summary_ops_v2 import create_db_writer
from tensorflow.python.ops.summary_ops_v2 import create_file_writer
from tensorflow.python.ops.summary_ops_v2 import create_summary_file_writer
from tensorflow.python.ops.summary_ops_v2 import eval_dir
from tensorflow.python.ops.summary_ops_v2 import flush
from tensorflow.python.ops.summary_ops_v2 import generic
from tensorflow.python.ops.summary_ops_v2 import graph
from tensorflow.python.ops.summary_ops_v2 import histogram
from tensorflow.python.ops.summary_ops_v2 import image
from tensorflow.python.ops.summary_ops_v2 import import_event
from tensorflow.python.ops.summary_ops_v2 import initialize
from tensorflow.python.ops.summary_ops_v2 import never_record_summaries
from tensorflow.python.ops.summary_ops_v2 import record_summaries_every_n_global_steps
from tensorflow.python.ops.summary_ops_v2 import scalar
from tensorflow.python.ops.summary_ops_v2 import should_record_summaries
from tensorflow.python.ops.summary_ops_v2 import summary_writer_initializer_op
from tensorflow.python.ops.summary_ops_v2 import SummaryWriter
