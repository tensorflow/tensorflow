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

"""Utility functions for training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path

from tensorflow.python import pywrap_tensorflow


def global_step(sess, global_step_tensor):
  """Small helper to get the global step.

  ```python
  # Creates a variable to hold the global_step.
  global_step_tensor = tf.Variable(10, trainable=False, name='global_step')
  # Creates a session.
  sess = tf.Session()
  # Initializes the variable.
  sess.run(global_step_tensor.initializer)
  print('global_step: %s' % tf.train.global_step(sess, global_step_tensor))

  global_step: 10
  ```

  Args:
    sess: A TensorFlow `Session` object.
    global_step_tensor:  `Tensor` or the `name` of the operation that contains
      the global step.

  Returns:
    The global step value.
  """
  return int(sess.run(global_step_tensor))


def write_graph(graph_def, logdir, name, as_text=True):
  """Writes a graph proto to a file.

  The graph is written as a binary proto unless `as_text` is `True`.

  ```python
  v = tf.Variable(0, name='my_variable')
  sess = tf.Session()
  tf.train.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt')
  ```

  Args:
    graph_def: A `GraphDef` protocol buffer.
    logdir: Directory where to write the graph. This can refer to remote
      filesystems, such as Google Cloud Storage (GCS).
    name: Filename for the graph.
    as_text: If `True`, writes the graph as an ASCII proto.
  """
  # gcs does not have the concept of directory at the moment.
  if not pywrap_tensorflow.file_exists(logdir) and not logdir.startswith("gs:"):
    pywrap_tensorflow.recursive_create_dir(logdir)
  path = os.path.join(logdir, name)
  if as_text:
    pywrap_tensorflow.write_string_to_file(path, str(graph_def))
  else:
    pywrap_tensorflow.write_string_to_file(path, graph_def.SerializeToString())
