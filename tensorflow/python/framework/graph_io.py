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

"""Utility functions for reading/writing graphs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path

from google.protobuf import text_format
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.util.tf_export import tf_export


@tf_export('io.write_graph', v1=['io.write_graph', 'train.write_graph'])
def write_graph(graph_or_graph_def, logdir, name, as_text=True):
  """Writes a graph proto to a file.

  The graph is written as a text proto unless `as_text` is `False`.

  ```python
  v = tf.Variable(0, name='my_variable')
  sess = tf.Session()
  tf.train.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt')
  ```

  or

  ```python
  v = tf.Variable(0, name='my_variable')
  sess = tf.Session()
  tf.train.write_graph(sess.graph, '/tmp/my-model', 'train.pbtxt')
  ```

  Args:
    graph_or_graph_def: A `Graph` or a `GraphDef` protocol buffer.
    logdir: Directory where to write the graph. This can refer to remote
      filesystems, such as Google Cloud Storage (GCS).
    name: Filename for the graph.
    as_text: If `True`, writes the graph as an ASCII proto.

  Returns:
    The path of the output proto file.
  """
  if isinstance(graph_or_graph_def, ops.Graph):
    graph_def = graph_or_graph_def.as_graph_def()
  else:
    graph_def = graph_or_graph_def

  # gcs does not have the concept of directory at the moment.
  if not file_io.file_exists(logdir) and not logdir.startswith('gs:'):
    file_io.recursive_create_dir(logdir)
  path = os.path.join(logdir, name)
  if as_text:
    file_io.atomic_write_string_to_file(path,
                                        text_format.MessageToString(graph_def))
  else:
    file_io.atomic_write_string_to_file(path, graph_def.SerializeToString())
  return path
