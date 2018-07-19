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
"""SavedModel simple load functionality."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.saved_model import loader


@tf_export('saved_model.simple_load')
def simple_load(session, export_dir):
  """Load a SavedModel artifact into a python TensorFlow Session.

  This function assumes that the export directory contains a model that has been
  serialized using the @{tf.saved_model.simple_save} function.

  More information about SavedModel and signatures can be found here in the
  @{$guide/saved_model} documentation.

  Complete Example:

  ```python
  import tensorflow as tf

  export_dir = 'example_model_directory'

  # Define a graph.
  graph1 = tf.Graph()
  with graph1.as_default():
    x = tf.placeholder(dtype=tf.int32)
    y = tf.placeholder(dtype=tf.int32)
    z = (x + y) * tf.Variable(5)
    init = tf.global_variables_initializer()

  # Save the graph out as a SavedModel.
  with tf.Session(graph=graph1) as sess1:
    sess1.run(init)
    tf.saved_model.simple_save(sess1,
            export_dir,
            inputs={"x": x, "y": y},
            outputs={"z": z})

  # Create an empty graph.
  graph2 = tf.Graph()

  # Load the SavedModel state into the empty graph.
  with tf.Session(graph=graph2) as sess2:
    inputs, outputs = simple_load(sess2, export_dir)
    result = sess2.run(outputs["z"], feed_dict={inputs["x"]: 2, inputs["y"]: 3})

  assert result == 25
  ```

  Args:
    session: The TensorFlow Session to load the serialized Variables into.
    export_dir: The string path in which the SavedModel was stored.

  Returns:
    inputs: dict mapping string input names to tensors. These are retrieved
        from the inputs SignatureDef.
    outputs:  dict mapping string output names to tensors. These are retrieved
        from the outputs SignatureDef.
  """
  model = loader.load(session, [tag_constants.SERVING], export_dir)
  serving_signature = model.signature_def[
      signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
  ]

  graph = session.graph

  inputs = {
      name: graph.get_tensor_by_name(sig.name)
      for name, sig in serving_signature.inputs.items()
  }
  outputs = {
      name: graph.get_tensor_by_name(sig.name)
      for name, sig in serving_signature.outputs.items()
  }

  return inputs, outputs
