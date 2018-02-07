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
"""SavedModel simple save functionality."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util.tf_export import tf_export


@tf_export('saved_model.simple_save')
def simple_save(session, export_dir, inputs, outputs, legacy_init_op=None):
  """Convenience function to build a SavedModel suitable for serving.

  In many common cases, saving models for serving will be as simple as:

      simple_save(session,
                  export_dir,
                  inputs={"x": x, "y": y},
                  outputs={"z": z})

  Although in many cases it's not necessary to understand all of the many ways
      to configure a SavedModel, this method has a few practical implications:
    - It will be treated as a graph for inference / serving (i.e. uses the tag
      `tag_constants.SERVING`)
    - The SavedModel will load in TensorFlow Serving and supports the
      [Predict
      API](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/predict.proto).
      To use the Classify, Regress, or MultiInference APIs, please
      use either
      [tf.Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator)
      or the lower level
      [SavedModel
      APIs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md).
    - Some TensorFlow ops depend on information on disk or other information
      called "assets". These are generally handled automatically by adding the
      assets to the `GraphKeys.ASSET_FILEPATHS` collection. Only assets in that
      collection are exported; if you need more custom behavior, you'll need to
      use the
      [SavedModelBuilder](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/builder.py).

  More information about SavedModel and signatures can be found here:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md.

  Args:
    session: The TensorFlow session from which to save the meta graph and
        variables.
    export_dir: The path to which the SavedModel will be stored.
    inputs: dict mapping string input names to tensors. These are added
        to the SignatureDef as the inputs.
    outputs:  dict mapping string output names to tensors. These are added
        to the SignatureDef as the outputs.
    legacy_init_op: Legacy support for op or group of ops to execute after the
        restore op upon a load.
  """
  signature_def_map = {
      signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          signature_def_utils.predict_signature_def(inputs, outputs)
  }
  b = builder.SavedModelBuilder(export_dir)
  b.add_meta_graph_and_variables(
      session,
      tags=[tag_constants.SERVING],
      signature_def_map=signature_def_map,
      assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
      legacy_init_op=legacy_init_op,
      clear_devices=True)
  b.save()
