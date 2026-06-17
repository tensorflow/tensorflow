# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Generates a toy v1 saved model for testing."""

import shutil
from absl import app
from absl import flags
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils

flags.DEFINE_string('saved_model_path', '', 'Path to save the model to.')
FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  shutil.rmtree(FLAGS.saved_model_path)

  # Create the graph
  x = array_ops.placeholder(dtypes.int32, shape=(1, 3), name='input')
  r = array_ops.identity(x)

  sess = session.Session()

  sm_builder = builder.SavedModelBuilder(FLAGS.saved_model_path)
  tensor_info_r = utils.build_tensor_info(r)

  toy_signature = (
      signature_def_utils.build_signature_def(
          outputs={'r': tensor_info_r},
          method_name=signature_constants.PREDICT_METHOD_NAME))

  sm_builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: toy_signature,
      },
      strip_default_attrs=True)
  sm_builder.save()


if __name__ == '__main__':
  app.run(main)
