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
"""Binary for showing C++ backward compatibility.

This creates a SavedModel using the "old" op and C++ kernel from multiplex_2.

https://www.tensorflow.org/guide/saved_model
https://www.tensorflow.org/api_docs/python/tf/saved_model/save
"""

import os
import shutil

from absl import app
from tensorflow.examples.custom_ops_doc.multiplex_2 import multiplex_2_op
from tensorflow.examples.custom_ops_doc.multiplex_4 import model_using_multiplex


def main(argv):
  del argv  # not used
  path = 'model_using_multiplex'
  if os.path.exists(path):
    shutil.rmtree(path, ignore_errors=True)
  model_using_multiplex.save(multiplex_2_op.multiplex, path)
  print('Saved model to', path)


if __name__ == '__main__':
  app.run(main)
