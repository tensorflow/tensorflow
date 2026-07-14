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

This loads a previously created SavedModel (esp. a model created by
multiplex_2_save.py which uses the "old" op and C++ kernel from multiplex_2)
and runs the model using the "new" multiplex_4 C++ kernel.

https://www.tensorflow.org/guide/saved_model
https://www.tensorflow.org/api_docs/python/tf/saved_model/save
"""

from absl import app
from tensorflow.examples.custom_ops_doc.multiplex_4 import model_using_multiplex


def main(argv):
  del argv  # not used
  path = 'model_using_multiplex'
  result = model_using_multiplex.load_and_use(path)
  print('Result:', result)


if __name__ == '__main__':
  app.run(main)
