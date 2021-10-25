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
"""Tests for Tensorflow -> CPURT compilation."""

import os
import time

from tensorflow.compiler.mlir.tfrt.jit.python_binding import tf_cpurt
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging

cpurt = tf_cpurt.TfCpurtExecutor()


class TfCompileTest(test.TestCase):

  def test_compile(self):
    mlirdir = os.path.join(resource_loader.get_data_files_path(),
                           'regression_tests')
    for filename in os.listdir(mlirdir):
      with open(os.path.join(mlirdir, filename), mode='r') as f:
        mlir_function = f.read()
        logging.info(f'processing {filename}')
        start = time.perf_counter()
        cpurt.compile(
            mlir_function,
            # TODO(akuegel): Currently we assume that the function is always
            # named 'test'. Better would be to process the IR and get the name
            # from there.
            'test',
            tf_cpurt.Specialization.ENABLED,
            vectorize=False)
        end = time.perf_counter()
        logging.info(f'compiled {filename} in {end-start:0.4f} seconds')

if __name__ == '__main__':
  test.main()
