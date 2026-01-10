# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf

from tensorflow.core.function.trace_type import trace_type_builder
from tensorflow.python.platform import test

class MockSymbolicTensor:
  """A class that mimics a Symbolic Tensor behavior during tracing.
  
  It has __array__ (which causes is_np_ndarray to return True in the buggy version)
  but raises NotImplementedError when called (mimicking the XLA crash).
  It also has a 'graph' attribute, which our fix in util.py detects.
  """
  def __init__(self):
    self.shape = (10,)
    self.dtype = tf.float32
    self.graph = "FakeGraph" # This triggers the exclusion in util.is_np_ndarray
  
  def __array__(self):
    # This simulates the crash seen in Issue #105728
    raise NotImplementedError("numpy() is only available when eager execution is enabled.")

class JitCompileIntegrationTest(test.TestCase):

  def testSymbolicTensorExclusion(self):
    """Regression test for GitHub Issue #105728."""
    fake_tensor = MockSymbolicTensor()
    
    # How fix works (util.py): 
    # util.is_np_ndarray sees .graph, returns False -> Skips __array__ -> Success.
    
    try:
      trace_type_builder.from_value(fake_tensor)
    except NotImplementedError:
       self.fail("trace_type_builder crashed on symbolic tensor! util.is_np_ndarray fix failed.")

if __name__ == '__main__':
  test.main()
