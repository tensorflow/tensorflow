# Copyright 2015 Google Inc. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
from tensorflow.python.platform.googletest import GetTempDir
from tensorflow.python.platform.googletest import main
from tensorflow.python.framework.test_util import TensorFlowTestCase as TestCase
from tensorflow.python.framework.test_util import IsGoogleCudaEnabled as IsBuiltWithCuda

from tensorflow.python.kernel_tests.gradient_checker import compute_gradient_error
from tensorflow.python.kernel_tests.gradient_checker import compute_gradient

get_temp_dir = GetTempDir
# pylint: enable=unused-import
