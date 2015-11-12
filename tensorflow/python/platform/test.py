from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform.googletest import GetTempDir
from tensorflow.python.platform.googletest import main
from tensorflow.python.framework.test_util import TensorFlowTestCase as TestCase
from tensorflow.python.framework.test_util import IsGoogleCudaEnabled as IsBuiltWithCuda

get_temp_dir = GetTempDir
