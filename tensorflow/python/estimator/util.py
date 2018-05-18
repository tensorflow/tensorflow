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

"""Utilities for Estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils

fn_args = function_utils.fn_args

# When we create a timestamped directory, there is a small chance that the
# directory already exists because another process is also creating these
# directories. In this case we just wait one second to get a new timestamp and
# try again. If this fails several times in a row, then something is seriously
# wrong.
MAX_DIRECTORY_CREATION_ATTEMPTS = 10


def get_timestamped_dir(dir_base):
  """Builds a path to a new subdirectory within the base directory.

  The subdirectory will be named using the current time.
  This guarantees monotonically increasing directory numbers even across
  multiple runs of the pipeline.
  The timestamp used is the number of seconds since epoch UTC.

  Args:
    dir_base: A string containing a directory to create the subdirectory under.

  Returns:
    The full path of the new subdirectory (which is not actually created yet).

  Raises:
    RuntimeError: if repeated attempts fail to obtain a unique timestamped
      directory name.
  """
  attempts = 0
  while attempts < MAX_DIRECTORY_CREATION_ATTEMPTS:
    timestamp = int(time.time())

    result_dir = os.path.join(
        compat.as_bytes(dir_base), compat.as_bytes(str(timestamp)))
    if not gfile.Exists(result_dir):
      # Collisions are still possible (though extremely unlikely): this
      # directory is not actually created yet, but it will be almost
      # instantly on return from this function.
      return result_dir
    time.sleep(1)
    attempts += 1
    logging.warn('Directory {} already exists; retrying (attempt {}/{})'.format(
        result_dir, attempts, MAX_DIRECTORY_CREATION_ATTEMPTS))
  raise RuntimeError('Failed to obtain a unique export directory name after '
                     '{} attempts.'.format(MAX_DIRECTORY_CREATION_ATTEMPTS))
