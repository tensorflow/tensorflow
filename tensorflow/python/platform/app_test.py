# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for our flags implementation."""
import sys

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
flags.DEFINE_boolean('myflag', False, '')

def main(argv):
  if (len(argv) != 3):
    print("Length of argv was not 3: ", argv)
    sys.exit(-1)

  if argv[1] != "--passthrough":
    print("--passthrough argument not in argv")
    sys.exit(-1)

  if argv[2] != "extra":
    print("'extra' argument not in argv")
    sys.exit(-1)


if __name__ == '__main__':
  sys.argv.extend(["--myflag", "--passthrough", "extra"])
  app.run()
