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
"""Wrapper for runninmg toco binary embedded in pip site-package.

NOTE: this mainly exists since PIP setup.py cannot install binaries to bin/.
It can only install Python "console-scripts." This will work as a console
script. See tools/pip_package/setup.py (search for CONSOLE_SCRIPTS).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys


def main():
  # Pip installs the binary in aux-bin off of main site-package install.
  # Just find it and exec, passing all arguments in the process.
  # TODO(aselle): it is unfortunate to use all of tensorflow to lookup binary.
  print("""TOCO from pip install is currently not working on command line.
Please use the python TOCO API or use
bazel run tensorflow/contrib/lite:toco -- <args> from a TensorFlow source dir.
""")
  sys.exit(1)
  # TODO(aselle): Replace this when we find a way to run toco without
  # blowing up executable size.
  # binary = os.path.join(tf.__path__[0], 'aux-bin/toco')
  # os.execvp(binary, sys.argv)
