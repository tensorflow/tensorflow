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

"""Generic entry point script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from tensorflow.python.platform import flags


def run(main=None, argv=None):
  """Runs the program with an optional 'main' function and 'argv' list."""
  f = flags.FLAGS

  # Extract the args from the optional `argv` list.
  args = argv[1:] if argv else None

  # Parse the known flags from that list, or from the command
  # line otherwise.
  # pylint: disable=protected-access
  flags_passthrough = f._parse_flags(args=args)
  # pylint: enable=protected-access

  main = main or sys.modules['__main__'].main

  # Call the main function, passing through any arguments
  # to the final program.
  sys.exit(main(sys.argv[:1] + flags_passthrough))
