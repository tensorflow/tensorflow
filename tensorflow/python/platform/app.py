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

import sys as _sys

from absl.app import run as _run

from tensorflow.python.platform import flags
from tensorflow.python.util.tf_export import tf_export


def _parse_flags_tolerate_undef(argv):
  """Parse args, returning any unknown flags (ABSL defaults to crashing)."""
  return flags.FLAGS(_sys.argv if argv is None else argv, known_only=True)


@tf_export(v1=['app.run'])
def run(main=None, argv=None):
  """Runs the program with an optional 'main' function and 'argv' list."""

  main = main or _sys.modules['__main__'].main

  _run(main=main, argv=argv, flags_parser=_parse_flags_tolerate_undef)
