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

# Import 'flags' into this module, for backwards compatibility. Users are
# encouraged to use tf.flags, not tf.app.flags.
# pylint: disable=unused-import
from tensorflow.python.platform import flags
# pylint: enable=unused-import
from tensorflow.python.platform import flags as _flags
from tensorflow.python.util.all_util import remove_undocumented


def run(main=None):
  f = _flags.FLAGS
  flags_passthrough = f._parse_flags()  # pylint: disable=protected-access
  main = main or _sys.modules['__main__'].main
  _sys.exit(main(_sys.argv[:1] + flags_passthrough))

_allowed_symbols = [
    'run',
]

# Add submodules.
_allowed_symbols.extend([
    'flags',
])

remove_undocumented(__name__, _allowed_symbols)
