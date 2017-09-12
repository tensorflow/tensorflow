# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TFGAN grouped API. Please see README.md for details and usage."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Collapse TFGAN into a tiered namespace.
from tensorflow.contrib.gan.python import features
from tensorflow.contrib.gan.python import losses
from tensorflow.contrib.gan.python import namedtuples
from tensorflow.contrib.gan.python import train

# pylint: disable=unused-import,wildcard-import
from tensorflow.contrib.gan.python.namedtuples import *
from tensorflow.contrib.gan.python.train import *
# pylint: enable=unused-import,wildcard-import

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    'features',
    'losses',
]
_allowed_symbols += train.__all__
_allowed_symbols += namedtuples.__all__
remove_undocumented(__name__, _allowed_symbols)
