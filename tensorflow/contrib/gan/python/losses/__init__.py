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
"""TFGAN losses and penalties.

Losses can be used with individual arguments or with GANModel tuples.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Collapse losses into a single namespace.
from tensorflow.contrib.gan.python.losses.python import losses_wargs as wargs
from tensorflow.contrib.gan.python.losses.python import tuple_losses

# pylint: disable=wildcard-import
from tensorflow.contrib.gan.python.losses.python.tuple_losses import *
# pylint: enable=wildcard-import

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = ['wargs'] + tuple_losses.__all__
remove_undocumented(__name__, _allowed_symbols)
