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

"""This module includes unsupported and experimental features which are exposed
but not part of the supported public API.  Anything in this module can change
without notice, even across a patch release.

## Utilities

@@constant_value
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform
from tensorflow.python.util.all_util import make_all

# pylint: disable=unused-import
from tensorflow.python.framework.tensor_util import constant_value

__all__ = make_all(__name__)
