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
"""tfprof is a tool that profile various aspect of TensorFlow model.

@@model_analyzer
@@tfprof_logger

@@ProfileContext
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
from tensorflow.contrib.tfprof import model_analyzer
from tensorflow.contrib.tfprof import tfprof_logger

from tensorflow.contrib.tfprof.model_analyzer import ProfileContext

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = ['model_analyzer', 'tfprof_logger', 'ProfileContext']

remove_undocumented(__name__, _allowed_symbols)
