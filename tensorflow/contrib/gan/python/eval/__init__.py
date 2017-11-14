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
"""TFGAN grouped API. Please see README.md for details and usage."""
# pylint: disable=,wildcard-import,unused-import

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Collapse eval into a single namespace.
from tensorflow.contrib.gan.python.eval.python import classifier_metrics
from tensorflow.contrib.gan.python.eval.python import eval_utils
from tensorflow.contrib.gan.python.eval.python import summaries

from tensorflow.contrib.gan.python.eval.python.classifier_metrics import *
from tensorflow.contrib.gan.python.eval.python.eval_utils import *
from tensorflow.contrib.gan.python.eval.python.summaries import *
# pylint: enable=wildcard-import,unused-import

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    'classifier_metrics',
    'summaries',
    'eval_utils',
] + classifier_metrics.__all__ + summaries.__all__ + eval_utils.__all__
remove_undocumented(__name__, _allowed_symbols)
