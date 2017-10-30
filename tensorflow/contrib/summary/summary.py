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

"""Contrib summary package.

The operations in this package are safe to use with eager execution turned or on
off.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
from tensorflow.contrib.summary.summary_ops import all_summary_ops
from tensorflow.contrib.summary.summary_ops import always_record_summaries
from tensorflow.contrib.summary.summary_ops import audio
from tensorflow.contrib.summary.summary_ops import create_summary_file_writer
from tensorflow.contrib.summary.summary_ops import generic
from tensorflow.contrib.summary.summary_ops import histogram
from tensorflow.contrib.summary.summary_ops import image
from tensorflow.contrib.summary.summary_ops import never_record_summaries
from tensorflow.contrib.summary.summary_ops import record_summaries_every_n_global_steps
from tensorflow.contrib.summary.summary_ops import scalar
from tensorflow.contrib.summary.summary_ops import should_record_summaries
from tensorflow.contrib.summary.summary_ops import summary_writer_initializer_op
