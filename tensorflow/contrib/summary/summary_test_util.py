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

"""Utilities to test summaries."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record
from tensorflow.python.platform import gfile


def events_from_file(logdir):
  """Returns all events in the single eventfile in logdir."""
  assert gfile.Exists(logdir)
  files = gfile.ListDirectory(logdir)
  assert len(files) == 1, "Found more than one file in logdir: %s" % files
  records = list(
      tf_record.tf_record_iterator(os.path.join(logdir, files[0])))
  result = []
  for r in records:
    event = event_pb2.Event()
    event.ParseFromString(r)
    result.append(event)
  return result
