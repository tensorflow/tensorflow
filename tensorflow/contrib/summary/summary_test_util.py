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

import functools
import os
import sqlite3

from tensorflow.contrib.summary import summary_ops
from tensorflow.core.util import event_pb2
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import tf_record
from tensorflow.python.platform import gfile


class SummaryDbTest(test_util.TensorFlowTestCase):
  """Helper for summary database testing."""

  def setUp(self):
    super(SummaryDbTest, self).setUp()
    self.db_path = os.path.join(self.get_temp_dir(), 'DbTest.sqlite')
    if os.path.exists(self.db_path):
      os.unlink(self.db_path)
    self.db = sqlite3.connect(self.db_path)
    self.create_summary_db_writer = functools.partial(
        summary_ops.create_summary_db_writer,
        db_uri=self.db_path,
        experiment_name='experiment',
        run_name='run',
        user_name='user')

  def tearDown(self):
    self.db.close()
    super(SummaryDbTest, self).tearDown()


def events_from_file(filepath):
  """Returns all events in a single event file.

  Args:
    filepath: Path to the event file.

  Returns:
    A list of all tf.Event protos in the event file.
  """
  records = list(tf_record.tf_record_iterator(filepath))
  result = []
  for r in records:
    event = event_pb2.Event()
    event.ParseFromString(r)
    result.append(event)
  return result


def events_from_logdir(logdir):
  """Returns all events in the single eventfile in logdir.

  Args:
    logdir: The directory in which the single event file is sought.

  Returns:
    A list of all tf.Event protos from the single event file.

  Raises:
    AssertionError: If logdir does not contain exactly one file.
  """
  assert gfile.Exists(logdir)
  files = gfile.ListDirectory(logdir)
  assert len(files) == 1, 'Found not exactly one file in logdir: %s' % files
  return events_from_file(os.path.join(logdir, files[0]))


def get_one(db, q, *p):
  return db.execute(q, p).fetchone()[0]


def get_all(db, q, *p):
  return unroll(db.execute(q, p).fetchall())


def unroll(list_of_tuples):
  return sum(list_of_tuples, ())
