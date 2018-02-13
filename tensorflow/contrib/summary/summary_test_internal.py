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
"""Internal helpers for tests in this directory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

import sqlite3

from tensorflow.contrib.summary import summary_ops
from tensorflow.python.framework import test_util


class SummaryDbTest(test_util.TensorFlowTestCase):
  """Helper for summary database testing."""

  def setUp(self):
    super(SummaryDbTest, self).setUp()
    self.db_path = os.path.join(self.get_temp_dir(), 'DbTest.sqlite')
    if os.path.exists(self.db_path):
      os.unlink(self.db_path)
    self.db = sqlite3.connect(self.db_path)
    self.create_db_writer = functools.partial(
        summary_ops.create_db_writer,
        db_uri=self.db_path,
        experiment_name='experiment',
        run_name='run',
        user_name='user')

  def tearDown(self):
    self.db.close()
    super(SummaryDbTest, self).tearDown()


def get_one(db, q, *p):
  return db.execute(q, p).fetchone()[0]


def get_all(db, q, *p):
  return unroll(db.execute(q, p).fetchall())


def unroll(list_of_tuples):
  return sum(list_of_tuples, ())
