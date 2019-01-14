# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""Tests for IgniteDataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.contrib.ignite import IgniteDataset
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class IgniteDatasetTest(test.TestCase):
  """The Apache Ignite servers have to setup before the test and tear down

     after the test manually. The docker engine has to be installed.

     To setup Apache Ignite servers:
     $ bash start_ignite.sh

     To tear down Apache Ignite servers:
     $ bash stop_ignite.sh
  """

  def test_ignite_dataset_with_plain_client(self):
    """Test Ignite Dataset with plain client.

    """
    self._clear_env()
    ds = IgniteDataset(cache_name="SQL_PUBLIC_TEST_CACHE", port=42300)
    self._check_dataset(ds)

  def _clear_env(self):
    """Clears environment variables used by Ignite Dataset.

    """
    if "IGNITE_DATASET_USERNAME" in os.environ:
      del os.environ["IGNITE_DATASET_USERNAME"]
    if "IGNITE_DATASET_PASSWORD" in os.environ:
      del os.environ["IGNITE_DATASET_PASSWORD"]
    if "IGNITE_DATASET_CERTFILE" in os.environ:
      del os.environ["IGNITE_DATASET_CERTFILE"]
    if "IGNITE_DATASET_CERT_PASSWORD" in os.environ:
      del os.environ["IGNITE_DATASET_CERT_PASSWORD"]

  def _check_dataset(self, dataset):
    """Checks that dataset provides correct data."""
    self.assertEqual(dtypes.int64, dataset.output_types["key"])
    self.assertEqual(dtypes.string, dataset.output_types["val"]["NAME"])
    self.assertEqual(dtypes.int64, dataset.output_types["val"]["VAL"])

    it = tf.compat.v1.data.make_one_shot_iterator(dataset)
    ne = it.get_next()

    with session.Session() as sess:
      rows = [sess.run(ne), sess.run(ne), sess.run(ne)]
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(ne)

    self.assertEqual({"key": 1, "val": {"NAME": b"TEST1", "VAL": 42}}, rows[0])
    self.assertEqual({"key": 2, "val": {"NAME": b"TEST2", "VAL": 43}}, rows[1])
    self.assertEqual({"key": 3, "val": {"NAME": b"TEST3", "VAL": 44}}, rows[2])


if __name__ == "__main__":
  test.main()
