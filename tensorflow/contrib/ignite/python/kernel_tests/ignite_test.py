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

from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from functools import reduce
from tensorflow.contrib.ignite.python.ops import ignite_dataset_ops

class IgniteDatasetTest(test.TestCase):
    def setUp(self):
        # The Ignite server has to be setup before the test
        # and tear down after the test manually.
        # The docker engine has to be installed.
        #
        # To setup the Ignite server:
        # $ bash ignite_test.sh start ignite_test
        #
        # To team down the Ignite server:
        # $ bash ignite_test.sh stop ignite_test
        # LD_LIBRARY_PATH should be set in such a way that it contains libjvm.so, for example
        # export LD_LIBRARY_PATH=/usr/lib/jvm/java-8-oracle/jre/lib/amd64/server/
        # IGNITE_HOME should point to Apache Ignite installation directory, for example
        # export IGNITE_HOME=~/apache-ignite-fabric-2.4.0-bin
        # TF_IGNITE_CLIENT_CONFIG should point to client ignite node config, for example
        # export TF_IGNITE_CLIENT_CONFIG=../../sample_configs/client.xml
        pass

    def testIgniteDataset(self):
        expected_caches = self.collect_values(["node1_entries/entries.txt", "node2_entries/entries.txt"])
        for cache in expected_caches:
            self.tst_cache(expected_caches[cache], cache)

    def tst_cache(self, expected_values, cache_name):
        values = {}

        size = reduce(lambda x, value: x + value, expected_values.values(), 0)

        ds = ignite_dataset_ops.IgniteDataset(cache_name)

        with self.test_session() as sess:
            it = ds.make_one_shot_iterator()
            next_element = it.get_next();
            for i in range(size):
                val = sess.run(next_element);
                if (val not in values):
                    values[val] = 0;
                values[val] = values[val] + 1
                print(val)
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(next_element)
        self.assertEqual(expected_values,values)

    def collect_values(self, file_names):
        res = {};

        for file_name in file_names:
            new_res = self.read_from_file(file_name)
            for new_res_cache_name in new_res:
                new_res_cache = new_res[new_res_cache_name]
                for key in new_res_cache:
                    if (new_res_cache_name in res) and (key in res[new_res_cache_name]):
                        res[new_res_cache_name][key] = res[new_res_cache_name][key] + new_res_cache[key]
                    else:
                        if new_res_cache_name not in res:
                            res[new_res_cache_name] = {}
                        res[new_res_cache_name][key] = new_res_cache[key]


        return res;

    def read_from_file(self, file_name):
        with open(file_name) as f:
            content = f.readlines()
        lines = [x.strip() for x in content]

        caches = {}

        cur_cache_name = ""
        for l in lines:
            # A cache name
            if (l.startswith("#")):
                cur_cache_name = l[1:].encode()
                caches[cur_cache_name] = {}
            # An entry
            else:
                sp = l.split()
                val = sp[1].encode()
                entries = caches[cur_cache_name]
                if (val not in entries):
                    entries[val] = 0;
                entries[val] = entries[val] + 1
        return caches

if __name__ == "__main__":
  test.main()
