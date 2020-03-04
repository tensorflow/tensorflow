# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for TPUStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import tpu_strategy as tpu_lib
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import errors
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_strategy_util


FLAGS = flags.FLAGS
flags.DEFINE_string("tpu", "", "Name of TPU to connect to.")
flags.DEFINE_string("project", None, "Name of GCP project with TPU.")
flags.DEFINE_string("zone", None, "Name of GCP zone with TPU.")


def get_tpu_strategy():
  resolver = tpu_cluster_resolver.TPUClusterResolver(
      tpu=FLAGS.tpu,
      zone=FLAGS.zone,
      project=FLAGS.project,
  )
  remote.connect_to_cluster(resolver)
  tpu_strategy_util.initialize_tpu_system(resolver)
  return tpu_lib.TPUStrategy(resolver)


class TpuStrategyTest(test.TestCase):

  def test_multiple_initialize_system(self):
    resolver = tpu_cluster_resolver.TPUClusterResolver(
        tpu=FLAGS.tpu,
        zone=FLAGS.zone,
        project=FLAGS.project,
    )
    remote.connect_to_cluster(resolver)
    tpu_strategy_util.initialize_tpu_system(resolver)

    with test.mock.patch.object(logging, "warning") as mock_log:
      tpu_strategy_util.initialize_tpu_system(resolver)
      self.assertRegex(str(mock_log.call_args), "already been initialized")

  def test_recover_from_compilation_failures(self):
    strategy = get_tpu_strategy()

    @def_function.function
    def compilation_failure_run():

      def computation():
        samples = random_ops.random_gamma([10], [0.5, 1.5])
        return samples

      return strategy.experimental_run_v2(computation)

    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 "TPU compilation failed"):
      compilation_failure_run()

    @def_function.function
    def good_run():

      def computation():
        samples = random_ops.random_normal([10])
        return samples

      return strategy.experimental_run_v2(computation)

    good_run()


if __name__ == "__main__":
  test.main()
