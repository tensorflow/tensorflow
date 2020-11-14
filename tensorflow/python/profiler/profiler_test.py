# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.profiler import option_builder

# pylint: disable=g-bad-import-order
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler.internal import model_analyzer_testlib as lib

builder = option_builder.ProfileOptionBuilder


class ProfilerTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testProfileBasic(self):
    ops.reset_default_graph()
    outfile = os.path.join(test.get_temp_dir(), 'dump')
    opts = (builder(builder.trainable_variables_parameter())
            .with_file_output(outfile)
            .with_accounted_types(['.*'])
            .select(['params', 'float_ops', 'micros', 'bytes',
                     'device', 'op_types', 'occurrence']).build())

    # Test the output without run_meta.
    sess = session.Session()
    r = lib.BuildFullModel()
    sess.run(variables.global_variables_initializer())

    # Test the output with run_meta.
    run_meta = config_pb2.RunMetadata()
    _ = sess.run(r,
                 options=config_pb2.RunOptions(
                     trace_level=config_pb2.RunOptions.FULL_TRACE),
                 run_metadata=run_meta)

    profiler = model_analyzer.Profiler(sess.graph)
    profiler.add_step(1, run_meta)
    profiler.profile_graph(opts)
    with gfile.Open(outfile, 'r') as f:
      profiler_str = f.read()

    model_analyzer.profile(
        sess.graph, cmd='graph', run_meta=run_meta, options=opts)
    with gfile.Open(outfile, 'r') as f:
      pma_str = f.read()
    self.assertEqual(pma_str, profiler_str)

    profiler.profile_name_scope(opts)
    with gfile.Open(outfile, 'r') as f:
      profiler_str = f.read()

    model_analyzer.profile(
        sess.graph, cmd='scope', run_meta=run_meta, options=opts)
    with gfile.Open(outfile, 'r') as f:
      pma_str = f.read()
    self.assertEqual(pma_str, profiler_str)

    profiler.profile_python(opts)
    with gfile.Open(outfile, 'r') as f:
      profiler_str = f.read()

    model_analyzer.profile(
        sess.graph, cmd='code', run_meta=run_meta, options=opts)
    with gfile.Open(outfile, 'r') as f:
      pma_str = f.read()
    self.assertEqual(pma_str, profiler_str)

    profiler.profile_operations(opts)
    with gfile.Open(outfile, 'r') as f:
      profiler_str = f.read()

    model_analyzer.profile(
        sess.graph, cmd='op', run_meta=run_meta, options=opts)
    with gfile.Open(outfile, 'r') as f:
      pma_str = f.read()
    self.assertEqual(pma_str, profiler_str)

    model_analyzer.profile(
        sess.graph, cmd='scope', run_meta=run_meta, options=opts)
    with gfile.Open(outfile, 'r') as f:
      pma_str = f.read()
    self.assertNotEqual(pma_str, profiler_str)

  def testMultiStepProfile(self):
    ops.reset_default_graph()
    opts = builder.time_and_memory(min_bytes=0)

    with session.Session() as sess:
      r1, r2, r3 = lib.BuildSplittableModel()
      sess.run(variables.global_variables_initializer())

      profiler = model_analyzer.Profiler(sess.graph)
      pb0 = profiler.profile_name_scope(opts)

      run_meta = config_pb2.RunMetadata()
      _ = sess.run(r1,
                   options=config_pb2.RunOptions(
                       trace_level=config_pb2.RunOptions.FULL_TRACE),
                   run_metadata=run_meta)
      profiler.add_step(1, run_meta)
      pb1 = profiler.profile_name_scope(opts)

      self.assertNotEqual(lib.SearchTFProfNode(pb1, 'DW'), None)
      self.assertEqual(lib.SearchTFProfNode(pb1, 'DW2'), None)
      self.assertEqual(lib.SearchTFProfNode(pb1, 'add'), None)

      run_meta2 = config_pb2.RunMetadata()
      _ = sess.run(r2,
                   options=config_pb2.RunOptions(
                       trace_level=config_pb2.RunOptions.FULL_TRACE),
                   run_metadata=run_meta2)
      profiler.add_step(2, run_meta2)
      pb2 = profiler.profile_name_scope(opts)

      self.assertNotEqual(lib.SearchTFProfNode(pb2, 'DW'), None)
      self.assertNotEqual(lib.SearchTFProfNode(pb2, 'DW2'), None)
      self.assertEqual(lib.SearchTFProfNode(pb2, 'add'), None)

      run_meta3 = config_pb2.RunMetadata()
      _ = sess.run(r3,
                   options=config_pb2.RunOptions(
                       trace_level=config_pb2.RunOptions.FULL_TRACE),
                   run_metadata=run_meta3)
      profiler.add_step(3, run_meta3)
      pb3 = profiler.profile_name_scope(opts)

      self.assertNotEqual(lib.SearchTFProfNode(pb3, 'DW'), None)
      self.assertNotEqual(lib.SearchTFProfNode(pb3, 'DW2'), None)
      self.assertNotEqual(lib.SearchTFProfNode(pb3, 'add'), None)

      self.assertEqual(lib.SearchTFProfNode(pb0, 'Conv2D'), None)
      self.assertGreater(lib.SearchTFProfNode(pb1, 'Conv2D').exec_micros, 0)
      self.assertEqual(lib.SearchTFProfNode(pb1, 'Conv2D_1'), None)
      self.assertGreater(lib.SearchTFProfNode(pb2, 'Conv2D_1').exec_micros, 0)
      self.assertEqual(lib.SearchTFProfNode(pb2, 'add'), None)
      self.assertGreater(lib.SearchTFProfNode(pb3, 'add').exec_micros, 0)

      advice_pb = profiler.advise(model_analyzer.ALL_ADVICE)
      self.assertTrue('AcceleratorUtilizationChecker' in advice_pb.checkers)
      self.assertTrue('ExpensiveOperationChecker' in advice_pb.checkers)
      self.assertTrue('OperationChecker' in advice_pb.checkers)

      checker = advice_pb.checkers['AcceleratorUtilizationChecker']
      if test.is_gpu_available():
        self.assertGreater(len(checker.reports), 0)
      else:
        self.assertEqual(len(checker.reports), 0)
      checker = advice_pb.checkers['ExpensiveOperationChecker']
      self.assertGreater(len(checker.reports), 0)

  @test_util.run_deprecated_v1
  def testMultipleProfilePerStep(self):
    ops.reset_default_graph()
    opts = (builder(builder.trainable_variables_parameter())
            .with_empty_output()
            .with_accounted_types(['.*'])
            .select(['micros', 'bytes', 'peak_bytes',
                     'residual_bytes', 'output_bytes']).build())

    r = lib.BuildSmallModel()
    sess = session.Session()
    profiler = model_analyzer.Profiler(sess.graph)

    init_var_run_meta = config_pb2.RunMetadata()
    sess.run(variables.global_variables_initializer(),
             options=config_pb2.RunOptions(
                 trace_level=config_pb2.RunOptions.FULL_TRACE),
             run_metadata=init_var_run_meta)

    train_run_meta = config_pb2.RunMetadata()
    sess.run(r,
             options=config_pb2.RunOptions(
                 trace_level=config_pb2.RunOptions.FULL_TRACE),
             run_metadata=train_run_meta)

    profiler.add_step(0, train_run_meta)
    ret1 = profiler.profile_name_scope(opts)
    n1 = lib.SearchTFProfNode(
        ret1, 'DW/Initializer/random_normal/RandomStandardNormal')
    # Without the var initialization run_meta, it doesn't have the
    # information of var_initialization.
    self.assertEqual(n1.exec_micros, 0)
    self.assertEqual(n1.requested_bytes, 0)
    self.assertEqual(n1.peak_bytes, 0)
    self.assertEqual(n1.residual_bytes, 0)

    profiler.add_step(0, init_var_run_meta)
    ret2 = profiler.profile_name_scope(opts)
    n2 = lib.SearchTFProfNode(
        ret2, 'DW/Initializer/random_normal/RandomStandardNormal')
    # After adding the var initialization run_meta.
    self.assertGreater(n2.exec_micros, 0)
    self.assertGreater(n2.requested_bytes, 0)
    self.assertGreater(n2.peak_bytes, 0)
    self.assertGreater(n2.residual_bytes, 0)


if __name__ == '__main__':
  test.main()
