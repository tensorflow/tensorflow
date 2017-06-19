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
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test

# pylint: disable=g-bad-import-order
from tensorflow.contrib.tfprof.python.tools.tfprof import model_analyzer
from tensorflow.contrib.tfprof.python.tools.tfprof.internal import model_analyzer_testlib as lib


class ProfilerTest(test.TestCase):

  def testProfileBasic(self):
    ops.reset_default_graph()
    opts = model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS.copy()
    opts['account_type_regexes'] = ['.*']
    opts['select'] = ['params', 'float_ops', 'micros', 'bytes',
                      'device', 'op_types', 'occurrence']
    outfile = os.path.join(test.get_temp_dir(), 'dump')
    opts['output'] = 'file:outfile=' + outfile

    # Test the output without run_meta.
    sess = session.Session()
    r = lib.BuildFullModel()
    sess.run(variables.global_variables_initializer())

    profiler = model_analyzer.Profiler(sess.graph)
    profiler.profile_name_scope(opts)
    with gfile.Open(outfile, 'r') as f:
      profiler_str = f.read()

    model_analyzer.print_model_analysis(
        sess.graph, tfprof_cmd='scope', tfprof_options=opts)
    with gfile.Open(outfile, 'r') as f:
      pma_str = f.read()
    self.assertEqual(pma_str, profiler_str)

    # Test the output with run_meta.
    run_meta = config_pb2.RunMetadata()
    _ = sess.run(r,
                 options=config_pb2.RunOptions(
                     trace_level=config_pb2.RunOptions.FULL_TRACE),
                 run_metadata=run_meta)

    profiler.add_step(1, run_meta)
    profiler.profile_graph(opts)
    with gfile.Open(outfile, 'r') as f:
      profiler_str = f.read()

    model_analyzer.print_model_analysis(
        sess.graph, tfprof_cmd='graph', run_meta=run_meta, tfprof_options=opts)
    with gfile.Open(outfile, 'r') as f:
      pma_str = f.read()
    self.assertEqual(pma_str, profiler_str)

    profiler.profile_python_codes(opts)
    with gfile.Open(outfile, 'r') as f:
      profiler_str = f.read()

    model_analyzer.print_model_analysis(
        sess.graph, tfprof_cmd='code', run_meta=run_meta, tfprof_options=opts)
    with gfile.Open(outfile, 'r') as f:
      pma_str = f.read()
    self.assertEqual(pma_str, profiler_str)

    profiler.profile_operations(opts)
    with gfile.Open(outfile, 'r') as f:
      profiler_str = f.read()

    model_analyzer.print_model_analysis(
        sess.graph, tfprof_cmd='op', run_meta=run_meta, tfprof_options=opts)
    with gfile.Open(outfile, 'r') as f:
      pma_str = f.read()
    self.assertEqual(pma_str, profiler_str)

    # Test the output difference between multi-step profile and 1-step profile.
    _ = sess.run(r,
                 options=config_pb2.RunOptions(
                     trace_level=config_pb2.RunOptions.FULL_TRACE),
                 run_metadata=run_meta)

    profiler.add_step(2, run_meta)
    profiler.profile_name_scope(opts)
    with gfile.Open(outfile, 'r') as f:
      profiler_str = f.read()

    model_analyzer.print_model_analysis(
        sess.graph, tfprof_cmd='scope', run_meta=run_meta, tfprof_options=opts)
    with gfile.Open(outfile, 'r') as f:
      pma_str = f.read()
    self.assertNotEqual(pma_str, profiler_str)

    opts2 = opts.copy()
    opts2['select'] = ['params', 'float_ops']
    profiler.profile_name_scope(opts2)
    with gfile.Open(outfile, 'r') as f:
      profiler_str = f.read()

    model_analyzer.print_model_analysis(
        sess.graph, tfprof_cmd='scope', run_meta=run_meta, tfprof_options=opts2)
    with gfile.Open(outfile, 'r') as f:
      pma_str = f.read()
    self.assertEqual(pma_str, profiler_str)

  def testMultiStepProfile(self):
    ops.reset_default_graph()
    opts = model_analyzer.PRINT_ALL_TIMING_MEMORY.copy()
    opts['account_type_regexes'] = ['.*']

    with session.Session() as sess, ops.device('/cpu:0'):
      r1, r2, r3 = lib.BuildSplitableModel()
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

      # TODO(xpan): Better test of advisor.
      profiler.advise()


if __name__ == '__main__':
  test.main()
