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

import gzip
import io
import os
import random
import re

import numpy as np

from tensorflow.core.profiler import profile_pb2
from tensorflow.core.profiler import tfprof_log_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder
from tensorflow.python.profiler import profile_context
from tensorflow.python.profiler.internal import model_analyzer_testlib as lib
from tensorflow.python.util import compat

builder = option_builder.ProfileOptionBuilder


class PrintModelAnalysisTest(test.TestCase):

  def testDumpToFile(self):
    ops.reset_default_graph()
    outfile = os.path.join(test.get_temp_dir(), 'dump')
    opts = builder(builder.trainable_variables_parameter()
                  ).with_file_output(outfile).build()

    with session.Session() as sess:
      _ = lib.BuildSmallModel()
      model_analyzer.profile(sess.graph, options=opts)

      with gfile.Open(outfile, 'r') as f:
        self.assertEqual(u'node name | # parameters\n'
                         '_TFProfRoot (--/451 params)\n'
                         '  DW (3x3x3x6, 162/162 params)\n'
                         '  DW2 (2x2x6x12, 288/288 params)\n'
                         '  ScalarW (1, 1/1 params)\n',
                         lib.CheckAndRemoveDoc(f.read()))

  def testSelectEverythingDetail(self):
    ops.reset_default_graph()
    dev = '/device:GPU:0' if test.is_gpu_available() else '/device:CPU:0'
    outfile = os.path.join(test.get_temp_dir(), 'dump')
    opts = (builder(builder.trainable_variables_parameter())
            .with_file_output(outfile)
            .with_accounted_types(['.*'])
            .select(['micros', 'bytes', 'params', 'float_ops', 'occurrence',
                     'device', 'op_types', 'input_shapes']).build())

    with profile_context.ProfileContext(test.get_temp_dir(),
                                        trace_steps=[],
                                        dump_steps=[]) as pctx:
      with session.Session() as sess, ops.device(dev):
        x = lib.BuildSmallModel()

        sess.run(variables.global_variables_initializer())
        pctx.trace_next_step()
        pctx.dump_next_step()
        _ = sess.run(x)

        pctx.profiler.profile_name_scope(options=opts)

        with gfile.Open(outfile, 'r') as f:
          # pylint: disable=line-too-long
          dump_str = lib.CheckAndRemoveDoc(f.read())
          outputs = dump_str.split('\n')

          self.assertEqual(outputs[0],
                           'node name | # parameters | # float_ops | requested bytes | total execution time | accelerator execution time | cpu execution time | assigned devices | op types | op count (run|defined) | input shapes')
          for o in outputs[1:]:
            if o.find('Conv2D ') > 0:
              metrics = o[o.find('(') +1: o.find(')')].split(',')
              # Make sure time is profiled.
              gap = 1 if test.is_gpu_available() else 2
              for i in range(3, 6, gap):
                mat = re.search('(.*)(?:us|ms|sec)/(.*)(?:us|ms|sec)', metrics[i])
                self.assertGreater(float(mat.group(1)), 0.0)
                self.assertGreater(float(mat.group(2)), 0.0)
              # Make sure device is profiled.
              if test.is_gpu_available():
                self.assertTrue(metrics[6].find('gpu') > 0)
                self.assertFalse(metrics[6].find('cpu') > 0)
              else:
                self.assertFalse(metrics[6].find('gpu') > 0)
                self.assertTrue(metrics[6].find('cpu') > 0)
              # Make sure float_ops is profiled.
              mat = re.search('(.*)k/(.*)k flops', metrics[1].strip())
              self.assertGreater(float(mat.group(1)), 0.0)
              self.assertGreater(float(mat.group(2)), 0.0)
              # Make sure op_count is profiled.
              self.assertEqual(metrics[8].strip(), '1/1|1/1')
              # Make sure input_shapes is profiled.
              self.assertEqual(metrics[9].strip(), '0:2x6x6x3|1:3x3x3x6')

            if o.find('DW (3x3x3x6') > 0:
              metrics = o[o.find('(') +1: o.find(')')].split(',')
              mat = re.search('(.*)/(.*) params', metrics[1].strip())
              self.assertGreater(float(mat.group(1)), 0.0)
              self.assertGreater(float(mat.group(2)), 0.0)
          # pylint: enable=line-too-long

    # Test that profiler restored from profile file gives the same result.
    gfile.Remove(outfile)
    profile_file = os.path.join(test.get_temp_dir(), 'profile_1')
    with lib.ProfilerFromFile(profile_file) as profiler:
      profiler.profile_name_scope(options=opts)
      with gfile.Open(outfile, 'r') as f:
        self.assertEqual(dump_str, lib.CheckAndRemoveDoc(f.read()))

  def testSelectEverything(self):
    ops.reset_default_graph()
    outfile = os.path.join(test.get_temp_dir(), 'dump')
    opts = (builder(builder.trainable_variables_parameter())
            .with_file_output(outfile)
            .with_accounted_types(['.*'])
            .select(['params', 'float_ops', 'occurrence', 'device', 'op_types',
                     'input_shapes']).build())

    rewriter_config = rewriter_config_pb2.RewriterConfig(
        disable_model_pruning=True)
    graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_config)
    config = config_pb2.ConfigProto(graph_options=graph_options)
    with session.Session(config=config) as sess, ops.device('/device:CPU:0'):
      x = lib.BuildSmallModel()

      sess.run(variables.global_variables_initializer())
      run_meta = config_pb2.RunMetadata()
      _ = sess.run(x,
                   options=config_pb2.RunOptions(
                       trace_level=config_pb2.RunOptions.FULL_TRACE),
                   run_metadata=run_meta)

      model_analyzer.profile(
          sess.graph, run_meta, options=opts)

  def testSimpleCodeView(self):
    ops.reset_default_graph()
    outfile = os.path.join(test.get_temp_dir(), 'dump')
    # TODO(xpan): Test 'micros'. Since the execution time changes each run,
    # it's a bit difficult to test it now.
    opts = (builder(builder.trainable_variables_parameter())
            .with_file_output(outfile)
            .with_accounted_types(['.*'])
            .with_node_names(show_name_regexes=['.*model_analyzer_testlib.*'])
            .account_displayed_op_only(False)
            .select(['bytes', 'params', 'float_ops', 'num_hidden_ops', 'device',
                     'input_shapes']).build())

    with session.Session() as sess:
      x = lib.BuildSmallModel()

      sess.run(variables.global_variables_initializer())
      run_meta = config_pb2.RunMetadata()
      _ = sess.run(x,
                   options=config_pb2.RunOptions(
                       trace_level=config_pb2.RunOptions.FULL_TRACE),
                   run_metadata=run_meta)

      model_analyzer.profile(
          sess.graph, run_meta, cmd='code', options=opts)

      with gfile.Open(outfile, 'r') as f:
        # pylint: disable=line-too-long
        self.assertEqual(
            'node name | requested bytes | # parameters | # float_ops | assigned devices | in',
            lib.CheckAndRemoveDoc(f.read())[0:80])
        # pylint: enable=line-too-long

  def testComplexCodeView(self):
    ops.reset_default_graph()
    outfile = os.path.join(test.get_temp_dir(), 'dump')
    opts = (builder(builder.trainable_variables_parameter())
            .with_file_output(outfile)
            .with_accounted_types(['.*'])
            .with_node_names(show_name_regexes=
                             ['.*model_analyzer_testlib.py.*'])
            .account_displayed_op_only(False)
            .select(['params', 'float_ops']).build())

    with profile_context.ProfileContext(test.get_temp_dir(),
                                        trace_steps=[],
                                        dump_steps=[]) as pctx:
      with session.Session() as sess:
        x = lib.BuildFullModel()

        sess.run(variables.global_variables_initializer())
        pctx.trace_next_step()
        _ = sess.run(x)
        tfprof_node = pctx.profiler.profile_python(options=opts)

        # pylint: disable=line-too-long
        with gfile.Open(outfile, 'r') as f:
          lines = f.read().split('\n')
          self.assertGreater(len(lines), 5)
          result = '\n'.join([l[:min(len(l), 80)] for l in lines])
          self.assertTrue(
              compat.as_text(lib.CheckAndRemoveDoc(result))
              .startswith('node name | # parameters | # float_ops'))

        self.assertLess(0, tfprof_node.total_exec_micros)
        self.assertEqual(2844, tfprof_node.total_parameters)
        #The graph is modifed when MKL is enabled,total_float_ops will
        #be different
        if test_util.IsMklEnabled():
          self.assertLess(101600, tfprof_node.total_float_ops)
        else:
          self.assertLess(145660, tfprof_node.total_float_ops)
        self.assertEqual(8, len(tfprof_node.children))
        self.assertEqual('_TFProfRoot', tfprof_node.name)
        self.assertEqual(
            'model_analyzer_testlib.py:63:BuildFullModel',
            tfprof_node.children[0].name)
        self.assertEqual(
            'model_analyzer_testlib.py:63:BuildFullModel (gradient)',
            tfprof_node.children[1].name)
        self.assertEqual(
            'model_analyzer_testlib.py:67:BuildFullModel',
            tfprof_node.children[2].name)
        self.assertEqual(
            'model_analyzer_testlib.py:67:BuildFullModel (gradient)',
            tfprof_node.children[3].name)
        self.assertEqual(
            'model_analyzer_testlib.py:69:BuildFullModel',
            tfprof_node.children[4].name)
        self.assertEqual(
            'model_analyzer_testlib.py:70:BuildFullModel',
            tfprof_node.children[5].name)
        self.assertEqual(
            'model_analyzer_testlib.py:70:BuildFullModel (gradient)',
            tfprof_node.children[6].name)
        self.assertEqual(
            'model_analyzer_testlib.py:72:BuildFullModel',
            tfprof_node.children[7].name)
        # pylint: enable=line-too-long

  def testCodeViewLeafGraphNode(self):
    ops.reset_default_graph()
    opts = (builder(builder.trainable_variables_parameter())
            .with_empty_output()
            .with_accounted_types(['.*'])
            .account_displayed_op_only(False)
            .select(['bytes', 'params', 'float_ops', 'device']).build())

    with session.Session() as sess:
      x = lib.BuildSmallModel()

      sess.run(variables.global_variables_initializer())
      run_meta = config_pb2.RunMetadata()
      _ = sess.run(x,
                   options=config_pb2.RunOptions(
                       trace_level=config_pb2.RunOptions.FULL_TRACE),
                   run_metadata=run_meta)

      tfprof_node = model_analyzer.profile(
          sess.graph, run_meta, cmd='code', options=opts)

      leaf = tfprof_node
      while leaf.children:
        self.assertEqual(0, len(leaf.graph_nodes))
        leaf = leaf.children[0]
      self.assertEqual(1, len(leaf.graph_nodes))

  def testTimeline(self):
    ops.reset_default_graph()
    outfile = os.path.join(test.get_temp_dir(), 'timeline')
    opts = (builder(builder.trainable_variables_parameter())
            .with_max_depth(100000)
            .with_step(0)
            .with_timeline_output(outfile)
            .with_accounted_types(['.*']).build())

    with session.Session() as sess:
      x = lib.BuildFullModel()

      sess.run(variables.global_variables_initializer())
      run_meta = config_pb2.RunMetadata()
      _ = sess.run(
          x,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE),
          run_metadata=run_meta)

      _ = model_analyzer.profile(
          sess.graph, run_meta, cmd='graph', options=opts)

      with gfile.Open(outfile + '_0', 'r') as f:
        # Test that a json file is created.
        # TODO(xpan): tfprof Timeline isn't quite correct on Windows.
        # Investigate why.
        if os.name != 'nt':
          self.assertLess(1000, len(f.read()))
        else:
          self.assertLess(1, len(f.read()))

  def testOpView(self):
    ops.reset_default_graph()
    outfile = os.path.join(test.get_temp_dir(), 'dump')

    opts = (builder(builder.trainable_variables_parameter())
            .with_file_output(outfile)
            .with_accounted_types(['.*'])
            .with_min_occurrence(10)
            .order_by('occurrence')
            .select(['params', 'micros', 'bytes',
                     'peak_bytes', 'residual_bytes',
                     'output_bytes', 'occurrence', 'input_shapes']).build())

    with session.Session() as sess:
      x = lib.BuildFullModel()

      sess.run(variables.global_variables_initializer())
      run_meta = config_pb2.RunMetadata()
      _ = sess.run(x,
                   options=config_pb2.RunOptions(
                       trace_level=config_pb2.RunOptions.FULL_TRACE),
                   run_metadata=run_meta)

      tfprof_node = model_analyzer.profile(
          sess.graph, run_meta, cmd='op', options=opts)

      with gfile.Open(outfile, 'r') as f:
        # pylint: disable=line-too-long
        self.assertEqual(
            'nodename|requestedbytes|peakbytes|residualbytes|outputbytes|totalexecutiontime|acceleratorexecutiontime|cpuexecutiontime|#parameters|opoccurrence(run|defined)|inputshapes',
            lib.CheckAndRemoveDoc(f.read()).replace('\t',
                                                    '').replace(' ', '')[0:170])
        # pylint: enable=line-too-long

      total_children = 0
      last_occurrence = 1e32
      input_shapes = 0
      last_total_micros = tfprof_node.total_exec_micros
      last_micros = tfprof_node.exec_micros
      while tfprof_node.children:
        for gnode in tfprof_node.graph_nodes:
          input_shapes += len(gnode.input_shapes)
        self.assertEqual(len(tfprof_node.children), 1)
        tfprof_node = tfprof_node.children[0]

        self.assertEqual(
            last_total_micros, tfprof_node.total_exec_micros + last_micros)
        last_total_micros = tfprof_node.total_exec_micros
        last_micros = tfprof_node.exec_micros

        total_children += 1
        self.assertLessEqual(len(tfprof_node.graph_nodes), last_occurrence)
        last_occurrence = len(tfprof_node.graph_nodes)

      self.assertGreater(input_shapes, 0)

  def testAdvisor(self):
    ops.reset_default_graph()

    with session.Session() as sess:
      x = lib.BuildFullModel()

      sess.run(variables.global_variables_initializer())
      run_meta = config_pb2.RunMetadata()
      _ = sess.run(
          x,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE),
          run_metadata=run_meta)

      advice_pb = model_analyzer.advise(sess.graph, run_meta)
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

  def pprof_test_helper(self, attribute, should_fail=False):
    ops.reset_default_graph()
    outfile = os.path.join(test.get_temp_dir(), attribute + '_pprof.pb.gz')
    opts = (builder(builder.time_and_memory())
            .select([attribute])
            .with_max_depth(100000)
            .with_node_names(trim_name_regexes=['ops.py.*'])
            .with_pprof_output(outfile).build())

    with session.Session() as sess:
      x = lib.BuildFullModel()

      sess.run(variables.global_variables_initializer())
      run_meta = config_pb2.RunMetadata()
      _ = sess.run(
          x,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE),
          run_metadata=run_meta)

      _ = model_analyzer.profile(
          sess.graph, run_meta, cmd='code', options=opts)

      if should_fail:
        self.assertFalse(gfile.Exists(outfile))
        return

      profile_pb = profile_pb2.Profile()
      with gfile.Open(outfile, 'rb') as f:
        with gzip.GzipFile(fileobj=io.BytesIO(f.read())) as gzipf:
          profile_pb.ParseFromString(gzipf.read())

      self.assertGreater(len(profile_pb.sample), 10)
      self.assertGreater(len(profile_pb.location), 10)
      self.assertGreater(len(profile_pb.function), 10)
      self.assertGreater(len(profile_pb.string_table), 30)

      has_rnn = False
      has_loop = False
      for s in profile_pb.string_table:
        if s.find('rnn') > 0:
          has_rnn = True
        if s.find('while') > 0:
          has_loop = True
        self.assertFalse(s.startswith('ops.py'))
      self.assertTrue(has_rnn)
      self.assertTrue(has_loop)

  def testPprof(self):
    for attr in ['micros', 'bytes', 'accelerator_micros', 'cpu_micros',
                 'params', 'float_ops']:
      self.pprof_test_helper(attr)
    for attr in ['op_types', 'device', 'input_shapes']:
      self.pprof_test_helper(attr, True)

  def testMinOption(self):
    ops.reset_default_graph()

    def check_min(nodes, mm=0, mam=0, mcm=0, mb=0, mpb=0, mrb=0, mob=0):
      for n in nodes:
        if mm > 0:
          self.assertGreaterEqual(n.exec_micros, mm)
        if mam > 0:
          self.assertGreaterEqual(n.accelerator_exec_micros, mam)
        if mcm > 0:
          self.assertGreaterEqual(n.cpu_exec_micros, mcm)
        if mb > 0:
          self.assertGreaterEqual(n.requested_bytes, mb)
        if mpb > 0:
          self.assertGreaterEqual(n.peak_bytes, mpb)
        if mrb > 0:
          self.assertGreaterEqual(n.residual_bytes, mrb)
        if mob > 0:
          self.assertGreaterEqual(n.output_bytes, mob)
        check_min(n.children, mm, mam, mcm, mb, mpb, mrb, mob)

    with session.Session() as sess:
      x = lib.BuildSmallModel()
      sess.run(variables.global_variables_initializer())
      run_meta = config_pb2.RunMetadata()
      _ = sess.run(x,
                   options=config_pb2.RunOptions(
                       trace_level=config_pb2.RunOptions.FULL_TRACE),
                   run_metadata=run_meta)

      min_val = random.randint(0, 10000)

      opts = builder(builder.time_and_memory(min_micros=min_val)
                    ).with_empty_output().build()
      tfprof_node = model_analyzer.profile(
          sess.graph, run_meta=run_meta, options=opts)
      check_min(tfprof_node.children, mm=min_val)

      opts = builder(builder.time_and_memory(min_accelerator_micros=min_val)
                    ).with_empty_output().build()
      tfprof_node = model_analyzer.profile(
          sess.graph, run_meta=run_meta, options=opts)
      check_min(tfprof_node.children, mam=min_val)

      opts = builder(builder.time_and_memory(min_cpu_micros=min_val)
                    ).with_empty_output().build()
      tfprof_node = model_analyzer.profile(
          sess.graph, run_meta=run_meta, options=opts)
      check_min(tfprof_node.children, mcm=min_val)

      opts = builder(builder.time_and_memory(min_bytes=min_val)
                    ).with_empty_output().build()
      tfprof_node = model_analyzer.profile(
          sess.graph, run_meta=run_meta, options=opts)
      check_min(tfprof_node.children, mb=min_val)

      opts = builder(builder.time_and_memory(min_peak_bytes=min_val)
                    ).with_empty_output().build()
      tfprof_node = model_analyzer.profile(
          sess.graph, run_meta=run_meta, options=opts)
      check_min(tfprof_node.children, mpb=min_val)

      opts = builder(builder.time_and_memory(min_residual_bytes=min_val)
                    ).with_empty_output().build()
      tfprof_node = model_analyzer.profile(
          sess.graph, run_meta=run_meta, options=opts)
      check_min(tfprof_node.children, mrb=min_val)

      opts = builder(builder.time_and_memory(min_output_bytes=min_val)
                    ).with_empty_output().build()
      tfprof_node = model_analyzer.profile(
          sess.graph, run_meta=run_meta, options=opts)
      check_min(tfprof_node.children, mob=min_val)

  def testSelectOption(self):
    ops.reset_default_graph()
    outfile = os.path.join(test.get_temp_dir(), 'dump')

    def check_selection(selected, not_selected):
      with gfile.Open(outfile, 'r') as f:
        s = f.read()
        for attr in selected:
          self.assertTrue(s.find(attr) > 0, s)
        for attr in not_selected:
          self.assertFalse(s.find(attr) > 0, s)

    with session.Session() as sess:
      x = lib.BuildSmallModel()
      sess.run(variables.global_variables_initializer())
      run_meta = config_pb2.RunMetadata()
      _ = sess.run(x,
                   options=config_pb2.RunOptions(
                       trace_level=config_pb2.RunOptions.FULL_TRACE),
                   run_metadata=run_meta)

      opts = builder(builder.time_and_memory()
                    ).with_file_output(outfile).select(['micros']).build()
      _ = model_analyzer.profile(
          sess.graph, run_meta=run_meta, options=opts)
      check_selection(['total execution time', 'accelerator execution time'],
                      ['bytes'])

      opts = builder(builder.time_and_memory()
                    ).with_file_output(outfile).select(['bytes']).build()
      _ = model_analyzer.profile(
          sess.graph, run_meta=run_meta, options=opts)
      check_selection(['requested bytes'],
                      ['peak bytes', 'residual bytes', 'output bytes'])

      opts = builder(builder.time_and_memory()).with_file_output(
          outfile).select(
              ['peak_bytes', 'residual_bytes', 'output_bytes']).build()
      _ = model_analyzer.profile(
          sess.graph, run_meta=run_meta, options=opts)
      check_selection(['peak bytes', 'residual bytes', 'output bytes'],
                      ['requested_bytes'])

  def _trainLoop(self, train_op, train_steps, time_dir, time_step,
                 memory_dir, memory_step, profile_dir, dump_step):
    with session.Session() as sess:
      sess.run(variables.global_variables_initializer())
      # start from 1 because variable_initializer took one step.
      for i in range(1, train_steps + 1):
        _ = sess.run(train_op)
        if i in time_step:
          ret = gfile.ListDirectory(time_dir)
          self.assertEqual(len(ret), 1)
          self.assertTrue(
              gfile.Open(os.path.join(time_dir, ret[0]), 'r').read()
              .find('execution time') > 0)
          _ = [gfile.Remove(os.path.join(time_dir, x)) for x in ret]
        else:
          self.assertEqual(len(gfile.ListDirectory(time_dir)), 0)
        if i in memory_step:
          ret = gfile.ListDirectory(memory_dir)
          self.assertEqual(len(ret), 1)
          self.assertTrue(
              gfile.Open(os.path.join(memory_dir, ret[0]), 'r').read()
              .find('requested bytes') > 0)
          _ = [gfile.Remove(os.path.join(memory_dir, x)) for x in ret]
        else:
          self.assertEqual(len(gfile.ListDirectory(memory_dir)), 0)
        if i in dump_step:
          ret = gfile.ListDirectory(profile_dir)
          self.assertAllEqual(ret, ['profile_%d' % i])
          _ = [gfile.Remove(os.path.join(profile_dir, x)) for x in ret]
        else:
          if i < dump_step[0]:
            self.assertFalse(gfile.Exists(profile_dir))
          else:
            self.assertEqual(len(gfile.ListDirectory(profile_dir)), 0)

  def testAutoProfiling(self):
    ops.reset_default_graph()
    time_dir = os.path.join(test.get_temp_dir(), 'time')
    memory_dir = os.path.join(test.get_temp_dir(), 'memory')
    profile_dir = os.path.join(test.get_temp_dir(), 'dir/dir2/profile')
    # TODO(xpan): Should we create parent directory for them?
    gfile.MkDir(time_dir)
    gfile.MkDir(memory_dir)

    time_opts = (builder(builder.time_and_memory())
                 .with_file_output(os.path.join(time_dir, 'profile'))
                 .select(['micros']).build())
    memory_opts = (builder(builder.time_and_memory())
                   .with_file_output(os.path.join(memory_dir, 'profile'))
                   .select(['bytes']).build())

    time_steps = [2, 3]
    memory_steps = [1, 3]
    dump_steps = [3, 4]

    x = lib.BuildSmallModel()
    with profile_context.ProfileContext(profile_dir,
                                        trace_steps=[1, 2, 3],
                                        dump_steps=[3, 4]) as pctx:
      pctx.add_auto_profiling('scope', time_opts, time_steps)
      pctx.add_auto_profiling('scope', memory_opts, memory_steps)

      self._trainLoop(x, 10, time_dir, time_steps,
                      memory_dir, memory_steps, profile_dir, dump_steps)

  def testOOM(self):
    if not test.is_gpu_available():
      return
    ops.reset_default_graph()
    with ops.device('/device:GPU:0'):
      a = random_ops.random_normal([1, 10000, 20000], name='test_random1')
      b = random_ops.random_normal([30000, 10000, 1], name='test_random2')
      c = a * b

    try:
      with session.Session() as sess:
        sess.run(c, options=config_pb2.RunOptions(
            report_tensor_allocations_upon_oom=True))
    except Exception as e:  # pylint: disable=broad-except
      exception_str = '%s' % e
      # This trace reports allocations for to random tensor.
      self.assertTrue(
          'OOM when allocating tensor with shape[30000,10000,20000]' in
          exception_str)
      mat = re.search('(.*)GiB from test_random2/RandomStandardNormal',
                      exception_str)
      self.assertGreater(float(mat.group(1)), 0.0)
      mat = re.search('(.*)MiB from test_random1/RandomStandardNormal',
                      exception_str)
      self.assertGreater(float(mat.group(1)), 0.0)

  def testDistributedOOM(self):
    if not test.is_gpu_available():
      return
    ops.reset_default_graph()

    workers, _ = test_util.create_local_cluster(2, 0)

    with ops.device('/job:worker/replica:0/task:0/gpu:0'):
      a = random_ops.random_normal([1, 10000, 20000], name='test_random1')
    with ops.device('/job:worker/replica:0/task:1/gpu:0'):
      b = random_ops.random_normal([30000, 10000, 1], name='test_random2')
      c = a * b

    try:
      with session.Session(workers[1].target) as sess:
        sess.run(c, options=config_pb2.RunOptions(
            report_tensor_allocations_upon_oom=True))
    except Exception as e:  # pylint: disable=broad-except
      exception_str = '%s' % e
      # test_random2 is reported because it's allocated in worker 1.
      self.assertTrue('Current usage from device: '
                      '/job:worker/replica:0/task:1/device:GPU:0, '
                      'allocator: GPU_0_bfc' in exception_str)
      mat = re.search('(.*)GiB from test_random2/RandomStandardNormal',
                      exception_str)
      self.assertGreater(float(mat.group(1)), 0.0)
      # test_random1 is not reported because it's allocated in worker 0.
      mat = re.search('(.*)MiB from test_random1/RandomStandardNormal',
                      exception_str)
      self.assertTrue(mat is None)

  def testTrackPersistentBytes(self):
    ops.reset_default_graph()
    a = array_ops.constant(np.ones((100, 100)))
    b = array_ops.constant(np.ones((100, 100)))
    c = a * b
    config = config_pb2.ConfigProto()
    config.graph_options.rewrite_options.min_graph_nodes = -1

    with session.Session(config=config) as sess:
      run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()
      sess.run(c, options=run_options, run_metadata=run_metadata)

      options = option_builder.ProfileOptionBuilder.time_and_memory()
      options['min_bytes'] = 0
      options['select'] = ('bytes', 'peak_bytes', 'output_bytes',
                           'residual_bytes')
      ret = model_analyzer.profile(
          sess.graph, run_meta=run_metadata, cmd='scope', options=options)

      run_metadata = config_pb2.RunMetadata()
      sess.run(c, options=run_options, run_metadata=run_metadata)
      ret2 = model_analyzer.profile(
          sess.graph, run_meta=run_metadata, cmd='scope', options=options)

      n = lib.SearchTFProfNode(ret, 'mul')
      n2 = lib.SearchTFProfNode(ret2, 'mul')
      self.assertGreater(n.peak_bytes, 0)
      self.assertGreater(n.output_bytes, 0)
      self.assertGreater(n.residual_bytes, 0)
      self.assertEqual(n.peak_bytes, n2.peak_bytes)
      self.assertEqual(n.output_bytes, n2.output_bytes)
      self.assertEqual(n.residual_bytes, n2.residual_bytes)

  def testTraceLoopBytes(self):
    if not test.is_gpu_available(): return
    ops.reset_default_graph()
    steps = 100

    with ops.device('/gpu:0'):
      x = array_ops.ones((100, 100), dtype=dtypes.float32)
      n = array_ops.constant(steps, dtype=dtypes.int32)
      x1 = array_ops.ones((100, 100))

      x *= x1
      def loop_body(i, x):
        x *= x
        return i + 1, x

      _, y = control_flow_ops.while_loop(
          lambda i, x: i < n, loop_body,
          [array_ops.constant(0), x])

    grad = gradients.gradients(y, [x1])

    with session.Session() as sess:
      run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()
      sess.run(grad, options=run_options, run_metadata=run_metadata)

      options = option_builder.ProfileOptionBuilder.time_and_memory()
      options['min_bytes'] = 0
      options['min_micros'] = 0
      options['select'] = ('bytes', 'peak_bytes', 'output_bytes',
                           'residual_bytes')
      options['output'] = 'none'
      ret_pb = model_analyzer.profile(
          sess.graph, run_meta=run_metadata, cmd='scope', options=options)
      self.assertGreater(ret_pb.total_requested_bytes, 1000000)

  def testEager(self):
    ops.reset_default_graph()
    with context.eager_mode():
      outfile = os.path.join(test.get_temp_dir(), 'dump')
      opts = builder(
          builder.time_and_memory()).with_file_output(outfile).build()
      context.enable_run_metadata()
      lib.BuildSmallModel()

      profiler = model_analyzer.Profiler()
      profiler.add_step(0, context.export_run_metadata())
      context.disable_run_metadata()
      profiler.profile_operations(opts)
      with gfile.Open(outfile, 'r') as f:
        out_str = f.read()
        self.assertTrue('Conv2D' in out_str)
        self.assertTrue('VarHandleOp' in out_str)

      with gfile.Open('/tmp/eager_profile', 'wb') as f:
        profile_pb = tfprof_log_pb2.ProfileProto()
        profile_pb.ParseFromString(profiler.serialize_to_string())
        profile_pb_str = '%s' % profile_pb
        self.assertTrue('Conv2D' in profile_pb_str)
        self.assertTrue('VarHandleOp' in profile_pb_str)


if __name__ == '__main__':
  test.main()
