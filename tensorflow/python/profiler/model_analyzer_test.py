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

from tensorflow.core.profiler import profile_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder
from tensorflow.python.profiler import profile_context
from tensorflow.python.profiler.internal import model_analyzer_testlib as lib

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
                         f.read())

  def testSelectEverthingDetail(self):
    ops.reset_default_graph()
    dev = '/gpu:0' if test.is_gpu_available() else '/cpu:0'
    outfile = os.path.join(test.get_temp_dir(), 'dump')
    opts = (builder(builder.trainable_variables_parameter())
            .with_file_output(outfile)
            .with_accounted_types(['.*'])
            .select(['micros', 'bytes', 'params', 'float_ops', 'occurrence',
                     'device', 'op_types', 'input_shapes']).build())

    config = config_pb2.ConfigProto()
    with session.Session(config=config) as sess, ops.device(dev):
      x = lib.BuildSmallModel()

      sess.run(variables.global_variables_initializer())
      run_meta = config_pb2.RunMetadata()
      _ = sess.run(x,
                   options=config_pb2.RunOptions(
                       trace_level=config_pb2.RunOptions.FULL_TRACE),
                   run_metadata=run_meta)

      model_analyzer.profile(
          sess.graph, run_meta, options=opts)

      with gfile.Open(outfile, 'r') as f:
        # pylint: disable=line-too-long
        outputs = f.read().split('\n')

        self.assertEqual(outputs[0],
                         'node name | # parameters | # float_ops | requested bytes | total execution time | accelerator execution time | cpu execution time | assigned devices | op types | op count (run|defined) | input shapes')
        for o in outputs[1:]:
          if o.find('Conv2D ') > 0:
            metrics = o[o.find('(') +1: o.find(')')].split(',')
            # Make sure time is profiled.
            gap = 1 if test.is_gpu_available() else 2
            for i in range(3, 6, gap):
              mat = re.search('(.*)[um]s/(.*)[um]s', metrics[i])
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
    with session.Session(config=config) as sess, ops.device('/cpu:0'):
      x = lib.BuildSmallModel()

      sess.run(variables.global_variables_initializer())
      run_meta = config_pb2.RunMetadata()
      _ = sess.run(x,
                   options=config_pb2.RunOptions(
                       trace_level=config_pb2.RunOptions.FULL_TRACE),
                   run_metadata=run_meta)

      model_analyzer.profile(
          sess.graph, run_meta, options=opts)

      with gfile.Open(outfile, 'r') as f:
        # pylint: disable=line-too-long
        self.assertEqual(
            'node name | # parameters | # float_ops | assigned devices | op types | op count (run|defined) | input shapes\n_TFProfRoot (--/451 params, --/10.44k flops, _kTFScopeParent, --/8|--/36, )\n  Conv2D (0/0 params, 5.83k/5.83k flops, /job:localhost/replica:0/task:0/cpu:0, /job:localhost/replica:0/task:0/cpu:0|Conv2D, 1/1|1/1, 0:2x6x6x3|1:3x3x3x6)\n  Conv2D_1 (0/0 params, 4.61k/4.61k flops, /job:localhost/replica:0/task:0/cpu:0, /job:localhost/replica:0/task:0/cpu:0|Conv2D, 1/1|1/1, 0:2x3x3x6|1:2x2x6x12)\n  DW (3x3x3x6, 162/162 params, 0/0 flops, /job:localhost/replica:0/task:0/cpu:0, /job:localhost/replica:0/task:0/cpu:0|VariableV2|_trainable_variables, 1/2|1/10, )\n    DW/Assign (0/0 params, 0/0 flops, Assign, 0/0|1/1, 0:3x3x3x6|1:3x3x3x6)\n    DW/Initializer (0/0 params, 0/0 flops, _kTFScopeParent, 0/0|1/7, )\n      DW/Initializer/random_normal (0/0 params, 0/0 flops, Add, 0/0|1/6, 0:3x3x3x6|1:1)\n        DW/Initializer/random_normal/RandomStandardNormal (0/0 params, 0/0 flops, RandomStandardNormal, 0/0|1/1, 0:4)\n        DW/Initializer/random_normal/mean (0/0 params, 0/0 flops, Const, 0/0|1/1, )\n        DW/Initializer/random_normal/mul (0/0 params, 0/0 flops, Mul, 0/0|1/1, 0:3x3x3x6|1:1)\n        DW/Initializer/random_normal/shape (0/0 params, 0/0 flops, Const, 0/0|1/1, )\n        DW/Initializer/random_normal/stddev (0/0 params, 0/0 flops, Const, 0/0|1/1, )\n    DW/read (0/0 params, 0/0 flops, /job:localhost/replica:0/task:0/cpu:0, /job:localhost/replica:0/task:0/cpu:0|Identity, 1/1|1/1, 0:3x3x3x6)\n  DW2 (2x2x6x12, 288/288 params, 0/0 flops, /job:localhost/replica:0/task:0/cpu:0, /job:localhost/replica:0/task:0/cpu:0|VariableV2|_trainable_variables, 1/2|1/10, )\n    DW2/Assign (0/0 params, 0/0 flops, Assign, 0/0|1/1, 0:2x2x6x12|1:2x2x6x12)\n    DW2/Initializer (0/0 params, 0/0 flops, _kTFScopeParent, 0/0|1/7, )\n      DW2/Initializer/random_normal (0/0 params, 0/0 flops, Add, 0/0|1/6, 0:2x2x6x12|1:1)\n        DW2/Initializer/random_normal/RandomStandardNormal (0/0 params, 0/0 flops, RandomStandardNormal, 0/0|1/1, 0:4)\n        DW2/Initializer/random_normal/mean (0/0 params, 0/0 flops, Const, 0/0|1/1, )\n        DW2/Initializer/random_normal/mul (0/0 params, 0/0 flops, Mul, 0/0|1/1, 0:2x2x6x12|1:1)\n        DW2/Initializer/random_normal/shape (0/0 params, 0/0 flops, Const, 0/0|1/1, )\n        DW2/Initializer/random_normal/stddev (0/0 params, 0/0 flops, Const, 0/0|1/1, )\n    DW2/read (0/0 params, 0/0 flops, /job:localhost/replica:0/task:0/cpu:0, /job:localhost/replica:0/task:0/cpu:0|Identity, 1/1|1/1, 0:2x2x6x12)\n  ScalarW (1, 1/1 params, 0/0 flops, VariableV2|_trainable_variables, 0/0|1/10, )\n    ScalarW/Assign (0/0 params, 0/0 flops, Assign, 0/0|1/1, 0:1|1:1)\n    ScalarW/Initializer (0/0 params, 0/0 flops, _kTFScopeParent, 0/0|1/7, )\n      ScalarW/Initializer/random_normal (0/0 params, 0/0 flops, Add, 0/0|1/6, 0:1|1:1)\n        ScalarW/Initializer/random_normal/RandomStandardNormal (0/0 params, 0/0 flops, RandomStandardNormal, 0/0|1/1, 0:0)\n        ScalarW/Initializer/random_normal/mean (0/0 params, 0/0 flops, Const, 0/0|1/1, )\n        ScalarW/Initializer/random_normal/mul (0/0 params, 0/0 flops, Mul, 0/0|1/1, 0:1|1:1)\n        ScalarW/Initializer/random_normal/shape (0/0 params, 0/0 flops, Const, 0/0|1/1, )\n        ScalarW/Initializer/random_normal/stddev (0/0 params, 0/0 flops, Const, 0/0|1/1, )\n    ScalarW/read (0/0 params, 0/0 flops, Identity, 0/0|1/1, 0:1)\n  _retval_Conv2D_1_0_0 (0/0 params, 0/0 flops, /job:localhost/replica:0/task:0/cpu:0, /job:localhost/replica:0/task:0/cpu:0|RunTimeOp, 1/1|1/1, )\n  init (0/0 params, 0/0 flops, NoOp, 0/0|1/1, 0:1|1:3x3x3x6|2:2x2x6x12)\n  zeros (0/0 params, 0/0 flops, /job:localhost/replica:0/task:0/cpu:0, /job:localhost/replica:0/task:0/cpu:0|Const, 1/1|1/1, )\n',
            f.read())
        # pylint: enable=line-too-long

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
            f.read()[0:80])
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

    with session.Session() as sess:
      x = lib.BuildFullModel()

      sess.run(variables.global_variables_initializer())
      run_meta = config_pb2.RunMetadata()
      _ = sess.run(x,
                   options=config_pb2.RunOptions(
                       trace_level=config_pb2.RunOptions.FULL_TRACE),
                   run_metadata=run_meta)

      tfprof_node = model_analyzer.profile(
          sess.graph, run_meta, cmd='code', options=opts)

      # pylint: disable=line-too-long
      with gfile.Open(outfile, 'r') as f:
        lines = f.read().split('\n')
        result = '\n'.join([l[:min(len(l), 80)] for l in lines])
        self.assertEqual('node name | # parameters | # float_ops\n_TFProfRoot (--/2.84k params, --/91.04k flops)\n  model_analyzer_testlib.py:58:BuildFullModel:seq.append(array_... (0/1.80k para\n    model_analyzer_testlib.py:35:BuildSmallModel:image = array_ops... (0/0 param\n    model_analyzer_testlib.py:39:BuildSmallModel:initializer=init_... (0/4 param\n    model_analyzer_testlib.py:43:BuildSmallModel:initializer=init_... (0/648 par\n    model_analyzer_testlib.py:44:BuildSmallModel:x = nn_ops.conv2d... (0/0 param\n    model_analyzer_testlib.py:48:BuildSmallModel:initializer=init_... (0/1.15k p\n    model_analyzer_testlib.py:49:BuildSmallModel:x = nn_ops.conv2d... (0/0 param\n  model_analyzer_testlib.py:58:BuildFullModel:seq.append(array_... (gradient) (0\n    model_analyzer_testlib.py:44:BuildSmallModel:x = nn_ops.conv2d... (gradient)\n    model_analyzer_testlib.py:49:BuildSmallModel:x = nn_ops.conv2d... (gradient)\n  model_analyzer_testlib.py:62:BuildFullModel:cell, array_ops.c... (0/1.04k para\n  model_analyzer_testlib.py:62:BuildFullModel:cell, array_ops.c... (gradient) (0\n  model_analyzer_testlib.py:64:BuildFullModel:target = array_op... (0/0 params, \n  model_analyzer_testlib.py:65:BuildFullModel:loss = nn_ops.l2_... (0/0 params, \n  model_analyzer_testlib.py:65:BuildFullModel:loss = nn_ops.l2_... (gradient) (0\n  model_analyzer_testlib.py:67:BuildFullModel:return sgd_op.min... (0/0 params, \n',
                         result)

      self.assertLess(0, tfprof_node.total_exec_micros)
      self.assertEqual(2844, tfprof_node.total_parameters)
      self.assertEqual(91040, tfprof_node.total_float_ops)
      self.assertEqual(8, len(tfprof_node.children))
      self.assertEqual('_TFProfRoot', tfprof_node.name)
      self.assertEqual(
          'model_analyzer_testlib.py:58:BuildFullModel:seq.append(array_...',
          tfprof_node.children[0].name)
      self.assertEqual(
          'model_analyzer_testlib.py:58:BuildFullModel:seq.append(array_... (gradient)',
          tfprof_node.children[1].name)
      self.assertEqual(
          'model_analyzer_testlib.py:62:BuildFullModel:cell, array_ops.c...',
          tfprof_node.children[2].name)
      self.assertEqual(
          'model_analyzer_testlib.py:62:BuildFullModel:cell, array_ops.c... (gradient)',
          tfprof_node.children[3].name)
      self.assertEqual(
          'model_analyzer_testlib.py:64:BuildFullModel:target = array_op...',
          tfprof_node.children[4].name)
      self.assertEqual(
          'model_analyzer_testlib.py:65:BuildFullModel:loss = nn_ops.l2_...',
          tfprof_node.children[5].name)
      self.assertEqual(
          'model_analyzer_testlib.py:65:BuildFullModel:loss = nn_ops.l2_... (gradient)',
          tfprof_node.children[6].name)
      self.assertEqual(
          'model_analyzer_testlib.py:67:BuildFullModel:return sgd_op.min...',
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

      with gfile.Open(outfile, 'r') as f:
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
            'nodename|requestedbytes|peakbytes|residualbytes|outputbytes|totalexecutiontime|acceleratorexecutiontime|cpuexecutiontime|#parameters|opoccurrence(run|defined)|inputshapes\nConst0B(0',
            f.read().replace('\t', '').replace(' ', '')[0:180])
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

      self.assertEqual(total_children, 15)
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
          self.assertAllEqual(sorted(ret),
                              ['graph.pbtxt', 'run_metadata', 'tfprof_log'])
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
    with profile_context.ProfileContext() as pctx:
      pctx.add_auto_profiling('scope', time_opts, time_steps)
      pctx.add_auto_profiling('scope', memory_opts, memory_steps)
      pctx.add_auto_profile_dump(profile_dir, dump_steps)

      self._trainLoop(x, 10, time_dir, time_steps,
                      memory_dir, memory_steps, profile_dir, dump_steps)


if __name__ == '__main__':
  test.main()
