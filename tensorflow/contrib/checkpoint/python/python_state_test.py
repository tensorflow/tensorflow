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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy

from tensorflow.contrib.checkpoint.python import python_state
from tensorflow.python.client import session
from tensorflow.python.eager import test
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.training.tracking import util


class NumpyStateTests(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testSaveRestoreNumpyState(self):
    directory = self.get_temp_dir()
    prefix = os.path.join(directory, "ckpt")
    save_state = python_state.NumpyState()
    saver = util.Checkpoint(numpy=save_state)
    save_state.a = numpy.ones([2, 2])
    save_state.b = numpy.ones([2, 2])
    save_state.b = numpy.zeros([2, 2])
    save_state.c = numpy.int64(3)
    self.assertAllEqual(numpy.ones([2, 2]), save_state.a)
    self.assertAllEqual(numpy.zeros([2, 2]), save_state.b)
    self.assertEqual(3, save_state.c)
    first_save_path = saver.save(prefix)
    save_state.a[1, 1] = 2.
    save_state.c = numpy.int64(4)
    second_save_path = saver.save(prefix)

    load_state = python_state.NumpyState()
    loader = util.Checkpoint(numpy=load_state)
    loader.restore(first_save_path).initialize_or_restore()
    self.assertAllEqual(numpy.ones([2, 2]), load_state.a)
    self.assertAllEqual(numpy.zeros([2, 2]), load_state.b)
    self.assertEqual(3, load_state.c)
    load_state.a[0, 0] = 42.
    self.assertAllEqual([[42., 1.], [1., 1.]], load_state.a)
    loader.restore(first_save_path).run_restore_ops()
    self.assertAllEqual(numpy.ones([2, 2]), load_state.a)
    loader.restore(second_save_path).run_restore_ops()
    self.assertAllEqual([[1., 1.], [1., 2.]], load_state.a)
    self.assertAllEqual(numpy.zeros([2, 2]), load_state.b)
    self.assertEqual(4, load_state.c)

  def testNoGraphPollution(self):
    graph = ops.Graph()
    with graph.as_default(), session.Session():
      directory = self.get_temp_dir()
      prefix = os.path.join(directory, "ckpt")
      save_state = python_state.NumpyState()
      saver = util.Checkpoint(numpy=save_state)
      save_state.a = numpy.ones([2, 2])
      save_path = saver.save(prefix)
      saver.restore(save_path)
      graph.finalize()
      saver.save(prefix)
      save_state.a = numpy.zeros([2, 2])
      saver.save(prefix)
      saver.restore(save_path)

  @test_util.run_in_graph_and_eager_modes
  def testNoMixedNumpyStateTF(self):
    save_state = python_state.NumpyState()
    save_state.a = numpy.ones([2, 2])
    with self.assertRaises(NotImplementedError):
      save_state.v = variables.Variable(1.)

  @test_util.run_in_graph_and_eager_modes
  def testDocstringExample(self):
    arrays = python_state.NumpyState()
    checkpoint = util.Checkpoint(numpy_arrays=arrays)
    arrays.x = numpy.zeros([3, 4])
    save_path = checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))
    arrays.x[1, 1] = 4.
    checkpoint.restore(save_path)
    self.assertAllEqual(numpy.zeros([3, 4]), arrays.x)

    second_checkpoint = util.Checkpoint(numpy_arrays=python_state.NumpyState())
    second_checkpoint.restore(save_path)
    self.assertAllEqual(numpy.zeros([3, 4]), second_checkpoint.numpy_arrays.x)


if __name__ == "__main__":
  test.main()
