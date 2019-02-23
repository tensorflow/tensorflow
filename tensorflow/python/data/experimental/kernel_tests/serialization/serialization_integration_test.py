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
"""Integration test for dataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.data.experimental.ops import iterator_ops as contrib_iterator_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_lib


class SerializationIntegrationTest(test.TestCase):

  def _build_input_pipeline(self, name, num_outputs):
    with ops.name_scope(name):
      ds = dataset_ops.Dataset.range(num_outputs).shuffle(
          10, reshuffle_each_iteration=False).prefetch(10)
      iterator = ds.make_initializable_iterator()
      saveable = contrib_iterator_ops.make_saveable_from_iterator(iterator)
      ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)
      return iterator.initializer, iterator.get_next()

  def _build_graph(self, num_pipelines, num_outputs):
    init_ops = []
    get_next_ops = []
    for i in range(num_pipelines):
      name = "input_pipeline_%d" % i
      init_op, get_next_op = self._build_input_pipeline(name, num_outputs)
      init_ops.append(init_op)
      get_next_ops.append(get_next_op)
    saver = saver_lib.Saver()
    return init_ops, get_next_ops, saver

  def _ckpt_path(self):
    return os.path.join(self.get_temp_dir(), "iterator")

  def testConcurrentSaves(self):
    num_pipelines = 100
    num_outputs = 100
    break_point = 10
    all_outputs = [[] for _ in range(num_pipelines)]
    with ops.Graph().as_default() as g:
      init_ops, get_next_ops, saver = self._build_graph(num_pipelines,
                                                        num_outputs)
      with self.session(graph=g) as sess:
        self.evaluate(init_ops)
        for _ in range(break_point):
          output = self.evaluate(get_next_ops)
          for i in range(num_pipelines):
            all_outputs[i].append(output[i])
        saver.save(sess, self._ckpt_path())

    with ops.Graph().as_default() as g:
      init_ops, get_next_ops, saver = self._build_graph(num_pipelines,
                                                        num_outputs)
      with self.session(graph=g) as sess:
        saver.restore(sess, self._ckpt_path())
        for _ in range(num_outputs - break_point):
          output = self.evaluate(get_next_ops)
          for i in range(num_pipelines):
            all_outputs[i].append(output[i])

    for output in all_outputs:
      self.assertSequenceEqual(sorted(output), range(num_outputs))


if __name__ == "__main__":
  test.main()
