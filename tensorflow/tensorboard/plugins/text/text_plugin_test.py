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
"""Integration tests for the Text Plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import six

from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.summary import summary
from tensorflow.python.summary import text_summary
from tensorflow.tensorboard.backend.event_processing import event_multiplexer
from tensorflow.tensorboard.plugins.text import text_plugin

GEMS = ["garnet", "amethyst", "pearl", "steven"]


class TextPluginTest(test.TestCase):

  def setUp(self):
    self.logdir = self.get_temp_dir()
    self.generate_testdata()
    multiplexer = event_multiplexer.EventMultiplexer()
    multiplexer.AddRunsFromDirectory(self.logdir)
    multiplexer.Reload()
    self.plugin = text_plugin.TextPlugin()
    self.apps = self.plugin.get_plugin_apps(multiplexer, None)

  def generate_testdata(self):
    ops.reset_default_graph()
    sess = session.Session()
    placeholder = array_ops.placeholder(dtypes.string, shape=[])
    summary_tensor = text_summary.text_summary("message", placeholder)

    run_names = ["fry", "leela"]
    for run_name in run_names:
      subdir = os.path.join(self.logdir, run_name)
      writer = summary.FileWriter(subdir)
      writer.add_graph(sess.graph)

      step = 0
      for gem in GEMS:
        message = run_name + " loves " + gem
        feed_dict = {placeholder: message}
        summ = sess.run(summary_tensor, feed_dict=feed_dict)
        writer.add_summary(summ, global_step=step)
        step += 1
      writer.close()

  def testIndex(self):
    index = self.plugin.index_impl()
    self.assertEqual(index, {
        "fry": ["message"],
        "leela": ["message"],
    })

  def testText(self):
    fry = self.plugin.text_impl("fry", "message")
    leela = self.plugin.text_impl("leela", "message")
    self.assertEqual(len(fry), 4)
    self.assertEqual(len(leela), 4)
    for i in range(4):
      self.assertEqual(fry[i]["step"], i)
      self.assertEqual(fry[i]["text"], six.b("fry loves " + GEMS[i]))
      self.assertEqual(leela[i]["step"], i)
      self.assertEqual(leela[i]["text"], six.b("leela loves " + GEMS[i]))


if __name__ == "__main__":
  test.main()
