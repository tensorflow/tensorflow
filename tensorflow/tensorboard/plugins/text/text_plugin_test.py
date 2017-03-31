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
import textwrap

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

  def assertConverted(self, actual, expected):
    expected_html = text_plugin.markdown_and_sanitize(expected)
    self.assertEqual(actual, expected_html)

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
        message = run_name + " *loves* " + gem
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
      self.assertConverted(fry[i]["text"], "fry *loves* " + GEMS[i])
      self.assertEqual(leela[i]["step"], i)
      self.assertConverted(leela[i]["text"], "leela *loves* " + GEMS[i])

  def assertTextConverted(self, actual, expected):
    self.assertEqual(text_plugin.markdown_and_sanitize(actual), expected)

  def testMarkdownConversion(self):
    emphasis = "*Italics1* _Italics2_ **bold1** __bold2__"
    emphasis_converted = ("<p><em>Italics1</em> <em>Italics2</em> "
                          "<strong>bold1</strong> <strong>bold2</strong></p>")

    self.assertEqual(
        text_plugin.markdown_and_sanitize(emphasis), emphasis_converted)

    md_list = textwrap.dedent("""\
    1. List item one.
    2. List item two.
      * Sublist
      * Sublist2
    1. List continues.
    """)
    md_list_converted = textwrap.dedent("""\
    <ol>
    <li>List item one.</li>
    <li>List item two.</li>
    <li>Sublist</li>
    <li>Sublist2</li>
    <li>List continues.</li>
    </ol>""")
    self.assertEqual(
        text_plugin.markdown_and_sanitize(md_list), md_list_converted)

    link = "[TensorFlow](http://tensorflow.org)"
    link_converted = '<p><a href="http://tensorflow.org">TensorFlow</a></p>'
    self.assertEqual(text_plugin.markdown_and_sanitize(link), link_converted)

    table = textwrap.dedent("""\
    An | Example | Table
    --- | --- | ---
    A | B | C
    1 | 2 | 3
    """)

    table_converted = textwrap.dedent("""\
    <table>
    <thead>
    <tr>
    <th>An</th>
    <th>Example</th>
    <th>Table</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>A</td>
    <td>B</td>
    <td>C</td>
    </tr>
    <tr>
    <td>1</td>
    <td>2</td>
    <td>3</td>
    </tr>
    </tbody>
    </table>""")

    self.assertEqual(text_plugin.markdown_and_sanitize(table), table_converted)

  def testSanitization(self):
    dangerous = "<script>alert('xss')</script>"
    sanitized = "&lt;script&gt;alert('xss')&lt;/script&gt;"
    self.assertEqual(text_plugin.markdown_and_sanitize(dangerous), sanitized)

    dangerous = textwrap.dedent("""\
    hello <a name="n"
    href="javascript:alert('xss')">*you*</a>""")
    sanitized = "<p>hello <a><em>you</em></a></p>"
    self.assertEqual(text_plugin.markdown_and_sanitize(dangerous), sanitized)


if __name__ == "__main__":
  test.main()
