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
import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.summary import summary
from tensorflow.python.summary import text_summary
from tensorflow.tensorboard.backend.event_processing import event_multiplexer
from tensorflow.tensorboard.plugins.text import text_plugin

GEMS = ['garnet', 'amethyst', 'pearl', 'steven']


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
    placeholder = array_ops.placeholder(dtypes.string)
    summary_tensor = text_summary.text_summary('message', placeholder)

    vector_summary = text_summary.text_summary('vector', placeholder)

    run_names = ['fry', 'leela']
    for run_name in run_names:
      subdir = os.path.join(self.logdir, run_name)
      writer = summary.FileWriter(subdir)
      writer.add_graph(sess.graph)

      step = 0
      for gem in GEMS:
        message = run_name + ' *loves* ' + gem
        feed_dict = {placeholder: message}
        summ = sess.run(summary_tensor, feed_dict=feed_dict)
        writer.add_summary(summ, global_step=step)
        step += 1

      vector_message = ['one', 'two', 'three', 'four']
      summ = sess.run(vector_summary, feed_dict={placeholder: vector_message})
      writer.add_summary(summ)
      writer.close()

  def testIndex(self):
    index = self.plugin.index_impl()
    self.assertEqual(index, {
        'fry': ['message', 'vector'],
        'leela': ['message', 'vector'],
    })

  def testText(self):
    fry = self.plugin.text_impl('fry', 'message')
    leela = self.plugin.text_impl('leela', 'message')
    self.assertEqual(len(fry), 4)
    self.assertEqual(len(leela), 4)
    for i in range(4):
      self.assertEqual(fry[i]['step'], i)
      self.assertConverted(fry[i]['text'], 'fry *loves* ' + GEMS[i])
      self.assertEqual(leela[i]['step'], i)
      self.assertConverted(leela[i]['text'], 'leela *loves* ' + GEMS[i])

    table = self.plugin.text_impl('fry', 'vector')[0]['text']
    self.assertEqual(table,
                     textwrap.dedent("""\
      <table>
      <tbody>
      <tr>
      <td><p>one</p></td>
      </tr>
      <tr>
      <td><p>two</p></td>
      </tr>
      <tr>
      <td><p>three</p></td>
      </tr>
      <tr>
      <td><p>four</p></td>
      </tr>
      </tbody>
      </table>"""))

  def assertTextConverted(self, actual, expected):
    self.assertEqual(text_plugin.markdown_and_sanitize(actual), expected)

  def testMarkdownConversion(self):
    emphasis = '*Italics1* _Italics2_ **bold1** __bold2__'
    emphasis_converted = ('<p><em>Italics1</em> <em>Italics2</em> '
                          '<strong>bold1</strong> <strong>bold2</strong></p>')

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

    link = '[TensorFlow](http://tensorflow.org)'
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
    hello <a name='n'
    href='javascript:alert('xss')'>*you*</a>""")
    sanitized = '<p>hello <a><em>you</em></a></p>'
    self.assertEqual(text_plugin.markdown_and_sanitize(dangerous), sanitized)

  def testTableGeneration(self):
    array2d = np.array([['one', 'two'], ['three', 'four']])
    expected_table = textwrap.dedent("""\
    <table>
    <tbody>
    <tr>
    <td>one</td>
    <td>two</td>
    </tr>
    <tr>
    <td>three</td>
    <td>four</td>
    </tr>
    </tbody>
    </table>""")
    self.assertEqual(text_plugin.make_table(array2d), expected_table)

    expected_table_with_headers = textwrap.dedent("""\
    <table>
    <thead>
    <tr>
    <th>c1</th>
    <th>c2</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>one</td>
    <td>two</td>
    </tr>
    <tr>
    <td>three</td>
    <td>four</td>
    </tr>
    </tbody>
    </table>""")

    actual_with_headers = text_plugin.make_table(array2d, headers=['c1', 'c2'])
    self.assertEqual(actual_with_headers, expected_table_with_headers)

    array_1d = np.array(['one', 'two', 'three', 'four', 'five'])
    expected_1d = textwrap.dedent("""\
    <table>
    <tbody>
    <tr>
    <td>one</td>
    </tr>
    <tr>
    <td>two</td>
    </tr>
    <tr>
    <td>three</td>
    </tr>
    <tr>
    <td>four</td>
    </tr>
    <tr>
    <td>five</td>
    </tr>
    </tbody>
    </table>""")
    self.assertEqual(text_plugin.make_table(array_1d), expected_1d)

    expected_1d_with_headers = textwrap.dedent("""\
    <table>
    <thead>
    <tr>
    <th>X</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>one</td>
    </tr>
    <tr>
    <td>two</td>
    </tr>
    <tr>
    <td>three</td>
    </tr>
    <tr>
    <td>four</td>
    </tr>
    <tr>
    <td>five</td>
    </tr>
    </tbody>
    </table>""")
    actual_1d_with_headers = text_plugin.make_table(array_1d, headers=['X'])
    self.assertEqual(actual_1d_with_headers, expected_1d_with_headers)

  def testMakeTableExceptions(self):
    # Verify that contents is being type-checked and shape-checked.
    with self.assertRaises(ValueError):
      text_plugin.make_table([])

    with self.assertRaises(ValueError):
      text_plugin.make_table('foo')

    with self.assertRaises(ValueError):
      invalid_shape = np.full((3, 3, 3), 'nope', dtype=np.dtype('S3'))
      text_plugin.make_table(invalid_shape)

    # Test headers exceptions in 2d array case.
    test_array = np.full((3, 3), 'foo', dtype=np.dtype('S3'))
    with self.assertRaises(ValueError):
      # Headers is wrong type.
      text_plugin.make_table(test_array, headers='foo')
    with self.assertRaises(ValueError):
      # Too many headers.
      text_plugin.make_table(test_array, headers=['foo', 'bar', 'zod', 'zoink'])
    with self.assertRaises(ValueError):
      # headers is 2d
      text_plugin.make_table(test_array, headers=test_array)

    # Also make sure the column counting logic works in the 1d array case.
    test_array = np.array(['foo', 'bar', 'zod'])
    with self.assertRaises(ValueError):
      # Too many headers.
      text_plugin.make_table(test_array, headers=test_array)

  def test_reduce_to_2d(self):

    def make_range_array(dim):
      """Produce an incrementally increasing multidimensional array.

      Args:
        dim: the number of dimensions for the array

      Returns:
        An array of increasing integer elements, with dim dimensions and size
        two in each dimension.

      Example: rangeArray(2) results in [[0,1],[2,3]].
      """
      return np.array(range(2**dim)).reshape([2] * dim)

    for i in range(2, 5):
      actual = text_plugin.reduce_to_2d(make_range_array(i))
      expected = make_range_array(2)
      np.testing.assert_array_equal(actual, expected)

  def test_text_array_to_html(self):

    convert = text_plugin.text_array_to_html
    scalar = np.array('foo')
    scalar_expected = '<p>foo</p>'
    self.assertEqual(convert(scalar), scalar_expected)

    vector = np.array(['foo', 'bar'])
    vector_expected = textwrap.dedent("""\
      <table>
      <tbody>
      <tr>
      <td><p>foo</p></td>
      </tr>
      <tr>
      <td><p>bar</p></td>
      </tr>
      </tbody>
      </table>""")
    self.assertEqual(convert(vector), vector_expected)

    d2 = np.array([['foo', 'bar'], ['zoink', 'zod']])
    d2_expected = textwrap.dedent("""\
      <table>
      <tbody>
      <tr>
      <td><p>foo</p></td>
      <td><p>bar</p></td>
      </tr>
      <tr>
      <td><p>zoink</p></td>
      <td><p>zod</p></td>
      </tr>
      </tbody>
      </table>""")
    self.assertEqual(convert(d2), d2_expected)

    d3 = np.array([[['foo', 'bar'], ['zoink', 'zod']], [['FOO', 'BAR'],
                                                        ['ZOINK', 'ZOD']]])

    warning = text_plugin.markdown_and_sanitize(text_plugin.WARNING_TEMPLATE %
                                                3)
    d3_expected = warning + textwrap.dedent("""\
      <table>
      <tbody>
      <tr>
      <td><p>foo</p></td>
      <td><p>bar</p></td>
      </tr>
      <tr>
      <td><p>zoink</p></td>
      <td><p>zod</p></td>
      </tr>
      </tbody>
      </table>""")
    self.assertEqual(convert(d3), d3_expected)


if __name__ == '__main__':
  test.main()
