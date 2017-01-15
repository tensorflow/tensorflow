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
"""Unit tests for the shared functions and classes for tfdbg CLI."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class BytesToReadableStrTest(test_util.TensorFlowTestCase):

  def testNoneSizeWorks(self):
    self.assertEqual(str(None), cli_shared.bytes_to_readable_str(None))

  def testSizesBelowOneKiloByteWorks(self):
    self.assertEqual("0", cli_shared.bytes_to_readable_str(0))
    self.assertEqual("500", cli_shared.bytes_to_readable_str(500))
    self.assertEqual("1023", cli_shared.bytes_to_readable_str(1023))

  def testSizesBetweenOneKiloByteandOneMegaByteWorks(self):
    self.assertEqual("1.00k", cli_shared.bytes_to_readable_str(1024))
    self.assertEqual("2.40k", cli_shared.bytes_to_readable_str(int(1024 * 2.4)))
    self.assertEqual("1023.00k", cli_shared.bytes_to_readable_str(1024 * 1023))

  def testSizesBetweenOneMegaByteandOneGigaByteWorks(self):
    self.assertEqual("1.00M", cli_shared.bytes_to_readable_str(1024**2))
    self.assertEqual("2.40M",
                     cli_shared.bytes_to_readable_str(int(1024**2 * 2.4)))
    self.assertEqual("1023.00M",
                     cli_shared.bytes_to_readable_str(1024**2 * 1023))

  def testSizeAboveOneGigaByteWorks(self):
    self.assertEqual("1.00G", cli_shared.bytes_to_readable_str(1024**3))
    self.assertEqual("2000.00G",
                     cli_shared.bytes_to_readable_str(1024**3 * 2000))

  def testReadableStrIncludesBAtTheEndOnRequest(self):
    self.assertEqual("0B", cli_shared.bytes_to_readable_str(0, include_b=True))
    self.assertEqual(
        "1.00kB", cli_shared.bytes_to_readable_str(
            1024, include_b=True))
    self.assertEqual(
        "1.00MB", cli_shared.bytes_to_readable_str(
            1024**2, include_b=True))
    self.assertEqual(
        "1.00GB", cli_shared.bytes_to_readable_str(
            1024**3, include_b=True))


class GetRunStartIntroAndDescriptionTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.const_a = constant_op.constant(11.0, name="a")
    self.const_b = constant_op.constant(22.0, name="b")
    self.const_c = constant_op.constant(33.0, name="c")

    self.sparse_d = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 1]], values=[1.0, 2.0], dense_shape=[3, 3])

  def tearDown(self):
    ops.reset_default_graph()

  def testSingleFetchNoFeeds(self):
    run_start_intro = cli_shared.get_run_start_intro(12, self.const_a, None, {})

    # Verify line about run() call number.
    self.assertTrue(run_start_intro.lines[1].endswith("run() call #12:"))

    # Verify line about fetch.
    const_a_name_line = run_start_intro.lines[4]
    self.assertEqual(self.const_a.name, const_a_name_line.strip())

    # Verify line about feeds.
    feeds_line = run_start_intro.lines[7]
    self.assertEqual("(Empty)", feeds_line.strip())

    # Verify lines about possible commands and their font attributes.
    self.assertEqual("run:", run_start_intro.lines[11][2:])
    annot = run_start_intro.font_attr_segs[11][0]
    self.assertEqual(2, annot[0])
    self.assertEqual(5, annot[1])
    self.assertEqual("run", annot[2][0].content)
    self.assertEqual("bold", annot[2][1])
    annot = run_start_intro.font_attr_segs[13][0]
    self.assertEqual(2, annot[0])
    self.assertEqual(8, annot[1])
    self.assertEqual("run -n", annot[2][0].content)
    self.assertEqual("bold", annot[2][1])
    self.assertEqual("run -t <T>:", run_start_intro.lines[15][2:])
    self.assertEqual([(2, 12, "bold")], run_start_intro.font_attr_segs[15])
    self.assertEqual("run -f <filter_name>:", run_start_intro.lines[17][2:])
    self.assertEqual([(2, 22, "bold")], run_start_intro.font_attr_segs[17])
    annot = run_start_intro.font_attr_segs[21][0]
    self.assertEqual(2, annot[0])
    self.assertEqual(16, annot[1])
    self.assertEqual("invoke_stepper", annot[2][0].content)

    # Verify short description.
    description = cli_shared.get_run_short_description(12, self.const_a, None)
    self.assertEqual("run #12: 1 fetch (a:0); 0 feeds", description)

    # Verify the main menu associated with the run_start_intro.
    self.assertIn(debugger_cli_common.MAIN_MENU_KEY,
                  run_start_intro.annotations)
    menu = run_start_intro.annotations[debugger_cli_common.MAIN_MENU_KEY]
    self.assertEqual("run", menu.caption_to_item("run").content)
    self.assertEqual("invoke_stepper",
                     menu.caption_to_item("invoke_stepper").content)
    self.assertEqual("exit", menu.caption_to_item("exit").content)

  def testSparseTensorAsFetchShouldHandleNoNameAttribute(self):
    run_start_intro = cli_shared.get_run_start_intro(1, self.sparse_d, None, {})
    self.assertEqual(str(self.sparse_d), run_start_intro.lines[4].strip())

  def testTwoFetchesListNoFeeds(self):
    fetches = [self.const_a, self.const_b]
    run_start_intro = cli_shared.get_run_start_intro(1, fetches, None, {})

    const_a_name_line = run_start_intro.lines[4]
    const_b_name_line = run_start_intro.lines[5]
    self.assertEqual(self.const_a.name, const_a_name_line.strip())
    self.assertEqual(self.const_b.name, const_b_name_line.strip())

    feeds_line = run_start_intro.lines[8]
    self.assertEqual("(Empty)", feeds_line.strip())

    # Verify short description.
    description = cli_shared.get_run_short_description(1, fetches, None)
    self.assertEqual("run #1: 2 fetches; 0 feeds", description)

  def testNestedListAsFetches(self):
    fetches = [self.const_c, [self.const_a, self.const_b]]
    run_start_intro = cli_shared.get_run_start_intro(1, fetches, None, {})

    # Verify lines about the fetches.
    self.assertEqual(self.const_c.name, run_start_intro.lines[4].strip())
    self.assertEqual(self.const_a.name, run_start_intro.lines[5].strip())
    self.assertEqual(self.const_b.name, run_start_intro.lines[6].strip())

    # Verify short description.
    description = cli_shared.get_run_short_description(1, fetches, None)
    self.assertEqual("run #1: 3 fetches; 0 feeds", description)

  def testNestedDictAsFetches(self):
    fetches = {"c": self.const_c, "ab": {"a": self.const_a, "b": self.const_b}}
    run_start_intro = cli_shared.get_run_start_intro(1, fetches, None, {})

    # Verify lines about the fetches. The ordering of the dict keys is
    # indeterminate.
    fetch_names = set()
    fetch_names.add(run_start_intro.lines[4].strip())
    fetch_names.add(run_start_intro.lines[5].strip())
    fetch_names.add(run_start_intro.lines[6].strip())

    self.assertEqual({"a:0", "b:0", "c:0"}, fetch_names)

    # Verify short description.
    description = cli_shared.get_run_short_description(1, fetches, None)
    self.assertEqual("run #1: 3 fetches; 0 feeds", description)

  def testTwoFetchesAsTupleNoFeeds(self):
    fetches = (self.const_a, self.const_b)
    run_start_intro = cli_shared.get_run_start_intro(1, fetches, None, {})

    const_a_name_line = run_start_intro.lines[4]
    const_b_name_line = run_start_intro.lines[5]
    self.assertEqual(self.const_a.name, const_a_name_line.strip())
    self.assertEqual(self.const_b.name, const_b_name_line.strip())

    feeds_line = run_start_intro.lines[8]
    self.assertEqual("(Empty)", feeds_line.strip())

    # Verify short description.
    description = cli_shared.get_run_short_description(1, fetches, None)
    self.assertEqual("run #1: 2 fetches; 0 feeds", description)

  def testTwoFetchesAsNamedTupleNoFeeds(self):
    fetches_namedtuple = namedtuple("fetches", "x y")
    fetches = fetches_namedtuple(self.const_b, self.const_c)
    run_start_intro = cli_shared.get_run_start_intro(1, fetches, None, {})

    const_b_name_line = run_start_intro.lines[4]
    const_c_name_line = run_start_intro.lines[5]
    self.assertEqual(self.const_b.name, const_b_name_line.strip())
    self.assertEqual(self.const_c.name, const_c_name_line.strip())

    feeds_line = run_start_intro.lines[8]
    self.assertEqual("(Empty)", feeds_line.strip())

    # Verify short description.
    description = cli_shared.get_run_short_description(1, fetches, None)
    self.assertEqual("run #1: 2 fetches; 0 feeds", description)

  def testWithFeedDict(self):
    feed_dict = {
        self.const_a: 10.0,
        self.const_b: 20.0,
    }

    run_start_intro = cli_shared.get_run_start_intro(1, self.const_c, feed_dict,
                                                     {})

    const_c_name_line = run_start_intro.lines[4]
    self.assertEqual(self.const_c.name, const_c_name_line.strip())

    # Verify lines about the feed dict.
    feed_a_line = run_start_intro.lines[7]
    feed_b_line = run_start_intro.lines[8]
    self.assertEqual(self.const_a.name, feed_a_line.strip())
    self.assertEqual(self.const_b.name, feed_b_line.strip())

    # Verify short description.
    description = cli_shared.get_run_short_description(1, self.const_c,
                                                       feed_dict)
    self.assertEqual("run #1: 1 fetch (c:0); 2 feeds", description)

  def testTensorFilters(self):
    feed_dict = {self.const_a: 10.0}
    tensor_filters = {
        "filter_a": lambda x: True,
        "filter_b": lambda x: False,
    }

    run_start_intro = cli_shared.get_run_start_intro(1, self.const_c, feed_dict,
                                                     tensor_filters)

    # Verify the listed names of the tensor filters.
    filter_names = set()
    filter_names.add(run_start_intro.lines[20].split(" ")[-1])
    filter_names.add(run_start_intro.lines[21].split(" ")[-1])

    self.assertEqual({"filter_a", "filter_b"}, filter_names)

    # Verify short description.
    description = cli_shared.get_run_short_description(1, self.const_c,
                                                       feed_dict)
    self.assertEqual("run #1: 1 fetch (c:0); 1 feed (a:0)", description)

    # Verify the command links for the two filters.
    command_set = set()
    annot = run_start_intro.font_attr_segs[20][0]
    command_set.add(annot[2].content)
    annot = run_start_intro.font_attr_segs[21][0]
    command_set.add(annot[2].content)
    self.assertEqual({"run -f filter_a", "run -f filter_b"}, command_set)

  def testGetRunShortDescriptionWorksForTensorFeedKey(self):
    short_description = cli_shared.get_run_short_description(
        1, self.const_a, {self.const_a: 42.0})
    self.assertEqual("run #1: 1 fetch (a:0); 1 feed (a:0)", short_description)

  def testGetRunShortDescriptionWorksForUnicodeFeedKey(self):
    short_description = cli_shared.get_run_short_description(
        1, self.const_a, {u"foo": 42.0})
    self.assertEqual("run #1: 1 fetch (a:0); 1 feed (foo)", short_description)


class GetErrorIntroTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.var_a = variables.Variable(42.0, name="a")

  def tearDown(self):
    ops.reset_default_graph()

  def testShapeError(self):
    tf_error = errors.OpError(None, self.var_a.initializer, "foo description",
                              None)

    error_intro = cli_shared.get_error_intro(tf_error)

    self.assertEqual("!!! An error occurred during the run !!!",
                     error_intro.lines[1])
    self.assertEqual([(0, len(error_intro.lines[1]), "blink")],
                     error_intro.font_attr_segs[1])

    self.assertEqual(2, error_intro.lines[4].index("ni -a -d -t a/Assign"))
    self.assertEqual(2, error_intro.font_attr_segs[4][0][0])
    self.assertEqual(22, error_intro.font_attr_segs[4][0][1])
    self.assertEqual("ni -a -d -t a/Assign",
                     error_intro.font_attr_segs[4][0][2][0].content)
    self.assertEqual("bold", error_intro.font_attr_segs[4][0][2][1])

    self.assertEqual(2, error_intro.lines[6].index("li -r a/Assign"))
    self.assertEqual(2, error_intro.font_attr_segs[6][0][0])
    self.assertEqual(16, error_intro.font_attr_segs[6][0][1])
    self.assertEqual("li -r a/Assign",
                     error_intro.font_attr_segs[6][0][2][0].content)
    self.assertEqual("bold", error_intro.font_attr_segs[6][0][2][1])

    self.assertEqual(2, error_intro.lines[8].index("lt"))
    self.assertEqual(2, error_intro.font_attr_segs[8][0][0])
    self.assertEqual(4, error_intro.font_attr_segs[8][0][1])
    self.assertEqual("lt", error_intro.font_attr_segs[8][0][2][0].content)
    self.assertEqual("bold", error_intro.font_attr_segs[8][0][2][1])

    self.assertStartsWith(error_intro.lines[11], "Op name:")
    self.assertTrue(error_intro.lines[11].endswith("a/Assign"))

    self.assertStartsWith(error_intro.lines[12], "Error type:")
    self.assertTrue(error_intro.lines[12].endswith(str(type(tf_error))))

    self.assertEqual("Details:", error_intro.lines[14])
    self.assertStartsWith(error_intro.lines[15], "foo description")


if __name__ == "__main__":
  googletest.main()
