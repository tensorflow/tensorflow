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
"""Tests for cli_config."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import tempfile

from tensorflow.python.debug.cli import cli_config
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import gfile
from tensorflow.python.platform import googletest


class CLIConfigTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._tmp_dir = tempfile.mkdtemp()
    self._tmp_config_path = os.path.join(self._tmp_dir, ".tfdbg_config")
    self.assertFalse(gfile.Exists(self._tmp_config_path))
    super(CLIConfigTest, self).setUp()

  def tearDown(self):
    file_io.delete_recursively(self._tmp_dir)
    super(CLIConfigTest, self).tearDown()

  def testConstructCLIConfigWithoutFile(self):
    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
    self.assertEqual(20, config.get("graph_recursion_depth"))
    self.assertEqual(True, config.get("mouse_mode"))
    with self.assertRaises(KeyError):
      config.get("property_that_should_not_exist")
    self.assertTrue(gfile.Exists(self._tmp_config_path))

  def testCLIConfigForwardCompatibilityTest(self):
    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
    with open(self._tmp_config_path, "rt") as f:
      config_json = json.load(f)
    # Remove a field to simulate forward compatibility test.
    del config_json["graph_recursion_depth"]
    with open(self._tmp_config_path, "wt") as f:
      json.dump(config_json, f)

    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
    self.assertEqual(20, config.get("graph_recursion_depth"))

  def testModifyConfigValue(self):
    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
    config.set("graph_recursion_depth", 9)
    config.set("mouse_mode", False)
    self.assertEqual(9, config.get("graph_recursion_depth"))
    self.assertEqual(False, config.get("mouse_mode"))

  def testModifyConfigValueWithTypeCasting(self):
    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
    config.set("graph_recursion_depth", "18")
    config.set("mouse_mode", "false")
    self.assertEqual(18, config.get("graph_recursion_depth"))
    self.assertEqual(False, config.get("mouse_mode"))

  def testModifyConfigValueWithTypeCastingFailure(self):
    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
    with self.assertRaises(ValueError):
      config.set("mouse_mode", "maybe")

  def testLoadFromModifiedConfigFile(self):
    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
    config.set("graph_recursion_depth", 9)
    config.set("mouse_mode", False)
    config2 = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
    self.assertEqual(9, config2.get("graph_recursion_depth"))
    self.assertEqual(False, config2.get("mouse_mode"))

  def testSummarizeFromConfig(self):
    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
    output = config.summarize()
    self.assertEqual(
        ["Command-line configuration:",
         "",
         "  graph_recursion_depth: %d" % config.get("graph_recursion_depth"),
         "  mouse_mode: %s" % config.get("mouse_mode")], output.lines)

  def testSummarizeFromConfigWithHighlight(self):
    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)
    output = config.summarize(highlight="mouse_mode")
    self.assertEqual(
        ["Command-line configuration:",
         "",
         "  graph_recursion_depth: %d" % config.get("graph_recursion_depth"),
         "  mouse_mode: %s" % config.get("mouse_mode")], output.lines)
    self.assertEqual((2, 12, ["underline", "bold"]),
                     output.font_attr_segs[3][0])
    self.assertEqual((14, 18, "bold"), output.font_attr_segs[3][1])

  def testSetCallback(self):
    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)

    test_value = {"graph_recursion_depth": -1}
    def callback(config):
      test_value["graph_recursion_depth"] = config.get("graph_recursion_depth")
    config.set_callback("graph_recursion_depth", callback)

    config.set("graph_recursion_depth", config.get("graph_recursion_depth") - 1)
    self.assertEqual(test_value["graph_recursion_depth"],
                     config.get("graph_recursion_depth"))

  def testSetCallbackInvalidPropertyName(self):
    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)

    with self.assertRaises(KeyError):
      config.set_callback("nonexistent_property_name", print)

  def testSetCallbackNotCallable(self):
    config = cli_config.CLIConfig(config_file_path=self._tmp_config_path)

    with self.assertRaises(TypeError):
      config.set_callback("graph_recursion_depth", 1)


if __name__ == "__main__":
  googletest.main()
