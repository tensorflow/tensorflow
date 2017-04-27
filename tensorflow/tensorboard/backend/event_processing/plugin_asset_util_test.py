# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

import os.path

from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.summary import plugin_asset
from tensorflow.python.summary.writer import writer
from tensorflow.tensorboard.backend.event_processing import event_multiplexer
from tensorflow.tensorboard.backend.event_processing import plugin_asset_util


class GenericContentPlugin(plugin_asset.PluginAsset):

  def __init__(self):
    self.contents = "hello world"

  def assets(self):
    return {"contents.txt": self.contents}


class PluginAlpha(GenericContentPlugin):
  plugin_name = "Alpha"


class PluginBeta(GenericContentPlugin):
  plugin_name = "Beta"


class PluginGamma(GenericContentPlugin):
  plugin_name = "Gamma"


class PluginAssetUtilitiesTest(test.TestCase):

  def testNonExistentDirectory(self):
    tempdir = self.get_temp_dir()
    fake_dir = os.path.join(tempdir, "nonexistent_dir")
    self.assertEqual([], plugin_asset_util.ListPlugins(fake_dir))
    self.assertEqual([], plugin_asset_util.ListAssets(fake_dir, "fake_plugin"))
    with self.assertRaises(KeyError):
      plugin_asset_util.RetrieveAsset(fake_dir, "fake_plugin", "fake_asset")

  def testSimplePluginCase(self):
    tempdir = self.get_temp_dir()
    with ops.Graph().as_default() as g:
      plugin_asset.get_plugin_asset(PluginAlpha)
      fw = writer.FileWriter(tempdir)
      fw.add_graph(g)
    self.assertEqual(["Alpha"], plugin_asset_util.ListPlugins(tempdir))
    assets = plugin_asset_util.ListAssets(tempdir, "Alpha")
    self.assertEqual(["contents.txt"], assets)
    contents = plugin_asset_util.RetrieveAsset(tempdir, "Alpha", "contents.txt")
    self.assertEqual("hello world", contents)

  def testEventMultiplexerIntegration(self):
    tempdir = self.get_temp_dir()
    with ops.Graph().as_default() as g:
      plugin_instance = plugin_asset.get_plugin_asset(PluginAlpha)
      plugin_instance.contents = "graph one"
      plugin_asset.get_plugin_asset(PluginBeta)

      fw = writer.FileWriter(os.path.join(tempdir, "one"))
      fw.add_graph(g)
      fw.close()

    with ops.Graph().as_default() as g:
      plugin_instance = plugin_asset.get_plugin_asset(PluginAlpha)
      plugin_instance.contents = "graph two"
      fw = writer.FileWriter(os.path.join(tempdir, "two"))
      fw.add_graph(g)
      fw.close()

    multiplexer = event_multiplexer.EventMultiplexer()
    multiplexer.AddRunsFromDirectory(tempdir)

    self.assertEqual(
        multiplexer.PluginAssets("Alpha"),
        {"one": ["contents.txt"], "two": ["contents.txt"]})
    self.assertEqual(
        multiplexer.RetrievePluginAsset("one", "Alpha", "contents.txt"),
        "graph one")
    self.assertEqual(
        multiplexer.RetrievePluginAsset("one", "Beta", "contents.txt"),
        "hello world")
    self.assertEqual(
        multiplexer.RetrievePluginAsset("two", "Alpha", "contents.txt"),
        "graph two")

    self.assertEqual(
        multiplexer.PluginAssets("Beta"),
        {"one": ["contents.txt"], "two": []})
    self.assertEqual(multiplexer.PluginAssets("Gamma"), {"one": [], "two": []})


if __name__ == "__main__":
  test.main()
