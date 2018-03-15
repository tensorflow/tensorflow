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

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.summary import plugin_asset


class _UnnamedPluginAsset(plugin_asset.PluginAsset):
  """An example asset with a dummy serialize method provided, but no name."""

  def assets(self):
    return {}


class _ExamplePluginAsset(_UnnamedPluginAsset):
  """Simple example asset."""
  plugin_name = "_ExamplePluginAsset"


class _OtherExampleAsset(_UnnamedPluginAsset):
  """Simple example asset."""
  plugin_name = "_OtherExampleAsset"


class _ExamplePluginThatWillCauseCollision(_UnnamedPluginAsset):
  plugin_name = "_ExamplePluginAsset"


class PluginAssetTest(test_util.TensorFlowTestCase):

  def testGetPluginAsset(self):
    epa = plugin_asset.get_plugin_asset(_ExamplePluginAsset)
    self.assertIsInstance(epa, _ExamplePluginAsset)
    epa2 = plugin_asset.get_plugin_asset(_ExamplePluginAsset)
    self.assertIs(epa, epa2)
    opa = plugin_asset.get_plugin_asset(_OtherExampleAsset)
    self.assertIsNot(epa, opa)

  def testUnnamedPluginFails(self):
    with self.assertRaises(ValueError):
      plugin_asset.get_plugin_asset(_UnnamedPluginAsset)

  def testPluginCollisionDetected(self):
    plugin_asset.get_plugin_asset(_ExamplePluginAsset)
    with self.assertRaises(ValueError):
      plugin_asset.get_plugin_asset(_ExamplePluginThatWillCauseCollision)

  def testGetAllPluginAssets(self):
    epa = plugin_asset.get_plugin_asset(_ExamplePluginAsset)
    opa = plugin_asset.get_plugin_asset(_OtherExampleAsset)
    self.assertItemsEqual(plugin_asset.get_all_plugin_assets(), [epa, opa])

  def testRespectsGraphArgument(self):
    g1 = ops.Graph()
    g2 = ops.Graph()
    e1 = plugin_asset.get_plugin_asset(_ExamplePluginAsset, g1)
    e2 = plugin_asset.get_plugin_asset(_ExamplePluginAsset, g2)

    self.assertEqual(e1, plugin_asset.get_all_plugin_assets(g1)[0])
    self.assertEqual(e2, plugin_asset.get_all_plugin_assets(g2)[0])

if __name__ == "__main__":
  googletest.main()
