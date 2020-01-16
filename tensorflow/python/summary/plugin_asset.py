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
"""TensorBoard Plugin asset abstract class.

TensorBoard plugins may need to provide arbitrary assets, such as
configuration information for specific outputs, or vocabulary files, or sprite
images, etc.

This module contains methods that allow plugin assets to be specified at graph
construction time. Plugin authors define a PluginAsset which is treated as a
singleton on a per-graph basis. The PluginAsset has an assets method which
returns a dictionary of asset contents. The tf.compat.v1.summary.FileWriter
(or any other Summary writer) will serialize these assets in such a way that
TensorBoard can retrieve them.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow.python.framework import ops

_PLUGIN_ASSET_PREFIX = "__tensorboard_plugin_asset__"


def get_plugin_asset(plugin_asset_cls, graph=None):
  """Acquire singleton PluginAsset instance from a graph.

  PluginAssets are always singletons, and are stored in tf Graph collections.
  This way, they can be defined anywhere the graph is being constructed, and
  if the same plugin is configured at many different points, the user can always
  modify the same instance.

  Args:
    plugin_asset_cls: The PluginAsset class
    graph: (optional) The graph to retrieve the instance from. If not specified,
      the default graph is used.

  Returns:
    An instance of the plugin_asset_class

  Raises:
    ValueError: If we have a plugin name collision, or if we unexpectedly find
      the wrong number of items in a collection.
  """
  if graph is None:
    graph = ops.get_default_graph()
  if not plugin_asset_cls.plugin_name:
    raise ValueError("Class %s has no plugin_name" % plugin_asset_cls.__name__)

  name = _PLUGIN_ASSET_PREFIX + plugin_asset_cls.plugin_name
  container = graph.get_collection(name)
  if container:
    if len(container) != 1:
      raise ValueError("Collection for %s had %d items, expected 1" %
                       (name, len(container)))
    instance = container[0]
    if not isinstance(instance, plugin_asset_cls):
      raise ValueError("Plugin name collision between classes %s and %s" %
                       (plugin_asset_cls.__name__, instance.__class__.__name__))
  else:
    instance = plugin_asset_cls()
    graph.add_to_collection(name, instance)
    graph.add_to_collection(_PLUGIN_ASSET_PREFIX, plugin_asset_cls.plugin_name)
  return instance


def get_all_plugin_assets(graph=None):
  """Retrieve all PluginAssets stored in the graph collection.

  Args:
    graph: Optionally, the graph to get assets from. If unspecified, the default
      graph is used.

  Returns:
    A list with all PluginAsset instances in the graph.

  Raises:
    ValueError: if we unexpectedly find a collection with the wrong number of
      PluginAssets.

  """
  if graph is None:
    graph = ops.get_default_graph()

  out = []
  for name in graph.get_collection(_PLUGIN_ASSET_PREFIX):
    collection = graph.get_collection(_PLUGIN_ASSET_PREFIX + name)
    if len(collection) != 1:
      raise ValueError("Collection for %s had %d items, expected 1" %
                       (name, len(collection)))
    out.append(collection[0])
  return out


@six.add_metaclass(abc.ABCMeta)
class PluginAsset(object):
  """This abstract base class allows TensorBoard to serialize assets to disk.

  Plugin authors are expected to extend the PluginAsset class, so that it:
  - has a unique plugin_name
  - provides an assets method that returns an {asset_name: asset_contents}
    dictionary. For now, asset_contents are strings, although we may add
    StringIO support later.

  LifeCycle of a PluginAsset instance:
  - It is constructed when get_plugin_asset is called on the class for
    the first time.
  - It is configured by code that follows the calls to get_plugin_asset
  - When the containing graph is serialized by the
    tf.compat.v1.summary.FileWriter, the writer calls assets and the
    PluginAsset instance provides its contents to be written to disk.
  """

  plugin_name = None

  @abc.abstractmethod
  def assets(self):
    """Provide all of the assets contained by the PluginAsset instance.

    The assets method should return a dictionary structured as
    {asset_name: asset_contents}. asset_contents is a string.

    This method will be called by the tf.compat.v1.summary.FileWriter when it
    is time to write the assets out to disk.
    """
    raise NotImplementedError()
