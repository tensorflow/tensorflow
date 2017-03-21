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
"""Load plugin assets from disk."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import gfile

_PLUGINS_DIR = "plugins"


def _IsDirectory(parent, item):
  """Helper that returns if parent/item is a directory."""
  return gfile.IsDirectory(os.path.join(parent, item))


def ListPlugins(logdir):
  """List all the plugins that have registered assets in logdir.

  If the plugins_dir does not exist, it returns an empty list. This maintains
  compatibility with old directories that have no plugins written.

  Args:
    logdir: A directory that was created by a TensorFlow events writer.

  Returns:
    a list of plugin names, as strings
  """
  plugins_dir = os.path.join(logdir, _PLUGINS_DIR)
  if not gfile.IsDirectory(plugins_dir):
    return []
  entries = gfile.ListDirectory(plugins_dir)
  return [x for x in entries if _IsDirectory(plugins_dir, x)]


def ListAssets(logdir, plugin_name):
  """List all the assets that are available for given plugin in a logdir.

  Args:
    logdir: A directory that was created by a TensorFlow summary.FileWriter.
    plugin_name: A string name of a plugin to list assets for.

  Returns:
    A string list of available plugin assets. If the plugin subdirectory does
    not exist (either because the logdir doesn't exist, or because the plugin
    didn't register) an empty list is returned.
  """
  plugin_dir = os.path.join(logdir, _PLUGINS_DIR, plugin_name)
  if not gfile.IsDirectory(plugin_dir):
    return []
  entries = gfile.ListDirectory(plugin_dir)
  return [x for x in entries if not _IsDirectory(plugin_dir, x)]


def RetrieveAsset(logdir, plugin_name, asset_name):
  """Retrieve a particular plugin asset from a logdir.

  Args:
    logdir: A directory that was created by a TensorFlow summary.FileWriter.
    plugin_name: The plugin we want an asset from.
    asset_name: The name of the requested asset.

  Returns:
    string contents of the plugin asset.

  Raises:
    KeyError: if the asset does not exist.
  """

  asset_path = os.path.join(logdir, _PLUGINS_DIR, plugin_name, asset_name)
  try:
    with gfile.Open(asset_path, "r") as f:
      return f.read()
  except errors_impl.NotFoundError:
    raise KeyError("Asset path %s not found" % asset_path)
  except errors_impl.OpError as e:
    raise KeyError("Couldn't read asset path: %s, OpError %s" % (asset_path, e))
