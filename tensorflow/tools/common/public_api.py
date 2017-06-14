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
"""Visitor restricting traversal to only the public tensorflow API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from tensorflow.python.util import tf_inspect


class PublicAPIVisitor(object):
  """Visitor to use with `traverse` to visit exactly the public TF API."""

  def __init__(self, visitor):
    """Constructor.

    `visitor` should be a callable suitable as a visitor for `traverse`. It will
    be called only for members of the public TensorFlow API.

    Args:
      visitor: A visitor to call for the public API.
    """
    self._visitor = visitor
    self._root_name = 'tf'

    # Modules/classes we want to suppress entirely.
    self._private_map = {
        # Some implementations have this internal module that we shouldn't
        # expose.
        'tf.flags': ['cpp_flags'],
    }

    # Modules/classes we do not want to descend into if we hit them. Usually,
    # system modules exposed through platforms for compatibility reasons.
    # Each entry maps a module path to a name to ignore in traversal.
    self._do_not_descend_map = {
        'tf': [
            'core',
            'examples',
            'flags',  # Don't add flags
            # TODO(drpng): This can be removed once sealed off.
            'platform',
            # TODO(drpng): This can be removed once sealed.
            'pywrap_tensorflow',
            # TODO(drpng): This can be removed once sealed.
            'user_ops',
            'python',
            'tools',
            'tensorboard',
        ],

        ## Everything below here is legitimate.
        # It'll stay, but it's not officially part of the API.
        'tf.app': ['flags'],
        # Imported for compatibility between py2/3.
        'tf.test': ['mock'],
    }

  @property
  def private_map(self):
    """A map from parents to symbols that should not be included at all.

    This map can be edited, but it should not be edited once traversal has
    begun.

    Returns:
      The map marking symbols to not include.
    """
    return self._private_map

  @property
  def do_not_descend_map(self):
    """A map from parents to symbols that should not be descended into.

    This map can be edited, but it should not be edited once traversal has
    begun.

    Returns:
      The map marking symbols to not explore.
    """
    return self._do_not_descend_map

  def set_root_name(self, root_name):
    """Override the default root name of 'tf'."""
    self._root_name = root_name

  def _is_private(self, path, name):
    """Return whether a name is private."""
    # TODO(wicke): Find out what names to exclude.
    return ((path in self._private_map and
             name in self._private_map[path]) or
            (name.startswith('_') and not re.match('__.*__$', name) or
             name in ['__base__', '__class__']))

  def _do_not_descend(self, path, name):
    """Safely queries if a specific fully qualified name should be excluded."""
    return (path in self._do_not_descend_map and
            name in self._do_not_descend_map[path])

  def __call__(self, path, parent, children):
    """Visitor interface, see `traverse` for details."""

    # Avoid long waits in cases of pretty unambiguous failure.
    if tf_inspect.ismodule(parent) and len(path.split('.')) > 10:
      raise RuntimeError('Modules nested too deep:\n%s.%s\n\nThis is likely a '
                         'problem with an accidental public import.' %
                         (self._root_name, path))

    # Includes self._root_name
    full_path = '.'.join([self._root_name, path]) if path else self._root_name

    # Remove things that are not visible.
    for name, child in list(children):
      if self._is_private(full_path, name):
        children.remove((name, child))

    self._visitor(path, parent, children)

    # Remove things that are visible, but which should not be descended into.
    for name, child in list(children):
      if self._do_not_descend(full_path, name):
        children.remove((name, child))
