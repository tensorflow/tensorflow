"""TODO(wicke): DO NOT SUBMIT without one-line documentation for public_api.

TODO(wicke): DO NOT SUBMIT without a detailed description of public_api.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect


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

  # Modules/classes we do not want to descend into if we hit them. Usually,
  # sytem modules exposed through platforms for compatibility reasons.
  # Each entry maps a module path to a name to ignore in traversal.
  _do_not_descend_map = {
      # TODO(drpng): This can be removed once sealed off
      '': ['platform', 'python', 'pywrap_tensorflow', 'sdca', 'tools', 'user_ops'],

      'core': ['protobuf'],

      # I believe these are google3-only issues that can be ignored if they
      # don't occur in OSS TF (and are not replaced with something worse).
      'flags': ['cpp_flags'],

      # Everything below here is legitimate.
      'app': 'flags',  # It'll stay, but it's not officially part of the API
      'test': ['mock'],  # Imported for compatibility between py2/3.
  }

  def _isprivate(self, name):
    return name.startswith('_')

  def _do_not_descend(self, path, name):
    return ((path in self._do_not_descend_map and
             name in self._do_not_descend_map[path]) or
            name.endswith('_pb2'))

  def __call__(self, path, parent, children):

    if inspect.ismodule(parent) and len(path.split('.')) > 10:
      raise RuntimeError('Modules nested too deep:\n%s\n\nThis is likely a '
                         'problem with an accidental public import.' % path)

    # Remove things that are not visible
    for name, child in list(children):
      if self._isprivate(name):
        children.remove((name, child))

    self._visitor(path, parent, children)

    # Remove things that are visible, but which should not be descended into
    for name, child in list(children):
      if self._do_not_descend(path, name):
        children.remove((name, child))
