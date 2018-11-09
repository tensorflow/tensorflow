# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Helper functions to help integrate TensorFlow components into TF API.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os


def package_hook(parent_package_str, child_package_str, error_msg=None):
  """Used to hook in an external package into the TensorFlow namespace.

  Example usage:
  ### tensorflow/__init__.py
  from tensorflow.python.tools import component_api_helper
  component_api_helper.package_hook(
      'tensorflow', 'tensorflow_estimator.python')
  component_api_helper(
      'tensorflow.contrib', 'tensorflow_estimator.contrib.python')
  del component_api_helper

  TODO(mikecase): This function has a minor issue, where if the child package
  does not exist alone in its directory, sibling packages to it will also be
  accessible from the parent. This is because we just add
  `child_pkg.__file__/..` to the subpackage search path. This should not be
  a big issue because of how our API generation scripts work (the child package
  we are hooking up should always be alone). But there might be a better way
  of doing this.

  Args:
    parent_package_str: Parent package name as a string such as 'tensorflow' or
      'tensorflow.contrib'. This will become the parent package for the
      component package being hooked in.
    child_package_str: Child package name as a string such as
      'tensorflow_estimator.python'. This package will be added as a subpackage
      of the parent.
    error_msg: Message to print if child package cannot be found.
  """
  parent_pkg = importlib.import_module(parent_package_str)
  try:
    child_pkg = importlib.import_module(child_package_str)
  except ImportError:
    if error_msg:
      print(error_msg)
    return

  def set_child_as_subpackage():
    """Sets child package as a subpackage of parent package.

    Will allow the following import statement to work.
    >>> import parent.child
    """
    child_pkg_path = [os.path.abspath(
        os.path.join(os.path.dirname(child_pkg.__file__), ".."))]
    try:
      parent_pkg.__path__ = child_pkg_path + parent_pkg.__path__
    except AttributeError:
      parent_pkg.__path__ = child_pkg_path

  def set_child_as_attr():
    """Sets child package as a attr of the parent package.

    Will allow for the following.
    >>> import parent
    >>> parent.child
    """
    child_pkg_attr_name = child_pkg.__name__.split(".")[-1]
    setattr(parent_pkg, child_pkg_attr_name, child_pkg)

  set_child_as_subpackage()
  set_child_as_attr()
