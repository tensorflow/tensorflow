# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Helper functions for modules."""
import os

import importlib


def get_parent_dir(module):
  return os.path.abspath(os.path.join(os.path.dirname(module.__file__), ".."))


def get_parent_dir_for_name(module_name):
  """Get parent directory for module with the given name.

  Args:
    module_name: Module name for e.g.
      tensorflow_estimator.python.estimator.api._v1.estimator.

  Returns:
    Path to the parent directory if module is found and None otherwise.
    Given example above, it should return:
      /pathtoestimator/tensorflow_estimator/python/estimator/api/_v1.
  """
  name_split = module_name.split(".")
  if not name_split:
    return None

  try:
    spec = importlib.util.find_spec(name_split[0])
  except ValueError:
    return None
  if not spec or not spec.origin:
    return None
  base_path = os.path.dirname(spec.origin)
  return os.path.join(base_path, *name_split[1:-1])
