# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Helper functions to add documentation to type aliases."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Not useful for builtin `help()`. But these get passed to the
# doc generator so that the description is still displayed on the site.
_EXTRA_DOCS: {int: str} = {}


def document(obj, doc):
  """Adds a docstring to typealias by overriding the `__doc__` attribute.

  Note: Overriding `__doc__` is only possible after python 3.7.

  Args:
    obj: Typealias object that needs to be documented.
    doc: Docstring of the typealias. It should follow the standard pystyle
      docstring rules.
  """
  try:
    obj.__doc__ = doc
  except AttributeError:
    _EXTRA_DOCS[id(obj)] = doc
