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
"""Collections utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


def tf_namedtuple(name, fieldnames_and_docs):
  """A `namedtuple` class factory that supports field-docstrings.

  ```
  cls = tf_namedtuple("MyNamedTuple",[("a", "Docs for a"),
                                      ("b", "Docs for b")])
  cls.a.__doc__  # ==> "Docs for a"
  ```

  Args:
    name: The name of the new class.
    fieldnames_and_docs: A sequence of `(fieldname, docstring)` pairs. The
      fieldnames are passed to `collections.namedtuple`.

  Returns:
    A namedtuple class.
  """
  fieldnames_and_docs = list(fieldnames_and_docs)
  fieldnames = [fieldname for fieldname, doc in fieldnames_and_docs]
  cls = collections.namedtuple(name, fieldnames)

  for fieldname, doc in fieldnames_and_docs:
    old_prop = getattr(cls, fieldname)
    new_prop = property(fget=old_prop.fget, fset=old_prop.fset,
                        fdel=old_prop.fdel, doc=doc)
    setattr(cls, fieldname, new_prop)

  return cls
