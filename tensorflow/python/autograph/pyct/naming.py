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
"""Symbol naming utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.pyct import qual_names


class Namer(object):
  """Symbol name generator."""

  def __init__(self, global_namespace):
    self.global_namespace = global_namespace
    self.generated_names = set()

  def new_symbol(self, name_root, reserved_locals):
    """See control_flow.SymbolNamer.new_symbol."""
    # reserved_locals may contain QNs.
    all_reserved_locals = set()
    for s in reserved_locals:
      if isinstance(s, qual_names.QN):
        all_reserved_locals.update(s.qn)
      elif isinstance(s, str):
        all_reserved_locals.add(s)
      else:
        raise ValueError('Unexpected symbol type "%s"' % type(s))

    pieces = name_root.split('_')
    if pieces[-1].isdigit():
      name_root = '_'.join(pieces[:-1])
      n = int(pieces[-1])
    else:
      n = 0
    new_name = name_root

    while (new_name in self.global_namespace or
           new_name in all_reserved_locals or new_name in self.generated_names):
      n += 1
      new_name = '%s_%d' % (name_root, n)

    self.generated_names.add(new_name)
    return new_name
