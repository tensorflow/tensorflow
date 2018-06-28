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
"""Conversion to A-normal form."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.autograph.pyct import transformer


class DummyGensym(object):
  """A dumb gensym that suffixes a stem by sequential numbers from 1000."""

  def __init__(self, entity_info):
    del entity_info
    # A proper implementation needs to account for:
    #   * entity_info.namespace
    #   * all the symbols defined in the AST
    #   * the symbols generated so far
    self._idx = 0

  def new_name(self, stem):
    self._idx += 1
    return stem + '_' + str(1000 + self._idx)


class AnfTransformer(transformer.Base):
  """Performs the actual conversion."""

  # TODO(mdan): Link to a reference.
  # TODO(mdan): Implement.

  def __init__(self, entity_info):
    """Creates a transformer.

    Args:
      entity_info: transformer.EntityInfo
    """
    super(AnfTransformer, self).__init__(entity_info)
    self._gensym = DummyGensym(entity_info)


def transform(node, entity_info):
  return AnfTransformer(entity_info).visit(node)
