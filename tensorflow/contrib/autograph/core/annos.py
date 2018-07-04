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
"""Annotations specific to AutoGraph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum


class NoValue(Enum):

  def __repr__(self):
    return self.name


class NodeAnno(NoValue):
  """Additional annotations used by AutoGraph converters.

  These are in addition to the basic annotations declared in pyct/anno.py and
  pyct/static_analysis/annos.py.
  """

  # The directives collection - see directives.py
  DIRECTIVES = (
      'Dict depicting static directive calls. See the directives converter.')
