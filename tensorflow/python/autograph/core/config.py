# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Global configuration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph import utils


PYTHON_LITERALS = {
    'None': None,
    'False': False,
    'True': True,
    'float': float,
}


def _internal_name(name):
  """This function correctly resolves internal and external names."""
  reference_name = utils.__name__

  reference_root = 'tensorflow.'
  # If the TF module is foo.tensorflow, then all other modules
  # are then assumed to be prefixed by 'foo'.

  if reference_name.startswith(reference_root):
    return name

  reference_begin = reference_name.find('.' + reference_root)
  assert reference_begin > 0

  root_prefix = reference_name[:reference_begin]
  return root_prefix + '.' + name


DEFAULT_UNCOMPILED_MODULES = set((
    ('tensorflow',),
    (_internal_name('tensorflow'),),
    # TODO(mdan): Remove once the conversion process is optimized.
    ('tensorflow_probability',),
    (_internal_name('tensorflow_probability'),),
    # TODO(b/130313089): Remove.
    ('numpy',),
    # TODO(mdan): Might need to add "thread" as well?
    ('threading',),
))
