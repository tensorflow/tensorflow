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

"""Tools to work with checkpoints."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework import deprecated
from tensorflow.contrib.framework.python.framework import checkpoint_utils


@deprecated('2016-08-22', 'Please use tf.contrib.framework.load_checkpoint '
            'instead')
def load_checkpoint(filepattern):
  """See `tf.contrib.framework.load_checkpoint`."""
  return checkpoint_utils.load_checkpoint(filepattern)


@deprecated('2016-08-22', 'Please use tf.contrib.framework.load_variable '
            'instead')
def load_variable(checkpoint_dir, name):
  """See `tf.contrib.framework.load_variable`."""
  return checkpoint_utils.load_variable(checkpoint_dir, name)


@deprecated('2016-08-22', 'Please use tf.contrib.framework.list_variables '
            'instead')
def list_variables(checkpoint_dir):
  """See `tf.contrib.framework.list_variables`."""
  return checkpoint_utils.list_variables(checkpoint_dir)


@deprecated('2016-08-22', 'Please use tf.contrib.framework.init_from_checkpoint'
            ' instead')
def init_from_checkpoint(checkpoint_dir, assignment_map):
  """See `tf.contrib.framework.init_from_checkpoint`."""
  checkpoint_utils.init_from_checkpoint(checkpoint_dir, assignment_map)
