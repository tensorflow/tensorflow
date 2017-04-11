# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

# Bring in all of the public TensorFlow interface into this
# module.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=wildcard-import
from tensorflow.python import *
# pylint: enable=wildcard-import

# Lazily import the `tf.contrib` module. This avoids loading all of the
# dependencies of `tf.contrib` at `import tensorflow` time.
class _LazyContribLoader(object):

  def __getattr__(self, item):
    global contrib
    # Replace the lazy loader with the imported module itself.
    import importlib  # pylint: disable=g-import-not-at-top
    contrib = importlib.import_module('tensorflow.contrib')
    return getattr(contrib, item)


contrib = _LazyContribLoader()

del absolute_import
del division
del print_function

# These symbols appear because we import the python package which
# in turn imports from tensorflow.core and tensorflow.python. They
# must come from this module. So python adds these symbols for the
# resolution to succeed.
# pylint: disable=undefined-variable
del python
del core
# pylint: enable=undefined-variable
