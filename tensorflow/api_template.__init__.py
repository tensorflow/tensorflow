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
"""Bring in all of the public TensorFlow interface into this module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-bad-import-order
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import

try:
  import os  # pylint: disable=g-import-not-at-top
  # Add `estimator` attribute to allow access to estimator APIs via
  # "tf.estimator..."
  from tensorflow.python.estimator.api import estimator  # pylint: disable=g-import-not-at-top

  # Add `estimator` to the __path__ to allow "from tensorflow.estimator..."
  # style imports.
  from tensorflow.python.estimator import api as estimator_api  # pylint: disable=g-import-not-at-top
  __path__ += [os.path.dirname(estimator_api.__file__)]
  del estimator_api
  del os
except (ImportError, AttributeError):
  print('tf.estimator package not installed.')

# API IMPORTS PLACEHOLDER

from tensorflow.python.util.lazy_loader import LazyLoader  # pylint: disable=g-import-not-at-top
contrib = LazyLoader('contrib', globals(), 'tensorflow.contrib')
del LazyLoader

from tensorflow.python.platform import flags  # pylint: disable=g-import-not-at-top
app.flags = flags  # pylint: disable=undefined-variable

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
