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

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

import os as _os

# pylint: disable=g-bad-import-order
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import

try:
  # Add `estimator` attribute to allow access to estimator APIs via
  # "tf.estimator..."
  from tensorflow.python.estimator.api import estimator  # pylint: disable=g-import-not-at-top

  # Add `estimator` to the __path__ to allow "from tensorflow.estimator..."
  # style imports.
  from tensorflow.python.estimator import api as estimator_api  # pylint: disable=g-import-not-at-top
  __path__ += [_os.path.dirname(estimator_api.__file__)]
  del estimator_api
except (ImportError, AttributeError):
  print('tf.estimator package not installed.')

# API IMPORTS PLACEHOLDER

from tensorflow.python.util.lazy_loader import LazyLoader  # pylint: disable=g-import-not-at-top
contrib = LazyLoader('contrib', globals(), 'tensorflow.contrib')
del LazyLoader
# The templated code that replaces the placeholder above sometimes
# sets the __all__ variable. If it does, we have to be sure to add
# "contrib".
if '__all__' in vars():
  vars()['__all__'].append('contrib')

from tensorflow.python.platform import flags  # pylint: disable=g-import-not-at-top
app.flags = flags  # pylint: disable=undefined-variable

# Make sure directory containing top level submodules is in
# the __path__ so that "from tensorflow.foo import bar" works.
_tf_api_dir = _os.path.dirname(_os.path.dirname(app.__file__))  # pylint: disable=undefined-variable
if _tf_api_dir not in __path__:
  __path__.append(_tf_api_dir)

# These symbols appear because we import the python package which
# in turn imports from tensorflow.core and tensorflow.python. They
# must come from this module. So python adds these symbols for the
# resolution to succeed.
# pylint: disable=undefined-variable
try:
  del python
  del core
except NameError:
  # Don't fail if these modules are not available.
  # For e.g. we are using this file for compat.v1 module as well and
  # 'python', 'core' directories are not under compat/v1.
  pass
# pylint: enable=undefined-variable
