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
"""Specifies sources of doc strings for API modules."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.util import tf_export


# Specifies docstring source for a module.
# Only one of docstring or docstring_module_name should be set.
# * If docstring is set, then we will use this docstring when
#   for the module.
# * If docstring_module_name is set, then we will copy the docstring
#   from docstring source module.
DocSource = collections.namedtuple(
    'DocSource', ['docstring', 'docstring_module_name'])
# Each attribute of DocSource is optional.
DocSource.__new__.__defaults__ = (None,) * len(DocSource._fields)

_TENSORFLOW_DOC_SOURCES = {
    'app': DocSource(docstring_module_name='platform.app'),
    'bitwise': DocSource(docstring_module_name='ops.bitwise_ops'),
    'compat': DocSource(docstring_module_name='util.compat'),
    'distribute': DocSource(docstring_module_name='distribute.distribute_lib'),
    'distributions': DocSource(
        docstring_module_name='ops.distributions.distributions'),
    'errors': DocSource(docstring_module_name='framework.errors'),
    'gfile': DocSource(docstring_module_name='platform.gfile'),
    'graph_util': DocSource(docstring_module_name='framework.graph_util'),
    'image': DocSource(docstring_module_name='ops.image_ops'),
    'keras.estimator': DocSource(docstring_module_name='keras.estimator'),
    'linalg': DocSource(docstring_module_name='ops.linalg_ops'),
    'logging': DocSource(docstring_module_name='ops.logging_ops'),
    'losses': DocSource(docstring_module_name='ops.losses.losses'),
    'manip': DocSource(docstring_module_name='ops.manip_ops'),
    'math': DocSource(docstring_module_name='ops.math_ops'),
    'metrics': DocSource(docstring_module_name='ops.metrics'),
    'nn': DocSource(docstring_module_name='ops.nn_ops'),
    'nn.rnn_cell': DocSource(docstring_module_name='ops.rnn_cell'),
    'python_io': DocSource(docstring_module_name='lib.io.python_io'),
    'ragged': DocSource(docstring_module_name='ops.ragged'),
    'resource_loader': DocSource(
        docstring_module_name='platform.resource_loader'),
    'sets': DocSource(docstring_module_name='ops.sets'),
    'signal': DocSource(docstring_module_name='ops.signal.signal'),
    'sparse': DocSource(docstring_module_name='ops.sparse_ops'),
    'strings': DocSource(docstring_module_name='ops.string_ops'),
    'summary': DocSource(docstring_module_name='summary.summary'),
    'sysconfig': DocSource(docstring_module_name='platform.sysconfig'),
    'test': DocSource(docstring_module_name='platform.test'),
    'train': DocSource(docstring_module_name='training.training'),
}

_ESTIMATOR_DOC_SOURCES = {
    'estimator': DocSource(
        docstring_module_name='estimator_lib'),
    'estimator.export': DocSource(
        docstring_module_name='export.export_lib'),
    'estimator.inputs': DocSource(
        docstring_module_name='inputs.inputs'),
}


def get_doc_sources(api_name):
  """Get a map from module to a DocSource object.

  Args:
    api_name: API you want to generate (e.g. `tensorflow` or `estimator`).

  Returns:
    Map from module name to DocSource object.
  """
  if api_name == tf_export.TENSORFLOW_API_NAME:
    return _TENSORFLOW_DOC_SOURCES
  if api_name == tf_export.ESTIMATOR_API_NAME:
    return _ESTIMATOR_DOC_SOURCES
  return {}
