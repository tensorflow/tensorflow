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

TENSORFLOW_DOC_SOURCES = {
    'app': DocSource(docstring_module_name='platform.app'),
    'compat': DocSource(docstring_module_name='util.compat'),
    'distributions': DocSource(
        docstring_module_name='ops.distributions.distributions'),
    'bitwise': DocSource(docstring_module_name='ops.bitwise_ops'),
    'errors': DocSource(docstring_module_name='framework.errors'),
    'gfile': DocSource(docstring_module_name='platform.gfile'),
    'graph_util': DocSource(docstring_module_name='framework.graph_util'),
    'image': DocSource(docstring_module_name='ops.image_ops'),
    'keras.estimator': DocSource(docstring_module_name='estimator.keras'),
    'linalg': DocSource(docstring_module_name='ops.linalg_ops'),
    'logging': DocSource(docstring_module_name='ops.logging_ops'),
    'losses': DocSource(docstring_module_name='ops.losses.losses'),
    'manip': DocSource(docstring_module_name='ops.manip_ops'),
    'math': DocSource(docstring_module_name='ops.math_ops'),
    'metrics': DocSource(docstring_module_name='ops.metrics'),
    'nn': DocSource(docstring_module_name='ops.nn_ops'),
    'nn.rnn_cell': DocSource(docstring_module_name='ops.rnn_cell'),
    'python_io': DocSource(docstring_module_name='lib.io.python_io'),
    'resource_loader': DocSource(
        docstring_module_name='platform.resource_loader'),
    'sets': DocSource(docstring_module_name='ops.sets'),
    'sparse': DocSource(docstring_module_name='ops.sparse_ops'),
    'spectral': DocSource(docstring_module_name='ops.spectral_ops'),
    'strings': DocSource(docstring_module_name='ops.string_ops'),
    'sysconfig': DocSource(docstring_module_name='platform.sysconfig'),
    'test': DocSource(docstring_module_name='platform.test'),
    'train': DocSource(docstring_module_name='training.training'),
    'train.queue_runner': DocSource(
        docstring_module_name='training.queue_runner'),
}
