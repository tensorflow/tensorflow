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
from tensorflow.python.util import tf_export


class DocSource(object):
  """Specifies docstring source for a module.

  Only one of docstring or docstring_module_name should be set.
  * If docstring is set, then we will use this docstring when
    for the module.
  * If docstring_module_name is set, then we will copy the docstring
    from docstring source module.
  """

  def __init__(self, docstring=None, docstring_module_name=None):
    self.docstring = docstring
    self.docstring_module_name = docstring_module_name

    if self.docstring is not None and self.docstring_module_name is not None:
      raise ValueError('Only one of `docstring` or `docstring_module_name` can '
                       'be set.')


_TENSORFLOW_DOC_SOURCES = {
    'app':
        DocSource(docstring='Import router for absl.app.'),
    'bitwise':
        DocSource(docstring_module_name='ops.bitwise_ops'),
    'compat':
        DocSource(docstring_module_name='util.compat'),
    'distribute':
        DocSource(docstring_module_name='distribute'),
    'distributions': DocSource(
        docstring='Core module for TensorFlow distribution objects and helpers.'
    ),
    'errors':
        DocSource(docstring_module_name='framework.errors'),
    'experimental.numpy':
        DocSource(docstring_module_name='ops.numpy_ops'),
    'gfile':
        DocSource(docstring='Import router for file_io.'),
    'graph_util':
        DocSource(docstring_module_name='framework.graph_util'),
    'image':
        DocSource(docstring_module_name='ops.image_ops'),
    'linalg':
        DocSource(docstring_module_name='ops.linalg_ops'),
    'logging':
        DocSource(docstring_module_name='ops.logging_ops'),
    'losses': DocSource(
        docstring=(
            'Loss operations for use in neural networks. Note: All the losses'
            ' are added to the `GraphKeys.LOSSES` collection by default.'
        )
    ),
    'manip':
        DocSource(docstring_module_name='ops.manip_ops'),
    'math':
        DocSource(docstring_module_name='ops.math_ops'),
    'metrics':
        DocSource(docstring='Evaluation-related metrics.'),
    'nest':
        DocSource(docstring_module_name='util.nest'),
    'nn':
        DocSource(docstring_module_name='ops.nn_ops'),
    'nn.rnn_cell':
        DocSource(docstring='Module for constructing RNN Cells.'),
    'python_io':
        DocSource(docstring_module_name='lib.io.python_io'),
    'ragged':
        DocSource(docstring_module_name='ops.ragged'),
    'resource_loader':
        DocSource(docstring='Resource management library.'),
    'sets':
        DocSource(docstring_module_name='ops.sets'),
    'signal':
        DocSource(docstring_module_name='ops.signal'),
    'sparse':
        DocSource(docstring_module_name='ops.sparse_ops'),
    'strings':
        DocSource(docstring_module_name='ops.string_ops'),
    'summary': DocSource(
        docstring=(
            'Operations for writing summary data, for use in analysis and'
            ' visualization. See the [Summaries and'
            ' TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard)'
            ' guide.'
        )
    ),
    'sysconfig':
        DocSource(docstring='System configuration library.'),
    'test':
        DocSource(docstring='Testing.'),
    'train': DocSource(
        docstring=(
            'Support for training models. See the'
            ' [Training](https://tensorflow.org/api_guides/python/train) guide.'
        )
    ),
}

_ESTIMATOR_DOC_SOURCES = {
    'estimator': DocSource(
        docstring_module_name='estimator_lib'),
    'estimator.export': DocSource(
        docstring_module_name='export.export_lib'),
    'estimator.inputs': DocSource(
        docstring_module_name='inputs.inputs'),
}

_KERAS_DOC_SOURCES = {
    'keras.optimizers.experimental':
        DocSource(docstring_module_name='optimizers.optimizer_experimental')
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
  if api_name == tf_export.KERAS_API_NAME:
    return _KERAS_DOC_SOURCES
  return {}
