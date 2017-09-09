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

"""A `Predictor` constructed from an `learn.python.estimator.Estimator`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.predictor import predictor
from tensorflow.python.estimator import model_fn
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import monitored_session


def _get_signature_def(
    serving_input_receiver, estimator, output_key=None):
  """Construct a `SignatureDef` proto."""
  if output_key is None:
    output_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
  # pylint: disable=protected-access
  estimator_spec = estimator.model_fn(
      serving_input_receiver.features, None, model_fn.ModeKeys.PREDICT,
      estimator.config)
  # pylint: enable=protected-access
  export_outputs = estimator_spec.export_outputs
  export_output = export_outputs.get(output_key)
  if export_output is None:
    raise KeyError('output_key must be one of {}; got {}'.format(
        export_outputs.keys(), output_key))
  return export_output.as_signature_def(serving_input_receiver.receiver_tensors)


class CoreEstimatorPredictor(predictor.Predictor):
  """A `Predictor` constructed from an `learn.python.estimator.Estimator`."""

  def __init__(self,
               estimator,
               serving_input_receiver_fn,
               output_key=None,
               graph=None):
    """Initialize a `CoreEstimatorPredictor`.

    Args:
      estimator: an instance of `learn.python.estimator.Estimator`.
      serving_input_receiver_fn: a function that takes no arguments and returns
        an instance of `ServingInputReceiver` compatible with `estimator`.
      output_key: Optional string specifying the export output to use. If
        `None`, then `DEFAULT_SERVING_SIGNATURE_DEF_KEY` is used.
      graph: Optional. The Tensorflow `graph` in which prediction should be
        done.
    """
    self._graph = graph or ops.Graph()
    with self._graph.as_default():
      serving_input_receiver = serving_input_receiver_fn()
      signature_def = _get_signature_def(
          serving_input_receiver, estimator, output_key)
      checkpoint_path = estimator.model_dir
      self._session = monitored_session.MonitoredSession(
          session_creator=monitored_session.ChiefSessionCreator(
              checkpoint_filename_with_path=checkpoint_path))

    feed_tensor_info = signature_def.inputs
    self._feed_tensors = {k: self._graph.get_tensor_by_name(v.name)
                          for k, v in feed_tensor_info.items()}
    fetch_tensor_info = signature_def.outputs
    self._fetch_tensors = {k: self._graph.get_tensor_by_name(v.name)
                           for k, v in fetch_tensor_info.items()}
