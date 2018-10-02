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

"""A `Predictor constructed from a `tf.contrib.learn.Estimator`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
from tensorflow.contrib.predictor import predictor
from tensorflow.python.framework import ops
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import monitored_session


class ContribEstimatorPredictor(predictor.Predictor):
  """A `Predictor constructed from a `tf.contrib.learn.Estimator`."""

  def __init__(self,
               estimator,
               prediction_input_fn,
               input_alternative_key=None,
               output_alternative_key=None,
               graph=None,
               config=None):
    """Initialize a `ContribEstimatorPredictor`.

    Args:
      estimator: an instance of `tf.contrib.learn.Estimator`.
      prediction_input_fn: a function that takes no arguments and returns an
        instance of `InputFnOps`.
      input_alternative_key: Optional. Specify the input alternative used for
        prediction.
      output_alternative_key: Specify the output alternative used for
        prediction. Not needed for single-headed models but required for
        multi-headed models.
      graph: Optional. The Tensorflow `graph` in which prediction should be
        done.
      config: `ConfigProto` proto used to configure the session.
    """
    self._graph = graph or ops.Graph()
    with self._graph.as_default():
      input_fn_ops = prediction_input_fn()
      # pylint: disable=protected-access
      model_fn_ops = estimator._get_predict_ops(input_fn_ops.features)
      # pylint: enable=protected-access
      checkpoint_path = checkpoint_management.latest_checkpoint(
          estimator.model_dir)
      self._session = monitored_session.MonitoredSession(
          session_creator=monitored_session.ChiefSessionCreator(
              config=config,
              checkpoint_filename_with_path=checkpoint_path))

    input_alternative_key = (
        input_alternative_key or
        saved_model_export_utils.DEFAULT_INPUT_ALTERNATIVE_KEY)
    input_alternatives, _ = saved_model_export_utils.get_input_alternatives(
        input_fn_ops)
    self._feed_tensors = input_alternatives[input_alternative_key]

    (output_alternatives,
     output_alternative_key) = saved_model_export_utils.get_output_alternatives(
         model_fn_ops, output_alternative_key)
    _, fetch_tensors = output_alternatives[output_alternative_key]
    self._fetch_tensors = fetch_tensors
