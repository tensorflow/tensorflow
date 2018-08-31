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

"""Tests for model_fn.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.estimator import model_fn
from tensorflow.python.estimator.export import export_output
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.keras import metrics
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import monitored_session
from tensorflow.python.training import session_run_hook


class _FakeHook(session_run_hook.SessionRunHook):
  """Fake implementation of `SessionRunHook`."""


class _InvalidHook(object):
  """Invalid hook (not a subclass of `SessionRunHook`)."""


class _InvalidScaffold(object):
  """Invalid scaffold (not a subclass of `Scaffold`)."""


class EstimatorSpecTrainTest(test.TestCase):
  """Tests EstimatorSpec in train mode."""

  def testRequiredArgumentsSet(self):
    """Tests that no errors are raised when all required arguments are set."""
    with ops.Graph().as_default(), self.cached_session():
      model_fn.EstimatorSpec(
          mode=model_fn.ModeKeys.TRAIN,
          loss=constant_op.constant(1.),
          train_op=control_flow_ops.no_op())

  def testAllArgumentsSet(self):
    """Tests that no errors are raised when all arguments are set."""
    with ops.Graph().as_default(), self.cached_session():
      loss = constant_op.constant(1.)
      predictions = {'loss': loss}
      classes = constant_op.constant('hello')
      metric_obj = metrics.Mean()
      metric_obj.update_state(loss)
      model_fn.EstimatorSpec(
          mode=model_fn.ModeKeys.TRAIN,
          predictions=predictions,
          loss=loss,
          train_op=control_flow_ops.no_op(),
          eval_metric_ops={
              'loss': (control_flow_ops.no_op(), loss),
              'mean': metric_obj,
          },
          export_outputs={
              'head_name': export_output.ClassificationOutput(classes=classes)
          },
          training_chief_hooks=[_FakeHook()],
          training_hooks=[_FakeHook()],
          scaffold=monitored_session.Scaffold(),
          evaluation_hooks=[_FakeHook()],
          prediction_hooks=[_FakeHook()])

  def testLossNumber(self):
    """Tests that error is raised when loss is a number (not Tensor)."""
    with ops.Graph().as_default(), self.cached_session():
      with self.assertRaisesRegexp(TypeError, 'loss must be Tensor'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.TRAIN,
            loss=1.,
            train_op=control_flow_ops.no_op())

  def testLoss1DTensor(self):
    """Tests that no errors are raised when loss is 1D tensor."""
    with ops.Graph().as_default(), self.cached_session():
      model_fn.EstimatorSpec(
          mode=model_fn.ModeKeys.TRAIN,
          loss=constant_op.constant([1.]),
          train_op=control_flow_ops.no_op())

  def testLossMissing(self):
    with ops.Graph().as_default(), self.cached_session():
      with self.assertRaisesRegexp(ValueError, 'Missing loss'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.TRAIN, train_op=control_flow_ops.no_op())

  def testLossNotScalar(self):
    with ops.Graph().as_default(), self.cached_session():
      with self.assertRaisesRegexp(ValueError, 'Loss must be scalar'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.TRAIN,
            loss=constant_op.constant([1., 2.]),
            train_op=control_flow_ops.no_op())

  def testLossSparseTensor(self):
    with ops.Graph().as_default(), self.cached_session():
      loss = sparse_tensor.SparseTensor(
          indices=[[0]],
          values=[0.],
          dense_shape=[1])
      with self.assertRaisesRegexp(TypeError, 'loss must be Tensor'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.TRAIN,
            loss=loss,
            train_op=control_flow_ops.no_op())

  def testLossFromDifferentGraph(self):
    with ops.Graph().as_default():
      loss = constant_op.constant(1.)
    with ops.Graph().as_default(), self.cached_session():
      with self.assertRaisesRegexp(
          ValueError, 'must be from the default graph'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.TRAIN,
            loss=loss,
            train_op=control_flow_ops.no_op())

  def testTrainOpMissing(self):
    with ops.Graph().as_default(), self.cached_session():
      with self.assertRaisesRegexp(ValueError, 'Missing train_op'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.TRAIN, loss=constant_op.constant(1.))

  def testTrainOpNotOperationAndTensor(self):
    with ops.Graph().as_default(), self.cached_session():
      with self.assertRaisesRegexp(TypeError,
                                   'train_op must be Operation or Tensor'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.TRAIN,
            loss=constant_op.constant(1.),
            train_op='Not an Operation or Tensor')

  def testTrainOpFromDifferentGraph(self):
    with ops.Graph().as_default():
      train_op = control_flow_ops.no_op()
    with ops.Graph().as_default(), self.cached_session():
      with self.assertRaisesRegexp(
          ValueError, 'must be from the default graph'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.TRAIN,
            loss=constant_op.constant(1.),
            train_op=train_op)

  def testTrainingChiefHookInvalid(self):
    with ops.Graph().as_default(), self.cached_session():
      with self.assertRaisesRegexp(
          TypeError, 'All hooks must be SessionRunHook instances'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.TRAIN,
            loss=constant_op.constant(1.),
            train_op=control_flow_ops.no_op(),
            training_chief_hooks=[_InvalidHook()])

  def testTrainingHookInvalid(self):
    with ops.Graph().as_default(), self.cached_session():
      with self.assertRaisesRegexp(
          TypeError, 'All hooks must be SessionRunHook instances'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.TRAIN,
            loss=constant_op.constant(1.),
            train_op=control_flow_ops.no_op(),
            training_hooks=[_InvalidHook()])

  def testScaffoldInvalid(self):
    with ops.Graph().as_default(), self.cached_session():
      with self.assertRaisesRegexp(
          TypeError, r'scaffold must be tf\.train\.Scaffold'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.TRAIN,
            loss=constant_op.constant(1.),
            train_op=control_flow_ops.no_op(),
            scaffold=_InvalidScaffold())

  def testReturnDefaultScaffold(self):
    with ops.Graph().as_default(), self.cached_session():
      estimator_spec = model_fn.EstimatorSpec(
          mode=model_fn.ModeKeys.TRAIN,
          loss=constant_op.constant(1.),
          train_op=control_flow_ops.no_op())
      self.assertIsNotNone(estimator_spec.scaffold)


class EstimatorSpecEvalTest(test.TestCase):
  """Tests EstimatorSpec in eval mode."""

  def testRequiredArgumentsSet(self):
    """Tests that no errors are raised when all required arguments are set."""
    with ops.Graph().as_default(), self.cached_session():
      loss = constant_op.constant(1.)
      model_fn.EstimatorSpec(
          mode=model_fn.ModeKeys.EVAL,
          predictions={'loss': loss},
          loss=loss)

  def testAllArgumentsSet(self):
    """Tests that no errors are raised when all arguments are set."""
    with ops.Graph().as_default(), self.cached_session():
      loss = constant_op.constant(1.)
      predictions = {'loss': loss}
      classes = constant_op.constant('hello')
      metric_obj = metrics.Mean()
      metric_obj.update_state(loss)
      model_fn.EstimatorSpec(
          mode=model_fn.ModeKeys.EVAL,
          predictions=predictions,
          loss=loss,
          train_op=control_flow_ops.no_op(),
          eval_metric_ops={
              'loss': (control_flow_ops.no_op(), loss),
              'mean': metric_obj,
          },
          export_outputs={
              'head_name': export_output.ClassificationOutput(classes=classes)
          },
          training_chief_hooks=[_FakeHook()],
          training_hooks=[_FakeHook()],
          scaffold=monitored_session.Scaffold(),
          evaluation_hooks=[_FakeHook()])

  def testEvaluationHookInvalid(self):
    with ops.Graph().as_default(), self.cached_session():
      with self.assertRaisesRegexp(
          TypeError, 'All hooks must be SessionRunHook instances'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.EVAL,
            loss=constant_op.constant(1.),
            evaluation_hooks=[_InvalidHook()])

  def testTupleMetric(self):
    """Tests that no errors are raised when a metric is tuple-valued."""
    with ops.Graph().as_default(), self.cached_session():
      loss = constant_op.constant(1.)
      model_fn.EstimatorSpec(
          mode=model_fn.ModeKeys.EVAL,
          loss=loss,
          eval_metric_ops={
              'some_metric': ((loss, loss, (constant_op.constant(2), loss)),
                              control_flow_ops.no_op())})

  def testLoss1DTensor(self):
    """Tests that no errors are raised when loss is 1D tensor."""
    with ops.Graph().as_default(), self.cached_session():
      loss = constant_op.constant([1.])
      model_fn.EstimatorSpec(
          mode=model_fn.ModeKeys.EVAL,
          predictions={'loss': loss},
          loss=loss)

  def testLossNumber(self):
    """Tests that error is raised when loss is a number (not Tensor)."""
    with ops.Graph().as_default(), self.cached_session():
      with self.assertRaisesRegexp(TypeError, 'loss must be Tensor'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.EVAL,
            predictions={'loss': constant_op.constant(1.)},
            loss=1.)

  def testLossMissing(self):
    with ops.Graph().as_default(), self.cached_session():
      with self.assertRaisesRegexp(ValueError, 'Missing loss'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.EVAL,
            predictions={'loss': constant_op.constant(1.)})

  def testLossNotScalar(self):
    with ops.Graph().as_default(), self.cached_session():
      loss = constant_op.constant([1., 2.])
      with self.assertRaisesRegexp(ValueError, 'Loss must be scalar'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.EVAL,
            predictions={'loss': loss},
            loss=loss)

  def testLossSparseTensor(self):
    with ops.Graph().as_default(), self.cached_session():
      loss = sparse_tensor.SparseTensor(
          indices=[[0]],
          values=[0.],
          dense_shape=[1])
      with self.assertRaisesRegexp(
          TypeError, 'loss must be Tensor'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.EVAL,
            predictions={'prediction': constant_op.constant(1.)},
            loss=loss)

  def testLossFromDifferentGraph(self):
    with ops.Graph().as_default():
      loss = constant_op.constant(1.)
    with ops.Graph().as_default(), self.cached_session():
      with self.assertRaisesRegexp(
          ValueError, 'must be from the default graph'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.EVAL,
            predictions={'prediction': constant_op.constant(1.)},
            loss=loss)

  def testReplaceRaisesConstructorChecks(self):
    with ops.Graph().as_default(), self.cached_session():
      loss = constant_op.constant(1.)
      spec = model_fn.EstimatorSpec(
          mode=model_fn.ModeKeys.EVAL, predictions={'loss': loss}, loss=loss)
      with self.assertRaisesRegexp(ValueError, 'Loss must be scalar'):
        spec._replace(loss=constant_op.constant([1., 2.]))

  def testReplaceDoesReplace(self):
    with ops.Graph().as_default(), self.cached_session():
      loss = constant_op.constant(1.)
      spec = model_fn.EstimatorSpec(
          mode=model_fn.ModeKeys.EVAL, predictions={'loss': loss}, loss=loss)
      new_spec = spec._replace(predictions={'m': loss})
      self.assertEqual(['m'], list(new_spec.predictions.keys()))

  def testReplaceNotAllowModeChange(self):
    with ops.Graph().as_default(), self.cached_session():
      loss = constant_op.constant(1.)
      spec = model_fn.EstimatorSpec(
          mode=model_fn.ModeKeys.EVAL, predictions={'loss': loss}, loss=loss)
      spec._replace(mode=model_fn.ModeKeys.EVAL)
      with self.assertRaisesRegexp(ValueError,
                                   'mode of EstimatorSpec cannot be changed'):
        spec._replace(mode=model_fn.ModeKeys.TRAIN)

  def testPredictionsMissingIsOkay(self):
    with ops.Graph().as_default(), self.cached_session():
      model_fn.EstimatorSpec(
          mode=model_fn.ModeKeys.EVAL, loss=constant_op.constant(1.))

  def testPredictionsTensor(self):
    """Tests that no error is raised when predictions is Tensor (not dict)."""
    with ops.Graph().as_default(), self.cached_session():
      loss = constant_op.constant(1.)
      model_fn.EstimatorSpec(
          mode=model_fn.ModeKeys.EVAL,
          predictions=loss,
          loss=loss)

  def testPredictionsNumber(self):
    with ops.Graph().as_default(), self.cached_session():
      with self.assertRaisesRegexp(
          TypeError, r'predictions\[number\] must be Tensor'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.EVAL,
            predictions={'number': 1.},
            loss=constant_op.constant(1.))

  def testPredictionsSparseTensor(self):
    with ops.Graph().as_default(), self.cached_session():
      predictions = {
          'sparse': sparse_tensor.SparseTensor(
              indices=[[0]],
              values=[0.],
              dense_shape=[1])}
      with self.assertRaisesRegexp(
          TypeError, r'predictions\[sparse\] must be Tensor'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.EVAL,
            predictions=predictions,
            loss=constant_op.constant(1.))

  def testPredictionsFromDifferentGraph(self):
    with ops.Graph().as_default():
      predictions = {'loss': constant_op.constant(1.)}
    with ops.Graph().as_default(), self.cached_session():
      with self.assertRaisesRegexp(
          ValueError, 'must be from the default graph'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.EVAL,
            predictions=predictions,
            loss=constant_op.constant(1.))

  def testEvalMetricOpsNoDict(self):
    with ops.Graph().as_default(), self.cached_session():
      loss = constant_op.constant(1.)
      with self.assertRaisesRegexp(
          TypeError, 'eval_metric_ops must be a dict'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.EVAL,
            predictions={'loss': loss},
            loss=loss,
            eval_metric_ops=loss)

  def testEvalMetricOpsNoTuple(self):
    with ops.Graph().as_default(), self.cached_session():
      loss = constant_op.constant(1.)
      with self.assertRaisesRegexp(
          TypeError,
          (r'Values of eval_metric_ops must be \(metric_value, update_op\) '
           'tuples')):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.EVAL,
            predictions={'loss': loss},
            loss=loss,
            eval_metric_ops={'loss': loss})

  def testEvalMetricOpsNoTensorOrOperation(self):
    with ops.Graph().as_default(), self.cached_session():
      loss = constant_op.constant(1.)
      with self.assertRaisesRegexp(TypeError, 'must be Operation or Tensor'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.EVAL,
            predictions={'loss': loss},
            loss=loss,
            eval_metric_ops={'loss': ('NonTensor', loss)})

  def testEvalMetricNestedNoTensorOrOperation(self):
    with ops.Graph().as_default(), self.cached_session():
      loss = constant_op.constant(1.)
      with self.assertRaisesRegexp(TypeError, 'must be Operation or Tensor'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.EVAL,
            predictions={'loss': loss},
            loss=loss,
            eval_metric_ops={'loss': ((('NonTensor',),),
                                      control_flow_ops.no_op())})

  def testEvalMetricOpsFromDifferentGraphWithMetricTuple(self):
    with ops.Graph().as_default():
      eval_metric_ops = {
          'loss': (control_flow_ops.no_op(), constant_op.constant(1.))}
    with ops.Graph().as_default(), self.cached_session():
      loss = constant_op.constant(1.)
      with self.assertRaisesRegexp(
          ValueError, 'must be from the default graph'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.EVAL,
            predictions={'loss': loss},
            loss=loss,
            eval_metric_ops=eval_metric_ops)

  def testEvalMetricOpsFromDifferentGraphWithMetricObject(self):
    with ops.Graph().as_default():
      metric_obj = metrics.Mean()
      metric_obj.update_state(constant_op.constant(1.))
      eval_metric_ops = {'metric': metric_obj}
    with ops.Graph().as_default(), self.cached_session():
      loss = constant_op.constant(1.)
      with self.assertRaisesRegexp(
          ValueError, 'must be from the default graph'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.EVAL,
            predictions={'loss': loss},
            loss=loss,
            eval_metric_ops=eval_metric_ops)

  def testEvalMetricOpsWithoutUpdates(self):
    with ops.Graph().as_default():
      eval_metric_ops = {'mean': metrics.Mean()}
    with ops.Graph().as_default(), self.cached_session():
      loss = constant_op.constant(1.)
      with self.assertRaisesRegexp(ValueError, 'Please call update_state(...)'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.EVAL,
            predictions={'loss': loss},
            loss=loss,
            eval_metric_ops=eval_metric_ops)


class EstimatorSpecInferTest(test.TestCase):
  """Tests EstimatorSpec in infer mode."""

  def testRequiredArgumentsSet(self):
    """Tests that no errors are raised when all required arguments are set."""
    with ops.Graph().as_default(), self.cached_session():
      model_fn.EstimatorSpec(
          mode=model_fn.ModeKeys.PREDICT,
          predictions={'loss': constant_op.constant(1.)})

  def testAllArgumentsSet(self):
    """Tests that no errors are raised when all arguments are set."""
    with ops.Graph().as_default(), self.cached_session():
      loss = constant_op.constant(1.)
      predictions = {'loss': loss}
      classes = constant_op.constant('hello')
      metric_obj = metrics.Mean()
      metric_obj.update_state(loss)
      model_fn.EstimatorSpec(
          mode=model_fn.ModeKeys.PREDICT,
          predictions=predictions,
          loss=loss,
          train_op=control_flow_ops.no_op(),
          eval_metric_ops={
              'loss': (control_flow_ops.no_op(), loss),
              'mean': metric_obj,
          },
          export_outputs={
              'head_name': export_output.ClassificationOutput(classes=classes)
          },
          training_chief_hooks=[_FakeHook()],
          training_hooks=[_FakeHook()],
          scaffold=monitored_session.Scaffold(),
          evaluation_hooks=[_FakeHook()],
          prediction_hooks=[_FakeHook()])

  def testPredictionHookInvalid(self):
    with ops.Graph().as_default(), self.cached_session():
      with self.assertRaisesRegexp(
          TypeError, 'All hooks must be SessionRunHook instances'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.PREDICT,
            predictions=constant_op.constant(1.),
            prediction_hooks=[_InvalidHook()])

  def testPredictionsMissing(self):
    with ops.Graph().as_default(), self.cached_session():
      with self.assertRaisesRegexp(ValueError, 'Missing predictions'):
        model_fn.EstimatorSpec(mode=model_fn.ModeKeys.PREDICT)

  def testPredictionsTensor(self):
    """Tests that no error is raised when predictions is Tensor (not dict)."""
    with ops.Graph().as_default(), self.cached_session():
      model_fn.EstimatorSpec(
          mode=model_fn.ModeKeys.PREDICT, predictions=constant_op.constant(1.))

  def testPredictionsNumber(self):
    with ops.Graph().as_default(), self.cached_session():
      with self.assertRaisesRegexp(
          TypeError, r'predictions\[number\] must be Tensor'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.PREDICT, predictions={'number': 1.})

  def testPredictionsSparseTensor(self):
    with ops.Graph().as_default(), self.cached_session():
      predictions = {
          'sparse': sparse_tensor.SparseTensor(
              indices=[[0]],
              values=[0.],
              dense_shape=[1])}
      with self.assertRaisesRegexp(
          TypeError, r'predictions\[sparse\] must be Tensor'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.PREDICT, predictions=predictions)

  def testExportOutputsNoDict(self):
    with ops.Graph().as_default(), self.cached_session():
      predictions = {'loss': constant_op.constant(1.)}
      classes = constant_op.constant('hello')
      with self.assertRaisesRegexp(
          TypeError, 'export_outputs must be dict'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs=export_output.ClassificationOutput(classes=classes))

  def testExportOutputsValueNotExportOutput(self):
    with ops.Graph().as_default(), self.cached_session():
      predictions = {'loss': constant_op.constant(1.)}
      with self.assertRaisesRegexp(
          TypeError,
          r"Values in export_outputs must be ExportOutput objects. "
          r"Given: {'head_name': {'loss': <tf.Tensor 'Const:0' shape=\(\) "
          r"dtype=float32>}}"):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={'head_name': predictions})

  def testExportOutputsSingleheadMissingDefault(self):
    with ops.Graph().as_default(), self.cached_session():
      predictions = {'loss': constant_op.constant(1.)}
      output_1 = constant_op.constant([1.])
      regression_output = export_output.RegressionOutput(value=output_1)
      export_outputs = {
          'head-1': regression_output,
          }
      estimator_spec = model_fn.EstimatorSpec(
          mode=model_fn.ModeKeys.PREDICT,
          predictions=predictions,
          export_outputs=export_outputs)
      expected_export_outputs = {
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          regression_output,
          'head-1': regression_output,
      }
      self.assertEqual(expected_export_outputs, estimator_spec.export_outputs)

  def testExportOutputsMultiheadWithDefault(self):
    with ops.Graph().as_default(), self.cached_session():
      predictions = {'loss': constant_op.constant(1.)}
      output_1 = constant_op.constant([1.])
      output_2 = constant_op.constant(['2'])
      output_3 = constant_op.constant(['3'])
      export_outputs = {
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          export_output.RegressionOutput(value=output_1),
          'head-2': export_output.ClassificationOutput(classes=output_2),
          'head-3': export_output.PredictOutput(outputs={
              'some_output_3': output_3
          })}
      estimator_spec = model_fn.EstimatorSpec(
          mode=model_fn.ModeKeys.PREDICT,
          predictions=predictions,
          export_outputs=export_outputs)
      self.assertEqual(export_outputs, estimator_spec.export_outputs)

  def testExportOutputsMultiheadMissingDefault(self):
    with ops.Graph().as_default(), self.cached_session():
      predictions = {'loss': constant_op.constant(1.)}
      output_1 = constant_op.constant([1.])
      output_2 = constant_op.constant(['2'])
      output_3 = constant_op.constant(['3'])
      export_outputs = {
          'head-1': export_output.RegressionOutput(value=output_1),
          'head-2': export_output.ClassificationOutput(classes=output_2),
          'head-3': export_output.PredictOutput(outputs={
              'some_output_3': output_3
          })}
      with self.assertRaisesRegexp(
          ValueError,
          'Multiple export_outputs were provided, but none of them is '
          'specified as the default.  Do this by naming one of them with '
          'signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY.'):
        model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs=export_outputs)

  def testDefaultExportOutputCreated(self):
    """Ensure that a default PredictOutput is created for export."""
    with ops.Graph().as_default(), self.cached_session():
      predictions = constant_op.constant(1.)
      self._assertDefaultExportOutputForPredictions(predictions)

  def testDefaultExportOutputCreatedDict(self):
    """Ensure that a default PredictOutput is created for export for dicts."""
    with ops.Graph().as_default(), self.cached_session():
      predictions = {'loss': constant_op.constant(1.),
                     'score': constant_op.constant(10.)}
      self._assertDefaultExportOutputForPredictions(predictions)

  def _assertDefaultExportOutputForPredictions(self, predictions):
    spec = model_fn.EstimatorSpec(
        mode=model_fn.ModeKeys.PREDICT, predictions=predictions)

    expected = export_output.PredictOutput(predictions).outputs
    serving_output = spec.export_outputs[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    self.assertEqual(serving_output.outputs, expected)

if __name__ == '__main__':
  test.main()
