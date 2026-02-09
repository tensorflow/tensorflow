# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

"""Tests for ops to trim segments."""

from absl.testing import parameterized

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow_text.python.ops import trimmer_ops


@test_util.run_all_in_graph_and_eager_modes
class WaterfallTrimmerOpsTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      # pyformat: disable
      dict(
          segments=[
              # segment 1
              [[1, 2, 3], [4, 5], [6]],
              # segment 2
              [[10], [20], [30, 40, 50]]
          ],
          expected=[
              # segment 1
              [[True, True, False], [True, False], [True]],
              # Segment 2
              [[False], [False], [True, True, False]]
          ],
          max_seq_length=[[2], [1], [3]],
      ),
      dict(
          segments=[
              # segment 1
              [[1, 2, 3], [4, 5], [6]],
              # segment 2
              [[10], [20], [30, 40, 50]]
          ],
          expected=[
              # segment 1
              [[True, True, False], [True, False], [True]],
              # Segment 2
              [[False], [False], [True, True, False]]
          ],
          max_seq_length=[2, 1, 3],
      ),
      dict(
          segments=[
              # first segment
              [[b"hello"], [b"name", b"is"],
               [b"what", b"time", b"is", b"it", b"?"]],
              # second segment
              [[b"whodis", b"?"], [b"bond", b",", b"james", b"bond"],
               [b"5:30", b"AM"]],
          ],
          max_seq_length=2,
          expected=[
              # first segment
              [[True], [True, True], [True, True, False, False, False]],
              # second segment
              [[True, False], [False, False, False, False], [False, False]],
          ],
      ),
      dict(
          descr="Test when segments are rank 3 RaggedTensors",
          segments=[
              # first segment
              [[[b"hello"], [b"there"]], [[b"name", b"is"]],
               [[b"what", b"time"], [b"is"], [b"it"], [b"?"]]],
              # second segment
              [[[b"whodis"], [b"?"]], [[b"bond"], [b","], [b"james"],
                                       [b"bond"]], [[b"5:30"], [b"AM"]]],
          ],
          max_seq_length=2,
          expected=[[[[True], [True]], [[True, True]],
                     [[True, True], [False], [False], [False]]],
                    [[[False], [False]], [[False], [False], [False], [False]],
                     [[False], [False]]]],
      ),
      dict(
          descr="Test when segments are rank 3 RaggedTensors and axis = 1",
          segments=[
              # first segment
              [[[b"hello"], [b"there"]], [[b"name", b"is"]],
               [[b"what", b"time"], [b"is"], [b"it"], [b"?"]]],
              # second segment
              [[[b"whodis"], [b"?"]], [[b"bond"], [b","], [b"james"],
                                       [b"bond"]], [[b"5:30"], [b"AM"]]],
          ],
          axis=1,
          max_seq_length=2,
          expected=[
              # 1st segment
              [[True, True], [True], [True, True, False, False]],
              # 2nd segment
              [[False, False], [True, False, False, False], [False, False]],
          ],
      ),
      # pyformat: enable
  ])
  def testGenerateMask(self,
                       segments,
                       max_seq_length,
                       expected,
                       axis=-1,
                       descr=None):
    max_seq_length = constant_op.constant(max_seq_length)
    segments = [ragged_factory_ops.constant(i) for i in segments]
    expected = [ragged_factory_ops.constant(i) for i in expected]
    trimmer = trimmer_ops.WaterfallTrimmer(max_seq_length, axis=axis)
    actual = trimmer.generate_mask(segments)
    for expected_mask, actual_mask in zip(expected, actual):
      self.assertAllEqual(actual_mask, expected_mask)

  @parameterized.parameters([
      dict(
          segments=[
              # first segment
              [[b"hello", b"there"], [b"name", b"is"],
               [b"what", b"time", b"is", b"it", b"?"]],
              # second segment
              [[b"whodis", b"?"], [b"bond", b",", b"james", b"bond"],
               [b"5:30", b"AM"]],
          ],
          max_seq_length=[1, 3, 4],
          expected=[
              # Expected first segment has shape [3, (1, 2, 4)]
              [[b"hello"], [b"name", b"is"], [b"what", b"time", b"is", b"it"]],
              # Expected second segment has shape [3, (0, 1, 0)]
              [[], [b"bond"], []],
          ]),
      dict(
          descr="Test max sequence length across the batch",
          segments=[
              # first segment
              [[b"hello", b"there"], [b"name", b"is"],
               [b"what", b"time", b"is", b"it", b"?"]],
              # second segment
              [[b"whodis", b"?"], [b"bond", b",", b"james", b"bond"],
               [b"5:30", b"AM"]],
          ],
          max_seq_length=2,
          expected=[
              # Expected first segment has shape [3, (2, 2, 2)]
              [[b"hello", b"there"], [b"name", b"is"], [b"what", b"time"]],
              # Expected second segment has shape [3, (0, 0, 0)]
              [[], [], []],
          ],
      ),
      dict(
          descr="Test when segments are rank 3 RaggedTensors",
          segments=[
              # first segment
              [[[b"hello"], [b"there"]], [[b"name", b"is"]],
               [[b"what", b"time"], [b"is"], [b"it"], [b"?"]]],
              # second segment
              [[[b"whodis"], [b"?"]], [[b"bond"], [b","], [b"james"],
                                       [b"bond"]], [[b"5:30"], [b"AM"]]],
          ],
          max_seq_length=2,
          expected=[
              # Expected first segment has shape [3, (2, 2, 2)]
              [[[b"hello"], [b"there"]], [[b"name", b"is"]],
               [[b"what", b"time"], [], [], []]],
              # Expected second segment has shape [3, (0, 0, 0)]
              [[[], []], [[], [], [], []], [[], []]]
          ],
      ),
      dict(
          descr="Test when segments are rank 3 RaggedTensors and axis = 1",
          segments=[
              # first segment
              [[[b"hello"], [b"there"]], [[b"name", b"is"]],
               [[b"what", b"time"], [b"is"], [b"it"], [b"?"]]],
              # second segment
              [[[b"whodis"], [b"?"]], [[b"bond"], [b","], [b"james"],
                                       [b"bond"]], [[b"5:30"], [b"AM"]]],
          ],
          axis=1,
          max_seq_length=2,
          expected=[
              [[[b"hello"], [b"there"]], [[b"name", b"is"]],
               [[b"what", b"time"], [b"is"]]],
              [[], [[b"bond"]], []],
          ],
      ),
  ])
  def testPerBatchBudgetTrimmer(self,
                                max_seq_length,
                                segments,
                                expected,
                                axis=-1,
                                descr=None):
    max_seq_length = constant_op.constant(max_seq_length)
    trimmer = trimmer_ops.WaterfallTrimmer(max_seq_length, axis=axis)
    segments = [ragged_factory_ops.constant(seg) for seg in segments]
    expected = [ragged_factory_ops.constant(exp) for exp in expected]
    actual = trimmer.trim(segments)
    for expected_seg, actual_seg in zip(expected, actual):
      self.assertAllEqual(expected_seg, actual_seg)


@test_util.run_all_in_graph_and_eager_modes
class RoundRobinTrimmerOpsTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      # pyformat: disable
      dict(
          descr="Basic test on rank 2 RTs",
          segments=[
              # segment 1
              [[1, 2, 3], [4, 5], [6]],
              # segment 2
              [[10], [20], [30, 40, 50]]
          ],
          expected=[
              # segment 1
              [[True, False, False], [True, False], [True]],
              # Segment 2
              [[True], [True], [True, False, False]]
          ],
          max_seq_length=2,
      ),
      dict(
          descr="Test where no truncation is needed",
          segments=[
              # segment 1
              [[1, 2, 3], [4, 5], [6]],
              # segment 2
              [[10], [20], [30, 40, 50]]
          ],
          expected=[
              # segment 1
              [[True, True, True], [True, True], [True]],
              # Segment 2
              [[True], [True], [True, True, True]]
          ],
          max_seq_length=100,
      ),
      dict(
          descr="Basic test w/ segments of rank 3 on axis=-1",
          segments=[
              # first segment
              # [batch, num_tokens, num_wordpieces]
              [[[b"hello", b"123"], [b"there"]]],
              # second segment
              [[[b"whodis", b"233"], [b"?"]]],
          ],
          max_seq_length=2,
          axis=-1,
          expected=[
              # segment 1
              [[[True, False], [False]]],
              # Segment 2
              [[[True, False], [False]]]
          ],
      ),
      dict(
          descr="Test 4 segments",
          segments=[
              # first segment
              [[b"a", b"b"]],
              # second segment
              [[b"one", b"two"]],
              # third segment
              [[b"un", b"deux", b"trois", b"quatre", b"cinque"]],
              # fourth segment
              [[b"unos", b"dos", b"tres", b"quatro", b"cincos"]],
          ],
          max_seq_length=10,
          expected=[
              # first segment
              [[True, True]],
              # second segment
              [[True, True]],
              # third segment
              [[True, True, True, False, False]],
              # fourth segment
              [[True, True, True, False, False]],
          ],
      ),
      dict(
          descr="Test rank 3 RTs, single batch",
          segments=[
              [[[3897], [4702]]],
              [[[[4248], [2829], [4419]]]],
          ],
          max_seq_length=7,
          expected=[
              [[[True], [True]]],
              [[[[True], [True], [True]]]],
          ],
      ),
      dict(
          descr="Test rank 2; test when one batch has "
                "elements < max_seq_length",
          segments=[[[11, 12, 13],
                     [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]],
                    [[11, 12, 13],
                     [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]]],
          max_seq_length=7,
          axis=-1,
          expected=[
              [[True, True, True],
               [True, True, True, True,
                False, False, False, False, False, False]],
              [[True, True, True],
               [True, True, True, False,
                False, False, False, False, False, False]]
          ],
      ),
      # pyformat: enable
  ])
  def testGenerateMask(self,
                       segments,
                       max_seq_length,
                       expected,
                       axis=-1,
                       descr=None):
    max_seq_length = constant_op.constant(max_seq_length)
    segments = [ragged_factory_ops.constant(i) for i in segments]
    expected = [ragged_factory_ops.constant(i) for i in expected]
    trimmer = trimmer_ops.RoundRobinTrimmer(max_seq_length, axis=axis)
    actual = trimmer.generate_mask(segments)
    for expected_mask, actual_mask in zip(expected, actual):
      self.assertAllEqual(actual_mask, expected_mask)

  @parameterized.parameters([
      # pyformat: disable
      dict(
          descr="Test w/ segments of rank 3 on axis=-1",
          segments=[
              # first segment
              [[[b"hello", b"123"], [b"there"]]],
              # second segment
              [[[b"whodis", b"233"], [b"?"]]],
          ],
          max_seq_length=2,
          axis=-1,
          expected=[
              [[[b"hello"], []]],
              [[[b"whodis"], []]],
          ]),
      dict(
          descr="Test rank 2; test when one batch has "
                "elements < max_seq_length",
          segments=[[[11, 12, 13],
                     [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]],
                    [[11, 12, 13],
                     [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]]],
          max_seq_length=7,
          axis=-1,
          expected=[
              [[11, 12, 13],
               [21, 22, 23, 24]],
              [[11, 12, 13],
               [21, 22, 23]]],
      ),
      dict(
          descr="Test rank 1 max sequence length",
          segments=[[[11, 12, 13],
                     [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]],
                    [[11, 12, 13],
                     [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]]],
          max_seq_length=[7],
          axis=-1,
          expected=[
              [[11, 12, 13],
               [21, 22, 23, 24]],
              [[11, 12, 13],
               [21, 22, 23]]],
      ),
      dict(
          descr="Test wordpiece trimming across 2 segments",
          segments=[
              # first segment
              [[[b"hello", b"123"], [b"there"]]],
              # second segment
              [[[b"whodis", b"233"], [b"?"]]],
          ],
          max_seq_length=3,
          axis=-1,
          expected=[
              [[[b"hello", b"123"], []]],
              [[[b"whodis"], []]],
          ]),
      dict(
          descr="Test whole word trimming across 2 segments",
          segments=[
              # first segment
              [[[b"hello", b"123"], [b"there"]]],
              # second segment
              [[[b"whodis", b"233"], [b"?"]]],
          ],
          max_seq_length=3,
          axis=-2,
          expected=[
              [[[b"hello", b"123"], [b"there"]]],
              [[[b"whodis", b"233"]]],
          ]),
      dict(
          descr="Basic test w/ segments of rank 2",
          segments=[
              # first segment
              [[b"hello", b"there"], [b"name", b"is"],
               [b"what", b"time", b"is", b"it", b"?"]],
              # second segment
              [[b"whodis", b"?"], [b"bond", b",", b"james", b"bond"],
               [b"5:30", b"AM"]],
          ],
          max_seq_length=2,
          expected=[
              # Expected first segment has shape [3, (1, 2, 4)]
              [[b"hello"], [b"name"], [b"what"]],
              # Expected second segment has shape [3, (0, 1, 0)]
              [[b"whodis"], [b"bond"], [b"5:30"]],
          ]),
      dict(
          descr="Basic test w/ segments of rank 3",
          segments=[
              # first segment
              [[[b"hello"], [b"there"]], [[b"name"], [b"is"]],
               [[b"what", b"time"], [b"is"], [b"it", b"?"]]],
              # second segment
              [[[b"whodis"], [b"?"]], [[b"bond"], [b","], [b"james"],
                                       [b"bond"]], [[b"5:30"], [b"AM"]]],
          ],
          max_seq_length=2,
          expected=[
              # Expected first segment has shape [3, (1, 2, 4), 1]
              [[[b"hello"], []], [[b"name"], []], [[b"what"], [], []]],
              # Expected second segment has shape [3, (0, 1, 0)]
              [[[b"whodis"], []], [[b"bond"], [], [], []], [[b"5:30"], []]],
          ]),
      dict(
          descr="Basic test w/ segments of rank 3 on axis=-2",
          segments=[
              # first segment
              [[[b"hello"], [b"there"]], [[b"name"], [b"is"]],
               [[b"what", b"time"], [b"is"], [b"it", b"?"]]],
              # second segment
              [[[b"whodis"], [b"?"]], [[b"bond"], [b","], [b"james"],
                                       [b"bond"]], [[b"5:30"], [b"AM"]]],
          ],
          max_seq_length=2,
          axis=-2,
          expected=[
              # Expected first segment has shape [3, (1, 2, 4), 1]
              [[[b"hello"]], [[b"name"]], [[b"what", b"time"]]],
              # Expected second segment has shape [3, (0, 1, 0)]
              [[[b"whodis"]], [[b"bond"]], [[b"5:30"]]],
          ]),
      dict(
          descr="Test 4 segments",
          segments=[
              # first segment
              [[b"a", b"b"]],
              # second segment
              [[b"one", b"two"]],
              # third segment
              [[b"un", b"deux", b"trois", b"quatre", b"cinque"]],
              # fourth segment
              [[b"unos", b"dos", b"tres", b"quatro", b"cincos"]],
          ],
          max_seq_length=10,
          expected=[
              [[b"a", b"b"]],
              [[b"one", b"two"]],
              [[b"un", b"deux", b"trois"]],
              [[b"unos", b"dos", b"tres"]],
          ],
      ),
      dict(
          descr="Test 4 segments of rank 3 on axis=-1",
          segments=[
              # first segment
              [[[b"a", b"b"], [b"c", b"d"]]],
              # second segment
              [[[b"one", b"two"], [b"three", b"four"]]],
              # third segment
              [[[b"un", b"deux", b"trois", b"quatre", b"cinque"], [b"six"]]],
              # fourth segment
              [[[b"unos", b"dos", b"tres", b"quatro", b"cincos"], [b"seis"]]],
          ],
          max_seq_length=10,
          axis=-1,
          expected=[
              # first segment
              [[[b"a", b"b"], [b"c"]]],
              # second segment
              [[[b"one", b"two"], [b"three"]]],
              # third segment
              [[[b"un", b"deux"], []]],
              # fourth segment
              [[[b"unos", b"dos"], []]],
          ],
      ),
      dict(
          descr="Test a negative max_sequence_length.",
          segments=[
              # first segment
              [[-1]],
          ],
          max_seq_length=-1,
          axis=-1,
          expected=[
              # first segment
              [[]],
          ],
      ),
      # pyformat: enable
  ])
  def testPerBatchBudgetTrimmer(self,
                                max_seq_length,
                                segments,
                                expected,
                                axis=-1,
                                descr=None):
    max_seq_length = constant_op.constant(max_seq_length)
    trimmer = trimmer_ops.RoundRobinTrimmer(max_seq_length, axis=axis)
    segments = [ragged_factory_ops.constant(seg) for seg in segments]
    expected = [ragged_factory_ops.constant(exp) for exp in expected]
    actual = trimmer.trim(segments)
    for expected_seg, actual_seg in zip(expected, actual):
      self.assertAllEqual(expected_seg, actual_seg)

  # These two tests started to segfault after new TF integration,
  # Investigate after brunch cut tf.text
  #
  # def testGenerateMaskTfLite(self):
  #   """Checks TFLite conversion and inference."""
  #
  #   class Model(tf.keras.Model):
  #
  #     def __init__(self, **kwargs):
  #       super().__init__(**kwargs)
  #       self.trimmer_ = tf_text.RoundRobinTrimmer(3)
  #
  #     @tf.function(
  #         input_signature=[
  #             tf.TensorSpec(shape=[None], dtype=tf.int32, name="in1vals"),
  #             tf.TensorSpec(shape=[None], dtype=tf.int64, name="in1splits"),
  #             tf.TensorSpec(shape=[None], dtype=tf.int32, name="in2vals"),
  #             tf.TensorSpec(shape=[None], dtype=tf.int64, name="in2splits"),
  #         ]
  #     )
  #     def call(self, in1vals, in1splits, in2vals, in2splits):
  #       in1 = tf.RaggedTensor.from_row_splits(in1vals, in1splits)
  #       in2 = tf.RaggedTensor.from_row_splits(in2vals, in2splits)
  #       [out1, out2] = self.trimmer_.generate_mask([in1, in2])
  #       return {"output_1": out1.flat_values, "output_2": out2.flat_values}
  #
  #   # Test input data.
  #   in1vals = np.array([1, 2, 3, 4, 5], dtype=np.intc)
  #   in2vals = np.array([10, 20, 30, 40, 50, 60], dtype=np.intc)
  #   in1splits = np.array([0, 3, 5])
  #   in2splits = np.array([0, 3, 6])
  #
  #   # Define a model.
  #   model = Model()
  #   # Do TF inference.
  #   tf_result = model(
  #       tf.constant(in1vals, dtype=tf.int32),
  #       tf.constant(in1splits, dtype=tf.int64),
  #       tf.constant(in2vals, dtype=tf.int32),
  #       tf.constant(in2splits, dtype=tf.int64),
  #   )
  #
  #   # Convert to TFLite.
  #   converter = tf.lite.TFLiteConverter.from_keras_model(model)
  #   converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
  #   converter.allow_custom_ops = True
  #   tflite_model = converter.convert()
  #
  #   # Do TFLite inference.
  #   interp = interpreter.InterpreterWithCustomOps(
  #       model_content=tflite_model,
  #       custom_op_registerers=tf_text.tflite_registrar.SELECT_TFTEXT_OPS,
  #   )
  #   print(interp.get_signature_list())
  #   trimmer = interp.get_signature_runner("serving_default")
  #   tflite_result = trimmer(
  #       in1vals=in1vals,
  #       in1splits=in1splits,
  #       in2vals=in2vals,
  #       in2splits=in2splits,
  #   )
  #
  #   # Assert the results are identical.
  #   self.assertAllEqual(tflite_result["output_1"], tf_result["output_1"])
  #   self.assertAllEqual(tflite_result["output_2"], tf_result["output_2"])

  # def testTrimTfLite(self):
  #   """Checks TFLite conversion and inference."""
  #
  #   class Model(tf.keras.Model):
  #
  #     def __init__(self, **kwargs):
  #       super().__init__(**kwargs)
  #       self.trimmer_ = tf_text.RoundRobinTrimmer(3)
  #
  #     @tf.function(
  #         input_signature=[
  #             tf.TensorSpec(shape=[None], dtype=tf.int32, name="in1vals"),
  #             tf.TensorSpec(shape=[None], dtype=tf.int64, name="in1splits"),
  #             tf.TensorSpec(shape=[None], dtype=tf.int32, name="in2vals"),
  #             tf.TensorSpec(shape=[None], dtype=tf.int64, name="in2splits"),
  #         ]
  #     )
  #     def call(self, in1vals, in1splits, in2vals, in2splits):
  #       in1 = tf.RaggedTensor.from_row_splits(in1vals, in1splits)
  #       in2 = tf.RaggedTensor.from_row_splits(in2vals, in2splits)
  #       [out1, out2] = self.trimmer_.trim([in1, in2])
  #       return {"output_1": out1.flat_values, "output_2": out2.flat_values}
  #
  #   # Test input data.
  #   in1vals = np.array([1, 2, 3, 4, 5], dtype=np.intc)
  #   in2vals = np.array([10, 20, 30, 40, 50, 60], dtype=np.intc)
  #   in1splits = np.array([0, 3, 5])
  #   in2splits = np.array([0, 3, 6])
  #
  #   # Define a model.
  #   model = Model()
  #   # Do TF inference.
  #   tf_result = model(
  #       tf.constant(in1vals, dtype=tf.int32),
  #       tf.constant(in1splits, dtype=tf.int64),
  #       tf.constant(in2vals, dtype=tf.int32),
  #       tf.constant(in2splits, dtype=tf.int64),
  #   )
  #
  #   # Convert to TFLite.
  #   converter = tf.lite.TFLiteConverter.from_keras_model(model)
  #   converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
  #   converter.allow_custom_ops = True
  #   tflite_model = converter.convert()
  #
  #   # Do TFLite inference.
  #   interp = interpreter.InterpreterWithCustomOps(
  #       model_content=tflite_model,
  #       custom_op_registerers=tf_text.tflite_registrar.SELECT_TFTEXT_OPS,
  #   )
  #   print(interp.get_signature_list())
  #   trimmer = interp.get_signature_runner("serving_default")
  #   tflite_result = trimmer(
  #       in1vals=in1vals,
  #       in1splits=in1splits,
  #       in2vals=in2vals,
  #       in2splits=in2splits,
  #   )
  #
  #   # Assert the results are identical.
  #   self.assertAllEqual(tflite_result["output_1"], tf_result["output_1"])
  #   self.assertAllEqual(tflite_result["output_2"], tf_result["output_2"])


@test_util.run_all_in_graph_and_eager_modes
class ShrinkLongestTrimmerTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      # pyformat: disable
      dict(
          descr="Basic test on rank 2 RTs",
          segments=[
              # segment 1
              [
                  [1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9, 0],
              ],
              # segment 2
              [
                  [10],
                  [20, 30, 40],
                  [50, 60, 70],
              ]
          ],
          expected=[
              # segment 1
              [
                  [True, True, True],
                  [True, True, False],
                  [True, True, False, False],
              ],
              # Segment 2
              [
                  [True],
                  [True, True, False],
                  [True, True, False],
              ]
          ],
          max_seq_length=4,
      ),
      dict(
          descr="Test where no truncation is needed",
          segments=[
              # segment 1
              [
                  [1, 2, 3],
                  [4, 5],
                  [6],
              ],
              # segment 2
              [
                  [10],
                  [20],
                  [30, 40, 50],
              ]
          ],
          expected=[
              # segment 1
              [[True, True, True], [True, True], [True]],
              # Segment 2
              [[True], [True], [True, True, True]]
          ],
          max_seq_length=100,
      ),
      dict(
          descr="Test 4 segments",
          segments=[
              # first segment
              [[b"a", b"b"]],
              # second segment
              [[b"one", b"two"]],
              # third segment
              [[b"un", b"deux", b"trois", b"quatre", b"cinque"]],
              # fourth segment
              [[b"unos", b"dos", b"tres", b"quatro", b"cincos"]],
          ],
          max_seq_length=10,
          expected=[
              # first segment
              [[True, True]],
              # second segment
              [[True, True]],
              # third segment
              [[True, True, True, False, False]],
              # fourth segment
              [[True, True, True, False, False]],
          ],
      ),
      dict(
          descr="Test rank 2; test when one batch has "
                "elements < max_seq_length",
          segments=[[[11, 12, 13],
                     [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]],
                    [[11, 12, 13],
                     [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]]],
          max_seq_length=7,
          axis=-1,
          expected=[
              [[True, True, True],
               [True, True, True, False,
                False, False, False, False, False, False]],
              [[True, True, True],
               [True, True, True, True,
                False, False, False, False, False, False]]
          ],
      ),
      # pyformat: enable
  ])
  def testGenerateMask(self,
                       segments,
                       max_seq_length,
                       expected,
                       axis=-1,
                       descr=None):
    max_seq_length = constant_op.constant(max_seq_length)
    segments = [ragged_factory_ops.constant(i) for i in segments]
    expected = [ragged_factory_ops.constant(i) for i in expected]
    trimmer = trimmer_ops.ShrinkLongestTrimmer(max_seq_length, axis=axis)
    actual = trimmer.generate_mask(segments)
    for expected_mask, actual_mask in zip(expected, actual):
      self.assertAllEqual(actual_mask, expected_mask)

  @parameterized.parameters([
      # pyformat: disable
      dict(
          descr="Test rank 2; test when one batch has "
                "elements < max_seq_length",
          segments=[
              # first segment
              [
                  [11, 12, 13],
                  [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
              ],
              # second segment
              [
                  [11, 12, 13],
                  [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
              ],
          ],
          max_seq_length=7,
          axis=-1,
          expected=[
              # first segment
              [
                  [11, 12, 13],
                  [21, 22, 23],
              ],
              # second segment
              [
                  [11, 12, 13],
                  [21, 22, 23, 24],
              ],
          ]
      ),
      dict(
          descr="Basic test w/ segments of rank 2",
          segments=[
              # first segment
              [
                  [b"hello", b"there"],
                  [b"name", b"is"],
                  [b"what", b"time", b"is", b"it", b"?"],
              ],
              # second segment
              [
                  [b"whodis", b"?"],
                  [b"bond", b",", b"james", b"bond"],
                  [b"5:30", b"AM"],
              ],
          ],
          max_seq_length=2,
          expected=[
              # first segment
              [
                  [b"hello"],
                  [b"name"],
                  [b"what"],
              ],
              # second segment
              [
                  [b"whodis"],
                  [b"bond"],
                  [b"5:30"],
              ],
          ]),
      dict(
          descr="Test 4 segments",
          segments=[
              # first segment
              [[b"a", b"b"]],
              # second segment
              [[b"one", b"two"]],
              # third segment
              [[b"un", b"deux", b"trois", b"quatre", b"cinque"]],
              # fourth segment
              [[b"unos", b"dos", b"tres", b"quatro", b"cincos"]],
          ],
          max_seq_length=10,
          expected=[
              [[b"a", b"b"]],
              [[b"one", b"two"]],
              [[b"un", b"deux", b"trois"]],
              [[b"unos", b"dos", b"tres"]],
          ],
      ),
      # pyformat: enable
  ])
  def testPerBatchBudgetTrimmer(self,
                                max_seq_length,
                                segments,
                                expected,
                                axis=-1,
                                descr=None):
    max_seq_length = constant_op.constant(max_seq_length)
    trimmer = trimmer_ops.ShrinkLongestTrimmer(max_seq_length, axis=axis)
    segments = [ragged_factory_ops.constant(seg) for seg in segments]
    expected = [ragged_factory_ops.constant(exp) for exp in expected]
    actual = trimmer.trim(segments)
    for expected_seg, actual_seg in zip(expected, actual):
      self.assertAllEqual(expected_seg, actual_seg)

if __name__ == "__main__":
  test.main()
