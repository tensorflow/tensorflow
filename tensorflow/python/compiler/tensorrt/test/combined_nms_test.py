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
"""Script to test TF-TensorRT conversion of CombinedNMS op."""

import os

from tensorflow.python.compiler.tensorrt import utils as trt_utils
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops_impl
from tensorflow.python.platform import test

import numpy as np

class CombinedNmsTest(trt_test.TfTrtIntegrationTestBase):
  """Test for CombinedNMS op in TF-TRT."""

  def setUp(self):
    super().setUp()
    # Input Config
    self.batch_size = 1
    self.num_boxes = 20
    self.num_classes = 1
    self.share_boxes = True
    # NMS Attrs
    self.max_output_size_per_class = 5
    self.max_total_size = 5
    self.score_threshold = 0.1
    self.iou_threshold = 0.5
    self.pad_per_class = False
    self.clip_boxes = False

  def GraphFn(self, boxes, scores):
    if self.clip_boxes:
      # Expand the range of box coordinates to properly test box clipping.
      boxes = 2 * boxes - 0.5

    nms_output = image_ops_impl.combined_non_max_suppression(
        boxes=boxes,
        scores=scores,
        max_output_size_per_class=self.max_output_size_per_class,
        max_total_size=self.max_total_size,
        iou_threshold=self.iou_threshold,
        score_threshold=self.score_threshold,
        pad_per_class=self.pad_per_class,
        clip_boxes=self.clip_boxes,
        name='combined_nms')
    return [
        array_ops.identity(output, name=('output_%d' % i))
        for i, output in enumerate(nms_output)
    ]

  def GetParams(self):
    output_size = self.max_total_size
    if self.pad_per_class:
      output_size = min(self.max_total_size,
                    self.num_classes * self.max_output_size_per_class)
    box_classes = 1 if self.share_boxes else self.num_classes

    boxes_shape = [self.batch_size, self.num_boxes, box_classes, 4]
    scores_shape = [self.batch_size, self.num_boxes, self.num_classes]

    nmsed_boxes_shape = [self.batch_size, output_size, 4]
    nmsed_scores_shape = [self.batch_size, output_size]
    nmsed_classes_shape = [self.batch_size, output_size]
    valid_detections_shape = [self.batch_size]

    return self.BuildParams(self.GraphFn, dtypes.float32,
                            [boxes_shape, scores_shape], [
                                nmsed_boxes_shape, nmsed_scores_shape,
                                nmsed_classes_shape, valid_detections_shape
                            ])

  def _GenerateRandomData(self, np_shape, np_dtype, spec, scale):
    # Numpy has a problem where its random sample generates data in fp64.
    # When this is casted down to fp32 or fp16, there's a good chance that
    # duplicate values will appear in the generated data. This causes issues
    # for NMS testing, as boxes with identical scores can be returned in a
    # different order, making it difficult to cross-match NMS results.
    # Therefore a custom random data smpler is used for these tests which
    # guarantees no duplicates are generated, to avoid this problem.

    def unique_random_sample(N, np_dtype, oversample_factor=2):
      # return np.random.random_sample(N).astype(np_dtype)
      sample = np.random.random_sample(N * oversample_factor).astype(np_dtype)
      sample = np.unique(sample)
      length = sample.shape[0]
      if length < N:
        if (oversample_factor > 16):
          # Fail safe, too much oversampling, append another sample.
          extra = unique_random_sample(N - length, np_dtype)
          return np.concatenate((sample, extra))
        else:
          # If too many duplicates were removed, oversample by more.
          return unique_random_sample(N, np_dtype, oversample_factor * 2)
      np.random.shuffle(sample)
      return sample[0:N]

    sample = unique_random_sample(np.prod(np_shape), np_dtype)
    return (scale * sample.reshape(np_shape))

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {
        'TRTEngineOp_0': [
            'combined_nms/CombinedNonMaxSuppression',
            'max_output_size_per_class', 'max_total_size', 'iou_threshold',
            'score_threshold'
        ]
    }

  def ShouldRunTest(self, run_params):
    should_run, reason = super().ShouldRunTest(run_params)
    should_run = should_run and \
        not trt_test.IsQuantizationMode(run_params.precision_mode)
    reason += ' and precision != INT8'
    # Only run for TRT 7.1.3 and above.
    return should_run and trt_utils.is_linked_tensorrt_version_greater_equal(
        7, 1, 3), reason + ' and >= TRT 7.1.3'


class CombinedNmsExecuteNativeSegmentTest(CombinedNmsTest):

  def setUp(self):
    super().setUp()
    os.environ['TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION'] = 'True'

  def tearDown(self):
    super().tearDown()
    os.environ['TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION'] = 'False'

  def GetMaxBatchSize(self, run_params):
    """Returns the max_batch_size that the converter should use for tests."""
    if run_params.dynamic_engine:
      return None

    # Build the engine with the allowed max_batch_size less than the actual
    # max_batch_size, to fore the runtime to execute the native segment. This
    # is to test that combined_non_max_suppression, which doesn't have a TF GPU
    # implementation, can be executed natively even though the it is in the
    # the graph for the TRTEngineOp with a GPU as a default device.
    return super().GetMaxBatchSize(run_params) - 1

  def ShouldRunTest(self, run_params):
    should_run, reason = super().ShouldRunTest(run_params)
    # max_batch_size is only useful for selecting static engines. As such,
    # we shouldn't run the test for dynamic engines.
    return should_run and \
        not run_params.dynamic_engine, reason + ' and static engines'

class CombinedNmsBoxesTest(CombinedNmsTest):
  """Test for CombinedNMS with a larger number of boxes."""

  def setUp(self):
    super().setUp()
    self.batch_size = 4
    self.num_boxes = 1000
    self.max_output_size_per_class = 100
    self.max_total_size = 100
    self.share_boxes = False
    self.clip_boxes = True

class CombinedNmsScoresTest(CombinedNmsTest):
  """Test for CombinedNMS with a larger number of scores."""

  def setUp(self):
    super().setUp()
    self.num_boxes = 1000
    self.num_classes = 7
    self.max_output_size_per_class = 7
    self.max_total_size = 100
    self.pad_per_class = False

class CombinedNmsTopKTest(CombinedNmsTest):
  """Test for CombinedNMS with a larger number of outputs and max TopK."""

  def setUp(self):
    super().setUp()
    self.num_boxes = 100000
    self.score_threshold = 0.0001
    self.iou_threshold = 0.9999
    self.max_output_size_per_class = 5000
    self.max_total_size = 5000

  def ShouldRunTest(self, run_params):
    # The TopK test is not verifiable in fp16, due to duplicate score matching.
    # TF and TRT will produce boxes with same scores in different order, making
    # it impossible to match the tensors.
    should_run, reason = super().ShouldRunTest(run_params)
    should_run = should_run and run_params.precision_mode != "FP16"
    reason += ' and precision == FP16'
    return should_run, reason

if __name__ == '__main__':
  test.main()
