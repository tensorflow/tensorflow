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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.compiler.tf2tensorrt.wrap_py_utils import get_linked_tensorrt_version
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops_impl
from tensorflow.python.platform import test


class CombinedNmsTest(trt_test.TfTrtIntegrationTestBase):
  """Test for CombinedNMS op in TF-TRT."""

  def GraphFn(self, boxes, scores):
    max_output_size_per_class = 3
    max_total_size = 3
    score_threshold = 0.1
    iou_threshold = 0.5
    # Shapes
    max_output_size_per_class_tensor = constant_op.constant(
        max_output_size_per_class,
        dtype=dtypes.int32,
        name='max_output_size_per_class')
    max_total_size_tensor = constant_op.constant(
        max_total_size, dtype=dtypes.int32, name='max_total_size')
    iou_threshold_tensor = constant_op.constant(
        iou_threshold, dtype=dtypes.float32, name='iou_threshold')
    score_threshold_tensor = constant_op.constant(
        score_threshold, dtype=dtypes.float32, name='score_threshold')
    nms_output = image_ops_impl.combined_non_max_suppression(
        boxes,
        scores,
        max_output_size_per_class_tensor,
        max_total_size_tensor,
        iou_threshold_tensor,
        score_threshold_tensor,
        name='combined_nms')
    return [
        array_ops.identity(output, name=('output_%d' % i))
        for i, output in enumerate(nms_output)
    ]

  def GetParams(self):
    # Parameters
    q = 1
    batch_size = 1
    num_boxes = 200
    num_classes = 2
    max_total_size = 3

    boxes_shape = [batch_size, num_boxes, q, 4]
    scores_shape = [batch_size, num_boxes, num_classes]
    nmsed_boxes_shape = [batch_size, max_total_size, 4]
    nmsed_scores_shape = [batch_size, max_total_size]
    nmsed_classes_shape = [batch_size, max_total_size]
    valid_detections_shape = [batch_size]
    return self.BuildParams(self.GraphFn, dtypes.float32,
                            [boxes_shape, scores_shape], [
                                nmsed_boxes_shape, nmsed_scores_shape,
                                nmsed_classes_shape, valid_detections_shape
                            ])

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
    # There is no CombinedNonMaxSuppression op for GPU at the moment, so
    # calibration will fail.
    # TODO(laigd): fix this.
    if trt_test.IsQuantizationMode(run_params.precision_mode):
      return False

    # Only run for TRT 5.1 and above.
    ver = get_linked_tensorrt_version()
    return ver[0] > 5 or (ver[0] == 5 and ver[1] >= 1)


if __name__ == '__main__':
  test.main()
