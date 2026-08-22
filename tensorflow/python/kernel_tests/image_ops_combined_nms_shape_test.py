# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for input rank validation in combined_non_max_suppression."""

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import image_ops_impl
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class CombinedNmsShapeTest(test.TestCase):

  def test_boxes_wrong_static_rank_raises(self):
    boxes = random_ops.random_uniform([4, 10, 4])  # rank-3, must be rank-4.
    scores = random_ops.random_uniform([4, 10, 1])
    with self.assertRaisesRegex(
        (errors_impl.InvalidArgumentError, ValueError),
        "(boxes must be 4-D|Shape must be rank 4)",
    ):
      image_ops_impl.combined_non_max_suppression(boxes, scores, 10, 10)

  def test_scores_wrong_static_rank_raises(self):
    boxes = random_ops.random_uniform([4, 10, 1, 4])
    scores = random_ops.random_uniform([4, 10])  # rank-2, must be rank-3.
    with self.assertRaisesRegex(
        (errors_impl.InvalidArgumentError, ValueError),
        "(scores must be 3-D|Shape must be rank 3)",
    ):
      image_ops_impl.combined_non_max_suppression(boxes, scores, 10, 10)

  def test_wrong_dynamic_rank_triggers_kernel_validation(self):
    # Unknown-rank input signatures bypass static shape inference, so the
    # rank checks in the C++ kernel are what raises here.
    @def_function.function(
        input_signature=[
            tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32),
            tensor_spec.TensorSpec(shape=None, dtype=dtypes.float32),
        ]
    )
    def run_nms(boxes_dyn, scores_dyn):
      return image_ops_impl.combined_non_max_suppression(
          boxes_dyn, scores_dyn, 10, 10
      )

    boxes = random_ops.random_uniform([4, 10, 4])  # rank-3, must be rank-4.
    scores = random_ops.random_uniform([4, 10, 1])
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError, "boxes must be 4-D"
    ):
      run_nms(boxes, scores)

    boxes_valid = random_ops.random_uniform([4, 10, 1, 4])
    scores_invalid = random_ops.random_uniform([4, 10])  # rank-2.
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError, "scores must be 3-D"
    ):
      run_nms(boxes_valid, scores_invalid)


if __name__ == "__main__":
  test.main()
