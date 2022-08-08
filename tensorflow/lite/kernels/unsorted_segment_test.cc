/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/kernels/unsorted_segment_test.h"

#include <limits.h>
#include <stdint.h>

#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

TEST_P(UnsortedSegmentTest, SegmentIdsSizeNotEqualToDataFirstDimensionFails) {
  UnsortedSegmentModel<int32_t> model =
      getModel({TensorType_INT32, {3, 2}}, {TensorType_INT32, {2}},
               {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1});
  model.PopulateTensor<int32_t>(model.num_segments(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteError);
}
TEST_P(UnsortedSegmentTest,
       LargestSegmentIdPlusOneGreaterThanNumSegmentsFails) {
  UnsortedSegmentModel<int32_t> model =
      getModel({TensorType_INT32, {2, 2}}, {TensorType_INT32, {2}},
               {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1});
  model.PopulateTensor<int32_t>(model.num_segments(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteError);
}
TEST_P(UnsortedSegmentTest, NumSegmentsNotScalarShapeFails) {
  UnsortedSegmentModel<int32_t> model =
      getModel({TensorType_INT32, {3, 2}}, {TensorType_INT32, {3}},
               {TensorType_INT32, {2}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1, 0});
  model.PopulateTensor<int32_t>(model.num_segments(), {2, 1});
  ASSERT_EQ(model.Invoke(), kTfLiteError);
}
}  // namespace
}  // namespace tflite
