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
#include <stdint.h>

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/kernels/unsorted_segment_test.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename T>
class UnsortedSegmentProdModel : public UnsortedSegmentModel<T> {
 public:
  UnsortedSegmentProdModel(const TensorData& data,
                           const TensorData& segment_ids,
                           const TensorData& num_segments)
      : UnsortedSegmentModel<T>(data, segment_ids, num_segments,
                                BuiltinOperator_UNSORTED_SEGMENT_PROD,
                                BuiltinOptions_UnsortedSegmentProdOptions) {}

  explicit UnsortedSegmentProdModel(
      const TensorData& data, const std::initializer_list<int>& segment_id_data,
      const std::initializer_list<int>& segment_id_shape,
      const std::initializer_list<int>& num_segments_data,
      const std::initializer_list<int>& num_segments_shape)
      : UnsortedSegmentModel<T>(data, segment_id_data, segment_id_shape,
                                num_segments_data, num_segments_shape,
                                BuiltinOperator_UNSORTED_SEGMENT_PROD,
                                BuiltinOptions_UnsortedSegmentProdOptions) {}
};

INSTANTIATE_TEST_SUITE_P(
    UnsortedSegmentProdTestP, UnsortedSegmentTest,
    testing::Values(BuiltinOperator_UNSORTED_SEGMENT_PROD));

TEST(UnsortedSegmentProdModelTest, Int32Test_Simple) {
  UnsortedSegmentProdModel<int32_t> model({TensorType_INT32, {8}},
                                          {TensorType_INT32, {8}},
                                          {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4, 4, 3, 2, 1});
  model.PopulateTensor<int32_t>(model.segment_ids(), {1, 0, 1, 7, 7, 7, 7, 7});
  model.PopulateTensor<int32_t>(model.num_segments(), {8});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({2, 3, 1, 1, 1, 1, 1, 96}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({8}));
}

TEST(UnsortedSegmentProdModelTest, TestSkipNegSegmentId) {
  UnsortedSegmentProdModel<int32_t> model({TensorType_INT32, {8}},
                                          {TensorType_INT32, {8}},
                                          {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4, 4, 3, 2, 1});
  model.PopulateTensor<int32_t>(model.segment_ids(), {1, 0, 1, 7, 7, 7, 7, -1});
  model.PopulateTensor<int32_t>(model.num_segments(), {8});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({2, 3, 1, 1, 1, 1, 1, 96}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({8}));
}

TEST(UnsortedSegmentProdModelTest, Int32Test_Simple2D) {
  UnsortedSegmentProdModel<int32_t> model({TensorType_INT32, {3, 4}},
                                          {TensorType_INT32, {3}},
                                          {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(),
                                {1, 2, 3, 4, 5, 6, 7, 8, 4, 3, 2, 1});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1, 0});
  model.PopulateTensor<int32_t>(model.num_segments(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({4, 6, 6, 4, 5, 6, 7, 8}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 4}));
}

TEST(UnsortedSegmentProdModelTest, FloatTest_Simple) {
  UnsortedSegmentProdModel<float> model({TensorType_FLOAT32, {8}},
                                        {TensorType_INT32, {8}},
                                        {TensorType_INT32, {1}});
  model.PopulateTensor<float>(model.data(),
                              {1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0});
  model.PopulateTensor<int32_t>(model.segment_ids(), {1, 0, 1, 7, 7, 7, 7, 7});
  model.PopulateTensor<int32_t>(model.num_segments(), {8});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(
                  ArrayFloatNear({2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 96.0})));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({8}));
}

TEST(UnsortedSegmentProdModelTest, FloatTest_Simple2D) {
  UnsortedSegmentProdModel<float> model({TensorType_FLOAT32, {3, 4}},
                                        {TensorType_INT32, {3}},
                                        {TensorType_INT32, {1}});
  model.PopulateTensor<float>(model.data(), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                             8.0, 4.0, 3.0, 2.0, 1.0});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1, 0});
  model.PopulateTensor<int32_t>(model.num_segments(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(
                  ArrayFloatNear({4.0, 6.0, 6.0, 4.0, 5.0, 6.0, 7.0, 8.0})));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 4}));
}

TEST(UnsortedSegmentProdModelTest, Int8Test_Simple) {
  UnsortedSegmentProdModel<int8_t> model(
      {TensorType_INT8, {8}}, {TensorType_INT32, {8}}, {TensorType_INT32, {1}});
  model.PopulateTensor<int8_t>(model.data(), {1, 2, 3, 4, 4, 3, 2, 1});
  model.PopulateTensor<int32_t>(model.segment_ids(), {1, 0, 1, 7, 7, 7, 7, 7});
  model.PopulateTensor<int32_t>(model.num_segments(), {8});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({2, 3, 1, 1, 1, 1, 1, 96}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({8}));
}

}  // namespace
}  // namespace tflite
