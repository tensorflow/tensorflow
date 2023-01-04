/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename T>
class SegmentSumOpModel : public SingleOpModel {
 public:
  SegmentSumOpModel(const TensorData& data, const TensorData& segment_ids) {
    data_id_ = AddInput(data);
    segment_ids_id_ = AddInput(segment_ids);
    output_id_ = AddOutput(data.type);
    SetBuiltinOp(BuiltinOperator_SEGMENT_SUM, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(data_id_), GetShape(segment_ids_id_)});
  }

  int data() const { return data_id_; }
  int segment_ids() const { return segment_ids_id_; }
  std::vector<T> GetOutput() { return ExtractVector<T>(output_id_); }
  std::vector<int32_t> GetOutputShape() { return GetTensorShape(output_id_); }

 protected:
  int data_id_;
  int segment_ids_id_;
  int output_id_;
};

TEST(SegmentSumOpModelTest, Int32Test_Simple) {
  SegmentSumOpModel<int32_t> model({TensorType_INT32, {3, 4}},
                                   {TensorType_INT32, {3}});
  model.PopulateTensor<int32_t>(model.data(),
                                {1, 2, 3, 4, 4, 3, 2, 1, 5, 6, 7, 8});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 0, 1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({5, 5, 5, 5, 5, 6, 7, 8}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 4}));
}

TEST(SegmentSumOpModelTest, Int32Test_OneDimension) {
  SegmentSumOpModel<int32_t> model({TensorType_INT32, {3}},
                                   {TensorType_INT32, {3}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 0, 1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({3, 3}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2}));
}

TEST(SegmentSumOpModelTest, Int32Test_ThreeDimensions) {
  SegmentSumOpModel<int32_t> model({TensorType_INT32, {3, 2, 1}},
                                   {TensorType_INT32, {3}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 0, 1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({4, 6, 5, 6}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 2, 1}));
}

TEST(SegmentSumOpModelTest, Float32Test_Simple) {
  SegmentSumOpModel<float> model({TensorType_FLOAT32, {3, 4}},
                                 {TensorType_INT32, {3}});
  model.PopulateTensor<float>(model.data(),
                              {1, 2, 3, 4, 4, 3, 2, 1, 5, 6, 7, 8});
  model.PopulateTensor<int>(model.segment_ids(), {0, 0, 1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
                                                   6.0f, 7.0f, 8.0f}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 4}));
}

TEST(SegmentSumOpModelTest, Float32Test_OneDimension) {
  SegmentSumOpModel<float> model({TensorType_FLOAT32, {3}},
                                 {TensorType_INT32, {3}});
  model.PopulateTensor<float>(model.data(), {1, 2, 3});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 0, 1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({3.0f, 3.0f}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2}));
}

TEST(SegmentSumOpModelTest, Float32Test_ThreeDimensions) {
  SegmentSumOpModel<float> model({TensorType_FLOAT32, {3, 2, 1}},
                                 {TensorType_INT32, {3}});
  model.PopulateTensor<float>(model.data(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 0, 1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({4.0f, 6.0f, 5.0f, 6.0f}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 2, 1}));
}

TEST(SegmentSumOpModelTest, TestFailIfSegmentsAreNotSorted) {
  SegmentSumOpModel<int32_t> model({TensorType_INT32, {3, 2}},
                                   {TensorType_INT32, {3}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 3, 1});
  ASSERT_EQ(model.Invoke(), kTfLiteError);
}

TEST(SegmentSumOpModelTest, TestFailIfSegmentsAreNotConsecutive) {
  SegmentSumOpModel<int32_t> model({TensorType_INT32, {3, 2}},
                                   {TensorType_INT32, {3}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 3, 5});
  ASSERT_EQ(model.Invoke(), kTfLiteError);
}

TEST(SegmentSumOpModelTest, TestFailIfSegmentsAreNegative) {
  SegmentSumOpModel<int32_t> model({TensorType_INT32, {3, 2}},
                                   {TensorType_INT32, {3}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int32_t>(model.segment_ids(), {-1, 0, 1});
  ASSERT_EQ(model.Invoke(), kTfLiteError);
}

TEST(SegmentSumOpModelTest, TestFailIfSegmentsAreNotTheRightCardinality) {
  SegmentSumOpModel<int32_t> model({TensorType_INT32, {3, 2}},
                                   {TensorType_INT32, {2}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1});
  ASSERT_EQ(model.Invoke(), kTfLiteError);
}

}  // namespace
}  // namespace tflite
