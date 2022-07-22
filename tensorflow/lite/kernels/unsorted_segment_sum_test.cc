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
#include <limits.h>
#include <stdint.h>

#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename T>
class UnsortedSegmentSumOpModel : public SingleOpModel {
 public:
  UnsortedSegmentSumOpModel(const TensorData& data,
                            const TensorData& segment_ids,
                            const TensorData& num_segments) {
    data_id_ = AddInput(data);
    segment_ids_id_ = AddInput(segment_ids);
    num_segments_id_ = AddInput(num_segments);
    output_id_ = AddOutput(data.type);
    SetBuiltinOp(BuiltinOperator_UNSORTED_SEGMENT_SUM, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(data_id_), GetShape(segment_ids_id_),
                      GetShape(num_segments_id_)});
  }

  int data() const { return data_id_; }
  int segment_ids() const { return segment_ids_id_; }
  int num_segments() const { return num_segments_id_; }
  std::vector<T> GetOutput() { return ExtractVector<T>(output_id_); }
  std::vector<int32_t> GetOutputShape() { return GetTensorShape(output_id_); }

 protected:
  int data_id_;
  int segment_ids_id_;
  int num_segments_id_;
  int output_id_;
};

TEST(UnsortedSegmentSumOpModelTest, Int32Test_Simple) {
  UnsortedSegmentSumOpModel<int32_t> model({TensorType_INT32, {7}},
                                           {TensorType_INT32, {7}},
                                           {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(), {5, 1, 7, 2, 3, 4, 10});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 0, 1, 1, 0, 1, 0});
  model.PopulateTensor<int32_t>(model.num_segments(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({19, 13}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2}));
}

TEST(UnsortedSegmentSumOpModelTest, Int32Test_Simple2D) {
  UnsortedSegmentSumOpModel<int32_t> model({TensorType_INT32, {3, 4}},
                                           {TensorType_INT32, {3}},
                                           {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(),
                                {1, 2, 3, 4, 5, 6, 7, 8, 4, 3, 2, 1});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1, 0});
  model.PopulateTensor<int32_t>(model.num_segments(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({5, 5, 5, 5, 5, 6, 7, 8}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 4}));
}

TEST(UnsortedSegmentSumOpModelTest, FloatTest_Simple) {
  UnsortedSegmentSumOpModel<float> model({TensorType_FLOAT32, {6}},
                                         {TensorType_INT32, {6}},
                                         {TensorType_INT32, {1}});
  model.PopulateTensor<float>(model.data(), {1.0, 2.0, 3.0, 4.0, 4.0, 3.0});
  model.PopulateTensor<int32_t>(model.segment_ids(), {1, 0, 1, 7, 7, 7});
  model.PopulateTensor<int32_t>(model.num_segments(), {8});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(
                  ArrayFloatNear({2.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0})));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({8}));
}

TEST(UnsortedSegmentSumOpModelTest, FloatTest_Simple2D) {
  UnsortedSegmentSumOpModel<float> model({TensorType_FLOAT32, {3, 4}},
                                         {TensorType_INT32, {3}},
                                         {TensorType_INT32, {1}});
  model.PopulateTensor<float>(model.data(), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                             8.0, 4.0, 3.0, 2.0, 1.0});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1, 0});
  model.PopulateTensor<int32_t>(model.num_segments(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(
                  ArrayFloatNear({5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 7.0, 8.0})));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 4}));
}

TEST(UnsortedSegmentSumOpModelTest,
     SegmentIdsSizeNotEqualToDataFirstDimensionFails) {
  UnsortedSegmentSumOpModel<int32_t> model({TensorType_INT32, {3, 2}},
                                           {TensorType_INT32, {2}},
                                           {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1});
  model.PopulateTensor<int32_t>(model.num_segments(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteError);
}

TEST(UnsortedSegmentSumOpModelTest, AllNegativeSegmentIdsZeroTensor) {
  UnsortedSegmentSumOpModel<int32_t> model({TensorType_INT32, {2, 2}},
                                           {TensorType_INT32, {2}},
                                           {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4});
  model.PopulateTensor<int32_t>(model.segment_ids(), {-1, -1});
  model.PopulateTensor<int32_t>(model.num_segments(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({0, 0}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2}));
}

TEST(UnsortedSegmentSumOpModelTest, SomeNegativeSegmentIdsIgnored) {
  UnsortedSegmentSumOpModel<int32_t> model({TensorType_INT32, {4}},
                                           {TensorType_INT32, {4}},
                                           {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4});
  model.PopulateTensor<int32_t>(model.segment_ids(), {-1, 0, -1, 1});
  model.PopulateTensor<int32_t>(model.num_segments(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({2, 4}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2}));
}

TEST(UnsortedSegmentSumOpModelTest,
     LargestSegmentIdPlusOneGreaterThanNumSegmentsFails) {
  UnsortedSegmentSumOpModel<int32_t> model({TensorType_INT32, {2, 2}},
                                           {TensorType_INT32, {2}},
                                           {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1});
  model.PopulateTensor<int32_t>(model.num_segments(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteError);
}

TEST(UnsortedSegmentSumOpModelTest,
     NumSegmentsGreaterThanNumIdsPadsWithZeroTensors) {
  UnsortedSegmentSumOpModel<int32_t> model({TensorType_INT32, {2, 2}},
                                           {TensorType_INT32, {2}},
                                           {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1});
  model.PopulateTensor<int32_t>(model.num_segments(), {3});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 0, 0}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({3, 2}));
}

TEST(UnsortedSegmentSumOpModelTest, NumSegmentsNotScalarShapeFails) {
  UnsortedSegmentSumOpModel<int32_t> model({TensorType_INT32, {3, 2}},
                                           {TensorType_INT32, {3}},
                                           {TensorType_INT32, {2}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1, 0});
  model.PopulateTensor<int32_t>(model.num_segments(), {2, 1});
  ASSERT_EQ(model.Invoke(), kTfLiteError);
}

}  // namespace
}  // namespace tflite
