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
class UnsortedSegmentMinOpModel : public SingleOpModel {
 public:
  UnsortedSegmentMinOpModel(const TensorData& data,
                            const TensorData& segment_ids,
                            const TensorData& num_segments) {
    data_id_ = AddInput(data);
    segment_ids_id_ = AddInput(segment_ids);
    num_segments_id_ = AddInput(num_segments);
    output_id_ = AddOutput(data.type);
    SetBuiltinOp(BuiltinOperator_UNSORTED_SEGMENT_MIN, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(data_id_), GetShape(segment_ids_id_),
                      GetShape(num_segments_id_)});
  }

  explicit UnsortedSegmentMinOpModel(
      const TensorData& data, const std::initializer_list<int>& segment_id_data,
      const std::initializer_list<int>& segment_id_shape,
      const std::initializer_list<int>& num_segments_data,
      const std::initializer_list<int>& num_segments_shape) {
    data_id_ = AddInput(data);
    segment_ids_id_ =
        AddConstInput(TensorType_INT32, segment_id_data, segment_id_shape);
    num_segments_id_ =
        AddConstInput(TensorType_INT32, num_segments_data, num_segments_shape);
    output_id_ = AddOutput(data.type);
    SetBuiltinOp(BuiltinOperator_UNSORTED_SEGMENT_MIN, BuiltinOptions_NONE, 0);
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

TEST(UnsortedSegmentMinOpModelTest, Int32Test_Simple) {
  UnsortedSegmentMinOpModel<int32_t> model({TensorType_INT32, {6}},
                                           {TensorType_INT32, {6}},
                                           {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(), {5, 3, 7, 8, 6, 4});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 0, 1, 1, 0, 1});
  model.PopulateTensor<int32_t>(model.num_segments(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({3, 4}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2}));
}

TEST(UnsortedSegmentMinOpModelTest, Int32Test_Simple2D) {
  UnsortedSegmentMinOpModel<int32_t> model({TensorType_INT32, {3, 4}},
                                           {TensorType_INT32, {3}},
                                           {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(),
                                {1, 2, 3, 4, 5, 6, 7, 8, 4, 3, 2, 1});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1, 0});
  model.PopulateTensor<int32_t>(model.num_segments(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 2, 1, 5, 6, 7, 8}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 4}));
}

TEST(UnsortedSegmentMinOpModelTest, FloatTest_Simple) {
  UnsortedSegmentMinOpModel<float> model({TensorType_FLOAT32, {8}},
                                         {TensorType_INT32, {8}},
                                         {TensorType_INT32, {1}});
  model.PopulateTensor<float>(model.data(),
                              {1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0});
  model.PopulateTensor<int32_t>(model.segment_ids(), {1, 0, 1, 7, 7, 7, 7, 7});
  model.PopulateTensor<int32_t>(model.num_segments(), {8});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray(ArrayFloatNear(
          {2.0, 1.0, std::numeric_limits<float>::max(),
           std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
           std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
           1.0})));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({8}));
}

TEST(UnsortedSegmentMinOpModelTest, FloatTest_Simple2D) {
  UnsortedSegmentMinOpModel<float> model({TensorType_FLOAT32, {3, 4}},
                                         {TensorType_INT32, {3}},
                                         {TensorType_INT32, {1}});
  model.PopulateTensor<float>(model.data(), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                             8.0, 4.0, 3.0, 2.0, 1.0});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1, 0});
  model.PopulateTensor<int32_t>(model.num_segments(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray(
                  ArrayFloatNear({1.0, 2.0, 2.0, 1.0, 5.0, 6.0, 7.0, 8.0})));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 4}));
}

TEST(UnsortedSegmentMinOpModelTest,
     TestFailIfSegmentsAreNotTheRightCardinality) {
  UnsortedSegmentMinOpModel<int32_t> model({TensorType_INT32, {3, 2}},
                                           {TensorType_INT32, {2}},
                                           {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1});
  model.PopulateTensor<int32_t>(model.num_segments(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteError);
}

TEST(UnsortedSegmentMinOpModelTest, SegmentsAreNegative) {
  UnsortedSegmentMinOpModel<int32_t> model({TensorType_INT32, {2, 2}},
                                           {TensorType_INT32, {2}},
                                           {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4});
  model.PopulateTensor<int32_t>(model.segment_ids(), {-1, -1});
  model.PopulateTensor<int32_t>(model.num_segments(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({std::numeric_limits<int32_t>::max(),
                                std::numeric_limits<int32_t>::max()}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2}));
}

TEST(UnsortedSegmentMinOpModelTest, TestFailIfSegmentIDGreaterThanNumSegments) {
  UnsortedSegmentMinOpModel<int32_t> model({TensorType_INT32, {2, 2}},
                                           {TensorType_INT32, {2}},
                                           {TensorType_INT32, {1}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1});
  model.PopulateTensor<int32_t>(model.num_segments(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteError);
}

TEST(UnsortedSegmentMinOpModelTest,
     TestFailIfNumSegmentsAreNotTheRightCardinality) {
  UnsortedSegmentMinOpModel<int32_t> model({TensorType_INT32, {3, 2}},
                                           {TensorType_INT32, {3}},
                                           {TensorType_INT32, {2}});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int32_t>(model.segment_ids(), {0, 1, 0});
  model.PopulateTensor<int32_t>(model.num_segments(), {2, 1});
  ASSERT_EQ(model.Invoke(), kTfLiteError);
}

TEST(UnsortedSegmentMinOpModelTest, ConstParamenterTest) {
  UnsortedSegmentMinOpModel<int32_t> model({TensorType_INT32, {3, 2}},
                                           {0, 1, 0}, {3}, {2}, {1});
  model.PopulateTensor<int32_t>(model.data(), {1, 2, 3, 4, 5, 6});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 2}));
}

}  // namespace
}  // namespace tflite
