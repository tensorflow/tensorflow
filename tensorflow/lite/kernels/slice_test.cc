/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename input_type, typename index_type>
class SliceOpModel : public SingleOpModel {
 public:
  SliceOpModel(std::initializer_list<int> input_shape,
               std::initializer_list<int> begin_shape,
               std::initializer_list<int> size_shape,
               TensorType tensor_index_type, TensorType tensor_input_type) {
    input_ = AddInput(tensor_input_type);
    begin_ = AddInput(tensor_index_type);
    size_ = AddInput(tensor_index_type);
    output_ = AddOutput(tensor_input_type);
    SetBuiltinOp(BuiltinOperator_SLICE, BuiltinOptions_SliceOptions,
                 CreateSliceOptions(builder_).Union());
    BuildInterpreter({input_shape, begin_shape, size_shape});
  }

  void SetInput(std::initializer_list<input_type> data) {
    PopulateTensor<input_type>(input_, data);
  }
  void SetStringInput(std::vector<string> data) {
    PopulateStringTensor(input_, data);
  }
  void SetBegin(std::initializer_list<index_type> data) {
    PopulateTensor<index_type>(begin_, data);
  }
  void SetSize(std::initializer_list<index_type> data) {
    PopulateTensor<index_type>(size_, data);
  }

  std::vector<input_type> GetOutput() {
    return ExtractVector<input_type>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int begin_;
  int size_;
  int output_;
};

TEST(SliceOpTest, In1D) {
  SliceOpModel<float, int32_t> m({4}, {1}, {1}, TensorType_INT32,
                                 TensorType_FLOAT32);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({1});
  m.SetSize({2});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 3}));
}

TEST(SliceOpTest, In2D) {
  SliceOpModel<float, int32_t> m({2, 3}, {2}, {2}, TensorType_INT32,
                                 TensorType_FLOAT32);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({1, 0});
  m.SetSize({1, 2});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({4, 5}));
}

TEST(SliceOpTest, In3D) {
  SliceOpModel<float, int32_t> m({2, 3, 2}, {3}, {4}, TensorType_INT32,
                                 TensorType_FLOAT32);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetSize({2, 3, 2});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 2}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
}

TEST(SliceOpTest, InputFloat) {
  SliceOpModel<float, int32_t> m({4, 1, 1, 1}, {4}, {4}, TensorType_INT32,
                                 TensorType_FLOAT32);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({1, 0, 0, 0});
  m.SetSize({3, 1, 1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1, 1, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 3, 4}));
}

TEST(SliceOpTest, IndexInt64) {
  SliceOpModel<float, int64_t> m({4, 1, 1, 1}, {4}, {4}, TensorType_INT64,
                                 TensorType_FLOAT32);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({1, 0, 0, 0});
  m.SetSize({3, 1, 1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1, 1, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 3, 4}));
}

// See these test cases under:
// https://www.tensorflow.org/versions/master/api_docs/python/tf/slice
TEST(SliceOpTest, InputInteger1) {
  SliceOpModel<int32_t, int32_t> m({3, 2, 3, 1}, {4}, {4}, TensorType_INT32,
                                   TensorType_INT32);
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  m.SetBegin({1, 0, 0, 0});
  m.SetSize({1, 1, 3, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3}));
}

TEST(SliceOpTest, InputInteger2) {
  SliceOpModel<int32_t, int32_t> m({3, 2, 3, 1}, {4}, {4}, TensorType_INT32,
                                   TensorType_INT32);
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  m.SetBegin({1, 0, 0, 0});
  m.SetSize({1, 2, 3, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 4, 4, 4}));
}

TEST(SliceOpTest, InputInteger3) {
  SliceOpModel<int32_t, int32_t> m({3, 2, 3, 1}, {4}, {4}, TensorType_INT32,
                                   TensorType_INT32);
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  m.SetBegin({1, 0, 0, 0});
  m.SetSize({2, 1, 3, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 5, 5, 5}));
}

TEST(SliceOpTest, SizeMinus1) {
  SliceOpModel<int32_t, int32_t> m({3, 2, 3, 1}, {4}, {4}, TensorType_INT32,
                                   TensorType_INT32);
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  m.SetBegin({1, 0, 0, 0});
  m.SetSize({2, 1, -1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 5, 5, 5}));
}

TEST(SliceOpTest, BeginNonZeroSizeMinus1Axis1) {
  SliceOpModel<int32_t, int32_t> m({3, 3, 2, 1}, {4}, {4}, TensorType_INT32,
                                   TensorType_INT32);
  m.SetInput({1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9});
  m.SetBegin({1, 1, 0, 0});
  m.SetSize({2, -1, 1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 1, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 6, 8, 9}));
}

TEST(SliceOpTest, BeginNonZeroSizeMinus1Axis2) {
  SliceOpModel<int32_t, int32_t> m({3, 2, 3, 1}, {4}, {4}, TensorType_INT32,
                                   TensorType_INT32);
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  m.SetBegin({1, 0, 1, 0});
  m.SetSize({2, 1, -1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 2, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 5, 5}));
}

TEST(SliceOpTest, BeginNonZeroSizeMinus1Axis3) {
  SliceOpModel<int32_t, int32_t> m({3, 1, 2, 3}, {4}, {4}, TensorType_INT32,
                                   TensorType_INT32);
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  m.SetBegin({1, 0, 0, 1});
  m.SetSize({2, 1, 1, -1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 5, 5}));
}

TEST(SliceOpTest, SliceUint8) {
  SliceOpModel<uint8_t, int32_t> m({3, 2, 3, 1}, {4}, {4}, TensorType_INT32,
                                   TensorType_UINT8);
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  m.SetBegin({1, 0, 0, 0});
  m.SetSize({2, 1, -1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 5, 5, 5}));
}

TEST(SliceOpTest, SliceInt8) {
  SliceOpModel<int8_t, int32_t> m({3, 2, 3, 1}, {4}, {4}, TensorType_INT32,
                                  TensorType_INT8);
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  m.SetBegin({1, 0, 0, 0});
  m.SetSize({2, 1, -1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 5, 5, 5}));
}

TEST(SliceOpTest, SliceString) {
  SliceOpModel<string, int32_t> m({3, 2, 3, 1}, {4}, {4}, TensorType_INT32,
                                  TensorType_STRING);
  m.SetStringInput({"0,0,0,0", "0,0,1,0", "0,0,2,0",  //
                    "0,1,0,0", "0,1,1,0", "0,1,2,0",  //
                    "1,0,0,0", "1,0,1,0", "1,0,2,0",  //
                    "1,1,0,0", "1,1,1,0", "1,1,2,0",  //
                    "2,0,0,0", "2,0,1,0", "2,0,2,0",  //
                    "2,1,0,0", "2,1,1,0", "2,1,2,0"});
  m.SetBegin({1, 0, 0, 0});
  m.SetSize({2, 1, -1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({"1,0,0,0", "1,0,1,0", "1,0,2,0",  //
                                "2,0,0,0", "2,0,1,0", "2,0,2,0"}));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
