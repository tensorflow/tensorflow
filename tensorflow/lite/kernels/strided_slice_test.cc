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

template <typename input_type = float,
          TensorType tensor_input_type = TensorType_FLOAT32>
class StridedSliceOpModel : public SingleOpModel {
 public:
  StridedSliceOpModel(std::initializer_list<int> input_shape,
                      std::initializer_list<int> begin_shape,
                      std::initializer_list<int> end_shape,
                      std::initializer_list<int> strides_shape, int begin_mask,
                      int end_mask, int ellipsis_mask, int new_axis_mask,
                      int shrink_axis_mask) {
    input_ = AddInput(tensor_input_type);
    begin_ = AddInput(TensorType_INT32);
    end_ = AddInput(TensorType_INT32);
    strides_ = AddInput(TensorType_INT32);
    output_ = AddOutput(tensor_input_type);
    SetBuiltinOp(
        BuiltinOperator_STRIDED_SLICE, BuiltinOptions_StridedSliceOptions,
        CreateStridedSliceOptions(builder_, begin_mask, end_mask, ellipsis_mask,
                                  new_axis_mask, shrink_axis_mask)
            .Union());
    BuildInterpreter({input_shape, begin_shape, end_shape, strides_shape});
  }

  void SetInput(std::initializer_list<input_type> data) {
    PopulateTensor<input_type>(input_, data);
  }
  void SetBegin(std::initializer_list<int32_t> data) {
    PopulateTensor<int32_t>(begin_, data);
  }
  void SetEnd(std::initializer_list<int32_t> data) {
    PopulateTensor<int32_t>(end_, data);
  }
  void SetStrides(std::initializer_list<int32_t> data) {
    PopulateTensor<int32_t>(strides_, data);
  }

  std::vector<input_type> GetOutput() {
    return ExtractVector<input_type>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int begin_;
  int end_;
  int strides_;
  int output_;
};

TEST(StridedSliceOpTest, UnsupportedInputSize) {
  EXPECT_DEATH(
      StridedSliceOpModel<>({2, 2, 2, 2, 2}, {5}, {5}, {5}, 0, 0, 0, 0, 0),
      "StridedSlice op only supports 1D-4D input arrays.");
}

TEST(StridedSliceOpTest, UnssupportedArgs) {
  EXPECT_DEATH(StridedSliceOpModel<>({3, 2}, {2}, {2}, {2}, 0, 0, 1, 0, 0),
               "ellipsis_mask is not implemented yet.");
  EXPECT_DEATH(StridedSliceOpModel<>({3, 2}, {2}, {2}, {2}, 0, 0, 0, 1, 0),
               "new_axis_mask is not implemented yet.");
}

TEST(StridedSliceOpTest, In1D) {
  StridedSliceOpModel<> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({1});
  m.SetEnd({3});
  m.SetStrides({1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 3}));
}

TEST(StridedSliceOpTest, In1D_EmptyOutput) {
  StridedSliceOpModel<> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({10});
  m.SetEnd({3});
  m.SetStrides({1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({0}));
}

TEST(StridedSliceOpTest, In1D_NegativeBegin) {
  StridedSliceOpModel<> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({-3});
  m.SetEnd({3});
  m.SetStrides({1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 3}));
}

TEST(StridedSliceOpTest, In1D_OutOfRangeBegin) {
  StridedSliceOpModel<> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({-5});
  m.SetEnd({3});
  m.SetStrides({1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3}));
}

TEST(StridedSliceOpTest, In1D_NegativeEnd) {
  StridedSliceOpModel<> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({1});
  m.SetEnd({-2});
  m.SetStrides({1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2}));
}

TEST(StridedSliceOpTest, In1D_OutOfRangeEnd) {
  StridedSliceOpModel<> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({-3});
  m.SetEnd({5});
  m.SetStrides({1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 3, 4}));
}

TEST(StridedSliceOpTest, In1D_BeginMask) {
  StridedSliceOpModel<> m({4}, {1}, {1}, {1}, 1, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({1});
  m.SetEnd({3});
  m.SetStrides({1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3}));
}

TEST(StridedSliceOpTest, In1D_NegativeBeginNegativeStride) {
  StridedSliceOpModel<> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({-2});
  m.SetEnd({-3});
  m.SetStrides({-1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3}));
}

TEST(StridedSliceOpTest, In1D_OutOfRangeBeginNegativeStride) {
  StridedSliceOpModel<> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({5});
  m.SetEnd({2});
  m.SetStrides({-1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({4}));
}

TEST(StridedSliceOpTest, In1D_NegativeEndNegativeStride) {
  StridedSliceOpModel<> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({2});
  m.SetEnd({-4});
  m.SetStrides({-1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 2}));
}

TEST(StridedSliceOpTest, In1D_OutOfRangeEndNegativeStride) {
  StridedSliceOpModel<> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({-3});
  m.SetEnd({-5});
  m.SetStrides({-1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 1}));
}

TEST(StridedSliceOpTest, In1D_EndMask) {
  StridedSliceOpModel<> m({4}, {1}, {1}, {1}, 0, 1, 0, 0, 0);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({1});
  m.SetEnd({3});
  m.SetStrides({1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 3, 4}));
}

TEST(StridedSliceOpTest, In1D_NegStride) {
  StridedSliceOpModel<> m({3}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3});
  m.SetBegin({-1});
  m.SetEnd({-4});
  m.SetStrides({-1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 2, 1}));
}

TEST(StridedSliceOpTest, In1D_EvenLenStride2) {
  StridedSliceOpModel<> m({2}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2});
  m.SetBegin({0});
  m.SetEnd({2});
  m.SetStrides({2});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1}));
}

TEST(StridedSliceOpTest, In1D_OddLenStride2) {
  StridedSliceOpModel<> m({3}, {1}, {1}, {1}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3});
  m.SetBegin({0});
  m.SetEnd({3});
  m.SetStrides({2});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3}));
}

TEST(StridedSliceOpTest, In2D_Identity) {
  StridedSliceOpModel<> m({2, 3}, {2}, {2}, {2}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({0, 0});
  m.SetEnd({2, 3});
  m.SetStrides({1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(StridedSliceOpTest, In2D) {
  StridedSliceOpModel<> m({2, 3}, {2}, {2}, {2}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({1, 0});
  m.SetEnd({2, 2});
  m.SetStrides({1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({4, 5}));
}

TEST(StridedSliceOpTest, In2D_Stride2) {
  StridedSliceOpModel<> m({2, 3}, {2}, {2}, {2}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({0, 0});
  m.SetEnd({2, 3});
  m.SetStrides({2, 2});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3}));
}

TEST(StridedSliceOpTest, In2D_NegStride) {
  StridedSliceOpModel<> m({2, 3}, {2}, {2}, {2}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({1, -1});
  m.SetEnd({2, -4});
  m.SetStrides({2, -1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({6, 5, 4}));
}

TEST(StridedSliceOpTest, In2D_BeginMask) {
  StridedSliceOpModel<> m({2, 3}, {2}, {2}, {2}, 1, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({1, 0});
  m.SetEnd({2, 2});
  m.SetStrides({1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 4, 5}));
}

TEST(StridedSliceOpTest, In2D_EndMask) {
  StridedSliceOpModel<> m({2, 3}, {2}, {2}, {2}, 0, 2, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({1, 0});
  m.SetEnd({2, 2});
  m.SetStrides({1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({4, 5, 6}));
}

TEST(StridedSliceOpTest, In2D_NegStrideBeginMask) {
  StridedSliceOpModel<> m({2, 3}, {2}, {2}, {2}, 2, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({1, -2});
  m.SetEnd({2, -4});
  m.SetStrides({1, -1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({6, 5, 4}));
}

TEST(StridedSliceOpTest, In2D_NegStrideEndMask) {
  StridedSliceOpModel<> m({2, 3}, {2}, {2}, {2}, 0, 2, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({1, -2});
  m.SetEnd({2, -3});
  m.SetStrides({1, -1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 4}));
}

TEST(StridedSliceOpTest, In3D_Identity) {
  StridedSliceOpModel<> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({2, 3, 2});
  m.SetStrides({1, 1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 2}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
}

TEST(StridedSliceOpTest, In3D_NegStride) {
  StridedSliceOpModel<> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({-1, -1, -1});
  m.SetEnd({-3, -4, -3});
  m.SetStrides({-1, -1, -1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 2}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}));
}

TEST(StridedSliceOpTest, In3D_Strided2) {
  StridedSliceOpModel<> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({2, 3, 2});
  m.SetStrides({2, 2, 2});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 5}));
}

TEST(StridedSliceOpTest, In1D_ShrinkAxisMask1) {
  StridedSliceOpModel<> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 1);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({1});
  m.SetEnd({2});
  m.SetStrides({1});
  m.Invoke();
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2}));
}

TEST(StridedSliceOpTest, In1D_ShrinkAxisMask1_NegativeSlice) {
  // This is equivalent to tf.range(4)[-1].
  StridedSliceOpModel<> m({4}, {1}, {1}, {1}, 0, 0, 0, 0, 1);
  m.SetInput({0, 1, 2, 3});
  m.SetBegin({-1});
  m.SetEnd({0});
  m.SetStrides({1});

  m.Invoke();
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3}));
}

TEST(StridedSliceOpTest, In2D_ShrinkAxis3_NegativeSlice) {
  // This is equivalent to tf.range(4)[:, tf.newaxis][-2, -1].
  StridedSliceOpModel<> m({4, 1}, {2}, {2}, {2}, 0, 0, 0, 0, 3);
  m.SetInput({0, 1, 2, 3});
  m.SetBegin({-2, -1});
  m.SetEnd({-1, 0});
  m.SetStrides({1, 1});

  m.Invoke();
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2}));
}

TEST(StridedSliceOpTest, In2D_ShrinkAxis2_BeginEndAxis1_NegativeSlice) {
  // This is equivalent to tf.range(4)[:, tf.newaxis][:, -1].
  StridedSliceOpModel<> m({4, 1}, {2}, {2}, {2}, 1, 1, 0, 0, 2);
  m.SetInput({0, 1, 2, 3});
  m.SetBegin({0, -1});
  m.SetEnd({0, 0});
  m.SetStrides({1, 1});

  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 1, 2, 3}));
}

TEST(StridedSliceOpTest, In1D_BeginMaskShrinkAxisMask1) {
  StridedSliceOpModel<> m({4}, {1}, {1}, {1}, 1, 0, 0, 0, 1);
  m.SetInput({1, 2, 3, 4});
  m.SetBegin({1});
  m.SetEnd({1});
  m.SetStrides({1});
  m.Invoke();
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1}));
}

TEST(StridedSliceOpTest, In2D_ShrinkAxisMask1) {
  StridedSliceOpModel<> m({2, 3}, {2}, {2}, {2}, 0, 0, 0, 0, 1);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({0, 0});
  m.SetEnd({1, 3});
  m.SetStrides({1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3}));
}

TEST(StridedSliceOpTest, In2D_ShrinkAxisMask2) {
  StridedSliceOpModel<> m({2, 3}, {2}, {2}, {2}, 0, 0, 0, 0, 2);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({0, 0});
  m.SetEnd({2, 1});
  m.SetStrides({1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 4}));
}

TEST(StridedSliceOpTest, In2D_ShrinkAxisMask3) {
  StridedSliceOpModel<> m({2, 3}, {2}, {2}, {2}, 0, 0, 0, 0, 3);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({0, 0});
  m.SetEnd({1, 1});
  m.SetStrides({1, 1});
  m.Invoke();
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1}));
}

TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis1) {
  StridedSliceOpModel<> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 0, 1);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({1, 3, 2});
  m.SetStrides({1, 1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis2) {
  StridedSliceOpModel<> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 0, 2);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({2, 1, 2});
  m.SetStrides({1, 1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 7, 8}));
}

TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis3) {
  StridedSliceOpModel<> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 0, 3);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({1, 1, 2});
  m.SetStrides({1, 1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2}));
}

TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis4) {
  StridedSliceOpModel<> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 0, 4);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({2, 3, 1});
  m.SetStrides({1, 1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3, 5, 7, 9, 11}));
}

TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis5) {
  StridedSliceOpModel<> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 0, 5);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({1, 3, 1});
  m.SetStrides({1, 1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3, 5}));
}

TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis6) {
  StridedSliceOpModel<> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 0, 6);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({2, 1, 1});
  m.SetStrides({1, 1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 7}));
}

TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis7) {
  StridedSliceOpModel<> m({2, 3, 2}, {3}, {3}, {3}, 0, 0, 0, 0, 7);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({1, 1, 1});
  m.SetStrides({1, 1, 1});
  m.Invoke();
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1}));
}

// This tests catches a very subtle bug that was fixed by cl/188403234.
TEST(StridedSliceOpTest, RunTwice) {
  StridedSliceOpModel<> m({2, 3}, {2}, {2}, {2}, 1, 0, 0, 0, 0);

  auto setup_inputs = [&m]() {
    m.SetInput({1, 2, 3, 4, 5, 6});
    m.SetBegin({1, 0});
    m.SetEnd({2, 2});
    m.SetStrides({1, 1});
  };

  setup_inputs();
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 4, 5}));

  setup_inputs();
  m.Invoke();
  // Prior to cl/188403234 this was {4, 5}.
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 4, 5}));
}

TEST(StridedSliceOpTest, In3D_IdentityShrinkAxis1Uint8) {
  StridedSliceOpModel<uint8_t, TensorType_UINT8> m({2, 3, 2}, {3}, {3}, {3}, 0, 0,
                                                 0, 0, 1);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.SetBegin({0, 0, 0});
  m.SetEnd({1, 3, 2});
  m.SetStrides({1, 1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}
}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
