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
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"

namespace tflite {
using testing::ElementsAreArray;

template <typename input_type>
class StridedSliceOpModel : public SingleOpModelWithHexagon {
 public:
  StridedSliceOpModel(const TensorData& input, const TensorData& output,
                      const TensorData& begin,
                      std::initializer_list<int> begin_data,
                      const TensorData& end,
                      std::initializer_list<int> end_data,
                      const TensorData& strides,
                      std::initializer_list<int> strides_data, int begin_mask,
                      int end_mask, int shrink_axis_mask) {
    input_ = AddInput(input);
    begin_ = AddConstInput(begin, begin_data);
    end_ = AddConstInput(end, end_data);
    strides_ = AddConstInput(strides, strides_data);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_STRIDED_SLICE,
                 BuiltinOptions_StridedSliceOptions,
                 CreateStridedSliceOptions(
                     builder_, begin_mask, end_mask, /*ellipsis_mask*/ 0,
                     /*new_axis_mask*/ 0, shrink_axis_mask)
                     .Union());
    BuildInterpreter({GetShape(input_), GetShape(begin_), GetShape(end_),
                      GetShape(strides_)});
  }

  void SetInput(std::initializer_list<input_type> data) {
    PopulateTensor<input_type>(input_, data);
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

TEST(StridedSliceOpModel, In1D_UInt8) {
  StridedSliceOpModel<uint8_t> m(
      /*input=*/{TensorType_UINT8, {4}, -10, 10},
      /*output=*/{TensorType_UINT8, {2}, -10, 10},
      /*begin*/ {TensorType_INT32, {1}},
      /*begin_data*/ {1},
      /*end*/ {TensorType_INT32, {1}},
      /*end_data*/ {3},
      /*strides*/ {TensorType_INT32, {1}},
      /*strides_data*/ {1},
      /*begin_mask*/ 0, /*end_mask*/ 0, /*shrink_axis_mask*/ 0);
  m.SetInput({1, 2, 3, 4});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 3}));
}

TEST(StridedSliceOpModel, In1D_NegativeBegin_Int8) {
  StridedSliceOpModel<int8_t> m(
      /*input=*/{TensorType_INT8, {4}, -10, 10},
      /*output=*/{TensorType_INT8, {2}, -10, 10},
      /*begin*/ {TensorType_INT32, {1}},
      /*begin_data*/ {-3},
      /*end*/ {TensorType_INT32, {1}},
      /*end_data*/ {3},
      /*strides*/ {TensorType_INT32, {1}},
      /*strides_data*/ {1},
      /*begin_mask*/ 0, /*end_mask*/ 0, /*shrink_axis_mask*/ 0);
  m.SetInput({1, 2, 3, 4});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 3}));
}

TEST(StridedSliceOpModel, In1D_NegativeEnd_Int8) {
  StridedSliceOpModel<int8_t> m(
      /*input=*/{TensorType_INT8, {4}, -10, 10},
      /*output=*/{TensorType_INT8, {1}, -10, 10},
      /*begin*/ {TensorType_INT32, {1}},
      /*begin_data*/ {1},
      /*end*/ {TensorType_INT32, {1}},
      /*end_data*/ {-2},
      /*strides*/ {TensorType_INT32, {1}},
      /*strides_data*/ {1},
      /*begin_mask*/ 0, /*end_mask*/ 0, /*shrink_axis_mask*/ 0);
  m.SetInput({1, 2, 3, 4});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2}));
}

TEST(StridedSliceOpModel, In2D_MultipleStrides_UInt8) {
  StridedSliceOpModel<uint8_t> m(
      /*input=*/{TensorType_UINT8, {2, 3}, -10, 10},
      /*output=*/{TensorType_UINT8, {1, 3}, -10, 10},
      /*begin*/ {TensorType_INT32, {2}},
      /*begin_data*/ {1, -1},
      /*end*/ {TensorType_INT32, {2}},
      /*end_data*/ {2, -4},
      /*strides*/ {TensorType_INT32, {2}},
      /*strides_data*/ {2, -1},
      /*begin_mask*/ 0, /*end_mask*/ 0, /*shrink_axis_mask*/ 0);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({6, 5, 4}));
}

TEST(StridedSliceOpModel, In2D_EndMask_Int8) {
  StridedSliceOpModel<int8_t> m(
      /*input=*/{TensorType_INT8, {2, 3}, -127, 128},
      /*output=*/{TensorType_INT8, {1, 3}, -127, 128},
      /*begin*/ {TensorType_INT32, {2}},
      /*begin_data*/ {1, 0},
      /*end*/ {TensorType_INT32, {2}},
      /*end_data*/ {2, 2},
      /*strides*/ {TensorType_INT32, {2}},
      /*strides_data*/ {1, 1},
      /*begin_mask*/ 0, /*end_mask*/ 2, /*shrink_axis_mask*/ 0);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({4, 5, 6}));
}

TEST(StridedSliceOpModel, In2D_NegStrideBeginMask_UInt8) {
  StridedSliceOpModel<uint8_t> m(
      /*input=*/{TensorType_UINT8, {2, 3}, -10, 10},
      /*output=*/{TensorType_UINT8, {1, 3}, -10, 10},
      /*begin*/ {TensorType_INT32, {2}},
      /*begin_data*/ {1, -2},
      /*end*/ {TensorType_INT32, {2}},
      /*end_data*/ {2, -4},
      /*strides*/ {TensorType_INT32, {2}},
      /*strides_data*/ {1, -1},
      /*begin_mask*/ 2, /*end_mask*/ 0, /*shrink_axis_mask*/ 0);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({6, 5, 4}));
}

TEST(StridedSliceOpModel, In2D_ShrinkAxis2_BeginEndAxis1_NegativeSlice_Int8) {
  StridedSliceOpModel<int8_t> m(
      /*input=*/{TensorType_INT8, {4, 1}, -10, 10},
      /*output=*/{TensorType_INT8, {4}, -10, 10},
      /*begin*/ {TensorType_INT32, {2}},
      /*begin_data*/ {0, -1},
      /*end*/ {TensorType_INT32, {2}},
      /*end_data*/ {0, 0},
      /*strides*/ {TensorType_INT32, {2}},
      /*strides_data*/ {1, 1},
      /*begin_mask*/ 1, /*end_mask*/ 1, /*shrink_axis_mask*/ 2);
  m.SetInput({0, 1, 2, 3});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 1, 2, 3}));
}

TEST(StridedSliceOpModel, In2D_ShrinkAxisMask3_Int8) {
  StridedSliceOpModel<int8_t> m(
      /*input=*/{TensorType_INT8, {2, 3}, -10, 10},
      /*output=*/{TensorType_INT8, {}, -10, 10},
      /*begin*/ {TensorType_INT32, {2}},
      /*begin_data*/ {0, 0},
      /*end*/ {TensorType_INT32, {2}},
      /*end_data*/ {1, 1},
      /*strides*/ {TensorType_INT32, {2}},
      /*strides_data*/ {1, 1},
      /*begin_mask*/ 0, /*end_mask*/ 0, /*shrink_axis_mask*/ 3);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.ApplyDelegateAndInvoke();
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1}));
}

TEST(StridedSliceOpModel, In3D_Identity_UInt8) {
  StridedSliceOpModel<uint8_t> m(
      /*input=*/{TensorType_UINT8, {2, 3, 2}, -15, 15},
      /*output=*/{TensorType_UINT8, {2, 3, 2}, -15, 15},
      /*begin*/ {TensorType_INT32, {3}},
      /*begin_data*/ {0, 0, 0},
      /*end*/ {TensorType_INT32, {3}},
      /*end_data*/ {2, 3, 2},
      /*strides*/ {TensorType_INT32, {3}},
      /*strides_data*/ {1, 1, 1},
      /*begin_mask*/ 0, /*end_mask*/ 0, /*shrink_axis_mask*/ 0);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 2}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
}

TEST(StridedSliceOpModel, In3D_IdentityShrinkAxis4_Int8) {
  StridedSliceOpModel<int8_t> m(
      /*input=*/{TensorType_INT8, {2, 3, 2}, -15, 15},
      /*output=*/{TensorType_INT8, {2, 3, 2}, -15, 15},
      /*begin*/ {TensorType_INT32, {3}},
      /*begin_data*/ {0, 0, 0},
      /*end*/ {TensorType_INT32, {3}},
      /*end_data*/ {2, 3, 1},
      /*strides*/ {TensorType_INT32, {3}},
      /*strides_data*/ {1, 1, 1},
      /*begin_mask*/ 0, /*end_mask*/ 0, /*shrink_axis_mask*/ 4);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3, 5, 7, 9, 11}));
}

}  // namespace tflite
