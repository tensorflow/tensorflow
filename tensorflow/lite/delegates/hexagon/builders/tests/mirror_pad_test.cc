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
#include <initializer_list>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
using testing::ElementsAreArray;

template <typename T>
class MirrorPadOpModel : public SingleOpModelWithHexagon {
 public:
  MirrorPadOpModel(const TensorData& input,
                   std::initializer_list<int> paddings_shape,
                   std::initializer_list<int> paddings,
                   const TensorData& output, const tflite::MirrorPadMode mode) {
    input_id_ = AddInput(input);
    padding_matrix_id_ =
        AddConstInput(TensorType_INT32, paddings, paddings_shape);
    output_id_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_MIRROR_PAD, BuiltinOptions_MirrorPadOptions,
                 CreateMirrorPadOptions(builder_, mode).Union());
    BuildInterpreter({GetShape(input_id_), GetShape(padding_matrix_id_)});
  }

  int input_tensor_id() { return input_id_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_id_); }

 protected:
  int input_id_;
  int padding_matrix_id_;
  int output_id_;
};

TEST(MirrorPadTest, EmptyPad_UInt8) {
  MirrorPadOpModel<uint8_t> model(
      {TensorType_UINT8, {2, 3}, -1.0, 1.0}, {2, 2}, {0, 0, 0, 0},
      {TensorType_UINT8, {}, -1.0, 1.0}, tflite::MirrorPadMode_REFLECT);
  model.PopulateTensor<uint8_t>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(MirrorPadTest, PadBothSides_Symmetric_Int8) {
  MirrorPadOpModel<int8_t> model({TensorType_INT8, {2, 3}, -1.0, 1.0}, {2, 2},
                                 {1, 1, 1, 1}, {TensorType_INT8, {}, -1.0, 1.0},
                                 tflite::MirrorPadMode_SYMMETRIC);
  model.PopulateTensor<int8_t>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1, 1, 2, 3, 3, 1, 1, 2, 3, 3,
                                4, 4, 5, 6, 6, 4, 4, 5, 6, 6}));
}

TEST(MirrorPadTest, PadBothSides_Reflect_UInt8) {
  MirrorPadOpModel<uint8_t> model(
      {TensorType_UINT8, {2, 3}, -1.0, 1.0}, {2, 2}, {1, 1, 1, 1},
      {TensorType_UINT8, {}, -1.0, 1.0}, tflite::MirrorPadMode_REFLECT);
  model.PopulateTensor<uint8_t>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({5, 4, 5, 6, 5, 2, 1, 2, 3, 2,
                                5, 4, 5, 6, 5, 2, 1, 2, 3, 2}));
}

TEST(MirrorPadTest, PadOneSide_left_Reflect_Int8) {
  MirrorPadOpModel<int8_t> model({TensorType_INT8, {2, 3}, -1.0, 1.0}, {2, 2},
                                 {1, 0, 1, 0}, {TensorType_INT8, {}, -1.0, 1.0},
                                 tflite::MirrorPadMode_REFLECT);
  model.PopulateTensor<int8_t>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({5, 4, 5, 6, 2, 1, 2, 3, 5, 4, 5, 6}));
}

TEST(MirrorPadTest, PadOneSide_right_Symmetric_UInt8) {
  MirrorPadOpModel<uint8_t> model(
      {TensorType_UINT8, {2, 3}, -1.0, 1.0}, {2, 2}, {0, 1, 0, 1},
      {TensorType_UINT8, {}, -1.0, 1.0}, tflite::MirrorPadMode_SYMMETRIC);
  model.PopulateTensor<uint8_t>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1, 2, 3, 3, 4, 5, 6, 6, 4, 5, 6, 6}));
}

TEST(MirrorPadTest, Pad_1D_Reflect_Int8) {
  MirrorPadOpModel<int8_t> model({TensorType_INT8, {3}, -1.0, 1.0}, {1, 2},
                                 {0, 2}, {TensorType_INT8, {}, -1.0, 1.0},
                                 tflite::MirrorPadMode_REFLECT);
  model.PopulateTensor<int8_t>(model.input_tensor_id(), {1, 2, 3});
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 2, 1}));
}

TEST(MirrorPadTest, Pad_1D_Symmetric_UInt8) {
  MirrorPadOpModel<uint8_t> model({TensorType_UINT8, {3}, -1.0, 1.0}, {1, 2},
                                  {0, 2}, {TensorType_UINT8, {}, -1.0, 1.0},
                                  tflite::MirrorPadMode_SYMMETRIC);
  model.PopulateTensor<uint8_t>(model.input_tensor_id(), {1, 2, 3});
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 3, 2}));
}

TEST(MirrorPadTest, PadBothSides_Reflect_Whole_UInt8) {
  MirrorPadOpModel<uint8_t> model(
      {TensorType_UINT8, {2, 3}, -1.0, 1.0}, {2, 2}, {1, 1, 2, 2},
      {TensorType_UINT8, {}, -1.0, 1.0}, tflite::MirrorPadMode_REFLECT);
  model.PopulateTensor<uint8_t>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1,
                                6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1}));
}

}  // namespace tflite
