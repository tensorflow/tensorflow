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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename T>
class BaseMirrorPadOpModel : public SingleOpModel {
 public:
  BaseMirrorPadOpModel(const TensorData& input,
                       const TensorData& padding_matrix,
                       const TensorData& output,
                       const tflite::MirrorPadMode mode) {
    input_id_ = AddInput(input);
    padding_matrix_id_ = AddInput(padding_matrix);
    output_id_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_MIRROR_PAD, BuiltinOptions_MirrorPadOptions,
                 CreateMirrorPadOptions(builder_, mode).Union());
    BuildInterpreter({GetShape(input_id_), GetShape(padding_matrix_id_)});
  }

  int input_tensor_id() { return input_id_; }
  int padding_matrix_tensor_id() { return padding_matrix_id_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_id_); }

 protected:
  int input_id_;
  int padding_matrix_id_;
  int output_id_;
};

TEST(MirrorPadTest, EmptyPad) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {2, 3}}, {TensorType_INT32, {2, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_REFLECT);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {0, 0, 0, 0});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(MirrorPadTest, PadOneSide_right_Reflect) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {2, 3}}, {TensorType_INT32, {2, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_REFLECT);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {0, 1, 0, 1});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1, 2, 3, 2, 4, 5, 6, 5, 1, 2, 3, 2}));
}

TEST(MirrorPadTest, PadOneSide_left_Reflect) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {2, 3}}, {TensorType_INT32, {2, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_REFLECT);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {1, 0, 1, 0});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({5, 4, 5, 6, 2, 1, 2, 3, 5, 4, 5, 6}));
}

TEST(MirrorPadTest, PadOneSide_right_Symmetric) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {2, 3}}, {TensorType_INT32, {2, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_SYMMETRIC);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {0, 1, 0, 1});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1, 2, 3, 3, 4, 5, 6, 6, 4, 5, 6, 6}));
}

TEST(MirrorPadTest, PadOneSide_left_Symmetric) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {2, 3}}, {TensorType_INT32, {2, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_SYMMETRIC);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {1, 0, 1, 0});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1, 1, 2, 3, 1, 1, 2, 3, 4, 4, 5, 6}));
}

TEST(MirrorPadTest, PadBothSides_Symmetric) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {2, 3}}, {TensorType_INT32, {2, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_SYMMETRIC);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {1, 1, 1, 1});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1, 1, 2, 3, 3, 1, 1, 2, 3, 3,
                                4, 4, 5, 6, 6, 4, 4, 5, 6, 6}));
}

TEST(MirrorPadTest, PadBothSides_Reflect) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {2, 3}}, {TensorType_INT32, {2, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_REFLECT);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {1, 1, 1, 1});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({5, 4, 5, 6, 5, 2, 1, 2, 3, 2,
                                5, 4, 5, 6, 5, 2, 1, 2, 3, 2}));
}

TEST(MirrorPadTest, PadBothSides_Symmetric_Whole) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {2, 3}}, {TensorType_INT32, {2, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_SYMMETRIC);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {2, 2, 3, 3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({6, 5, 4, 4, 5, 6, 6, 5, 4, 3, 2, 1, 1, 2, 3, 3, 2, 1,
                        3, 2, 1, 1, 2, 3, 3, 2, 1, 6, 5, 4, 4, 5, 6, 6, 5, 4,
                        6, 5, 4, 4, 5, 6, 6, 5, 4, 3, 2, 1, 1, 2, 3, 3, 2, 1}));
}

TEST(MirrorPadTest, PadBothSides_Reflect_Whole) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {2, 3}}, {TensorType_INT32, {2, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_REFLECT);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {1, 1, 2, 2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1,
                                6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1}));
}

TEST(MirrorPadTest, Pad_Symmetric) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {2, 3}}, {TensorType_INT32, {2, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_SYMMETRIC);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {1, 1, 2, 2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({2, 1, 1, 2, 3, 3, 2, 2, 1, 1, 2, 3, 3, 2,
                                5, 4, 4, 5, 6, 6, 5, 5, 4, 4, 5, 6, 6, 5}));
}

TEST(MirrorPadTest, Pad_1D_Reflect) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {3}}, {TensorType_INT32, {1, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_REFLECT);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {0, 2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 2, 1}));
}

TEST(MirrorPadTest, Pad_1D_Symmetric) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {3}}, {TensorType_INT32, {1, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_SYMMETRIC);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {0, 2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 3, 2}));
}

TEST(MirrorPadTest, Pad_1D_Symmetric_Multiple_Invoke) {
  BaseMirrorPadOpModel<int> model(
      {TensorType_INT32, {3}}, {TensorType_INT32, {1, 2}},
      {TensorType_INT32, {}}, tflite::MirrorPadMode_SYMMETRIC);
  model.PopulateTensor<int>(model.input_tensor_id(), {1, 2, 3});
  model.PopulateTensor<int>(model.padding_matrix_tensor_id(), {0, 2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 2, 3, 3, 2}));
  model.PopulateTensor<int>(model.input_tensor_id(), {4, 5, 6});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({4, 5, 6, 6, 5}));
}

}  // namespace
}  // namespace tflite
