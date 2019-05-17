/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

template <typename T>
class MatrixDiagOpModel : public SingleOpModel {
 public:
  explicit MatrixDiagOpModel(const TensorData& input) {
    input_ = AddInput(input);
    output_ = AddOutput({input.type, {}});

    SetBuiltinOp(BuiltinOperator_MATRIX_DIAG, BuiltinOptions_MatrixDiagOptions,
                 CreateMatrixDiagOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }
  TfLiteType GetOutputType() {
    TfLiteTensor* t = interpreter_->tensor(output_);
    return t->type;
  }

 private:
  int input_;
  int output_;
};

// Use the machinery of TYPED_TEST_SUITE to test all supported types.
// See
// https://github.com/google/googletest/blob/master/googletest/docs/advanced.md#typed-tests
// for details.
template <typename T>
class MatrixDiagOpTest : public ::testing::Test {};

using TypesUnderTest =
    ::testing::Types<TypeUnion<int32_t>, TypeUnion<float>, TypeUnion<int16_t>,
                     TypeUnion<int8_t>, TypeUnion<uint8_t>>;
TYPED_TEST_SUITE(MatrixDiagOpTest, TypesUnderTest);

TYPED_TEST(MatrixDiagOpTest, ThreeByThreeDiag) {
  MatrixDiagOpModel<typename TypeParam::ScalarType> model(
      {TypeParam::tensor_type, {3}});
  model.template PopulateTensor<typename TypeParam::ScalarType>(model.input(),
                                                                {1, 2, 3});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 3));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 0, 0,  //
                                                   0, 2, 0,  //
                                                   0, 0, 3}));
  EXPECT_THAT(model.GetOutputType(), TypeParam::tflite_type);
}

// Additional special cases.
TEST(MatrixDiagTest, Int32TestTwoDimDiag) {
  MatrixDiagOpModel<int32_t> model({TensorType_INT32, {2, 4}});
  model.PopulateTensor<int32_t>(model.input(), {1, 2, 3, 4, 5, 6, 7, 8});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(2, 4, 4));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 0, 0, 0,  //
                                                   0, 2, 0, 0,  //
                                                   0, 0, 3, 0,  //
                                                   0, 0, 0, 4,  //
                                                   5, 0, 0, 0,  //
                                                   0, 6, 0, 0,  //
                                                   0, 0, 7, 0,  //
                                                   0, 0, 0, 8}));
  EXPECT_THAT(model.GetOutputType(), TfLiteType::kTfLiteInt32);
}

TEST(MatrixDiagTest, DegenenerateCase) {
  MatrixDiagOpModel<uint8_t> model({TensorType_UINT8, {1}});
  model.PopulateTensor<uint8_t>(model.input(), {1});
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1}));
  EXPECT_THAT(model.GetOutputType(), TfLiteType::kTfLiteUInt8);
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
