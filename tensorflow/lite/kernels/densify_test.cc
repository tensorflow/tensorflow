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
#include <cstdint>
#include <initializer_list>

#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/optimize/sparsity/format_converter.h"

namespace tflite {

namespace ops {
namespace builtin {

TfLiteRegistration* Register_DENSIFY();

}  // namespace builtin
}  // namespace ops

namespace {

using ::testing::ElementsAreArray;

template <typename T>
class DensifyOpModel : public SingleOpModel {
 public:
  DensifyOpModel(TensorType type, std::initializer_list<int> shape,
                 std::initializer_list<T> input_data, int version = 1) {
    const TensorData io_tensor_data = {type, shape};
    input_ = AddConstSparseInput(type, shape, input_data);
    output_ = AddOutput(io_tensor_data);

    SetBuiltinOp(BuiltinOperator_DENSIFY, BuiltinOptions_DensifyOptions,
                 CreateDensifyOptions(builder_).Union());

    resolver_ = absl::make_unique<SingleOpResolver>(
        BuiltinOperator_DENSIFY, ops::builtin::Register_DENSIFY(), version);

    BuildInterpreter({shape});
  }

  std::vector<T> GetInput() { return ExtractVector<T>(input_); }
  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }

 private:
  int input_;
  int output_;
};

TEST(DensifyOpTest, Float) {
  std::initializer_list<float> dense_values = {6, 0, 9, 8, 0, 0,
                                               0, 0, 5, 0, 0, 7};
  std::initializer_list<float> sparse_values = {6, 9, 8, 5, 7};
  DensifyOpModel<float> m(TensorType_FLOAT32, {3, 4}, dense_values);
  m.Invoke();
  EXPECT_THAT(m.GetInput(), ElementsAreArray(sparse_values));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(dense_values));
}

TEST(DensifyOpTest, Int8) {
  std::initializer_list<int8_t> dense_values = {6, 0, 9, 8, 0, 0,
                                                0, 0, 5, 0, 0, 7};
  std::initializer_list<int8_t> sparse_values = {6, 9, 8, 5, 7};
  DensifyOpModel<int8_t> m(TensorType_INT8, {3, 4}, dense_values);
  m.Invoke();
  EXPECT_THAT(m.GetInput(), ElementsAreArray(sparse_values));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(dense_values));
}

}  // namespace
}  // namespace tflite
