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
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

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
  DensifyOpModel(const TensorData& input, std::initializer_list<T> input_data,
                 int version = 1) {
    input_ = AddConstSparseInput(input, input_data);
    output_ = AddOutput({input.type, input.shape});

    SetBuiltinOp(BuiltinOperator_DENSIFY, BuiltinOptions_DensifyOptions,
                 CreateDensifyOptions(builder_).Union());

    resolver_ = absl::make_unique<SingleOpResolver>(
        BuiltinOperator_DENSIFY, ops::builtin::Register_DENSIFY(), version);

    BuildInterpreter({input.shape});
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
  TensorData input = {};
  input.type = TensorType_FLOAT32;
  input.shape = {3, 4};
  input.traversal_order = {0, 1};
  input.format = {kTfLiteDimDense, kTfLiteDimSparseCSR};
  DensifyOpModel<float> m(input, dense_values);
  m.Invoke();
  EXPECT_THAT(m.GetInput(), ElementsAreArray(sparse_values));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(dense_values));
}

TEST(DensifyOpTest, Float3D) {
  std::initializer_list<float> dense_values = {6, 0, 9, 8, 0, 0,
                                               0, 0, 5, 0, 0, 7};
  std::initializer_list<float> sparse_values = {6, 9, 8, 5, 7};
  TensorData input = {};
  input.type = TensorType_FLOAT32;
  input.shape = {3, 2, 2};
  input.traversal_order = {0, 1, 2};
  input.format = {kTfLiteDimDense, kTfLiteDimDense, kTfLiteDimSparseCSR};
  DensifyOpModel<float> m(input, dense_values);
  m.Invoke();
  EXPECT_THAT(m.GetInput(), ElementsAreArray(sparse_values));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(dense_values));
}

TEST(DensifyOpTest, Int8) {
  std::initializer_list<int8_t> dense_values = {6, 0, 9, 8, 0, 0,
                                                0, 0, 5, 0, 0, 7};
  std::initializer_list<int8_t> sparse_values = {6, 9, 8, 5, 7};
  TensorData input = {};
  input.type = TensorType_INT8;
  input.shape = {3, 4};
  input.traversal_order = {0, 1};
  input.format = {kTfLiteDimDense, kTfLiteDimSparseCSR};
  DensifyOpModel<int8_t> m(input, dense_values);
  m.Invoke();
  EXPECT_THAT(m.GetInput(), ElementsAreArray(sparse_values));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(dense_values));
}

}  // namespace
}  // namespace tflite
