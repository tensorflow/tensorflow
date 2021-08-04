/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/shim/tflite_tensor_view.h"

#include <cstdint>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/shim/test_util.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace shim {
namespace {

using ::testing::Eq;

TEST(TfLiteTensorW, Bool) {
  ::tflite::Interpreter interpreter;
  interpreter.AddTensors(1);
  interpreter.AllocateTensors();
  auto* tflite_tensor = interpreter.tensor(0);
  ReallocDynamicTensor<bool>({3, 2}, tflite_tensor);
  tflite_tensor->name = "test_bool";
  auto owned_tflite_tensor = UniqueTfLiteTensor(tflite_tensor);

  // Test move assignment
  auto t_premove_or = TensorView::New(tflite_tensor);
  ASSERT_TRUE(t_premove_or.ok()) << t_premove_or.status();
  auto t = std::move(t_premove_or.value());

  auto data = t.Data<bool>();
  for (int32_t i = 0; i < 3 * 2; ++i) data[i] = (i % 5 == 0);

  ASSERT_THAT(TfliteTensorDebugString(tflite_tensor),
              Eq("[[1, 0], [0, 0], [0, 1]]"));
}

template <typename IntType>
void IntTest() {
  ::tflite::Interpreter interpreter;
  interpreter.AddTensors(1);
  interpreter.AllocateTensors();
  auto* tflite_tensor = interpreter.tensor(0);
  ReallocDynamicTensor<IntType>({3, 2}, tflite_tensor);
  tflite_tensor->name = "test_int";
  auto owned_tflite_tensor = UniqueTfLiteTensor(tflite_tensor);

  // Test move assignment
  auto t_premove_or = TensorView::New(tflite_tensor);
  ASSERT_TRUE(t_premove_or.ok()) << t_premove_or.status();
  auto t = std::move(t_premove_or.value());

  auto data = t.Data<IntType>();
  for (int32_t i = 0; i < 3 * 2; ++i) data[i] = i;

  ASSERT_THAT(TfliteTensorDebugString(tflite_tensor),
              Eq("[[0, 1], [2, 3], [4, 5]]"));
}

TEST(TfLiteTensorW, Int8) { IntTest<int8_t>(); }
TEST(TfLiteTensorW, UInt8) { IntTest<uint8_t>(); }
TEST(TfLiteTensorW, Int16) { IntTest<int16_t>(); }
TEST(TfLiteTensorW, Int32) { IntTest<int32_t>(); }
TEST(TfLiteTensorW, Int64) { IntTest<int64_t>(); }

template <typename FloatType>
void FloatTest() {
  ::tflite::Interpreter interpreter;
  interpreter.AddTensors(1);
  interpreter.AllocateTensors();
  auto* tflite_tensor = interpreter.tensor(0);
  ReallocDynamicTensor<FloatType>({3, 2}, tflite_tensor);
  tflite_tensor->name = "test_float";
  auto owned_tflite_tensor = UniqueTfLiteTensor(tflite_tensor);

  auto t_or = TensorView::New(tflite_tensor);
  ASSERT_TRUE(t_or.ok()) << t_or.status();
  auto& t = t_or.value();

  auto data = t.Data<FloatType>();
  for (int32_t i = 0; i < 3 * 2; ++i) data[i] = static_cast<FloatType>(i) / 2.;

  ASSERT_THAT(TfliteTensorDebugString(tflite_tensor),
              Eq("[[0, 0.5], [1, 1.5], [2, 2.5]]"));
}

TEST(TfLiteTensorW, Float) { FloatTest<float>(); }
TEST(TfLiteTensorW, Double) { FloatTest<double>(); }

TEST(TfLiteTensorW, Str) {
  ::tflite::Interpreter interpreter;
  interpreter.AddTensors(1);
  interpreter.AllocateTensors();
  auto* tflite_tensor = interpreter.tensor(0);
  ReallocDynamicTensor<std::string>({3, 2}, tflite_tensor);
  tflite_tensor->name = "test_str";
  auto owned_tflite_tensor = UniqueTfLiteTensor(tflite_tensor);

  {
    auto t_or = TensorView::New(tflite_tensor);
    ASSERT_TRUE(t_or.ok()) << t_or.status();
    auto& t = t_or.value();
    auto t_mat = t.As<::tensorflow::tstring, 2>();
    t.Data<::tensorflow::tstring>()[0] = "a";
    t.Data<::tensorflow::tstring>()[1] = "bc";
    t_mat(1, 0) = "def";
    t.Data<::tensorflow::tstring>()[3] = "g";
    t.Data<::tensorflow::tstring>()[4] = "";
    t_mat(2, 1) = "hi";
  }

  {
    auto t_or = TensorView::New(tflite_tensor);
    ASSERT_TRUE(t_or.ok()) << t_or.status();
    auto& t = t_or.value();
    EXPECT_THAT(t.Data<::tensorflow::tstring>(),
                ::testing::ElementsAre("a", "bc", "def", "g", "", "hi"));
  }

  const auto const_tflite_tensor = tflite_tensor;
  {
    const auto t_or = TensorView::New(const_tflite_tensor);
    ASSERT_TRUE(t_or.ok()) << t_or.status();
    const auto& t = t_or.value();
    EXPECT_THAT(t.Data<::tensorflow::tstring>(),
                ::testing::ElementsAre("a", "bc", "def", "g", "", "hi"));
  }

  EXPECT_THAT(TfliteTensorDebugString(tflite_tensor),
              Eq("[[a, bc], [def, g], [, hi]]"));
}

TEST(TfLiteTensorW, EmptyStr) {
  ::tflite::Interpreter interpreter;
  interpreter.AddTensors(1);
  interpreter.AllocateTensors();
  auto* tflite_tensor = interpreter.tensor(0);
  ReallocDynamicTensor<std::string>(/*shape=*/{0}, tflite_tensor);
  tflite_tensor->name = "test_str";
  auto owned_tflite_tensor = UniqueTfLiteTensor(tflite_tensor);

  // Placing tensor_view instance in a block to ensure its dtor runs
  {
    auto t_or = TensorView::New(tflite_tensor);
    ASSERT_TRUE(t_or.ok()) << t_or.status();
  }

  EXPECT_THAT(GetStringCount(tflite_tensor), Eq(0));
}

}  // namespace
}  // namespace shim
}  // namespace tflite
