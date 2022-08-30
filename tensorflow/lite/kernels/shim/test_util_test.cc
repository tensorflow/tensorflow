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
#include "tensorflow/lite/kernels/shim/test_util.h"

#include <string>

#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"

namespace tflite {
namespace {

TEST(TfliteTensorDebugString, Basic) {
  ::tflite::Interpreter interpreter;
  interpreter.AddTensors(3);
  interpreter.AllocateTensors();

  auto t_int32 = UniqueTfLiteTensor(interpreter.tensor(0));
  PopulateTfLiteTensor<int32_t>({1, 2, 3, 4, 5}, {5}, t_int32.get());
  EXPECT_EQ("[1, 2, 3, 4, 5]", TfliteTensorDebugString(t_int32.get()));

  auto t_int64 = UniqueTfLiteTensor(interpreter.tensor(1));
  PopulateTfLiteTensor<int32_t>({1, 2, 3, 4}, {2, 2}, t_int64.get());
  EXPECT_EQ("[[1, 2], [3, 4]]", TfliteTensorDebugString(t_int64.get()));

  auto t_str = UniqueTfLiteTensor(interpreter.tensor(2));
  PopulateTfLiteTensor<std::string>({"ab", "cde", "f"}, {1, 3}, t_str.get());
  EXPECT_EQ("[[ab, cde, f]]", TfliteTensorDebugString(t_str.get()));
}

TEST(TfliteTensorDebugString, MaxVal) {
  ::tflite::Interpreter interpreter;
  interpreter.AddTensors(2);
  interpreter.AllocateTensors();

  auto t_int32 = UniqueTfLiteTensor(interpreter.tensor(0));
  PopulateTfLiteTensor<int32_t>({1, 2, 3, 4}, {4}, t_int32.get());
  EXPECT_EQ("[1, 2, 3, 4]",
            TfliteTensorDebugString(t_int32.get(), /*max_values=*/4));

  t_int32 = UniqueTfLiteTensor(interpreter.tensor(0));
  PopulateTfLiteTensor<int32_t>({1, 2, 3, 4, 5}, {5}, t_int32.get());
  EXPECT_EQ("[1, 2, 3, 4, ...]",
            TfliteTensorDebugString(t_int32.get(), /*max_values=*/4));
}

}  // namespace
}  // namespace tflite
