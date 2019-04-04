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
#include "tensorflow/lite/tools/optimize/model_utils.h"
#include <memory>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/optimize/test_util.h"

namespace tflite {
namespace optimize {
namespace utils {
namespace {

TEST(ModelUtilsTest, QuantizationParametersExist) {
  TensorT tensor;
  tensor.quantization = absl::make_unique<QuantizationParametersT>();
  tensor.quantization->scale.push_back(0.5);
  tensor.quantization->scale.push_back(1.5);
  EXPECT_FALSE(QuantizationParametersExist(&tensor));
  tensor.quantization->zero_point.push_back(1);
  tensor.quantization->zero_point.push_back(-1);
  EXPECT_TRUE(QuantizationParametersExist(&tensor));
}

}  // namespace
}  // namespace utils
}  // namespace optimize
}  // namespace tflite

int main(int argc, char** argv) { return RUN_ALL_TESTS(); }
