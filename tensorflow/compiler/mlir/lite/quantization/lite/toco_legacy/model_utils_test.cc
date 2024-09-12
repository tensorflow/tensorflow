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
// This file is the MLIR copy of part of
// third_party/tensorflow/lite/tools/optimize/model_utils_test.cc as part of the
// effort to decouple TFLite from MLIR.

#include "tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy/model_utils.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"

namespace mlir {
namespace lite {
namespace toco_legacy {

namespace {

using std::string;

// LINT.IfChange(HasMinMaxTest)
TEST(ModelUtilsTest, HasMinMax) {
  tflite::TensorT tensor;
  tensor.quantization = std::make_unique<tflite::QuantizationParametersT>();
  tensor.quantization->min.push_back(0.5);
  EXPECT_FALSE(mlir::lite::toco_legacy::HasMinMax(&tensor));
  tensor.quantization->max.push_back(1.5);
  EXPECT_TRUE(mlir::lite::toco_legacy::HasMinMax(&tensor));
}
// LINT.ThenChange(//tensorflow/lite/tools/optimize/model_utils_test.cc:HasMinMaxTest)

}  // namespace

}  // namespace toco_legacy
}  // namespace lite
}  // namespace mlir
