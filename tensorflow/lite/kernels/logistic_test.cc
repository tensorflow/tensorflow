/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include 

#include 
#include 
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;

// Logistic kernel registration.
TfLiteRegistration* Register_LOGISTIC();

class LogisticOpModel : public SingleOpModel {
 public:
  LogisticOpModel(TensorData input, TensorData output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_LOGISTIC, BuiltinOptions_NONE, 0);
    resolver_ = std::make_unique(BuiltinOperator_LOGISTIC,
                                                   Register_LOGISTIC());
    BuildInterpreter({GetShape(input_)});
  }

  void SetInput(const std::vector& data) {
    PopulateTensor(input_, data);
  }

  std::vector GetOutput() { return ExtractVector(output_); }
};

// Regression test for GitHub Issue #118391: TFLite's float32 Sigmoid/Logistic
// activation must saturate to 0.0 or 1.0 for extreme finite inputs instead of
// returning NaN, which was caused by integer overflow during range-reduction in
// vectorized compiler routines when calculating std::exp(-x).
TEST(LogisticOpTest, FloatLargeInputsSaturate) {
  LogisticOpModel m({TensorType_FLOAT32, {1, 4}}, {TensorType_FLOAT32, {1, 4}});
  m.SetInput({5e29f, 1e25f, -5e29f, -1e25f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ::testing::ElementsAre(
      ::testing::FloatNear(1.0f, 1e-6),
      ::testing::FloatNear(1.0f, 1e-6),
      ::testing::FloatNear(0.0f, 1e-6),
      ::testing::FloatNear(0.0f, 1e-6)
  ));
}

}  // namespace
}  // namespace tflite
