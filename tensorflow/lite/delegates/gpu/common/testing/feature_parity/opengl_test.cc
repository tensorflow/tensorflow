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

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/common/testing/feature_parity/feature_parity.h"
#include "tensorflow/lite/delegates/gpu/common/testing/feature_parity/utils.h"
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"

namespace tflite {

class OpenGLBackend : public testing::TestWithParam<TestParams> {};

TEST_P(OpenGLBackend, DelegateTest) {
  TfLiteGpuDelegateOptions options = TfLiteGpuDelegateOptions();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteGpuDelegateDelete)>
      gl_delegate(TfLiteGpuDelegateCreate(&options), TfLiteGpuDelegateDelete);
  const TestParams& param = GetParam();
  const std::vector<uint8_t>& model_binary = param.model;
  ASSERT_NE(model_binary.empty(), true);
  std::unique_ptr<Interpreter> tflite_cpu_interpreter;
  ASSERT_OK(
      BuildInterpreter(GetModel(model_binary.data()), &tflite_cpu_interpreter));
  ASSERT_OK(AllocateTensors(&tflite_cpu_interpreter));
  std::unique_ptr<Interpreter> opengl_interpreter;
  ASSERT_OK(
      BuildInterpreter(GetModel(model_binary.data()), &opengl_interpreter));
  // Ensures that tensors are allocated.
  ASSERT_OK(ModifyGraphWithDelegate(&opengl_interpreter, gl_delegate.get()));
  // Inputs are initialized with consequent values of the fixed range.
  InitializeInputs(/*left=*/0, /*right=*/100, &tflite_cpu_interpreter);
  InitializeInputs(/*left=*/0, /*right=*/100, &opengl_interpreter);
  ASSERT_OK(Invoke(&tflite_cpu_interpreter));
  ASSERT_OK(Invoke(&opengl_interpreter));
  for (int i = 0; i < tflite_cpu_interpreter->outputs().size(); ++i) {
    int id = tflite_cpu_interpreter->outputs()[i];
    const TfLiteTensor* cpu = tflite_cpu_interpreter->tensor(id);
    const TfLiteTensor* gpu = opengl_interpreter->tensor(id);
    EXPECT_THAT(*gpu, TensorEq(testing::FloatNear(1e-6), *cpu))
        << " for output tensor #" << i << " with id " << id;
  }
}

INSTANTIATE_TEST_SUITE_P(FeatureParityTests, OpenGLBackend,
                         testing::ValuesIn(GetFeatureParity()),
                         [](const testing::TestParamInfo<TestParams>& info) {
                           return info.param.name;
                         });

}  // namespace tflite
