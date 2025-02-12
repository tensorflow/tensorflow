/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_c_api.h"

#include <sys/mman.h>

#include <algorithm>
#include <initializer_list>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class SingleOpModelWithNnapiDelegateCApi : public SingleOpModel {
 public:
  SingleOpModelWithNnapiDelegateCApi() {
    options_ = TfLiteNnapiDelegateOptionsDefault();
    options_.disallow_nnapi_cpu = false;
  }

  explicit SingleOpModelWithNnapiDelegateCApi(
      const TfLiteNnapiDelegateOptions& options) {
    options_ = options;
    options_.disallow_nnapi_cpu = false;
  }

  ~SingleOpModelWithNnapiDelegateCApi() {
    if (nnapi_delegate_) {
      TfLiteNnapiDelegateDelete(nnapi_delegate_);
    }
    nnapi_delegate_ = nullptr;
  }

 protected:
  void BuildInterpreterWithNNAPI(std::vector<std::vector<int>> input_shapes) {
    if (nnapi_delegate_) {
      TfLiteNnapiDelegateDelete(nnapi_delegate_);
    }
    nnapi_delegate_ = TfLiteNnapiDelegateCreate(&options_);
    SetDelegate(nnapi_delegate_);
    BuildInterpreter(input_shapes, /*num_threads=*/-1, options_.allow_fp16,
                     /*apply_delegate=*/true, /*allocate_and_delegate=*/true);
  }

 private:
  TfLiteNnapiDelegateOptions options_;
  TfLiteDelegate* nnapi_delegate_ = nullptr;
};

class FloatAddOpModel : public SingleOpModelWithNnapiDelegateCApi {
 public:
  FloatAddOpModel(const TensorData& input1, const TensorData& input2,
                  const TensorData& output,
                  ActivationFunctionType activation_type) {
    Init(input1, input2, output, activation_type);
  }

  FloatAddOpModel(const TfLiteNnapiDelegateOptions& options,
                  const TensorData& input1, const TensorData& input2,
                  const TensorData& output,
                  ActivationFunctionType activation_type)
      : SingleOpModelWithNnapiDelegateCApi(options) {
    Init(input1, input2, output, activation_type);
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 protected:
  int input1_;
  int input2_;
  int output_;

 private:
  // Performs initialization logic shared across all constructors.
  void Init(const TensorData& input1, const TensorData& input2,
            const TensorData& output, ActivationFunctionType activation_type) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_ADD, BuiltinOptions_AddOptions,
                 CreateAddOptions(builder_, activation_type).Union());
    BuildInterpreterWithNNAPI({GetShape(input1_), GetShape(input2_)});
  }
};

// Basic test for the NNAPI delegate C APIs.
TEST(NNAPIDelegate, C_API) {
  TfLiteNnapiDelegateOptions options = TfLiteNnapiDelegateOptionsDefault();
  options.execution_preference =
      TfLiteNnapiDelegateOptions::ExecutionPreference::kLowPower;

  FloatAddOpModel m(options, {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1.9, 0.4, 1.0, 1.3}));
}

// Basic test for the NNAPI delegate C API with accelerator_name specified.
TEST(NNAPIDelegate, C_API_WithAcceleratorName) {
  TfLiteNnapiDelegateOptions options = TfLiteNnapiDelegateOptionsDefault();
  options.execution_preference =
      TfLiteNnapiDelegateOptions::ExecutionPreference::kLowPower;
  options.accelerator_name = "nnapi-reference";

  FloatAddOpModel m(options, {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1.9, 0.4, 1.0, 1.3}));
}

// Basic test for the NNAPI delegate C API with compilation caching enabled.
TEST(NNAPIDelegate, C_API_WithCompilationCaching) {
  TfLiteNnapiDelegateOptions options = TfLiteNnapiDelegateOptionsDefault();
  options.execution_preference =
      TfLiteNnapiDelegateOptions::ExecutionPreference::kLowPower;
  options.cache_dir = "/data/local/tmp";
  options.model_token = "NNAPIDelegate.C_API_WithCompilationCaching";

  // 1st run
  {
    FloatAddOpModel m(options, {TensorType_FLOAT32, {1, 2, 2, 1}},
                      {TensorType_FLOAT32, {1, 2, 2, 1}},
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
    m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1.9, 0.4, 1.0, 1.3}));
  }
  // 2nd run
  {
    FloatAddOpModel m(options, {TensorType_FLOAT32, {1, 2, 2, 1}},
                      {TensorType_FLOAT32, {1, 2, 2, 1}},
                      {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
    m.PopulateTensor<float>(m.input1(), {-1.0, 0.1, 0.7, 0.8});
    m.PopulateTensor<float>(m.input2(), {0.2, 0.2, 0.4, 0.2});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({-0.8, 0.3, 1.1, 1.0}));
  }
}
}  // namespace
}  // namespace tflite
