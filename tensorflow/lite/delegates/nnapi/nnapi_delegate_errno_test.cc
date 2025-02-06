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
#include <sys/mman.h>

#include <memory>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_mock_test.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

class SingleOpModelWithNNAPI : public SingleOpModel {
 public:
  explicit SingleOpModelWithNNAPI(const NnApi* nnapi) {
    options_.disallow_nnapi_cpu = false;
    stateful_delegate_ =
        std::make_unique<StatefulNnApiDelegate>(nnapi, options_);
    this->SetDelegate(stateful_delegate_.get());
  }
  ~SingleOpModelWithNNAPI() { stateful_delegate_.reset(); }

  StatefulNnApiDelegate* GetDelegate() { return stateful_delegate_.get(); }

  void SetBufferHandle(int index, TfLiteBufferHandle handle) {
    interpreter_->SetBufferHandle(index, handle, stateful_delegate_.get());
  }

 private:
  std::unique_ptr<StatefulNnApiDelegate> stateful_delegate_;
  StatefulNnApiDelegate::Options options_;
};

class FloatAddOpModel : public SingleOpModelWithNNAPI {
 public:
  FloatAddOpModel(const NnApi* nnapi, const TensorData& input1,
                  const TensorData& input2, const TensorData& output,
                  ActivationFunctionType activation_type,
                  bool allow_fp32_relax_to_fp16 = false)
      : SingleOpModelWithNNAPI(nnapi) {
    Init(input1, input2, output, activation_type, allow_fp32_relax_to_fp16);
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
            const TensorData& output, ActivationFunctionType activation_type,
            bool allow_fp32_relax_to_fp16 = false) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_ADD, BuiltinOptions_AddOptions,
                 CreateAddOptions(builder_, activation_type).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)}, /*num_threads=*/-1,
                     allow_fp32_relax_to_fp16, /*apply_delegate=*/true);
  }
};

struct NnApiErrnoTest : ::tflite::delegate::nnapi::NnApiDelegateMockTest {};

TEST_F(NnApiErrnoTest, IsZeroWhenNoErrorOccurs) {
  FloatAddOpModel m(nnapi_mock_->GetNnApi(), {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_EQ(m.GetDelegate()->GetNnApiErrno(), 0);
}

TEST_F(NnApiErrnoTest, HasTheStatusOfTheNnApiCallFailedCallingInit) {
  nnapi_mock_->ExecutionCreateReturns<8>();

  FloatAddOpModel m(nnapi_mock_->GetNnApi(), {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);

  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});

  EXPECT_EQ(m.Invoke(), kTfLiteError);
  EXPECT_EQ(m.GetDelegate()->GetNnApiErrno(), 8);
}

TEST_F(NnApiErrnoTest, HasTheStatusOfTheNnApiCallFailedCallingInvoke) {
  nnapi_mock_->ModelFinishReturns<-4>();

  FloatAddOpModel m(nnapi_mock_->GetNnApi(), {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);

  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});

  // Failure is detected and the delegate is disabled.
  // Execution runs without it and succeeds
  EXPECT_EQ(m.Invoke(), kTfLiteOk);
  // The delegate should store the value of the failure
  EXPECT_EQ(m.GetDelegate()->GetNnApiErrno(), -4);
}

TEST_F(NnApiErrnoTest, ErrnoIsResetWhenRestoringDelegateForModel) {
  nnapi_mock_->ModelFinishReturns<-4>();

  FloatAddOpModel m(nnapi_mock_->GetNnApi(), {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);

  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});

  // Failure is detected and the delegate is disabled.
  // Execution runs without it and succeeds
  EXPECT_EQ(m.Invoke(), kTfLiteOk);
  // The delegate should store the value of the failure
  EXPECT_EQ(m.GetDelegate()->GetNnApiErrno(), -4);

  nnapi_mock_->ModelFinishReturns<0>();

  // Need to restore the delegate since it was disabled because of the
  // previous failure.
  m.ApplyDelegate();
  EXPECT_EQ(m.Invoke(), kTfLiteOk);

  // The error is still the last one recorded
  EXPECT_EQ(m.GetDelegate()->GetNnApiErrno(), 0);
}

TEST_F(NnApiErrnoTest, ErrnoIsUpdatedInCaseOfAnotherFailure) {
  nnapi_mock_->ModelFinishReturns<-4>();

  FloatAddOpModel m(nnapi_mock_->GetNnApi(), {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);

  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});

  // Failure is detected and the delegate is disabled.
  // Execution runs without it and succeeds
  EXPECT_EQ(m.Invoke(), kTfLiteOk);
  // The delegate should store the value of the failure
  EXPECT_EQ(m.GetDelegate()->GetNnApiErrno(), -4);

  nnapi_mock_->ModelFinishReturns<-5>();

  // Need to restore the delegate since it was disabled because of the
  // previous failure.
  m.ApplyDelegate();
  EXPECT_EQ(m.Invoke(), kTfLiteOk);

  // The error is still the last one recorded
  EXPECT_EQ(m.GetDelegate()->GetNnApiErrno(), -5);
}

}  // namespace
}  // namespace tflite
