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

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <ostream>
#include <string>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_mock_test.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

struct NnApiFailureHandlingTest
    : ::tflite::delegate::nnapi::NnApiDelegateMockTest {};

// This is a model with two ops:
//
//  input1 ---->
//                ADD --
//  input2   -->        |
//                       -->
//                          SUB --> output
//  input3 ---------------->
//
class AddSubOpsAcceleratedModel : public MultiOpModel {
 public:
  AddSubOpsAcceleratedModel(const TensorData& input1, const TensorData& input2,
                            const TensorData& input3, const TensorData& output,
                            ActivationFunctionType activation_type,
                            const NnApi* nnapi,
                            const std::string& accelerator_name,
                            bool allow_fp32_relax_to_fp16 = false)
      : MultiOpModel() {
    StatefulNnApiDelegate::Options options;
    options.accelerator_name = accelerator_name.c_str();
    stateful_delegate_ =
        std::make_unique<StatefulNnApiDelegate>(nnapi, options);
    SetDelegate(stateful_delegate_.get());
    Init(input1, input2, input3, output, activation_type,
         allow_fp32_relax_to_fp16);
  }
  ~AddSubOpsAcceleratedModel() { stateful_delegate_.reset(); }

  int input1() { return input1_; }
  int input2() { return input2_; }
  int input3() { return input3_; }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 protected:
  int input1_;
  int input2_;
  int input3_;
  int output_;

 private:
  std::unique_ptr<StatefulNnApiDelegate> stateful_delegate_;

  // Performs initialization logic shared across all constructors.
  void Init(const TensorData& input1, const TensorData& input2,
            const TensorData& input3, const TensorData& output,
            ActivationFunctionType activation_type,
            bool allow_fp32_relax_to_fp16 = false) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    input3_ = AddInput(input3);
    const int add_output = AddInnerTensor<float>(output);
    output_ = AddOutput(output);
    AddBuiltinOp(BuiltinOperator_ADD, BuiltinOptions_AddOptions,
                 CreateAddOptions(builder_, activation_type).Union(),
                 {input1_, input2_}, {add_output});
    AddBuiltinOp(BuiltinOperator_SUB, BuiltinOptions_SubOptions,
                 CreateSubOptions(builder_, activation_type).Union(),
                 {add_output, input3_}, {output_});
    BuildInterpreter({GetShape(input1_), GetShape(input2_), GetShape(input3_)},
                     /*num_threads=*/-1, allow_fp32_relax_to_fp16,
                     /*apply_delegate=*/false);
    ApplyDelegate();
  }
};

TEST_F(NnApiFailureHandlingTest, DelegateShouldFailImmediatelyIfUnableToAddOp) {
  static int add_op_invocation_count = 0;
  nnapi_mock_->SetNnapiSupportedDevice("test-device");

  nnapi_mock_->StubAddOperationWith(
      [](ANeuralNetworksModel* model, ANeuralNetworksOperationType type,
         uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount,
         const uint32_t* outputs) -> int {
        ++add_op_invocation_count;
        return ANEURALNETWORKS_BAD_DATA;
      });

  AddSubOpsAcceleratedModel m(
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {1, 2, 2, 1}},
      {TensorType_FLOAT32, {1, 2, 2, 1}}, {TensorType_FLOAT32, {}},
      ActivationFunctionType_NONE, nnapi_mock_->GetNnApi(),
      /*accelerator_name=*/"test-device");
  std::vector<float> input1{-2.0, 0.2, 0.7, 0.9};
  std::vector<float> input2{0.1, 0.2, 0.3, 0.5};
  m.PopulateTensor<float>(m.input1(), input1);
  m.PopulateTensor<float>(m.input2(), input2);
  m.PopulateTensor<float>(m.input3(), input2);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_EQ(add_op_invocation_count, 1);
}

}  // namespace
}  // namespace tflite
