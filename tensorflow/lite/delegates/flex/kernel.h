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
#ifndef TENSORFLOW_LITE_DELEGATES_FLEX_KERNEL_H_
#define TENSORFLOW_LITE_DELEGATES_FLEX_KERNEL_H_

#include <memory>

#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"

namespace tflite {
namespace flex {

namespace testing {
class KernelTest;  // friend class declaration.
}  // namespace testing

struct OpData;
struct OpNode;

class DelegateKernel : public SimpleDelegateKernelInterface {
 public:
  DelegateKernel();
  ~DelegateKernel() override;

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override;
  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override;
  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override;

 private:
  friend class tflite::flex::testing::KernelTest;

  // Validates that the computed output tensor shape for the Flex node matches
  // the existing output shape assigned to the output tensor.
  TfLiteStatus ValidateOutputTensorShapeConsistency(
      TfLiteContext* context) const;

  // Executes the Tensorflow op based on the inputs/outputs/attributes
  // information represented in the `node_data`.
  tensorflow::Status ExecuteFlexOp(TfLiteContext* context, OpNode* node_data);

  // Returns the tensor release map held in `op_data_`;
  const std::map<int, int>& GetTensorReleaseMap() const;

  std::unique_ptr<OpData> op_data_;
};

}  // namespace flex
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_FLEX_KERNEL_H_
