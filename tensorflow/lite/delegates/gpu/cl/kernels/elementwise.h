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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_ELEMENTWISE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_ELEMENTWISE_H_

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"

namespace tflite {
namespace gpu {
namespace cl {

// Class for simple one input operations without any parameters, for example
// log, sin, cos and etc.
class ElementwiseOneInput : public ElementwiseOperation {
 public:
  explicit ElementwiseOneInput(const OperationDef& definition,
                               const OperationType& op_type)
      : ElementwiseOperation(definition), op_type_(op_type) {}

  // Move only
  ElementwiseOneInput(ElementwiseOneInput&& operation);
  ElementwiseOneInput& operator=(ElementwiseOneInput&& operation);
  ElementwiseOneInput(const ElementwiseOneInput&) = delete;
  ElementwiseOneInput& operator=(const ElementwiseOneInput&) = delete;

  std::string GetCoreCode(const LinkingContext& context) const override;

 private:
  OperationType op_type_;
};

ElementwiseOneInput CreateElementwiseOneInput(const OperationDef& definition,
                                              const OperationType& op_type);

// Class for simple two input operations without any parameters, for example
// sub, div and etc.
class ElementwiseTwoInput : public ElementwiseOperation {
 public:
  explicit ElementwiseTwoInput(const OperationDef& definition,
                               const OperationType& op_type)
      : ElementwiseOperation(definition), op_type_(op_type) {}

  // Move only
  ElementwiseTwoInput(ElementwiseTwoInput&& operation);
  ElementwiseTwoInput& operator=(ElementwiseTwoInput&& operation);
  ElementwiseTwoInput(const ElementwiseTwoInput&) = delete;
  ElementwiseTwoInput& operator=(const ElementwiseTwoInput&) = delete;

  void SetLinkIndex(int index) override;
  std::string GetCoreCode(const LinkingContext& context) const override;
  std::string GetArgsDeclaration() const override;
  Status BindArguments(CLKernel* kernel) override;

 private:
  int link_index_;
  OperationType op_type_;
};

ElementwiseTwoInput CreateElementwiseTwoInput(const OperationDef& definition,
                                              const OperationType& op_type);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_ELEMENTWISE_H_
