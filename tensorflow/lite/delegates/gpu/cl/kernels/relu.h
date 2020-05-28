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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_RELU_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_RELU_H_

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/flt_type.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"

namespace tflite {
namespace gpu {
namespace cl {

class ReLU : public ElementwiseOperation {
 public:
  // Move only
  ReLU(ReLU&& operation);
  ReLU& operator=(ReLU&& operation);
  ReLU(const ReLU&) = delete;
  ReLU& operator=(const ReLU&) = delete;

  void SetLinkIndex(int index) override;
  std::string GetCoreCode(const LinkingContext& context) const override;
  std::string GetArgsDeclaration() const override;
  absl::Status BindArguments(CLKernel* kernel) override;

  friend ReLU CreateReLU(const CreationContext& creation_context,
                         const OperationDef& definition,
                         const ReLUAttributes& attr);

 private:
  ReLU(const OperationDef& definition, const ReLUAttributes& attr,
       CalculationsPrecision scalar_precision);

  FLT alpha_;
  FLT clip_;
};

ReLU CreateReLU(const CreationContext& creation_context,
                const OperationDef& definition, const ReLUAttributes& attr);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_RELU_H_
