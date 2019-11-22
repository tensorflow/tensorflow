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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_APPLY_MASK_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_APPLY_MASK_H_

#include "tensorflow/lite/delegates/gpu/cl/cl_kernel.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

class ApplyMask : public ElementwiseOperation {
 public:
  // Move only
  ApplyMask(ApplyMask&& operation);
  ApplyMask& operator=(ApplyMask&& operation);
  ApplyMask(const ApplyMask&) = delete;
  ApplyMask& operator=(const ApplyMask&) = delete;

  void SetLinkIndex(int index) override;
  std::string GetCoreCode(const LinkingContext& context) const override;
  std::string GetArgsDeclaration() const override;
  Status BindArguments(CLKernel* kernel) override;

 private:
  friend ApplyMask CreateApplyMask(const OperationDef& definition,
                                   const BHWC& src_shape,
                                   const BHWC& mask_shape);

  enum class MaskType { LAYER, CHANNELS, TENSOR };

  explicit ApplyMask(const OperationDef& definition, MaskType mask_type)
      : ElementwiseOperation(definition), mask_type_(mask_type) {}

  MaskType mask_type_;
  int link_index_;
};

ApplyMask CreateApplyMask(const OperationDef& definition, const BHWC& src_shape,
                          const BHWC& mask_shape);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_APPLY_MASK_H_
