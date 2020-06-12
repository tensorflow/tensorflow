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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_ADD_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_ADD_H_

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

// Add operation inherited from ElementwiseOperation, but it is more
// complicated than usual elementwise, that is why it has own versions for
// Compile. Add operation support not equal tensors on input (for possibility to
// remove Padding operation with zeroes in Z dimension)
class Add : public ElementwiseOperation {
 public:
  Add(const OperationDef& definition, const std::vector<int>& channels,
      int dst_channels);

  absl::Status Compile(const CreationContext& creation_context) override;

  // Move only
  Add(Add&& operation);
  Add& operator=(Add&& operation);
  Add(const Add&) = delete;
  Add& operator=(const Add&) = delete;

  void SetLinkIndex(int index) override;
  std::string GetCoreCode(const LinkingContext& context) const override;
  std::string GetArgsDeclaration() const override;
  absl::Status BindArguments(CLKernel* kernel) override;
  absl::Status SetArgs(int link_id, Arguments* args) override;
  bool IsLinkable() const override { return dst_depth_ == src_depthes_[0]; }

 private:
  std::string GetElementWiseCode(
      const OperationDef& op_def,
      const std::vector<ElementwiseOperation*>& linked_operations);

  int link_index_;
  std::vector<int> src_depthes_;
  int dst_depth_;
};

Add CreateAdd(const OperationDef& definition, const std::vector<int>& channels,
              int dst_channels);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_ADD_H_
