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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_HARD_SWISH_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_HARD_SWISH_H_

#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"

namespace tflite {
namespace gpu {
namespace cl {

class HardSwish : public ElementwiseOperation {
 public:
  static std::unique_ptr<HardSwish> Create(const OperationDef& op_def) {
    auto h_swish = absl::make_unique<HardSwish>(op_def);
    h_swish->SetLinkIndex(0);
    return h_swish;
  }

  HardSwish() = delete;
  explicit HardSwish(const OperationDef& op_def)
      : ElementwiseOperation(op_def) {}
  HardSwish(const HardSwish&) = delete;
  HardSwish(HardSwish&& h_swish) : ElementwiseOperation(std::move(h_swish)) {}

  HardSwish& operator=(const HardSwish&) = delete;
  HardSwish& operator=(HardSwish&& h_swish) {
    if (this != &h_swish) ElementwiseOperation::operator=(std::move(h_swish));
    return *this;
  }

  std::string GetCoreCode(const std::string& src, const std::string& z_coord,
                          const std::string& address) const override {
    return absl::Substitute(
        "$0 *= clamp($0 / 6.0f + (FLT4)(0.5f), (FLT4)(0.0f), (FLT4)(1.0f));\n",
        src);
  }
};

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_HARD_SWISH_H_
