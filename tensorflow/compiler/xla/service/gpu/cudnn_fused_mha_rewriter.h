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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_FUSED_MHA_REWRITER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_FUSED_MHA_REWRITER_H_

#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {


// cuDNN currently supports and the rewriter matches the following patterns for Multi-headed attention:
// 1. BMM1 - BMM2
// 2. BMM1 - Scale - Bias - Mask - Softmax - BMM2 (To be added)
// 3. BMM1 - Scale - Bias - Mask - Softmax - Dropout - BMM2 (To be added)
// 4. BMM1 - Scale - Mask - Softmax - BMM2 (To be added)
// 5. BMM1 - Scale - Mask - Softmax - Dropout - BMM2 (To be added)
// 6. BMM1 - Softmax - Dropout - BMM2 (To be added)

// Note that cuDNN 8.8 has the following limitations: 
// 1. The non contracting dimensions for BMM1 need to be less than or equal to 512.
// 2. The contracting dimensions for BMM1 need to be 64.
// 3. The non contracting dimension for BMM2 needs to be 64 for the input matrix.

class CudnnFusedMHARewriter : public HloModulePass {
 public:
  explicit CudnnFusedMHARewriter(se::CudaComputeCapability cc)
      : compute_capability_(cc) {}

  absl::string_view name() const override {
    return "cudnn-fused-multi-headed-attention-rewriter";
  }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const se::CudaComputeCapability compute_capability_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_FUSED_MHA_REWRITER_H_
