/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_CUDNN_FUSED_MHA_REWRITER_H_
#define XLA_SERVICE_GPU_CUDNN_FUSED_MHA_REWRITER_H_

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/stream_executor/dnn.h"

namespace xla {
namespace gpu {

class CudnnFusedMHARewriter : public HloModulePass {
 public:
  explicit CudnnFusedMHARewriter(se::CudaComputeCapability cc,
                                 se::StreamExecutor* stream_executor)
      : compute_capability_(cc), stream_executor_(stream_executor) {}

  explicit CudnnFusedMHARewriter(se::CudaComputeCapability cc,
                                 se::dnn::VersionInfo cudnn_version)
      : compute_capability_(cc), cudnn_version_(cudnn_version) {}

  absl::string_view name() const override {
    return "cudnn-fused-multi-headed-attention-rewriter";
  }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const se::CudaComputeCapability compute_capability_;
  se::StreamExecutor* stream_executor_ = nullptr;
  const se::dnn::VersionInfo cudnn_version_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_CUDNN_FUSED_MHA_REWRITER_H_
