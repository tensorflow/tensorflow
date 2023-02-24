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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GEMM_BROADCAST_FOLDING_REWRITER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GEMM_BROADCAST_FOLDING_REWRITER_H_

#include <optional>

#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// cuBLAS GEMM has support for strided batched calls, where the stride is used
// to determine the offset between the batches.
//
// This allows (kCustomCall:gemm A kBroadcast(B)) or
//             (kCustomCall:gemm kBroadcast(A) B)
// to be rewritten as (kCustomCall:gemm A B) with a zero stride for the
// broadcasted operand if the broadcast operates on all the batch dimensions.
//
// This pattern matches the above case and removes the unnecessary broadcast.
class GemmBroadcastFoldingRewriter : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "cublas-gemm-broadcast-folding-rewriter";
  }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GEMM_BROADCAST_FOLDING_REWRITER_H_
