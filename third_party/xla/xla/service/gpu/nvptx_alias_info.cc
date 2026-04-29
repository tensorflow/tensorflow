/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/nvptx_alias_info.h"

#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/strings/match.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/shape_util.h"

namespace xla::gpu {
std::optional<bool> NVPTXAliasInfo::MayAlias(
    const HloInstruction* operand, const ShapeIndex& operand_index,
    const HloInstruction* user, const ShapeIndex& user_index) const {
  switch (user->opcode()) {
    // NCCL all-reduce and collective-broadcast can be performed in-place.
    // NVSHMEM all-reduce needs out-of-place buffers.
    case HloOpcode::kAllReduce: {
      if (absl::StrContainsIgnoreCase(user->raw_backend_config_string(),
                                      "nvshmem")) {
        return false;
      }
      [[fallthrough]];
    }
    case HloOpcode::kCollectiveBroadcast:
      return user->operand_count() == 1 ||
             (user_index.size() == 1 &&
              user->operand(user_index[0]) == operand);
    case HloOpcode::kCustomCall:
      // The matrix bias operand can be overwritten in-place.
      // See https://docs.nvidia.com/cuda/cublas/#cublasltmatmul.
      // Note that we disallow aliasing the matmul workspace buffer
      // with the bias buffer.
      if (user->custom_call_target() == kCublasLtMatmulCallTarget) {
        GemmBackendConfig config =
            std::move(user->backend_config<GpuBackendConfig>())
                ->gemm_backend_config();
        // For cublasLt matmul, operand(2) is the matrix 'C' in D = alpha * (A *
        // B) + beta * C. We only allow aliasing if beta is non-zero (so 'C' is
        // actually used), the operand is 'C', and it is used only once by this
        // instruction. Finally, we check that we are looking at the
        // first output of the custom call, which is the result incontrary to
        // the second output which is the workspace buffer.
        return (config.beta() != 0.) && operand == user->operand(2) &&
               absl::c_count(user->operands(), operand) == 1 &&
               user_index.size() == 1 && user_index[0] == 0;
      }
      return false;
    default:
      return GpuAliasInfo::MayAlias(operand, operand_index, user, user_index);
  }
}
}  // namespace xla::gpu
