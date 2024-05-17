/* Copyright 2023 The OpenXLA Authors.

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
#ifndef XLA_SERVICE_GPU_FUSIONS_COPY_H_
#define XLA_SERVICE_GPU_FUSIONS_COPY_H_

#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/Value.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/ir_emitter_context.h"

namespace xla {
namespace gpu {

// Special case of a fusion consisting only of `kCopy` instructions that can be
// implemented using `memcpy`s.
class MemcpyFusion : public FusionInterface {
 public:
  MemcpyFusion(std::vector<BufferAllocation::Slice> src_buffers,
               std::vector<BufferAllocation::Slice> dst_buffers,
               std::vector<mlir::Value> srcs, std::vector<mlir::Value> dsts)
      : src_buffers_(std::move(src_buffers)),
        dst_buffers_(std::move(dst_buffers)),
        srcs_(std::move(srcs)),
        dsts_(std::move(dsts)) {}

  absl::StatusOr<FusionEmissionResult> Emit(
      IrEmitterContext& ir_emitter_context,
      const HloFusionInstruction& fusion) const final;

 private:
  std::vector<BufferAllocation::Slice> src_buffers_;
  std::vector<BufferAllocation::Slice> dst_buffers_;

  // These are only used by the LMHLO code path and are empty if emitting from
  // HLO.
  std::vector<mlir::Value> srcs_;
  std::vector<mlir::Value> dsts_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_COPY_H_
