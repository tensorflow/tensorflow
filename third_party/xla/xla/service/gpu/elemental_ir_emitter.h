/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_ELEMENTAL_IR_EMITTER_H_
#define XLA_SERVICE_GPU_ELEMENTAL_IR_EMITTER_H_

#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/elemental_ir_emitter.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

class GpuElementalIrEmitter : public ElementalIrEmitter {
 public:
  GpuElementalIrEmitter(IrEmitterContext& ir_emitter_context,
                        llvm::IRBuilderBase* b);

 protected:
  absl::StatusOr<std::vector<llvm::Value*>> EmitThreadLocalCall(
      const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
      absl::string_view, bool /*is_reducer*/) override;

  bool fast_min_max() override {
    return ir_emitter_context_.debug_options().xla_gpu_enable_fast_min_max();
  }

 private:
  IrEmitterContext& ir_emitter_context_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_ELEMENTAL_IR_EMITTER_H_
