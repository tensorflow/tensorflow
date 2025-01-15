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

#include "xla/service/gpu/elemental_ir_emitter.h"

#include <vector>

// IWYU pragma: no_include "llvm/IR/Attributes.gen.inc"
// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/elemental_ir_emitter.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/ir_emitter_nested.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

GpuElementalIrEmitter::GpuElementalIrEmitter(
    IrEmitterContext& ir_emitter_context, llvm::IRBuilderBase* b)
    : ElementalIrEmitter(ir_emitter_context.llvm_module(), b),
      ir_emitter_context_(ir_emitter_context) {}

absl::StatusOr<std::vector<llvm::Value*>>
GpuElementalIrEmitter::EmitThreadLocalCall(
    const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
    absl::string_view, bool /*is_reducer*/) {
  return CallNestedComputationWithScalars(b(), ir_emitter_context_, callee,
                                          parameters);
}

}  // namespace gpu
}  // namespace xla
