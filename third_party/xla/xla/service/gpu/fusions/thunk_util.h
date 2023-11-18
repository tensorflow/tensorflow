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
#ifndef XLA_SERVICE_GPU_FUSIONS_THUNK_UTIL_H_
#define XLA_SERVICE_GPU_FUSIONS_THUNK_UTIL_H_

#include <memory>
#include <optional>

#include "mlir/IR/Value.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/thunk.h"
#include "xla/statusor.h"

namespace xla {
namespace gpu {

// Attempts to build an initializer constant for the given value. Returns an
// empty optional if the value is not a constant.
StatusOr<std::optional<std::unique_ptr<Thunk>>> BuildConstantInitializerThunk(
    IrEmitterContext& ir_emitter_context, mlir::Operation* op,
    const HloInstruction* instr, const HloInstruction* init_value,
    mlir::Value dest, BufferAllocation::Slice dest_slice);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_THUNK_UTIL_H_
