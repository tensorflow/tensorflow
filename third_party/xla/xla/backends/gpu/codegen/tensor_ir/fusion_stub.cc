/* Copyright 2026 The OpenXLA Authors.

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

#include "absl/status/status.h"
#include "xla/backends/gpu/codegen/tensor_ir/fusion.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/ir_emitter_context.h"

namespace xla::gpu {

// Stands in for the TensorIR emitter in builds without CUDA: the tensor_ir
// runtime and compiler libraries include cuda.h.
AsyncThunkSequence TensorIrFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  return absl::UnimplementedError(
      "The TensorIR fusion emitter is only available in CUDA builds.");
}

}  // namespace xla::gpu
