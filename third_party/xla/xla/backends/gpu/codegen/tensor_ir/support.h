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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_SUPPORT_H_
#define XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_SUPPORT_H_

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/decision.h"

namespace xla::gpu::tensor_ir {

using CodegenDecision = Decision;

// Returns `Decision::Allow` if the fusion computation is supported by the
// TensorIR fusion emitter.
CodegenDecision IsSupportedFusionComputation(const HloComputation& comp);

// Returns `Decision::Allow` if the given instruction is supported by the
// TensorIR fusion emitter. If `instr` is a fusion instruction, verifies the
// fused computation.
CodegenDecision IsInstructionSupportedForFusion(const HloInstruction& instr);

}  // namespace xla::gpu::tensor_ir

#endif  // XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_SUPPORT_H_
