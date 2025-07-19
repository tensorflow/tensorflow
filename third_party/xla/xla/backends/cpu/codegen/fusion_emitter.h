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

#ifndef XLA_BACKENDS_CPU_CODEGEN_FUSION_EMITTER_H_
#define XLA_BACKENDS_CPU_CODEGEN_FUSION_EMITTER_H_

#include "absl/status/statusor.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/mlir_kernel_definition.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"

namespace xla::cpu {

absl::StatusOr<MlirKernelDefinition> EmitFusionKernel(
    mlir::MLIRContext& context, const HloFusionInstruction& fusion,
    const BufferAssignment* buffer_assignment, bool use_unique_c_name);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_FUSION_EMITTER_H_
