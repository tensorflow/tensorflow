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

#ifndef XLA_BACKENDS_GPU_CODEGEN_LLVM_LLVM_EMITTER_H_
#define XLA_BACKENDS_GPU_CODEGEN_LLVM_LLVM_EMITTER_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "xla/backends/gpu/codegen/gpu_kernel_spec.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/dense_data_intermediate.h"
#include "xla/service/gpu/ir_emitter_context.h"

namespace xla::gpu {

// Emit a constant with a given number of element, given byte size of the
// element, given symbol name and content.
void AppendGlobalConstant(llvm::Module* module, int64_t num_elements,
                          int64_t bytes_per_element,
                          absl::string_view symbol_name, int allocation_idx,
                          DenseDataIntermediate content, bool emit_initializer);

// Emit a kernel to increment the global state for Philox RNG
// algorithm.
absl::StatusOr<KernelDefinition<LlvmKernelSource, GpuKernelSpec>>
EmitRngGetAndUpdateStateLLVMIR(
    const HloRngGetAndUpdateStateInstruction* hlo,
    IrEmitterContext* ir_emitter_context,
    const emitters::KernelArguments& kernel_arguments);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_LLVM_LLVM_EMITTER_H_
