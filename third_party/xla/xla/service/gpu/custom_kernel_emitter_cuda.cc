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

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/codegen/kernels/ptx_custom_kernel.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/custom_kernel_emitter.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_call.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

absl::StatusOr<std::unique_ptr<Thunk>> EmitPtxCustomKernelThunk(
    const HloCustomCallInstruction* instr, IrEmitterContext* context) {
  absl::string_view backend_config_str = instr->raw_backend_config_string();
  if (backend_config_str.empty()) {
    return absl::InvalidArgumentError(
        "PTX custom call backend config is empty");
  }

  TF_ASSIGN_OR_RETURN(
      KernelCall call,
      KernelCall::Parse(backend_config_str, context->mlir_context()));
  if (call.kernel_type != KernelCall::KernelType::kPtxSource) {
    return absl::InvalidArgumentError(
        "PTX custom call backend config is not a PTX source");
  }

  emitters::KernelArguments::BufferAlignment buffer_alignment =
      GetDefaultBufferAlignment();
  TF_ASSIGN_OR_RETURN(emitters::KernelArguments kernel_arguments,
                      emitters::KernelArguments::Create(
                          context->buffer_assignment(), buffer_alignment, instr,
                          call.output_indices));

  TF_ASSIGN_OR_RETURN(
      CustomKernel ptx_custom_kernel,
      kernel::GetOwnedPtxCustomKernel(
          call.name, call.kernel_data, kernel_arguments.args().size(),
          call.block_dim, call.thread_dim, call.shared_mem));

  Thunk::ThunkInfo thunk_info =
      Thunk::ThunkInfo::WithProfileAnnotation(instr, context->GetNextThunkId());
  return std::make_unique<CustomKernelThunk>(
      std::move(thunk_info), ptx_custom_kernel, kernel_arguments);
}

}  // namespace gpu
}  // namespace xla
