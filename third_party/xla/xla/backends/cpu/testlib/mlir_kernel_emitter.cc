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

#include "xla/backends/cpu/testlib/mlir_kernel_emitter.h"

#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {
MlirKernelEmitter::MlirKernelEmitter(absl::string_view mlir,
                                     absl::string_view kernel_name,
                                     se::ThreadDim thread_dim,
                                     absl::Span<const KernelArg> args)
    : mlir_(mlir),
      kernel_name_(kernel_name),
      thread_dim_(thread_dim),
      args_(args.begin(), args.end()) {
  for (const MlirKernelEmitter::KernelArg& arg : args_) {
    buffer_allocations_.emplace_back(buffer_allocations_.size(), arg.size_bytes,
                                     /*color=*/0);
  }
}

absl::StatusOr<KernelDefinition> MlirKernelEmitter::EmitKernelDefinition() {
  std::unique_ptr<mlir::MLIRContext> context = FusionCompiler::CreateContext();

  TF_ASSIGN_OR_RETURN(
      MlirKernelSource source,
      MlirKernelSource::ParseFromString(mlir_, std::move(context)));

  // Convert kernel arguments to fake allocations and buffer uses.
  KernelSpec::Buffers argument_buffers;
  KernelSpec::Buffers result_buffers;

  for (const auto& [arg, allocation] : llvm::zip(args_, buffer_allocations_)) {
    BufferAllocation::Slice slice(&allocation, 0, arg.size_bytes);
    if (arg.memory_access == BufferUse::MemoryAccess::kRead) {
      argument_buffers.push_back(slice);
    } else {
      result_buffers.push_back(slice);
    }
  }

  KernelSpec kernel_spec(kernel_name_, thread_dim_, std::move(argument_buffers),
                         std::move(result_buffers), /*invariant_arguments=*/{});
  return KernelDefinition(
      std::move(kernel_spec),
      std::make_unique<MlirKernelSource>(std::move(source)));
}
}  // namespace xla::cpu
