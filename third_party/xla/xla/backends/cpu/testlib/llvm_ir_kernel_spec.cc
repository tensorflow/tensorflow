/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/testlib/llvm_ir_kernel_spec.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_ir_kernel_source.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::cpu {

LlvmIrKernelSpec::LlvmIrKernelSpec(
    se::ThreadDim thread_dim, std::vector<BufferAllocation> buffer_allocations,
    BufferUses buffer_uses, std::unique_ptr<LlvmIrKernelSource> kernel_source)
    : KernelSpec(se::ClusterDim(), se::BlockDim(), thread_dim, std::nullopt,
                 std::move(buffer_uses)),
      buffer_allocations_(std::move(buffer_allocations)),
      kernel_source_(std::move(kernel_source)) {}

}  // namespace xla::cpu
