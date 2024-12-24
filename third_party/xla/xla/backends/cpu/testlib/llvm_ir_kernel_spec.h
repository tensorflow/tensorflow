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

#ifndef XLA_BACKENDS_CPU_TESTLIB_LLVM_IR_KERNEL_SPEC_H_
#define XLA_BACKENDS_CPU_TESTLIB_LLVM_IR_KERNEL_SPEC_H_

#include <memory>
#include <vector>

#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_ir_kernel_source.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::cpu {

// A KernelSpec that wraps an LlvmIrKernelSource and owns fake buffer
// allocations for all kernel arguments.
class LlvmIrKernelSpec final : public xla::KernelSpec {
 public:
  LlvmIrKernelSpec(se::ThreadDim thread_dim,
                   std::vector<BufferAllocation> buffer_allocations,
                   BufferUses buffer_uses,
                   std::unique_ptr<LlvmIrKernelSource> kernel_source);

  LlvmIrKernelSpec(LlvmIrKernelSpec&& other) = default;
  LlvmIrKernelSpec& operator=(LlvmIrKernelSpec&& other) noexcept = default;

  LlvmIrKernelSource& kernel_source() override { return *kernel_source_; }

 private:
  std::vector<BufferAllocation> buffer_allocations_;
  std::unique_ptr<LlvmIrKernelSource> kernel_source_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_TESTLIB_LLVM_IR_KERNEL_SPEC_H_
