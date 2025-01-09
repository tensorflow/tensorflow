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

#ifndef XLA_BACKENDS_CPU_TESTLIB_LLVM_IR_KERNEL_EMITTER_H_
#define XLA_BACKENDS_CPU_TESTLIB_LLVM_IR_KERNEL_EMITTER_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/codegen/kernel_emitter.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/runtime/buffer_use.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::cpu {

// An XLA kernel emitter that emits a kernel by parsing the given LLVM IR module
// into the dedicated LLVM context and module instance. This kernel emitter is
// intended to be used for testing purposes only: (1) load pre-compiled LLVM IR
// into the XLA kernel spec; (2) Execute it with user provided input buffers.
class LlvmIrKernelEmitter : public KernelEmitter {
 public:
  // When loading kernel IR into the KernelSpec we create a separate buffer
  // allocation for every kernel argument. We don't use buffer assignment in
  // kernel testlib, but we still need to return a valid BufferUses vector.
  struct KernelArg {
    size_t size_bytes;
    BufferUse::MemoryAccess memory_access;
  };

  LlvmIrKernelEmitter(absl::string_view llvm_ir, absl::string_view kernel_name,
                      se::ThreadDim thread_dim,
                      absl::Span<const KernelArg> args);

  absl::StatusOr<std::unique_ptr<KernelSpec>> EmitKernelSpec() final;

 private:
  std::string llvm_ir_;
  std::string kernel_name_;
  se::ThreadDim thread_dim_;
  std::vector<KernelArg> args_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_TESTLIB_LLVM_IR_KERNEL_EMITTER_H_
