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

#ifndef XLA_SERVICE_CPU_IR_EMITTER2_H_
#define XLA_SERVICE_CPU_IR_EMITTER2_H_

#include <string>

#include "absl/status/statusor.h"
#include "llvm/IR/Module.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla::cpu {

// IrEmitter emits host kernels form HLO instructions into the LLVM module(s).
//
// Host kernel is simply a function that implements StreamExecutor HostKernel
// interface (defined as C API for ABI stability), and XLA:CPU runtime is
// responsible for launching host kernels on the host as a part of the Thunk
// sequence execution.
//
// In addition to a host kernel function itself, host kernel defines how much
// concurrency it can support by picking the right thread and block sizes.
// Runtime might launch host kernel blocks and threads on a thread pool, with an
// assumption that threads and blocks that are close to each other in three
// dimensional space are likely to touch the same memory, and thus should be
// executed on the same thread (or same NUMA node).
//
// At run time thunks resolve kernel functions by name in the compiled LLVM
// module.
//
// WARNING: This is under construction and will eventually replace IrEmitter.
class IrEmitter2 {
 public:
  class ElementalIrEmitter;

  explicit IrEmitter2(llvm::Module* module);

  // A symbol name in the LLVM module that defines a host kernel.
  struct HostKernelSym {
    std::string name;
  };

  // Emits an elemental host kernel for the given HLO instruction.
  absl::StatusOr<HostKernelSym> EmitElementalHostKernel(
      const HloInstruction* instr);

 private:
  llvm::Module* module_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_IR_EMITTER2_H_
