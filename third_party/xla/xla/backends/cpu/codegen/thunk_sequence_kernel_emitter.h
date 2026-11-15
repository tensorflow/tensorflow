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

#ifndef XLA_BACKENDS_CPU_CODEGEN_THUNK_SEQUENCE_KERNEL_EMITTER_H_
#define XLA_BACKENDS_CPU_CODEGEN_THUNK_SEQUENCE_KERNEL_EMITTER_H_

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/Module.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/codegen/kernel_emitter.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/service/decision.h"

namespace xla::cpu {

// Compiles a thunk sequence into a single host kernel that runs the sequence
// with no runtime dispatch.
//
// The kernel takes disjoint read only `arguments` followed by writable
// `results` as its parameters. The C API entry passes their base pointers to an
// internal `<name>_body` function, whose parameters are `noalias`, and
// `readonly` for arguments. Copy, while, conditional and call thunks become
// control flow and memcpy. Kernel thunks become calls through the Kernel C API,
// with the workgroups of each launch run serially on the calling thread.
//
// The emitted module only declares the kernels it calls. LinkKernels then
// links their definitions in, which lets the caller compile them separately
// (and in parallel) first.
class ThunkSequenceKernelEmitter final
    : public KernelEmitter<LlvmKernelSource> {
 public:
  // Returns whether `thunks` can be compiled into a kernel with the given
  // disjoint parameters, writing only to `results`. `kernels` names the kernels
  // whose sources will be linked in.
  static Decision CanEmit(const ThunkSequence& thunks,
                          const absl::flat_hash_set<std::string>& kernels,
                          const KernelSpec::Buffers& arguments,
                          const KernelSpec::Buffers& results);

  // A compiled kernel: its entry point and the module defining it.
  struct KernelModule {
    std::string name;
    const llvm::Module* module;
  };

  // Links the definitions of the kernels called by `module` and gives them
  // internal linkage, so that LLVM can inline them. `kernels` must define every
  // kernel the module calls. They are cloned, not consumed, and those that the
  // module does not call are skipped. Per kernel backend options are merged
  // into `module`.
  static absl::Status LinkKernels(llvm::Module& module,
                                  absl::Span<const KernelModule> kernels);

  ThunkSequenceKernelEmitter(std::string name, std::string memory_region_name,
                             KernelSpec::Buffers arguments,
                             KernelSpec::Buffers results,
                             const ThunkSequence* thunks,
                             const TargetMachineFeatures* target_machine,
                             KernelApiIrBuilder::Options options);

  absl::string_view name() const final {
    return "thunk_sequence_kernel_emitter";
  }

  absl::StatusOr<KernelDefinition> EmitKernelDefinition() final;

 private:
  std::string kernel_name_;
  std::string memory_region_name_;
  KernelSpec::Buffers arguments_;
  KernelSpec::Buffers results_;
  const ThunkSequence* thunks_;
  const TargetMachineFeatures* target_machine_;
  KernelApiIrBuilder::Options options_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_THUNK_SEQUENCE_KERNEL_EMITTER_H_
