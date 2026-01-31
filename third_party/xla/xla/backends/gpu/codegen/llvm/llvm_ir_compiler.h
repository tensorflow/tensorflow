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

#ifndef XLA_BACKENDS_GPU_CODEGEN_LLVM_LLVM_IR_COMPILER_H_
#define XLA_BACKENDS_GPU_CODEGEN_LLVM_LLVM_IR_COMPILER_H_

#include <cstdint>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "llvm/IR/Module.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

// LlvmIrCompiler abstracts compilation of LLVM IR to target binary.
//
// If debug options require it, implementations are responsible for logging
// compilation details and dumping debug artifacts.
using LlvmIrCompiler = absl::AnyInvocable<absl::StatusOr<std::vector<uint8_t>>(
    llvm::Module& module, const stream_executor::DeviceDescription& descr,
    const DebugOptions& opts)>;

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_LLVM_LLVM_IR_COMPILER_H_
