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
#ifndef XLA_BACKENDS_GPU_CODEGEN_NESTED_KERNEL_UTILS_H_
#define XLA_BACKENDS_GPU_CODEGEN_NESTED_KERNEL_UTILS_H_

#include <cstddef>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Shared launcher/splicing helpers used by `PhasedBlockClusterFusion` and
// `ThreadBlockSpecializedFusion`.

// Returns the nested kernel function name for the given entry function and
// index.
std::string GetNestedKernelFunctionName(absl::string_view entry_function_name,
                                        size_t index);

// Lowers a launcher module (containing dispatch logic to nested kernels) to the
// LLVM GPU dialect.
absl::Status LowerLauncherModuleToLlvmDialect(
    mlir::ModuleOp launcher_module,
    const stream_executor::DeviceDescription& device_info, bool should_verify);

// Splices a lowered nested kernel module into `launcher_module`, replacing any
// forward declaration of `nested_fn_name` in `launcher_module` with the
// `alwaysinline` internal definition from `nested_module`.
absl::Status SpliceNestedKernelModuleIntoLauncher(
    mlir::ModuleOp launcher_module, mlir::ModuleOp nested_module,
    absl::string_view nested_fn_name);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_NESTED_KERNEL_UTILS_H_
