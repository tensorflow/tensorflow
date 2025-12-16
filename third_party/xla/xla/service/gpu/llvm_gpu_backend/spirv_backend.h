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

// LLVM-based compiler backend.
#ifndef XLA_SERVICE_GPU_LLVM_GPU_BACKEND_SPIRV_BACKEND_H_
#define XLA_SERVICE_GPU_LLVM_GPU_BACKEND_SPIRV_BACKEND_H_

#include <functional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla::gpu::spirv {

absl::StatusOr<std::string> CompileToSPIRV(
    llvm::Module* module, stream_executor::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options);

// Filters out "unsupported_extensions" from the extensions list
std::vector<std::string> RemoveUnsupportedExtensionsFromAll(
    llvm::Triple triple, const std::vector<std::string> unsupported_extensions);

// Returns the LLVM command line flags that we use for compilation.
std::vector<std::string> GetSPIRVBackendOptions(
    const DebugOptions& debug_options);

}  // namespace xla::gpu::spirv

#endif  // XLA_SERVICE_GPU_LLVM_GPU_BACKEND_SPIRV_BACKEND_H_
