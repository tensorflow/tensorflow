/* Copyright 2017 The OpenXLA Authors.

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
#ifndef XLA_SERVICE_GPU_LLVM_GPU_BACKEND_AMDGPU_BACKEND_H_
#define XLA_SERVICE_GPU_LLVM_GPU_BACKEND_AMDGPU_BACKEND_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "llvm/IR/Module.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla::gpu::amdgpu {
// Get path to libdevice file.
std::string LibDevicePath(std::string gcn_arch_name,
                          const std::string& rocdl_dir_path);
// Compiles the argument module and returns it with LLVM AMDGPU backend.
// rocdl_dir_path is the parent directory of ROCm-Device-Libs bitcode libraries.
// The contents of the module may be changed.
absl::StatusOr<std::vector<uint8_t>> CompileToHsaco(
    llvm::Module* module, stream_executor::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options,
    const std::string& module_config_cache_key);
}  // namespace xla::gpu::amdgpu

#endif  // XLA_SERVICE_GPU_LLVM_GPU_BACKEND_AMDGPU_BACKEND_H_
