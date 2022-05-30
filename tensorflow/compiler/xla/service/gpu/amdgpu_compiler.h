/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_AMDGPU_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_AMDGPU_COMPILER_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/gpu/gpu_compiler.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace gpu {

// AMDGPUCompiler generates efficient GPU executables for AMDGPU target.
class AMDGPUCompiler : public GpuCompiler {
 public:
  AMDGPUCompiler();
  ~AMDGPUCompiler() override {}

  Status OptimizeHloConvolutionCanonicalization(
      HloModule* hlo_module, se::StreamExecutor* stream_exec,
      se::DeviceMemoryAllocator* device_allocator) override;

  Status OptimizeHloPostLayoutAssignment(
      HloModule* hlo_module, se::StreamExecutor* stream_exec,
      se::DeviceMemoryAllocator* device_allocator) override;

  GpuVersion GetGpuVersion(se::StreamExecutor* stream_exec) override;

  StatusOr<std::pair<std::string, std::vector<uint8_t>>> CompileTargetBinary(
      const HloModuleConfig& module_config, llvm::Module* llvm_module,
      GpuVersion gpu_version, se::StreamExecutor* stream_exec, bool relocatable,
      const HloModule* debug_module) override;

 private:
  // The parent directory of ROCm-Device-Libs IR libraries.
  std::string rocdl_dir_;

  AMDGPUCompiler(const AMDGPUCompiler&) = delete;
  AMDGPUCompiler& operator=(const AMDGPUCompiler&) = delete;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_AMDGPU_COMPILER_H_
