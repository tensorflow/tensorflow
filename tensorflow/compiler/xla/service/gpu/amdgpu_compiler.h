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

#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/llvm_compiler.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace xla {
namespace gpu {

// The GPU compiler generates efficient GPU executables.
class AMDGPUCompiler : public LLVMCompiler {
 public:
  AMDGPUCompiler();
  ~AMDGPUCompiler() override {}

  // Bring in
  // StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
  //     std::vector<std::unique_ptr<HloModule>> modules,
  //     std::vector<std::vector<se::StreamExecutor*>>
  //        stream_execs)
  using LLVMCompiler::Compile;

  StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      DeviceMemoryAllocator* device_allocator) override;

  StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      DeviceMemoryAllocator* device_allocator) override;

  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::vector<std::unique_ptr<HloModule>> module,
                     AotCompilationOptions const& options) override;

  perftools::gputools::Platform::Id PlatformId() const override;

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override {
    // Capture just the pointer size, not the entire AMDGPUCompiler object.
    int64 pointer_size = pointer_size_;
    return [pointer_size](const Shape& shape) {
      return ShapeUtil::ByteSizeOf(shape, pointer_size);
    };
  }

  // The triple that represents our target.
  static const char* kTargetTriple;

  // The data layout of the emitted module. Copied from computeDataLayout in
  // AMDGPUTargetMachine.cpp.
  static const char* kDataLayout;

 private:
  // The size in bytes of a pointer. Used by ShapeSizeBytesFunction.
  const int64 pointer_size_;

  tensorflow::mutex mutex_;

  // The parent directory of ROCm-Device-Libs IR libraries.
  string rocdl_dir_;

  TF_DISALLOW_COPY_AND_ASSIGN(AMDGPUCompiler);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_AMDGPU_COMPILER_H_
