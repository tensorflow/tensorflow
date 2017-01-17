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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_PARALLEL_CPU_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_PARALLEL_CPU_EXECUTABLE_H_

#include <stddef.h>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/simple_orc_jit.h"
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace xla {
namespace cpu {

// CPU-targeting parallel implementation of the XLA Executable interface.
//
// Wraps a JIT-ed object that can be executed "on device". We JIT for the host
// architecture, so JIT-ed code and host code share the same ABI.
class ParallelCpuExecutable : public Executable {
 public:
  ParallelCpuExecutable(
      std::unique_ptr<SimpleOrcJIT> jit,
      std::unique_ptr<BufferAssignment> assignment,
      std::unique_ptr<HloModule> hlo_module,
      std::unique_ptr<HloModuleConfig> module_config,
      std::unique_ptr<std::map<HloInstruction*, string>> instruction_functions,
      std::unordered_map<const HloInstruction*, size_t> hlo_to_profile_idx,
      std::unordered_map<const HloInstruction*,
                         std::unique_ptr<unsigned char[]>>
          aligned_constants);
  ~ParallelCpuExecutable() override {}

  StatusOr<perftools::gputools::DeviceMemoryBase> ExecuteOnStream(
      const ExecutableRunOptions* run_options,
      tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>
          arguments,
      HloExecutionProfile* hlo_execution_profile) override;

  StatusOr<std::unique_ptr<ShapedBuffer>> ExecuteOnStream(
      const ExecutableRunOptions* run_options,
      tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
      HloExecutionProfile* hlo_execution_profile) override;

  Status ExecuteOnStream(
      const ExecutableRunOptions* run_options,
      tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
      ShapedBuffer* result_buffer,
      HloExecutionProfile* hlo_execution_profile) override;

  StatusOr<perftools::gputools::DeviceMemoryBase> ExecuteAsyncOnStream(
      const ExecutableRunOptions* run_options,
      tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>
          arguments) override;

  // This should be called after set_ir_module_string.
  const string& ir_module_string() const { return ir_module_string_; }

  void set_ir_module_string(const string& ir_module_string) {
    ir_module_string_ = ir_module_string;
  }

 private:
  // The JIT containing compiled modules.
  tensorflow::mutex jit_mutex_;
  std::unique_ptr<SimpleOrcJIT> jit_ GUARDED_BY(jit_mutex_);

  // Buffer assignment for the buffers we need to allocate.
  std::unique_ptr<BufferAssignment> assignment_;

  // The LLVM IR, in string format, of the unoptimized module generated for this
  // ParallelCpuExecutable. We save a string instead of an llvm::Module* because
  // leaving llvm::Module* in a singleton can cause the heap checker to emit
  // false positives.
  string ir_module_string_;

  // Map containing the JITted function names for each HLO instruction.
  std::unique_ptr<std::map<HloInstruction*, string>> functions_names_;

  // Maps HLOs to their index into the profile counter array.
  const std::unordered_map<const HloInstruction*, size_t> hlo_to_profile_idx_;

  // Map from HLO Constant instructions to a pointer to their literal data.
  // The data stored in the protocol buffer might be insufficiently aligned,
  // we create a sufficiently aligned copy and store it in this map.
  std::unordered_map<const HloInstruction*, std::unique_ptr<unsigned char[]>>
      aligned_constants_;

  TF_DISALLOW_COPY_AND_ASSIGN(ParallelCpuExecutable);
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_PARALLEL_CPU_EXECUTABLE_H_
