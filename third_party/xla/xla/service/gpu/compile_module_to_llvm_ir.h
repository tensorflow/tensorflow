/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_COMPILE_MODULE_TO_LLVM_IR_H_
#define XLA_SERVICE_GPU_COMPILE_MODULE_TO_LLVM_IR_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/executable.pb.h"
#include "xla/service/gpu/execution_stream_assignment.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

struct CompileModuleResults {
  std::unique_ptr<llvm::Module> llvm_module;
  std::unique_ptr<llvm::Module> llvm_module_constants;
  std::unique_ptr<BufferAssignment> buffer_assignment;
  std::unique_ptr<ExecutionStreamAssignment> execution_stream_assignment;
  std::vector<BufferAllocation> allocations;
  std::unique_ptr<SequentialThunk> executable;
  std::vector<GpuExecutable::ConstantInfo> constants;
  absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo> output_info;
  Shape output_shape;
  std::string module_name;
  CompilationCacheProto kernel_compilation_cache;

  // If true, the compiled module uses buffer allocations owned by
  // buffer_assignment. Otherwise the compiled module uses buffer allocations
  // stored in allocations.
  bool use_original_allocations;
};

absl::Status LoadCache(IrEmitterContext& ir_emitter_context,
                       absl::string_view cache_file_path);

absl::StatusOr<CompileModuleResults> CompileModuleToLlvmIr(
    const HloModule* hlo_module, llvm::LLVMContext* llvm_context,
    const std::string& target_triple, const std::string& data_layout,
    const se::Platform* platform, const se::DeviceDescription& device_desc,
    const GpuAliasInfo* alias_info,
    const BufferValue::SizeFunction& buffer_size_bytes_function,
    bool split_constants_module = false);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_COMPILE_MODULE_TO_LLVM_IR_H_
