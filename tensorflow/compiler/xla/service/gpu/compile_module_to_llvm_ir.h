/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_COMPILE_MODULE_TO_LLVM_IR_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_COMPILE_MODULE_TO_LLVM_IR_H_

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/gpu/executable.pb.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {

struct CompileModuleResults {
  std::unique_ptr<llvm::Module> llvm_module;
  std::unique_ptr<BufferAssignment> buffer_assignment;
  std::vector<BufferAllocation> allocations;
  std::variant<GpuExecutable::OwnedThunkSequence,
               GpuExecutable::OwnedGpuRuntimeProgram>
      executable;
  EntryFunctionAttributes entry_func_attrs;
  std::vector<GpuExecutable::ConstantInfo> constants;
  absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo> output_info;
  Shape output_shape;
  std::string module_name;
};

Status GetMlirAllocationInfo(
    mlir::func::FuncOp func, std::vector<BufferAllocation>* allocations,
    absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>* output_info,
    Shape* output_shape, EntryFunctionAttributes* entry_func_attrs);

// Removes all globals from the given module that are both uninitialized and
// have no uses within that module.
void RemoveUnusedAndUninitializedGlobals(
    llvm::Module* llvm_module,
    const std::vector<GpuExecutable::ConstantInfo>& constants);

StatusOr<GpuExecutable::OwnedGpuRuntimeProgram> LowerToJitRt(
    mlir::ModuleOp mlir_module, llvm::StringRef entry_function_name,
    llvm::ArrayRef<int64_t> buffer_sizes, const HloModuleConfig& module_config,
    std::unique_ptr<ThunkSequence> thunk_sequence,
    const HloModule* hlo_module_for_dump = nullptr);

std::optional<bool> DummyCanShareBufferFunction(const HloInstruction*,
                                                const HloInstruction*,
                                                const ShapeIndex&);

// Compile `hlo_module` using XLA GPU and return the LLVM module thus generated.
// The GpuExecutable (and the Thunks that are part of it) are not returned.
StatusOr<std::unique_ptr<llvm::Module>> CompileModuleToLlvmIr(
    HloModule* hlo_module, llvm::LLVMContext* llvm_context,
    const std::string& target_triple, const std::string& data_layout,
    const std::string& platform_name, se::Platform::Id platform_id,
    GpuDeviceInfo gpu_device_info,
    se::CudaComputeCapability cuda_compute_capability,
    se::RocmComputeCapability rocm_compute_capability, int pointer_size);

Status CompileModuleToLlvmIrImpl(
    HloModule* hlo_module, llvm::LLVMContext* llvm_context,
    const std::string& target_triple, const std::string& data_layout,
    const std::string& platform_name, se::Platform::Id platform_id,
    GpuDeviceInfo gpu_device_info,
    se::CudaComputeCapability cuda_compute_capability,
    se::RocmComputeCapability rocm_compute_capability,
    const HloDataflowAnalysis::CanShareBuffer& can_share_buffer_function,
    int pointer_size, CompileModuleResults* results,
    se::StreamExecutor* stream_exec = nullptr);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_COMPILE_MODULE_TO_LLVM_IR_H_
