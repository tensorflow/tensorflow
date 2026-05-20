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

#ifndef XLA_SERVICE_GPU_IR_EMITTER_CONTEXT_H_
#define XLA_SERVICE_GPU_IR_EMITTER_CONTEXT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/host_execute_thunk.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/runtime/object_pool.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/call_inliner.h"
#include "xla/service/gpu/execution_stream_assignment.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/name_uniquer.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
// Maps host offloading start ops to their async events so we can emit done
// thunk sharing events with corresponding start thunk.
using InstructionToHostExecuteAsyncEvents =
    absl::flat_hash_map<const HloInstruction*,
                        std::shared_ptr<HostExecuteAsyncEvents>>;

// IrEmitterContext encapsulates common (mutable and immutable) data structures
// used by both IrEmitterNested and IrEmitterUnnested, such as the buffer
// assignment and the name uniquer.
class IrEmitterContext {
 public:
  IrEmitterContext(const HloModule* hlo_module,
                   const BufferAssignment* buffer_assignment,
                   const ExecutionStreamAssignment* execution_stream_assignment,
                   absl::string_view platform_name,
                   const se::DeviceDescription& gpu_device_info,
                   mlir::MLIRContext* mlir_context,
                   llvm::LLVMContext* llvm_context, bool emit_kernels,
                   llvm::Triple target_triple, std::string data_layout,
                   KernelCompiler* compiler,
                   xla::cpu::TargetMachineOptions cpu_target_machine_options,
                   ObjectPool<std::unique_ptr<mlir::MLIRContext>>* pool)
      : hlo_module_(hlo_module),
        buffer_assignment_(buffer_assignment),
        execution_stream_assignment_(execution_stream_assignment),
        platform_name_(platform_name),
        gpu_device_info_(gpu_device_info),
        mlir_context_(mlir_context),
        llvm_context_(llvm_context),
        data_layout_(std::move(data_layout)),
        target_triple_(std::move(target_triple)),
        emit_kernels_(emit_kernels),
        compiler_(compiler),
        cpu_target_machine_options_(std::move(cpu_target_machine_options)),
        mlir_context_pool_(pool) {}

  // Disallow copy and assign.
  IrEmitterContext(const IrEmitterContext&) = delete;
  IrEmitterContext& operator=(const IrEmitterContext&) = delete;

  std::unique_ptr<IrEmitterContext> SubContext(
      llvm::LLVMContext* llvm_context) {
    return std::make_unique<IrEmitterContext>(
        hlo_module_, buffer_assignment_, execution_stream_assignment_,
        platform_name_, gpu_device_info_, mlir_context_, llvm_context,
        emit_kernels_, target_triple_, data_layout_, compiler_,
        cpu_target_machine_options_, mlir_context_pool_);
  }

  // Simple accessors.
  const HloModule& hlo_module() const { return *hlo_module_; }
  const BufferAssignment& buffer_assignment() const {
    return *buffer_assignment_;
  }
  const ExecutionStreamAssignment& execution_stream_assignment() const {
    return *execution_stream_assignment_;
  }
  absl::string_view platform_name() const { return platform_name_; }
  const se::DeviceDescription& gpu_device_info() const {
    return gpu_device_info_;
  }
  const se::GpuComputeCapability& gpu_compute_capability() const {
    return gpu_device_info_.gpu_compute_capability();
  }

  const xla::cpu::TargetMachineOptions& cpu_target_machine_options() const {
    return cpu_target_machine_options_;
  }

  mlir::MLIRContext* mlir_context() { return mlir_context_; }
  llvm::LLVMContext* llvm_context() { return llvm_context_; }

  const std::string& data_layout() { return data_layout_; }
  const llvm::Triple& target_triple() { return target_triple_; }

  absl::StatusOr<InlinedModule*> get_inlined_module() {
    if (inlined_module_ == nullptr) {
      ASSIGN_OR_RETURN(InlinedModule inlined_module,
                       GetInlinedModule(hlo_module_));
      inlined_module_ =
          std::make_unique<InlinedModule>(std::move(inlined_module));
    }
    return inlined_module_.get();
  }

  std::vector<GpuExecutable::ConstantInfo>& constants() { return constants_; }

  const DebugOptions& debug_options() const {
    return hlo_module_->config().debug_options();
  }

  KernelReuseCache& kernel_cache() { return kernel_cache_; }

  InstructionToHostExecuteAsyncEvents&
  instruction_to_host_execute_async_events() {
    return instruction_to_host_execute_async_events_;
  }

  bool emit_kernels() const { return emit_kernels_; }

  ThunkId GetNextThunkId() { return thunk_id_generator_.GetNextThunkId(); }

  // Compute the kernel name. The opcode string may contain "-" which cannot be
  // in a PTX function name, so sanitize the name before uniquifying it.
  std::string GetSanitizedUniqueName(const std::string& suggested_name) {
    return name_uniquer_.GetUniqueName(
        llvm_ir::SanitizeFunctionName(suggested_name));
  }

  std::unique_ptr<llvm::Module> CreateLLVMModule(
      const std::string& module_name) {
    auto llvm_module =
        std::make_unique<llvm::Module>(module_name, *llvm_context_);
    llvm_module->setTargetTriple(target_triple_);
    llvm_module->setDataLayout(data_layout_);
    return llvm_module;
  }

  KernelCompiler* kernel_compiler() { return compiler_; }

  BorrowedMlirContext BorrowMlirContext() {
    auto context = mlir_context_pool_->GetOrCreate();
    // Due to generator, always ok.
    CHECK_OK(context.status());
    return std::move(*context);
  }

 private:
  const HloModule* hlo_module_;
  const BufferAssignment* buffer_assignment_;
  const ExecutionStreamAssignment* execution_stream_assignment_;
  absl::string_view platform_name_;
  const se::DeviceDescription& gpu_device_info_;
  mlir::MLIRContext* mlir_context_;
  llvm::LLVMContext* llvm_context_;
  NameUniquer name_uniquer_;
  std::vector<GpuExecutable::ConstantInfo> constants_;
  KernelReuseCache kernel_cache_;
  std::unique_ptr<InlinedModule> inlined_module_;
  const std::string data_layout_;
  llvm::Triple target_triple_;

  InstructionToHostExecuteAsyncEvents instruction_to_host_execute_async_events_;

  // We should not emit kernels when loading thunks from a compilation result.
  const bool emit_kernels_;

  // Generates unique IDs for thunk creation.
  ThunkIdGenerator thunk_id_generator_;

  KernelCompiler* compiler_;
  const xla::cpu::TargetMachineOptions cpu_target_machine_options_;
  ObjectPool<std::unique_ptr<mlir::MLIRContext>>* mlir_context_pool_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_IR_EMITTER_CONTEXT_H_
