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
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/execution_stream_assignment.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/name_uniquer.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {
// Maps async start ops to their async events so we can emit done thunk
// sharing events with corresponding start thunk. Async events may be null if
// the start op is degenerate (so not emitted). For Send and Recv, this maps
// <isRecv, channel_id> to the asyn events, as multiple Recv and Recv-done or
// multiple Send and Send-done may map to the same async events and a Recv-done
// or Send-done operand may not be its corresponding Recv or Send, when a
// Send-Recv chain inside a loop is pipelined.
using CollectivesAsyncEvents =
    absl::flat_hash_map<std::variant<mlir::Operation*, const HloInstruction*,
                                     std::pair<bool, uint64_t>>,
                        std::shared_ptr<CollectiveThunk::AsyncEvents>>;

// IrEmitterContext encapsulates common (mutable and immutable) data structures
// used by both IrEmitterNested and IrEmitterUnnested, such as the buffer
// assignment and the name uniquer.
class IrEmitterContext {
 public:
  IrEmitterContext(const HloModule* hlo_module,
                   const BufferAssignment* buffer_assignment,
                   const ExecutionStreamAssignment* execution_stream_assignment,
                   std::string platform_name,
                   const se::DeviceDescription& gpu_device_info,
                   mlir::MLIRContext* mlir_context, llvm::Module* llvm_module,
                   llvm::Module* llvm_module_constants, bool emit_kernels)
      : hlo_module_(hlo_module),
        buffer_assignment_(buffer_assignment),
        execution_stream_assignment_(execution_stream_assignment),
        platform_name_(std::move(platform_name)),
        gpu_device_info_(gpu_device_info),
        mlir_context_(mlir_context),
        llvm_module_(llvm_module),
        llvm_module_constants_(llvm_module_constants),
        emit_kernels_(emit_kernels) {}
  // Disallow copy and assign.
  IrEmitterContext(const IrEmitterContext&) = delete;
  IrEmitterContext& operator=(const IrEmitterContext&) = delete;

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
  se::CudaComputeCapability cuda_compute_capability() const {
    auto* cc =
        std::get_if<se::CudaComputeCapability>(&gpu_compute_capability());
    return cc != nullptr ? *cc : se::CudaComputeCapability();
  }
  se::RocmComputeCapability rocm_compute_capability() const {
    auto* cc =
        std::get_if<se::RocmComputeCapability>(&gpu_compute_capability());
    return cc != nullptr ? *cc : se::RocmComputeCapability();
  }
  mlir::MLIRContext* mlir_context() { return mlir_context_; }
  llvm::Module* llvm_module() { return llvm_module_; }
  // A separate module can optionally be used to emit constants.
  llvm::Module* llvm_module_constants() {
    return (llvm_module_constants_ == nullptr) ? llvm_module_
                                               : llvm_module_constants_;
  }
  NameUniquer* name_uniquer() { return &name_uniquer_; }

  std::vector<GpuExecutable::ConstantInfo>& constants() { return constants_; }

  // Emit a constant with a given number of element, given byte size of the
  // element, given symbol name and content.
  void emit_constant(int64_t num_elements, int64_t bytes_per_element,
                     absl::string_view symbol_name, int allocation_idx,
                     DenseDataIntermediate content, llvm::IRBuilderBase* b);

  const DebugOptions& debug_options() const {
    return hlo_module_->config().debug_options();
  }

  KernelReuseCache& kernel_cache() { return kernel_cache_; }
  CollectivesAsyncEvents& collectives_async_events() {
    return collectives_async_events_;
  }

  bool emit_kernels() const { return emit_kernels_; }

 private:
  const HloModule* hlo_module_;
  const BufferAssignment* buffer_assignment_;
  const ExecutionStreamAssignment* execution_stream_assignment_;
  std::string platform_name_;
  const se::DeviceDescription& gpu_device_info_;
  mlir::MLIRContext* mlir_context_;
  llvm::Module* llvm_module_;
  llvm::Module* llvm_module_constants_;
  NameUniquer name_uniquer_;
  std::vector<GpuExecutable::ConstantInfo> constants_;
  KernelReuseCache kernel_cache_;

  CollectivesAsyncEvents collectives_async_events_;

  // We should not emit kernels when loading thunks from a compilation result.
  const bool emit_kernels_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_IR_EMITTER_CONTEXT_H_
