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

#ifndef XLA_SERVICE_GPU_IR_EMITTER_CONTEXT_H_
#define XLA_SERVICE_GPU_IR_EMITTER_CONTEXT_H_

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/name_uniquer.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// IrEmitterContext encapsulates common (mutable and immutable) data structures
// used by both IrEmitterNested and IrEmitterUnnested, such as the buffer
// assignment and the name uniquer.
class IrEmitterContext {
 public:
  IrEmitterContext(const HloModule* hlo_module,
                   const BufferAssignment* buffer_assignment,
                   std::string platform_name,
                   const se::DeviceDescription& gpu_device_info,
                   mlir::MLIRContext* mlir_context, llvm::Module* llvm_module,
                   bool emit_ir_from_hlo)
      : hlo_module_(hlo_module),
        buffer_assignment_(buffer_assignment),
        platform_name_(std::move(platform_name)),
        gpu_device_info_(gpu_device_info),
        mlir_context_(mlir_context),
        llvm_module_(llvm_module),
        emit_ir_from_hlo_(emit_ir_from_hlo) {}
  // Disallow copy and assign.
  IrEmitterContext(const IrEmitterContext&) = delete;
  IrEmitterContext& operator=(const IrEmitterContext&) = delete;

  // Simple accessors.
  const HloModule& hlo_module() const { return *hlo_module_; }
  const BufferAssignment& buffer_assignment() const {
    return *buffer_assignment_;
  }
  absl::string_view platform_name() const { return platform_name_; }
  const se::DeviceDescription& gpu_device_info() const {
    return gpu_device_info_;
  }
  se::CudaComputeCapability cuda_compute_capability() const {
    auto* cc = std::get_if<se::CudaComputeCapability>(
        &gpu_device_info_.gpu_compute_capability());
    return cc != nullptr ? *cc : se::CudaComputeCapability();
  }
  se::RocmComputeCapability rocm_compute_capability() const {
    auto* cc = std::get_if<se::RocmComputeCapability>(
        &gpu_device_info_.gpu_compute_capability());
    return cc != nullptr ? *cc : se::RocmComputeCapability();
  }
  mlir::MLIRContext* mlir_context() { return mlir_context_; }
  llvm::Module* llvm_module() { return llvm_module_; }
  NameUniquer* name_uniquer() { return &name_uniquer_; }

  std::vector<GpuExecutable::ConstantInfo>& constants() { return constants_; }

  absl::Span<const BufferAllocation* const> allocations() const {
    return allocations_;
  }

  void set_allocations(absl::Span<const BufferAllocation* const> allocations) {
    allocations_ = allocations;
  }

  // Emit a constant with a given number of element, given byte size of the
  // element, given symbol name and content.
  void emit_constant(int64_t num_elements, int64_t bytes_per_element,
                     absl::string_view symbol_name, int allocation_idx,
                     DenseDataIntermediate content, llvm::IRBuilder<>* b);

  const DebugOptions& debug_options() const {
    return hlo_module_->config().debug_options();
  }

  bool emit_ir_from_hlo() const { return emit_ir_from_hlo_; }

 private:
  const HloModule* hlo_module_;
  const BufferAssignment* buffer_assignment_;

  // Stores pointer to buffer allocations in the order of the LMHLO entry args.
  // LMHLO-based emitters need the ordering to locate the buffer allocation.
  // This should be removed once LMHLO-based emitters are removed.
  absl::Span<const BufferAllocation* const> allocations_;

  std::string platform_name_;
  const se::DeviceDescription& gpu_device_info_;
  mlir::MLIRContext* mlir_context_;
  llvm::Module* llvm_module_;
  NameUniquer name_uniquer_;
  std::vector<GpuExecutable::ConstantInfo> constants_;
  const bool emit_ir_from_hlo_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_IR_EMITTER_CONTEXT_H_
