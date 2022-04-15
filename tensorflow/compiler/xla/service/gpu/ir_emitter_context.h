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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_CONTEXT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_CONTEXT_H_

#include "llvm/IR/Module.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"

namespace xla {
namespace gpu {

// IrEmitterContext encapsulates common (mutable and immutable) data structures
// used by both IrEmitterNested and IrEmitterUnnested, such as the buffer
// assignment and the name uniquer.
class IrEmitterContext {
 public:
  IrEmitterContext(const HloModule* hlo_module,
                   const BufferAssignment* buffer_assignment,
                   std::string platform_name, GpuDeviceInfo gpu_device_info,
                   se::CudaComputeCapability cuda_compute_capability,
                   se::RocmComputeCapability rocm_compute_capability,
                   mlir::MLIRContext* mlir_context, llvm::Module* llvm_module)
      : hlo_module_(hlo_module),
        buffer_assignment_(buffer_assignment),
        platform_name_(std::move(platform_name)),
        gpu_device_info_(gpu_device_info),
        cuda_compute_capability_(cuda_compute_capability),
        rocm_compute_capability_(rocm_compute_capability),
        mlir_context_(mlir_context),
        llvm_module_(llvm_module) {}
  // Disallow copy and assign.
  IrEmitterContext(const IrEmitterContext&) = delete;
  IrEmitterContext& operator=(const IrEmitterContext&) = delete;

  // Simple accessors.
  const HloModule& hlo_module() const { return *hlo_module_; }
  const BufferAssignment& buffer_assignment() const {
    return *buffer_assignment_;
  }
  absl::string_view platform_name() const { return platform_name_; }
  GpuDeviceInfo gpu_device_info() const { return gpu_device_info_; }
  se::CudaComputeCapability cuda_compute_capability() const {
    return cuda_compute_capability_;
  }
  se::RocmComputeCapability rocm_compute_capability() const {
    return rocm_compute_capability_;
  }
  mlir::MLIRContext* mlir_context() { return mlir_context_; }
  llvm::Module* llvm_module() { return llvm_module_; }
  NameUniquer* name_uniquer() { return &name_uniquer_; }

  std::vector<GpuExecutable::ConstantInfo>& constants() { return constants_; }

  absl::Span<const BufferAllocation> allocations() const {
    if (buffer_assignment_) {
      return buffer_assignment_->Allocations();
    }
    return allocations_;
  }

  void set_allocations(absl::Span<const BufferAllocation> allocations) {
    CHECK_EQ(nullptr, buffer_assignment_);
    allocations_ = allocations;
  }

 private:
  const HloModule* hlo_module_;
  const BufferAssignment* buffer_assignment_;
  absl::Span<const BufferAllocation> allocations_;
  std::string platform_name_;
  GpuDeviceInfo gpu_device_info_;
  se::CudaComputeCapability cuda_compute_capability_;
  se::RocmComputeCapability rocm_compute_capability_;
  mlir::MLIRContext* mlir_context_;
  llvm::Module* llvm_module_;
  NameUniquer name_uniquer_;
  std::vector<GpuExecutable::ConstantInfo> constants_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_CONTEXT_H_
