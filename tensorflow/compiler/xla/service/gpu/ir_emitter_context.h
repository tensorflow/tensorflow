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

#include "external/llvm/include/llvm/IR/Module.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/partition_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/temp_buffer_offsets.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

// IrEmitterContext encapsulates common (mutable and immutable) data structures
// used by both IrEmitterNested and IrEmitterUnnested, such as the buffer
// assignment and the name uniquer.
class IrEmitterContext {
 public:
  IrEmitterContext(const HloModule* hlo_module,
                   const BufferAssignment* buffer_assignment,
                   const TempBufferOffsets* temp_buffer_offsets,
                   const perftools::gputools::DeviceDescription* device_desc,
                   llvm::Module* llvm_module)
      : hlo_module_(hlo_module),
        buffer_assignment_(buffer_assignment),
        temp_buffer_offsets_(temp_buffer_offsets),
        device_desc_(device_desc),
        llvm_module_(llvm_module) {}
  // Disallow copy and assign.
  IrEmitterContext(const IrEmitterContext&) = delete;
  IrEmitterContext& operator=(const IrEmitterContext&) = delete;

  // Simple accessors.
  const HloModule& hlo_module() const { return *hlo_module_; }
  const BufferAssignment& buffer_assignment() const {
    return *buffer_assignment_;
  }
  const TempBufferOffsets& temp_buffer_offsets() const {
    return *temp_buffer_offsets_;
  }
  const perftools::gputools::DeviceDescription& device_description() const {
    return *device_desc_;
  }
  llvm::Module* llvm_module() { return llvm_module_; }
  NameUniquer* name_uniquer() { return &name_uniquer_; }

 private:
  const HloModule* hlo_module_;
  const BufferAssignment* buffer_assignment_;
  const TempBufferOffsets* temp_buffer_offsets_;
  const perftools::gputools::DeviceDescription* device_desc_;
  llvm::Module* llvm_module_;
  NameUniquer name_uniquer_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_CONTEXT_H_
