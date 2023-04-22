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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_EXECUTABLE_H_

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/simple_orc_jit.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"

namespace xla {
namespace cpu {

// CPU-targeting implementation of the XLA Executable interface.
//
// Wraps a JIT-ed object that can be executed "on device". We JIT for the host
// architecture, so JIT-ed code and host code share the same ABI.
class CpuExecutable : public Executable {
 public:
  CpuExecutable(std::unique_ptr<SimpleOrcJIT> jit,
                std::unique_ptr<const BufferAssignment> assignment,
                std::unique_ptr<HloModule> hlo_module,
                const string& entry_function_name,
                std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
                std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map);
  ~CpuExecutable() override;

  StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments,
      HloExecutionProfile* hlo_execution_profile) override;

  // Calls the generated function performing the computation with the given
  // arguments using the supplied buffers.
  Status ExecuteComputeFunction(
      const ExecutableRunOptions* run_options,
      absl::Span<MaybeOwningDeviceMemory const> buffers,
      HloExecutionProfile* hlo_execution_profile);

  // This should be called after set_ir_module_string.
  const string& ir_module_string() const { return ir_module_string_; }

  void set_ir_module_string(const string& ir_module_string) {
    ir_module_string_ = ir_module_string;
  }

  static int64 ShapeSizeBytes(const Shape& shape);

  // Type of the computation function we expect in the JIT.
  using ComputeFunctionType =
      void (*)(void* /*result*/, const ExecutableRunOptions* /*run_options*/,
               const void** /*args*/, void** /*buffer_table*/,
               int64* /*profile_counters*/);

  const ComputeFunctionType& compute_function() const {
    return compute_function_;
  }

  const BufferAssignment& buffer_assignment() const { return *assignment_; }

  int64 SizeOfGeneratedCodeInBytes() const override;

 private:
  // Creates an array suitable for passing as the "buffer_table" argument to the
  // JIT compiled function pointer.
  //
  // Returns (unowning_buffers, owning_buffers) where:
  //
  //  - unowning_buffers.data() can be passed as the buffer_table argument as-is
  //    and includes pointers to the scratch storage required by the
  //    computation, the live-out buffer into which the result will be written
  //    and entry computation parameters.
  //
  //  - owning_buffers contains owning pointers to the buffers that were
  //    allocated by this routine.  This routine allocates buffers for temporary
  //    storage and the live-out buffer into which the computation writes it
  //    result.
  //
  //  - buffers_to_free: buffers whose ownership was donated by the caller that
  //    are to be freed by the caller.
  StatusOr<std::vector<MaybeOwningDeviceMemory>> CreateBufferTable(
      se::DeviceMemoryAllocator* memory_allocator, int device_ordinal,
      absl::Span<ExecutionInput const> arguments);

  // Creates an Execution output holding ScopedShapedBuffer for holding the
  // result of the computation, moving buffers out of allocated_buffers and into
  // the result as appropriate.  The addresses are set according to buffer
  // assignment.
  StatusOr<ExecutionOutput> CreateResultShapedBuffer(
      const ServiceExecutableRunOptions* run_options,
      absl::Span<MaybeOwningDeviceMemory> buffers,
      absl::Span<ExecutionInput> arguments);

  // Returns the instruction value set of the root instruction of the entry
  // computation. Uses dataflow analysis from buffer assignment.
  const InstructionValueSet& GetRootValueSet() const;

  // The JIT containing compiled modules.
  const std::unique_ptr<SimpleOrcJIT> jit_;

  // Buffer assignment for the buffers we need to allocate.
  const std::unique_ptr<const BufferAssignment> assignment_;

  std::shared_ptr<const BufferAssignmentProto> buffer_assignment_;

  // The LLVM IR, in string format, of the unoptimized module generated for this
  // CpuExecutable. We save a string instead of an llvm::Module* because leaving
  // llvm::Module* in a singleton can cause the heap checker to emit false
  // positives.
  string ir_module_string_;

  // Unique identifier.
  string module_name_;

  ComputeFunctionType compute_function_;

  // Entry function name for the computation.
  const string entry_function_name_;

  TF_DISALLOW_COPY_AND_ASSIGN(CpuExecutable);
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_EXECUTABLE_H_
