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

#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/simple_orc_jit.h"
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

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
                std::unique_ptr<const HloModule> hlo_module,
                const string& entry_function_name,
                std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
                std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map);
  ~CpuExecutable() override {}

  StatusOr<ScopedShapedBuffer> ExecuteOnStream(
      const ServiceExecutableRunOptions* run_options,
      tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
      HloExecutionProfile* hlo_execution_profile) override;

  StatusOr<ScopedShapedBuffer> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments) override;

  // This should be called after set_ir_module_string.
  const string& ir_module_string() const { return ir_module_string_; }

  void set_ir_module_string(const string& ir_module_string) {
    ir_module_string_ = ir_module_string;
  }

  static int64 ShapeSizeBytes(const Shape& shape);

  // Type of the computation function we expect in the JIT.
  using ComputeFunctionType = void (*)(
      void* /*result*/, const ExecutableRunOptions* /*run_options*/,
      const void** /*args*/, void** /*temps*/, int64* /*profile_counters*/);

  const ComputeFunctionType& compute_function() const {
    return compute_function_;
  }

  const BufferAssignment& buffer_assignment() const { return *assignment_; }

 private:
  // Creates an array suitable for passing as the "temps" argument to the JIT
  // compiled function pointer.
  //
  // Returns (unowning_buffers, owning_buffers) where:
  //
  //  - unowning_buffers.data() can be passed as the temps argument as-is and
  //    includes pointers to the scratch storage required by the computation,
  //    the live-out buffer into which the result will be written and entry
  //    computation parameters.
  //
  //  - owning_buffers contains owning pointers to the buffers that were
  //    allocated by this routine.  This routine allocates buffers for temporary
  //    storage and the live-out buffer into which the computation writes it
  //    result.
  StatusOr<std::pair<std::vector<se::DeviceMemoryBase>,
                     std::vector<OwningDeviceMemory>>>
  CreateTempArray(DeviceMemoryAllocator* memory_allocator, int device_ordinal,
                  tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments);

  // Calls the generated function performing the computation with the given
  // arguments using the supplied buffers.
  Status ExecuteComputeFunction(
      const ExecutableRunOptions* run_options,
      tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> buffers,
      HloExecutionProfile* hlo_execution_profile);

  // Creates a ScopedShapedBuffer for holding the result of the computation,
  // moving buffers out of allocated_buffers and into the result as appropriate.
  // The addresses are set according to buffer assignment.
  StatusOr<ScopedShapedBuffer> CreateResultShapedBuffer(
      const ServiceExecutableRunOptions* run_options,
      tensorflow::gtl::MutableArraySlice<OwningDeviceMemory> buffers);

  // Returns the points-to set of the root instruction of the entry
  // computation. Uses points-to analysis from buffer assignment.
  const PointsToSet& GetRootPointsToSet() const;

  // The JIT containing compiled modules.
  const std::unique_ptr<SimpleOrcJIT> jit_;

  // Buffer assignment for the buffers we need to allocate.
  const std::unique_ptr<const BufferAssignment> assignment_;

  // The LLVM IR, in string format, of the unoptimized module generated for this
  // CpuExecutable. We save a string instead of an llvm::Module* because leaving
  // llvm::Module* in a singleton can cause the heap checker to emit false
  // positives.
  string ir_module_string_;

  ComputeFunctionType compute_function_;

  // Entry function name for the computation.
  const string entry_function_name_;

  TF_DISALLOW_COPY_AND_ASSIGN(CpuExecutable);
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_EXECUTABLE_H_
