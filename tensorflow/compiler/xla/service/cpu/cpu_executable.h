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
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/runtime/jit_executable.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/simple_orc_jit.h"
#include "tensorflow/compiler/xla/service/cpu/xla_framework.h"
#include "tensorflow/compiler/xla/service/custom_call_status_internal.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory_allocator.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace cpu {

// BufferDesc for passing raw `buffer` (i.e. void ptr + size) arguments.
class BufferDesc {
 public:
  BufferDesc(void* data, size_t size) : data_(data), size_(size) {}
  void* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  void* data_;
  size_t size_;
};

class XlaRuntimeCpuExecutable {
 public:
  explicit XlaRuntimeCpuExecutable(
      std::unique_ptr<runtime::JitExecutable> jit_executable,
      const XlaFrameworkMapping& xla_framework_mapping)
      : executable_(std::move(jit_executable)),
        xla_framework_mapping_(xla_framework_mapping) {}

  explicit XlaRuntimeCpuExecutable(
      std::unique_ptr<runtime::Executable> executable,
      const XlaFrameworkMapping& xla_framework_mapping)
      : executable_(std::move(executable)),
        xla_framework_mapping_(xla_framework_mapping) {}

  Status Execute(const std::vector<BufferDesc>& descriptor_table,
                 const ExecutableRunOptions* run_options);

  runtime::Executable& GetExecutable() {
    if (std::holds_alternative<std::unique_ptr<runtime::JitExecutable>>(
            executable_)) {
      runtime::JitExecutable* jit_executable =
          std::get<std::unique_ptr<runtime::JitExecutable>>(executable_).get();
      return *jit_executable->DefaultExecutable();
    } else {
      runtime::Executable* aot_executable =
          std::get<std::unique_ptr<runtime::Executable>>(executable_).get();
      return *aot_executable;
    }
  }

  StatusOr<std::string_view> GetObjFile() const {
    if (!std::holds_alternative<std::unique_ptr<runtime::JitExecutable>>(
            executable_)) {
      return InternalError("No JitExecutable");
    }

    runtime::JitExecutable* jit_executable =
        std::get<std::unique_ptr<runtime::JitExecutable>>(executable_).get();
    std::unique_ptr<llvm::MemoryBuffer> obj_file =
        jit_executable->DefaultExecutable()->obj_file();
    if (!obj_file)
      return InternalError("XlaRuntimeCpuExecutable didn't save the obj file");

    return std::string_view(obj_file->getBuffer());
  }

  StatusOr<std::string_view> GetMlirModule() const {
    if (!std::holds_alternative<std::unique_ptr<runtime::JitExecutable>>(
            executable_)) {
      return InternalError("No JitExecutable");
    }

    runtime::JitExecutable* jit_executable =
        std::get<std::unique_ptr<runtime::JitExecutable>>(executable_).get();
    return jit_executable->mlir_module();
  }

  XlaFrameworkMapping xla_framework_mapping() { return xla_framework_mapping_; }

 private:
  // In JIT compilation mode `JitExecutable` is used. In AOT compilation mode
  // `Executable` is used.
  std::variant<std::unique_ptr<runtime::JitExecutable>,
               std::unique_ptr<runtime::Executable>>
      executable_;

  XlaFrameworkMapping xla_framework_mapping_;
};

// CPU-targeting implementation of the XLA Executable interface.
//
// Wraps a JIT-ed object that can be executed "on device". We JIT for the host
// architecture, so JIT-ed code and host code share the same ABI.
class CpuExecutable : public Executable {
 public:
  CpuExecutable(std::unique_ptr<SimpleOrcJIT> jit,
                std::unique_ptr<const BufferAssignment> assignment,
                std::unique_ptr<HloModule> hlo_module,
                const std::string& entry_function_name,
                std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
                std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map);
  // XLA Runtime constructor.
  CpuExecutable(
      std::unique_ptr<HloModule> hlo_module,
      std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
      std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map,
      std::unique_ptr<const BufferAssignment> assignment,
      std::unique_ptr<XlaRuntimeCpuExecutable> xla_runtime_executable);

  ~CpuExecutable() override;

  bool IsXlaRuntime() const { return xla_runtime_executable_ != nullptr; }

  Status ExecuteXlaRuntime(const std::vector<BufferDesc>& descriptor_table,
                           const ExecutableRunOptions* run_options = nullptr) {
    return xla_runtime_executable_->Execute(descriptor_table, run_options);
  }

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

  // Returns an Executable that is loaded from an object file (XLA program
  // compiled to a native function using the XLA Runtime stack).
  static StatusOr<std::unique_ptr<Executable>> LoadFromObjFile(
      std::unique_ptr<HloModule> hlo_module, absl::string_view obj_file,
      absl::string_view mlir_module,
      std::unique_ptr<BufferAssignment> buffer_assignment,
      XlaFrameworkMapping xla_framework_mapping,
      runtime::JitExecutable::Options opts);

  // This should be called after set_ir_module_string.
  const std::string& ir_module_string() const { return ir_module_string_; }

  void set_ir_module_string(const std::string& ir_module_string) {
    ir_module_string_ = ir_module_string;
  }

  static int64_t ShapeSizeBytes(const Shape& shape);

  // Type of the computation function we expect in the JIT.
  using ComputeFunctionType =
      void (*)(void* /*result*/, const ExecutableRunOptions* /*run_options*/,
               const void** /*args*/, void** /*buffer_table*/,
               XlaCustomCallStatus* /*status*/, int64_t* /*profile_counters*/);

  const ComputeFunctionType& compute_function() const {
    return compute_function_;
  }

  const BufferAssignment& buffer_assignment() const { return *assignment_; }

  int64_t SizeOfGeneratedCodeInBytes() const override;

  StatusOr<std::string_view> GetObjFile() const {
    if (!IsXlaRuntime()) return InternalError("Not an XLA Runtime executable");
    return xla_runtime_executable_->GetObjFile();
  }

  StatusOr<std::string_view> GetMlirModule() const {
    if (!IsXlaRuntime()) return InternalError("Not an XLA Runtime executable");
    return xla_runtime_executable_->GetMlirModule();
  }

  StatusOr<XlaFrameworkMapping> GetXlaFrameworkMapping() const {
    if (!IsXlaRuntime()) return InternalError("Not an XLA Runtime executable");
    return xla_runtime_executable_->xla_framework_mapping();
  }

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
  std::string ir_module_string_;

  // Unique identifier.
  std::string module_name_;

  ComputeFunctionType compute_function_;

  // Entry function name for the computation.
  const std::string entry_function_name_;

  // If not null, XLA Runtime is enabled.
  std::unique_ptr<XlaRuntimeCpuExecutable> xla_runtime_executable_;

  CpuExecutable(const CpuExecutable&) = delete;
  CpuExecutable& operator=(const CpuExecutable&) = delete;
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_EXECUTABLE_H_
