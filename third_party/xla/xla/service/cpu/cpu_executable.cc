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

#include "xla/service/cpu/cpu_executable.h"

#include <stdint.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/dynamic_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/Support/Error.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/buffer_allocations.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/service/cpu/simple_orc_jit.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_status_internal.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_execution_profile.h"
#include "xla/service/hlo_value.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/xla_debug_info_manager.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/host/host_kernel_c_api.h"
#include "xla/stream_executor/host/host_stream.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace cpu {

using ConstantAllocation = CpuExecutable::ConstantAllocation;
using HostKernels = CpuExecutable::HostKernels;

HostKernels::HostKernels(SimpleOrcJIT* jit) : jit_(jit) {}

absl::StatusOr<SE_HOST_Kernel*> HostKernels::Find(std::string_view name) {
  VLOG(2) << "Find host kernel with a name " << name;

  llvm::Expected<llvm::orc::ExecutorSymbolDef> sym =
      jit_->FindCompiledSymbol(std::string(name));
  if (!sym) {
    return absl::InvalidArgumentError(
        absl::StrCat("Can't resolve host kernel with a name ", name,
                     " in the jit compiled module."));
  }
  return reinterpret_cast<SE_HOST_Kernel*>(sym->getAddress().getValue());
}

se::DeviceMemoryBase ConstantAllocation::AsDeviceMemoryBase() const {
  if (auto* empty = std::get_if<std::monostate>(&data)) {
    return se::DeviceMemoryBase();
  }

  if (auto* owned = std::get_if<std::vector<uint8_t>>(&data)) {
    return se::DeviceMemoryBase(
        const_cast<void*>(reinterpret_cast<const void*>(owned->data())),
        owned->size());
  }

  auto* view = std::get_if<absl::Span<const uint8_t>>(&data);
  return se::DeviceMemoryBase(
      const_cast<void*>(reinterpret_cast<const void*>(view->data())),
      view->size());
}

absl::StatusOr<std::unique_ptr<CpuExecutable>> CpuExecutable::Create(
    std::unique_ptr<SimpleOrcJIT> jit,
    std::unique_ptr<const BufferAssignment> assignment,
    std::unique_ptr<HloModule> hlo_module,
    const std::string& entry_function_name,
    std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
    std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map) {
  VLOG(2) << "Create CpuExecutable from a jit compiled function: "
          << entry_function_name << ", module=" << hlo_module->name();

  std::unique_ptr<CpuExecutable> executable(new CpuExecutable(
      std::move(hlo_module), std::move(hlo_profile_printer_data),
      std::move(hlo_profile_index_map), std::move(assignment)));
  executable->jit_ = std::move(jit);
  executable->module_name_ = entry_function_name;

  // Resolve symbols in the constructor rather than at execution time to avoid
  // races because FindSymbol is not thread safe.
  llvm::Expected<llvm::orc::ExecutorSymbolDef> sym =
      executable->jit_->FindCompiledSymbol(entry_function_name);
  // We expect to find the symbol provided with entry_function_name; otherwise
  // this is an internal error.
  if (!sym) {
    return absl::InvalidArgumentError(
        absl::StrCat("Symbol ", entry_function_name, " not found."));
  }
  // getAddress can do work under the hood in the jit, so it needs to be
  // guarded by the mutex.
  executable->compute_function_ =
      reinterpret_cast<ComputeFunctionType>(sym->getAddress().getValue());
  VLOG(1) << "compute_function_ at address "
          << reinterpret_cast<void*>(executable->compute_function_);
  executable->jit_->DoneCompiling();
  return executable;
}

absl::StatusOr<std::unique_ptr<CpuExecutable>> CpuExecutable::Create(
    std::unique_ptr<SimpleOrcJIT> jit,
    std::unique_ptr<const BufferAssignment> assignment,
    std::unique_ptr<HloModule> hlo_module, ThunkSequence thunks,
    std::vector<ConstantAllocation> constants,
    std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
    std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map) {
  VLOG(2) << "Create CpuExecutable from a thunk sequence; module="
          << hlo_module->name() << ", constants=" << constants.size();

  std::unique_ptr<CpuExecutable> executable(new CpuExecutable(
      std::move(hlo_module), std::move(hlo_profile_printer_data),
      std::move(hlo_profile_index_map), std::move(assignment)));

  executable->jit_ = std::move(jit);
  executable->thunks_ = std::move(thunks);
  executable->host_kernels_ = HostKernels(executable->jit_.get());

  // Re-index constants by their allocation index to allow efficient lookup.
  for (auto& constant : constants) {
    if (executable->constants_.size() <= constant.index) {
      executable->constants_.resize(constant.index + 1);
    }
    executable->constants_[constant.index] = std::move(constant);
  }

  return executable;
}

CpuExecutable::CpuExecutable(
    std::unique_ptr<HloModule> hlo_module,
    std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
    std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map,
    std::unique_ptr<const BufferAssignment> assignment)
    : Executable(std::move(hlo_module), std::move(hlo_profile_printer_data),
                 std::move(hlo_profile_index_map)),
      assignment_(std::move(assignment)) {
  if (assignment_ && has_module()) {
    XlaDebugInfoManager::Get()->RegisterModule(shared_module(),
                                               assignment_->ToProto());
  }
}

CpuExecutable::~CpuExecutable() {
  if (has_module()) {
    XlaDebugInfoManager::Get()->UnregisterModule(module().unique_id());
  }
}

static absl::StatusOr<MaybeOwningDeviceMemory> MemoryForAllocation(
    const BufferAllocation& allocation,
    absl::Span<const ExecutionInput> arguments,
    absl::Span<const ConstantAllocation> constants,
    se::DeviceMemoryAllocator* memory_allocator, int device_ordinal) {
  VLOG(3) << allocation.ToString();
  if (allocation.is_entry_computation_parameter()) {
    se::DeviceMemoryBase out = arguments[allocation.parameter_number()]
                                   .Buffer(allocation.param_shape_index())
                                   .AsDeviceMemoryBase();
    CHECK_LE(allocation.size(), out.size())
        << "Size mismatch on param " << allocation.parameter_number()
        << " at shape index " << allocation.param_shape_index().ToString();
    VLOG(3) << "allocation is a parameter";
    return MaybeOwningDeviceMemory{out};
  } else if (allocation.is_constant()) {
    VLOG(3) << "allocation is a constant";
    if (allocation.index() < constants.size()) {
      return MaybeOwningDeviceMemory(
          constants[allocation.index()].AsDeviceMemoryBase());
    }
    return MaybeOwningDeviceMemory{se::DeviceMemoryBase{}};
  } else if (allocation.is_thread_local()) {
    VLOG(3) << "buffer is thread-local";
    return MaybeOwningDeviceMemory{se::DeviceMemoryBase{}};
  }

  int64_t buffer_size = allocation.size();
  TF_ASSIGN_OR_RETURN(se::OwningDeviceMemory out,
                      memory_allocator->Allocate(device_ordinal, buffer_size));
  VLOG(3) << "buffer allocated " << buffer_size << " bytes [" << out->opaque()
          << "]";

  // Since the output buffer and all the temporary buffers were written into
  // by the JITed code, memory sanitizer has no way of knowing their memory was
  // initialized. Mark them initialized so that memory sanitizer doesn't flag
  // loads from these buffers.
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(out->opaque(), buffer_size);
  return MaybeOwningDeviceMemory{std::move(out)};
}

absl::StatusOr<std::vector<MaybeOwningDeviceMemory>>
CpuExecutable::CreateBufferTable(se::DeviceMemoryAllocator* memory_allocator,
                                 int device_ordinal,
                                 absl::Span<ExecutionInput const> arguments) {
  std::vector<MaybeOwningDeviceMemory> buffers(
      assignment_->Allocations().size());
  VLOG(3) << "Allocating " << assignment_->Allocations().size()
          << " allocations for module " << module().name();
  for (BufferAllocation::Index i = 0; i < assignment_->Allocations().size();
       ++i) {
    const BufferAllocation& allocation = assignment_->GetAllocation(i);
    TF_ASSIGN_OR_RETURN(buffers[i],
                        MemoryForAllocation(allocation, arguments, constants_,
                                            memory_allocator, device_ordinal));
  }

  if (VLOG_IS_ON(3)) {
    TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice result_slice,
                        assignment_->GetUniqueTopLevelOutputSlice());
    VLOG(3) << "result index: " << result_slice.index();
  }
  return std::move(buffers);
}

absl::Status CpuExecutable::ExecuteComputeFunction(
    const ExecutableRunOptions* run_options,
    absl::Span<MaybeOwningDeviceMemory const> buffers,
    HloExecutionProfile* hlo_execution_profile) {
  uint64_t start_micros = tsl::Env::Default()->NowMicros();

  size_t profile_counters_size =
      hlo_execution_profile ? hlo_execution_profile->profile_counters().size()
                            : 0;
  int64_t* profile_counters =
      hlo_execution_profile
          ? hlo_execution_profile->mutable_profile_counters()->data()
          : nullptr;

  // Call the computation function following the calling convention. See the
  // definition of 'ComputeFunctionType' for the details of the calling
  // convention of JITed functions.
  std::vector<void*> buffer_pointers;
  for (auto& buffer : buffers) {
    buffer_pointers.push_back(
        const_cast<void*>(buffer.AsDeviceMemoryBase().opaque()));
  }

  VLOG(3) << "Executing compute function:";
  VLOG(3) << absl::StrFormat("  Number of buffer table entries: %u",
                             buffer_pointers.size());
  auto ptr_printer = [](std::string* out, const void* p) {
    absl::StrAppend(out, absl::StrFormat("%p", p));
  };
  VLOG(3) << absl::StrFormat("  Buffer table: [%s]",
                             absl::StrJoin(buffer_pointers, ", ", ptr_printer));
  VLOG(3) << absl::StrFormat("  Number of profile counters: %u",
                             profile_counters_size);
  VLOG(3) << absl::StrFormat("  Profile counters: %p", profile_counters);

  auto record_profile = [&]() {
    uint64_t end_micros = tsl::Env::Default()->NowMicros();
    if (run_options->execution_profile()) {
      const double nanoseconds = (end_micros - start_micros) * 1000.0;
      run_options->execution_profile()->set_compute_time_ns(
          std::max(nanoseconds, 1.0));
      // If hlo profiling was disabled then the cycle count is left empty.
      if (hlo_execution_profile) {
        run_options->execution_profile()->set_compute_cycle_count(
            hlo_execution_profile->total_cycles_executed(
                *module().entry_computation()));
      }
    }
  };

  XlaCustomCallStatus status;
  // For the entry computation (like all global computations), all inputs and
  // outputs are in the buffer table, and both the result pointer and args
  // array pointers are unused (so we set them to 'nullptr').
  compute_function_(nullptr, run_options, nullptr, buffer_pointers.data(),
                    &status, profile_counters);
  record_profile();
  std::optional<absl::string_view> error_message =
      CustomCallStatusGetMessage(&status);
  if (error_message) {
    return Internal("CustomCall failed: %s", *error_message);
  }

  return absl::OkStatus();
}

absl::Status CpuExecutable::ExecuteThunks(
    const ExecutableRunOptions* run_options,
    absl::Span<MaybeOwningDeviceMemory const> buffers,
    HloExecutionProfile* hlo_execution_profile) {
  uint64_t start_ns = tsl::Env::Default()->NowNanos();

  size_t profile_counters_size =
      hlo_execution_profile ? hlo_execution_profile->profile_counters().size()
                            : 0;
  int64_t* profile_counters =
      hlo_execution_profile
          ? hlo_execution_profile->mutable_profile_counters()->data()
          : nullptr;

  BufferAllocations allocations(buffers);

  VLOG(3) << "Executing XLA:CPU thunks:";
  VLOG(3) << absl::StrFormat("  Number of buffer allocations: %u",
                             buffers.size());
  auto mem_printer = [](std::string* out, const MaybeOwningDeviceMemory& mem) {
    absl::StrAppend(out,
                    absl::StrFormat("%p", mem.AsDeviceMemoryBase().opaque()));
  };
  VLOG(3) << absl::StrFormat("  Buffer allocations: [%s]",
                             absl::StrJoin(buffers, ", ", mem_printer));
  VLOG(3) << absl::StrFormat("  Number of profile counters: %u",
                             profile_counters_size);
  VLOG(3) << absl::StrFormat("  Profile counters: %p", profile_counters);

  Thunk::ExecuteParams execute_params = {&*host_kernels_, &allocations};
  absl::Status executed = thunks_->Execute(execute_params);

  if (run_options->execution_profile()) {
    uint64_t end_ns = tsl::Env::Default()->NowNanos();
    run_options->execution_profile()->set_compute_time_ns(
        std::max<int64_t>(end_ns - start_ns, 1));
    // If hlo profiling was disabled then the cycle count is left empty.
    if (hlo_execution_profile) {
      run_options->execution_profile()->set_compute_cycle_count(
          hlo_execution_profile->total_cycles_executed(
              *module().entry_computation()));
    }
  }

  return executed;
}

absl::StatusOr<ExecutionOutput> CpuExecutable::CreateResultShapedBuffer(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<MaybeOwningDeviceMemory> buffers,
    absl::Span<ExecutionInput> arguments) {
  se::Stream* stream = run_options->stream();
  ExecutionOutput result(/*on_device_shape=*/result_shape(),
                         run_options->allocator(),
                         stream->parent()->device_ordinal());
  const HloInputOutputAliasConfig& input_output_alias =
      module().input_output_alias_config();
  HloInstruction* root = hlo_module_->entry_computation()->root_instruction();
  const Shape& root_shape = root->shape();

  // Move se::OwningDeviceMemory values which contain the array(s) of the result
  // into the respective location in ScopedShapedBuffer which is returned to the
  // caller.
  for (auto& p : result.MutableResult()->buffers()) {
    const ShapeIndex& index = p.first;
    se::DeviceMemoryBase& result_buffer = p.second;
    const HloValueSet& sources = this->GetRootValueSet().element(index);
    // The points to set is unambiguous so the set should be a
    // singleton.
    CHECK_EQ(1, sources.values().size());
    const HloValue* value_source = sources.values()[0];
    HloInstruction* src = value_source->instruction();

    // The source for this result buffer can be a nested buffer such as
    // a tuple element.
    TF_ASSIGN_OR_RETURN(
        const BufferAllocation::Slice slice,
        this->assignment_->GetUniqueSlice(src, value_source->index()));
    const BufferAllocation::Index buffer_index = slice.index();

    // TODO(cheshire): duplication with other backends.
    std::optional<HloInputOutputAliasConfig::Alias> alias =
        input_output_alias.GetAliasedParameter(index);
    if (alias) {
      CHECK_LT(alias->parameter_number, arguments.size());
      ExecutionInput& input = arguments[alias->parameter_number];
      MaybeOwningDeviceMemory* maybe_owning_memory =
          input.MutableBuffer(alias->parameter_index);
      if (alias->must_alias() && !maybe_owning_memory->HasOwnership()) {
        return InvalidArgument(
            "An input was configured to be must-alias at "
            "compile time but not donated at runtime: %s",
            alias->ToString());
      }
      if (std::optional<se::OwningDeviceMemory> owning =
              maybe_owning_memory->Release()) {
        // If the caller passes the ownership of the device memory, reuse it
        // as the output buffer. It is up to the caller whether or not to
        // donate a buffer; the aliasing information describes which buffers
        // may alias, not buffers that must alias.
        se::DeviceMemoryBase argument_buffer = owning->Release();
        *maybe_owning_memory = argument_buffer;
        result_buffer = argument_buffer;
        // The caller is giving us the
        // input buffer, but in case of error of the execute call, we should
        // not be releasing it as it contains valid data (for example, it is a
        // parameter which the user wants us to alias, in a gradient update
        // computation). So we store the index into the result in the aliased
        // vactor, which will be fed to the ExecutionOutput, which will be
        // using the indices to drop the addresses from its own
        // ScopedShapedBuffer result, if the ExecutionOutput is not committed.
        result.AddAliasedIndex(index);
      } else {
        VLOG(3) << "Using copy-protection: aliasing is specified, but the "
                   "buffer is not donated; allocating a fresh buffer";
        int64_t allocation_size =
            ShapeUtil::ByteSizeOf(ShapeUtil::GetSubshape(root_shape, index));
        TF_ASSIGN_OR_RETURN(
            se::OwningDeviceMemory allocated_buffer,
            run_options->allocator()->Allocate(
                stream->parent()->device_ordinal(), allocation_size));
        result_buffer = allocated_buffer.Release();
        MaybeOwningDeviceMemory& registered_buffer = buffers[buffer_index];
        CHECK_EQ(result_buffer.size(),
                 registered_buffer.AsDeviceMemoryBase().size());
        std::memcpy(/*dest=*/result_buffer.opaque(),
                    /*src=*/registered_buffer.AsDeviceMemoryBase().opaque(),
                    /*n=*/result_buffer.size());
        registered_buffer = result_buffer;
      }
    }

    if (result_buffer.is_null()) {
      MaybeOwningDeviceMemory& buffer = buffers[buffer_index];
      if (std::optional<se::OwningDeviceMemory> owned_buffer =
              buffer.Release()) {
        result_buffer = owned_buffer->Release();
        buffer = result_buffer;
      } else {
        result_buffer = buffer.AsDeviceMemoryBase();
        result.AddAliasedIndex(index);
      }
    }
  }
  return std::move(result);
}

absl::StatusOr<ExecutionOutput> CpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  if (GetRootValueSet().IsAmbiguous()) {
    return Unimplemented("Points-to set of root instruction is ambiguous");
  }

  if (hlo_module_) {
    const HloComputation* entry_comp = hlo_module_->entry_computation();
    CHECK_EQ(entry_comp->num_parameters(), arguments.size())
        << "Wrong number of arguments passed when running executable";
    for (int64_t i = 0; i < entry_comp->num_parameters(); ++i) {
      const Shape& expected_shape =
          entry_comp->parameter_instruction(i)->shape();
      const Shape& actual_shape = arguments[i].Buffers().shape();
      TF_RET_CHECK(
          ShapeUtil::DynamicShapeIsCompatible(actual_shape, expected_shape))
          << "Shape mismatch on argument " << i << ", "
          << expected_shape.ToString(/*print_layout=*/true) << " vs. "
          << actual_shape.ToString(/*print_layout=*/true);
    }
  }

  auto* host_stream =
      dynamic_cast<se::host::HostStream*>(run_options->stream());
  se::Stream* stream = run_options->stream();
  se::DeviceMemoryAllocator* memory_allocator = run_options->allocator();
  TF_ASSIGN_OR_RETURN(
      std::vector<MaybeOwningDeviceMemory> buffers,
      CreateBufferTable(memory_allocator, stream->parent()->device_ordinal(),
                        arguments));

  TF_ASSIGN_OR_RETURN(
      ExecutionOutput result,
      CreateResultShapedBuffer(run_options, absl::MakeSpan(buffers),
                               absl::MakeSpan(arguments)));

  // Logically we want this lambda to capture `buffers` by move, ultimately our
  // functor needs to be wrapped in an std::function, and that requires its
  // functor to be copyable.  Thus we perpetrate the hack of capturing buffers
  // "by shared pointer".
  //
  // We also need to change the types of some of the variables we capture:
  // run_options needs to change from a pointer to a value type, and arguments
  // needs to change from a Span into a vector.  We use a struct instead
  // of a lambda to make this explicit.
  struct AsyncRunTask {
    CpuExecutable* executable;
    ServiceExecutableRunOptions run_options;
    std::shared_ptr<std::vector<MaybeOwningDeviceMemory>> task_buffers;
    HloExecutionProfile* hlo_execution_profile;

    absl::Status operator()() {
      if (executable->has_compute_function()) {
        return executable->ExecuteComputeFunction(
            &run_options.run_options(), *task_buffers, hlo_execution_profile);
      } else if (executable->has_thunks()) {
        return executable->ExecuteThunks(&run_options.run_options(),
                                         *task_buffers, hlo_execution_profile);
      } else {
        return Internal("No compute function or thunks found.");
      }
    }
  };
  host_stream->EnqueueTaskWithStatus(
      AsyncRunTask{this, *run_options,
                   std::make_shared<std::vector<MaybeOwningDeviceMemory>>(
                       std::move(buffers)),
                   hlo_execution_profile});

  MarkToBeReleasedArguments(absl::MakeSpan(arguments), result);
  return std::move(result);
}

/*static*/ int64_t CpuExecutable::ShapeSizeBytes(const Shape& shape) {
  // On the cpu, opaques are pointers.
  if (shape.IsOpaque()) {
    return sizeof(void*);
  }
  if (shape.is_static() || shape.IsTuple()) {
    return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
  }
  // Each dynamic dimension size is represented as a S32.
  int64_t metadata_size = sizeof(int32_t) * shape.dimensions_size();
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*)) + metadata_size;
}

const InstructionValueSet& CpuExecutable::GetRootValueSet() const {
  return assignment_->dataflow_analysis().GetInstructionValueSet(
      module().entry_computation()->root_instruction());
}

int64_t CpuExecutable::SizeOfGeneratedCodeInBytes() const {
  return jit_ ? jit_->SizeOfGeneratedCodeInBytes() : 0;
}

}  // namespace cpu
}  // namespace xla
