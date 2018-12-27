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

#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"

#include <stdint.h>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/host/host_stream.h"

namespace xla {
namespace cpu {

CpuExecutable::CpuExecutable(
    std::unique_ptr<SimpleOrcJIT> jit,
    std::unique_ptr<const BufferAssignment> assignment,
    std::unique_ptr<HloModule> hlo_module, const string& entry_function_name,
    std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
    std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map)
    : Executable(std::move(hlo_module), std::move(hlo_profile_printer_data),
                 std::move(hlo_profile_index_map)),
      jit_(std::move(jit)),
      assignment_(std::move(assignment)) {
  // Resolve symbols in the constructor rather than at execution time to avoid
  // races because FindSymbol is not thread safe.
  llvm::JITSymbol sym = jit_->FindCompiledSymbol(entry_function_name);
  // We expect to find the symbol provided with entry_function_name; otherwise
  // this is an internal error.
  CHECK(sym) << "Symbol " << entry_function_name << " not found.";
  // getAddress can do work under the hood in the jit, so it needs to be
  // guarded by the mutex.
  compute_function_ =
      reinterpret_cast<ComputeFunctionType>(cantFail(sym.getAddress()));
  VLOG(1) << "compute_function_ at address "
          << reinterpret_cast<void*>(compute_function_);
}

StatusOr<std::pair<std::vector<se::DeviceMemoryBase>,
                   std::vector<OwningDeviceMemory>>>
CpuExecutable::CreateBufferTable(
    DeviceMemoryAllocator* memory_allocator, int device_ordinal,
    absl::Span<const ShapedBuffer* const> arguments) {
  std::vector<se::DeviceMemoryBase> unowning_buffers(
      assignment_->Allocations().size());
  std::vector<OwningDeviceMemory> owning_buffers(
      assignment_->Allocations().size());
  VLOG(3) << "Allocating " << assignment_->Allocations().size()
          << " allocations for module " << module().name();
  for (BufferAllocation::Index i = 0; i < assignment_->Allocations().size();
       ++i) {
    auto& allocation = assignment_->GetAllocation(i);

    VLOG(3) << allocation.ToString();

    if (allocation.is_entry_computation_parameter()) {
      unowning_buffers[i] = arguments[allocation.parameter_number()]->buffer(
          allocation.param_shape_index());
      VLOG(3) << "allocation #" << i << " is a parameter";
      continue;
    }

    if (allocation.is_constant()) {
      VLOG(3) << "allocation #" << i << " is a constant";
      continue;
    }

    if (allocation.is_thread_local()) {
      VLOG(3) << "buffer #" << i << " is thread-local";
      continue;
    }

    int64 buffer_size = allocation.size();
    if (!owning_buffers[i].is_null()) {
      VLOG(3) << "buffer #" << i
              << " is in the preallocated result ShapedBuffer";
    } else {
      TF_ASSIGN_OR_RETURN(owning_buffers[i], memory_allocator->Allocate(
                                                 device_ordinal, buffer_size));
      unowning_buffers[i] = owning_buffers[i].AsDeviceMemoryBase();

      VLOG(3) << "buffer #" << i << " allocated " << buffer_size << " bytes ["
              << owning_buffers[i].opaque() << "]";
    }

    // Since the output buffer and all the temporary buffers were written into
    // by the JITed code, msan has no way of knowing their memory was
    // initialized. Mark them initialized so that msan doesn't flag loads from
    // these buffers.
    TF_ANNOTATE_MEMORY_IS_INITIALIZED(owning_buffers[i].opaque(), buffer_size);
  }

  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice result_slice,
                      assignment_->GetUniqueTopLevelOutputSlice());
  VLOG(3) << "result index: " << result_slice.index();

  return {{std::move(unowning_buffers), std::move(owning_buffers)}};
}

Status CpuExecutable::ExecuteComputeFunction(
    const ExecutableRunOptions* run_options,
    absl::Span<const se::DeviceMemoryBase> buffers,
    HloExecutionProfile* hlo_execution_profile) {
  // The calling convention for JITed functions is:
  //
  //  void function(void* result, const void* run_options, void** args_array,
  //                void** buffer_table)
  //
  // result: Points at the result.
  // run_options: the ExecutableRunOptions object.
  // args_array: null
  // buffer_table: An array of pointers, containing pointers to temporary
  //   buffers required by the executable adn pointers to entry computation
  //   parameters.
  //

  uint64 start_micros = tensorflow::Env::Default()->NowMicros();

  size_t profile_counters_size =
      hlo_execution_profile ? hlo_execution_profile->profile_counters().size()
                            : 0;
  int64* profile_counters =
      hlo_execution_profile
          ? hlo_execution_profile->mutable_profile_counters()->data()
          : nullptr;

  // Call the computation function following the calling convention.
  std::vector<void*> buffer_pointers;
  for (auto& buffer : buffers) {
    buffer_pointers.push_back(const_cast<void*>(buffer.opaque()));
  }
  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice result_slice,
                      assignment_->GetUniqueTopLevelOutputSlice());
  void* result_buffer = buffer_pointers[result_slice.index()];
  if (VLOG_IS_ON(3)) {
    VLOG(3) << "Executing compute function:";
    VLOG(3) << absl::StrFormat(
        "  func(void* result, void* params[null], void* buffer_table[%u], "
        "uint64 profile_counters[%u])",
        buffer_pointers.size(), profile_counters_size);
    VLOG(3) << absl::StrFormat("    result = %p", result_buffer);
    auto ptr_printer = [](string* out, const void* p) {
      absl::StrAppend(out, absl::StrFormat("%p", p));
    };
    VLOG(3) << "    params = nullptr";
    VLOG(3) << absl::StrFormat(
        "    buffer_table = [%s]",
        absl::StrJoin(buffer_pointers, ", ", ptr_printer));
    VLOG(3) << absl::StrFormat("    profile_counters = %p", profile_counters);
  }

  compute_function_(result_buffer, run_options, nullptr, buffer_pointers.data(),
                    profile_counters);

  uint64 end_micros = tensorflow::Env::Default()->NowMicros();

  {
    tensorflow::mutex_lock lock(mutex_);
    const double nanoseconds = (end_micros - start_micros) * 1000.0;
    execution_profile_.set_compute_time_ns(std::max(nanoseconds, 1.0));
    // If hlo profiling was disabled then the cycle count is left empty.
    if (hlo_execution_profile) {
      execution_profile_.set_compute_cycle_count(
          hlo_execution_profile->total_cycles_executed(
              *module().entry_computation()));
    }
  }

  return Status::OK();
}

StatusOr<ScopedShapedBuffer> CpuExecutable::CreateResultShapedBuffer(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<OwningDeviceMemory> buffers) {
  se::Stream* stream = run_options->stream();
  ScopedShapedBuffer result_buffer(
      /*on_host_shape=*/result_shape(),
      /*on_device_shape=*/result_shape(), run_options->allocator(),
      stream->parent()->device_ordinal());

  // Move OwningDeviceMemory values which contain the array(s) of the result
  // into the respective location in ScopedShapedBuffer which is returned to the
  // caller.
  TF_RETURN_IF_ERROR(result_buffer.buffers().ForEachMutableElementWithStatus(
      [&](const ShapeIndex& index, se::DeviceMemoryBase* device_memory) {
        const auto& sources = this->GetRootPointsToSet().element(index);
        // The points to set is unambiguous so the set should be a
        // singleton.
        CHECK_EQ(1, sources.size());
        const LogicalBuffer* buffer_source = sources[0];
        HloInstruction* src = buffer_source->instruction();

        // The source for this result buffer can be a nested buffer such as
        // a tuple element. The source instruction should have a
        // non-parameter buffer assigned.
        TF_ASSIGN_OR_RETURN(
            const BufferAllocation::Slice slice,
            this->assignment_->GetUniqueSlice(src, buffer_source->index()));
        CHECK(!slice.allocation()->is_entry_computation_parameter());

        const BufferAllocation::Index buffer_index = slice.index();
        OwningDeviceMemory& buffer = buffers[buffer_index];
        CHECK(!buffer.is_null() || buffer.size() == 0);
        *device_memory = buffer.Forget();
        return Status::OK();
      }));
  return std::move(result_buffer);
}

StatusOr<ScopedShapedBuffer> CpuExecutable::ExecuteOnStream(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  TF_ASSIGN_OR_RETURN(
      auto result,
      ExecuteAsyncOnStreamImpl(run_options, arguments, hlo_execution_profile));
  TF_RETURN_IF_ERROR(run_options->stream()->BlockHostUntilDone());
  return std::move(result);
}

StatusOr<ScopedShapedBuffer> CpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments) {
  if (hlo_profiling_enabled()) {
    return Unimplemented(
        "Asynchronous execution on stream with hlo profiling is not yet "
        "supported on CPU.");
  }
  return ExecuteAsyncOnStreamImpl(run_options, arguments, nullptr);
}

StatusOr<ScopedShapedBuffer> CpuExecutable::ExecuteAsyncOnStreamImpl(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  if (GetRootPointsToSet().IsAmbiguous()) {
    return Unimplemented("Points-to set of root instruction is ambiguous");
  }

  auto* host_stream = dynamic_cast<se::host::HostStream*>(
      run_options->stream()->implementation());
  se::Stream* stream = run_options->stream();
  DeviceMemoryAllocator* memory_allocator = run_options->allocator();
  std::vector<OwningDeviceMemory> owning_buffers;
  std::vector<se::DeviceMemoryBase> unowning_buffers;
  TF_ASSIGN_OR_RETURN(
      std::tie(unowning_buffers, owning_buffers),
      CreateBufferTable(memory_allocator, stream->parent()->device_ordinal(),
                        arguments));

  TF_ASSIGN_OR_RETURN(
      ScopedShapedBuffer result,
      CreateResultShapedBuffer(run_options, absl::MakeSpan(owning_buffers)));

  // At this point, `unowning_buffers` contains unowning pointers to all of our
  // buffers, and `buffers` contains owning pointers to the non-live-out
  // buffers.  Enqueue a task which keeps alive the non-live-out buffers.
  //
  // Logically we want this lambda to capture `buffers` by move, ultimately our
  // functor needs to be wrapped in an std::function, and that requires its
  // functor to be copyable.  Thus we perpitrate the hack of capturing buffers
  // "by shared pointer".
  //
  // We also need to change the types of some of the variables we capture:
  // run_options needs to change from a pointer to a value type, and arguments
  // needs to change from a Span into a vector.  We use a struct instead
  // of a lambda to make this explicit.
  struct AsyncRunTask {
    CpuExecutable* executable;
    ServiceExecutableRunOptions run_options;
    std::vector<se::DeviceMemoryBase> unowning_buffers;
    std::shared_ptr<std::vector<OwningDeviceMemory>> buffers;
    HloExecutionProfile* hlo_execution_profile;

    void operator()() {
      // Failing a CHECK here is not great, but I don't see an obvious way to
      // return a failed Status asynchronously.
      TF_CHECK_OK(executable->ExecuteComputeFunction(
          &run_options.run_options(), unowning_buffers, hlo_execution_profile));
    }
  };
  host_stream->EnqueueTask(
      AsyncRunTask{this, *run_options, std::move(unowning_buffers),
                   std::make_shared<std::vector<OwningDeviceMemory>>(
                       std::move(owning_buffers)),
                   hlo_execution_profile});

  return std::move(result);
}

/*static*/ int64 CpuExecutable::ShapeSizeBytes(const Shape& shape) {
  // On the cpu, opaques are pointers.
  if (shape.IsOpaque()) {
    return sizeof(void*);
  }
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
}

const PointsToSet& CpuExecutable::GetRootPointsToSet() const {
  return assignment_->points_to_analysis().GetPointsToSet(
      module().entry_computation()->root_instruction());
}

}  // namespace cpu
}  // namespace xla
