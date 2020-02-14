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
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
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
#include "tensorflow/stream_executor/device_memory_allocator.h"
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

StatusOr<std::tuple<std::vector<se::DeviceMemoryBase>,
                    std::vector<se::OwningDeviceMemory>,
                    std::vector<se::OwningDeviceMemory>>>
CpuExecutable::CreateBufferTable(
    se::DeviceMemoryAllocator* memory_allocator, int device_ordinal,
    std::vector<ShapeTree<MaybeOwningDeviceMemory>> arguments) {
  std::vector<se::DeviceMemoryBase> unowning_buffers(
      assignment_->Allocations().size());
  std::vector<se::OwningDeviceMemory> owning_buffers(
      assignment_->Allocations().size());
  VLOG(3) << "Allocating " << assignment_->Allocations().size()
          << " allocations for module " << module().name();
  for (BufferAllocation::Index i = 0; i < assignment_->Allocations().size();
       ++i) {
    auto& allocation = assignment_->GetAllocation(i);

    VLOG(3) << allocation.ToString();

    if (allocation.is_entry_computation_parameter()) {
      unowning_buffers[i] = arguments[allocation.parameter_number()]
                                .element(allocation.param_shape_index())
                                .AsDeviceMemoryBase();
      CHECK_EQ(allocation.size(), unowning_buffers[i].size())
          << "Size mismatch on param " << allocation.parameter_number()
          << " at shape index " << allocation.param_shape_index().ToString();
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
      unowning_buffers[i] = *owning_buffers[i];

      VLOG(3) << "buffer #" << i << " allocated " << buffer_size << " bytes ["
              << owning_buffers[i]->opaque() << "]";
    }

    // Since the output buffer and all the temporary buffers were written into
    // by the JITed code, msan has no way of knowing their memory was
    // initialized. Mark them initialized so that msan doesn't flag loads from
    // these buffers.
    TF_ANNOTATE_MEMORY_IS_INITIALIZED(owning_buffers[i]->opaque(), buffer_size);
  }

  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice result_slice,
                      assignment_->GetUniqueTopLevelOutputSlice());
  VLOG(3) << "result index: " << result_slice.index();

  std::vector<se::OwningDeviceMemory> buffers_to_free;
  for (ShapeTree<MaybeOwningDeviceMemory>& argument : arguments) {
    for (std::pair<ShapeIndex, MaybeOwningDeviceMemory>& buffer : argument) {
      auto maybe_owning_buffer = buffer.second.Release();
      if (maybe_owning_buffer) {
        buffers_to_free.push_back(std::move(*maybe_owning_buffer));
      }
    }
  }
  return std::make_tuple(std::move(unowning_buffers), std::move(owning_buffers),
                         std::move(buffers_to_free));
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

  return Status::OK();
}

StatusOr<ScopedShapedBuffer> CpuExecutable::CreateResultShapedBuffer(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<se::OwningDeviceMemory> buffers) {
  se::Stream* stream = run_options->stream();
  ScopedShapedBuffer result_buffer(
      /*on_host_shape=*/result_shape(),
      /*on_device_shape=*/result_shape(), run_options->allocator(),
      stream->parent()->device_ordinal());
  const HloInputOutputAliasConfig& input_output_alias =
      module().input_output_alias_config();

  // Move se::OwningDeviceMemory values which contain the array(s) of the result
  // into the respective location in ScopedShapedBuffer which is returned to the
  // caller.
  TF_RETURN_IF_ERROR(result_buffer.buffers().ForEachMutableElementWithStatus(
      [&](const ShapeIndex& index, se::DeviceMemoryBase* device_memory) {
        const auto& sources = this->GetRootValueSet().element(index);
        // The points to set is unambiguous so the set should be a
        // singleton.
        CHECK_EQ(1, sources.values().size());
        const HloValue* value_source = sources.values()[0];
        HloInstruction* src = value_source->instruction();

        // The source for this result buffer can be a nested buffer such as
        // a tuple element. The source instruction should have a
        // non-parameter buffer assigned.
        TF_ASSIGN_OR_RETURN(
            const BufferAllocation::Slice slice,
            this->assignment_->GetUniqueSlice(src, value_source->index()));
        const BufferAllocation::Index buffer_index = slice.index();
        se::OwningDeviceMemory& buffer = buffers[buffer_index];
        if (!slice.allocation()->is_entry_computation_parameter()) {
          // If the buffer coming out of the result is from a parameter, the
          // owning buffer will be null, and that means the caller aliased some
          // parameter buffer to an output one (via the
          // HloInputOutputAliasConfig API). If that is the case, the caller
          // will receive a partially complete scoped shaped buffer, which they
          // will have to fill up on return. Unfortunately the interface to the
          // execute APIs are ShapedBuffer pointer based, which assumes caller
          // ownership, and hence a buffer coming from there cannot be part of
          // the new ScopedShapedBuffer we create for the result (which assumes
          // ownership).
          *device_memory = buffer.Release();
        } else {
          auto output_alias = input_output_alias.GetAliasedOutput(
              slice.allocation()->parameter_number(),
              slice.allocation()->param_shape_index());
          CHECK(output_alias)
              << "Output buffer is coming from parameter "
              << slice.allocation()->parameter_number() << " at index "
              << slice.allocation()->param_shape_index()
              << ", but no alias exists";
          CHECK_EQ(*output_alias, index);
        }
        return Status::OK();
      }));
  return std::move(result_buffer);
}

StatusOr<ExecutionOutput> CpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ShapeTree<MaybeOwningDeviceMemory>> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  if (GetRootValueSet().IsAmbiguous()) {
    return Unimplemented("Points-to set of root instruction is ambiguous");
  }

  if (hlo_module_) {
    const HloComputation* entry_comp = hlo_module_->entry_computation();
    CHECK_EQ(entry_comp->num_parameters(), arguments.size())
        << "Wrong number of arguments passed when running executable";
    for (int64 i = 0; i < entry_comp->num_parameters(); ++i) {
      const Shape& expected_shape =
          entry_comp->parameter_instruction(i)->shape();
      const Shape& actual_shape = arguments[i].shape();
      CHECK(
          Shape::Equal().IgnoreDynamicDimension()(expected_shape, actual_shape))
          << absl::StreamFormat(
                 "Shape mismatch on argument %d.  Expected %s, but was %s.", i,
                 expected_shape.ToString(/*print_layout=*/true),
                 actual_shape.ToString(/*print_layout=*/true));
    }
  }

  auto* host_stream = dynamic_cast<se::host::HostStream*>(
      run_options->stream()->implementation());
  se::Stream* stream = run_options->stream();
  se::DeviceMemoryAllocator* memory_allocator = run_options->allocator();
  std::vector<se::OwningDeviceMemory> owning_buffers;
  std::vector<se::DeviceMemoryBase> unowning_buffers;
  std::vector<se::OwningDeviceMemory> buffers_to_release;
  TF_ASSIGN_OR_RETURN(
      std::tie(unowning_buffers, owning_buffers, buffers_to_release),
      CreateBufferTable(memory_allocator, stream->parent()->device_ordinal(),
                        std::move(arguments)));

  TF_ASSIGN_OR_RETURN(
      ScopedShapedBuffer result,
      CreateResultShapedBuffer(run_options, absl::MakeSpan(owning_buffers)));

  // At this point, `unowning_buffers` contains unowning pointers to all of our
  // buffers, and `buffers` contains owning pointers to the non-live-out
  // buffers.  Enqueue a task which keeps alive the non-live-out buffers.
  //
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
    std::vector<se::DeviceMemoryBase> unowning_buffers;
    std::shared_ptr<std::vector<se::OwningDeviceMemory>> buffers;
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
                   std::make_shared<std::vector<se::OwningDeviceMemory>>(
                       std::move(owning_buffers)),
                   hlo_execution_profile});

  return ExecutionOutput(std::move(result), std::move(buffers_to_release), {},
                         se::OwningDeviceMemory());
}

/*static*/ int64 CpuExecutable::ShapeSizeBytes(const Shape& shape) {
  // On the cpu, opaques are pointers.
  if (shape.IsOpaque()) {
    return sizeof(void*);
  }
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
}

const InstructionValueSet& CpuExecutable::GetRootValueSet() const {
  return assignment_->dataflow_analysis().GetInstructionValueSet(
      module().entry_computation()->root_instruction());
}

}  // namespace cpu
}  // namespace xla
