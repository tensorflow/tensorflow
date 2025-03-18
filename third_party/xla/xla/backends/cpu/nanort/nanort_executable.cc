/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/nanort/nanort_executable.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/thread_pool_task_runner.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_layout.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {

using ::tsl::profiler::TraceMe;
using ::tsl::profiler::TraceMeEncode;

using ArgumentIndex = std::pair<size_t, ShapeIndex>;

// Resolves the mapping from argument index to allocation index.
static absl::StatusOr<std::vector<size_t>> ResolveArgumentsMapping(
    const HloModule& module, const BufferAssignment& buffer_assignment) {
  const ComputationLayout& entry_layout = module.entry_computation_layout();

  VLOG(3) << "Resolve executable arguments mapping:";

  // Mapping from argument index to flattened executable argument index.
  absl::flat_hash_map<ArgumentIndex, size_t> executable_arg_index;
  for (size_t i = 0; i < entry_layout.parameter_count(); ++i) {
    ShapeUtil::ForEachLeafShape(
        entry_layout.parameter_shape(i),
        [&](const Shape&, const ShapeIndex& index) {
          size_t arg_index = executable_arg_index.size();
          executable_arg_index[ArgumentIndex{i, index}] = arg_index;
        });
  }

  std::vector<size_t> argument_to_allocation_index(executable_arg_index.size());
  for (const BufferAllocation& allocation : buffer_assignment.Allocations()) {
    if (allocation.is_entry_computation_parameter()) {
      ArgumentIndex idx{allocation.parameter_number(),
                        allocation.param_shape_index()};

      // Skip buffer allocations assigned to non-leaf parameters (tuples).
      auto arg_idx = executable_arg_index.find(idx);
      if (arg_idx == executable_arg_index.end()) continue;

      VLOG(3) << absl::StreamFormat(
          " - parameter %d at shape index %s:"
          " argument index = %d allocation index = %d",
          allocation.parameter_number(),
          allocation.param_shape_index().ToString(), arg_idx->second,
          allocation.index());

      argument_to_allocation_index[arg_idx->second] = allocation.index();
    }
  }

  return argument_to_allocation_index;
}

// Resolves the mapping from result index to allocation index.
static absl::StatusOr<std::vector<size_t>> ResolveResultMapping(
    const HloModule& module, const BufferAssignment& buffer_assignment) {
  const ComputationLayout& entry_layout = module.entry_computation_layout();

  VLOG(3) << "Resolve executable results mapping:";

  // Mapping from result index to flattened executable result index.
  absl::flat_hash_map<ShapeIndex, size_t> executable_res_index;
  ShapeUtil::ForEachLeafShape(entry_layout.result_shape(),
                              [&](const Shape&, const ShapeIndex& index) {
                                size_t res_index = executable_res_index.size();
                                executable_res_index[index] = res_index;
                              });

  const InstructionValueSet& root_value_set =
      buffer_assignment.dataflow_analysis().GetInstructionValueSet(
          module.entry_computation()->root_instruction());

  std::vector<size_t> result_to_allocation_index(executable_res_index.size());

  TF_RETURN_IF_ERROR(ShapeUtil::ForEachLeafShapeWithStatus(
      entry_layout.result_shape(),
      [&](const Shape&, const ShapeIndex& index) -> absl::Status {
        // Skip buffer allocations assigned to non-leaf results (tuples).
        auto res_idx = executable_res_index.find(index);
        if (res_idx == executable_res_index.end()) return absl::OkStatus();

        const HloValueSet& sources = root_value_set.element(index);

        if (sources.values().size() != 1) {
          return Internal(
              "Expected a single value for result at shape index %s",
              index.ToString());
        }

        const HloValue* value = sources.values().front();
        TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice,
                            buffer_assignment.GetUniqueSlice(
                                value->instruction(), value->index()));

        DCHECK_EQ(slice.size(), slice.allocation()->size())
            << "Result slice size must match result allocation size";

        VLOG(3) << absl::StreamFormat(
            " - result at shape index %s:"
            " result index = %d allocation index = %d",
            index.ToString(), res_idx->second, slice.index());

        result_to_allocation_index[res_idx->second] = slice.index();
        return absl::OkStatus();
      }));

  return result_to_allocation_index;
}

static absl::StatusOr<std::optional<size_t>> ResolveTempAllocationIndex(
    const BufferAssignment& buffer_assignment) {
  VLOG(3) << "Resolve temp allocation index:";

  std::optional<size_t> temp_allocation_index;
  for (const BufferAllocation& allocation : buffer_assignment.Allocations()) {
    if (allocation.IsPreallocatedTempBuffer()) {
      if (temp_allocation_index.has_value()) {
        return Internal("Multiple temp buffer allocations found");
      }
      VLOG(3) << " - temp buffer allocation index = " << allocation.index();
      temp_allocation_index = allocation.index();
    }
  }

  if (!temp_allocation_index.has_value()) {
    VLOG(3) << " - no temp buffer allocation found";
  }

  return temp_allocation_index;
}

NanoRtExecutable::ExecuteOptions&
NanoRtExecutable::ExecuteOptions::set_intra_op_thread_pool(
    const Eigen::ThreadPoolDevice* intra_op_thread_pool) {
  intra_op_thread_pool_ = intra_op_thread_pool;
  task_runner_ = intra_op_thread_pool_ ? std::make_unique<ThreadPoolTaskRunner>(
                                             intra_op_thread_pool_->getPool())
                                       : nullptr;
  return *this;
}

const Eigen::ThreadPoolDevice*
NanoRtExecutable::ExecuteOptions::intra_op_thread_pool() const {
  return intra_op_thread_pool_;
}

ThreadPoolTaskRunner* NanoRtExecutable::ExecuteOptions::task_runner() const {
  return task_runner_.get();
}

absl::StatusOr<std::unique_ptr<NanoRtExecutable>> NanoRtExecutable::Create(
    std::unique_ptr<Executable> executable) {
  const HloModule& module = executable->module();

  VLOG(3) << "Create NanoRtExecutable: name = " << module.name();

  // NanoRtExecutable requires a CPU executable with thunks.
  auto* cpu_executable = tsl::down_cast<cpu::CpuExecutable*>(executable.get());
  if (cpu_executable == nullptr) {
    return Internal("NanoRtExecutable requires CPU executable");
  }
  if (!cpu_executable->has_thunks()) {
    return Internal("NanoRtExecutable requires CPU executable to use thunks");
  }

  // Mappings from argument/result index to buffer allocation index.
  TF_ASSIGN_OR_RETURN(
      std::vector<size_t> argument_to_allocation_index,
      ResolveArgumentsMapping(module, cpu_executable->buffer_assignment()));
  TF_ASSIGN_OR_RETURN(
      std::vector<size_t> result_to_allocation_index,
      ResolveResultMapping(module, cpu_executable->buffer_assignment()));

  TF_ASSIGN_OR_RETURN(
      std::optional<size_t> temp_allocation_index,
      ResolveTempAllocationIndex(cpu_executable->buffer_assignment()));

  const auto& buffer_assignment = cpu_executable->buffer_assignment();
  size_t num_allocations = buffer_assignment.Allocations().size();

  std::vector<size_t> allocation_sizes(num_allocations);
  for (const BufferAllocation& allocation : buffer_assignment.Allocations()) {
    allocation_sizes[allocation.index()] = allocation.size();
  }

  return absl::WrapUnique(new NanoRtExecutable(
      std::move(executable), std::move(allocation_sizes),
      std::move(argument_to_allocation_index),
      std::move(result_to_allocation_index), temp_allocation_index));
}

NanoRtExecutable::NanoRtExecutable(
    std::unique_ptr<Executable> executable,
    std::vector<size_t> allocation_sizes,
    std::vector<size_t> argument_to_allocation_index,
    std::vector<size_t> result_to_allocation_index,
    std::optional<size_t> temp_allocation_index)
    : executable_(std::move(executable)),
      allocation_sizes_(std::move(allocation_sizes)),
      argument_to_allocation_index_(std::move(argument_to_allocation_index)),
      result_to_allocation_index_(std::move(result_to_allocation_index)),
      temp_allocation_index_(temp_allocation_index) {}

static se::DeviceMemoryBase ToDeviceMemory(
    const NanoRtExecutable::Argument& argument) {
  return se::DeviceMemoryBase(
      const_cast<void*>(reinterpret_cast<const void*>(argument.data().data())),
      argument.data().size());
}

static se::DeviceMemoryBase ToDeviceMemory(
    const NanoRtExecutable::Result& result) {
  return se::DeviceMemoryBase(reinterpret_cast<void*>(result.data().data()),
                              result.data().size());
}

static se::DeviceMemoryBase ToDeviceMemory(
    const NanoRtExecutable::PreallocatedTemp& temp) {
  return se::DeviceMemoryBase(reinterpret_cast<void*>(temp.data()),
                              temp.size());
}

tsl::AsyncValueRef<NanoRtExecutable::ExecuteEvent> NanoRtExecutable::Execute(
    absl::Span<const Argument> arguments, absl::Span<const Result> results,
    PreallocatedTemp temp, const ExecuteOptions& options) {
  TraceMe trace([&] {
    return TraceMeEncode("NanoRtExecutable::Execute",
                         {{"name", executable_->module().name()}});
  });

  auto* executable = tsl::down_cast<cpu::CpuExecutable*>(executable_.get());

  size_t num_arguments = argument_to_allocation_index_.size();
  size_t num_results = result_to_allocation_index_.size();

  if (ABSL_PREDICT_FALSE(arguments.size() != num_arguments)) {
    return InvalidArgument("Expected %d arguments, got %d", num_arguments,
                           arguments.size());
  }

  if (ABSL_PREDICT_FALSE(results.size() != num_results)) {
    return InvalidArgument("Expected %d results, got %d", num_results,
                           results.size());
  }

  // Prepare buffer allocations for arguments, results, and temp.
  cpu::BufferAllocations::Buffers buffers(allocation_sizes_.size());

  for (size_t i = 0; i < num_arguments; ++i) {
    size_t idx = argument_to_allocation_index_[i];
    buffers[idx] = ToDeviceMemory(arguments[i]);

    if (ABSL_PREDICT_FALSE(buffers[idx].size() != allocation_sizes_[idx])) {
      return InvalidArgument("Argument %d size mismatch: expected %d, got %d",
                             i, allocation_sizes_[idx], buffers[idx].size());
    }
  }

  for (size_t i = 0; i < num_results; ++i) {
    size_t idx = result_to_allocation_index_[i];
    buffers[idx] = ToDeviceMemory(results[i]);

    if (ABSL_PREDICT_FALSE(buffers[idx].size() != allocation_sizes_[idx])) {
      return InvalidArgument("Result %d size mismatch: expected %d, got %d", i,
                             allocation_sizes_[idx], buffers[idx].size());
    }
  }

  if (temp_allocation_index_) {
    size_t idx = *temp_allocation_index_;
    buffers[idx] = ToDeviceMemory(temp);

    if (ABSL_PREDICT_FALSE(buffers[idx].size() != allocation_sizes_[idx])) {
      return InvalidArgument("Temp size mismatch: expected %d, got %d",
                             allocation_sizes_[idx], buffers[idx].size());
    }
  }

  for (const auto& constant : executable->constants()) {
    // Constants are re-indexed by the buffer allocation index at CpuExecutable
    // construction time, and `executable->constants()` actually returns the
    // vector of buffer allocations, and only allocations corresponding to
    // constants have a valid index.
    if (constant.index >= 0) {
      buffers[constant.index] = constant.AsDeviceMemoryBase();
    }
  }

  struct ExecutionContext {
    ExecutionContext(cpu::BufferAllocations::Buffers buffers,
                     FunctionLibrary* function_library,
                     const ExecuteOptions& options)
        : allocations(std::move(buffers)),
          execute_params({function_library, &allocations,
                          /*xfeed=*/nullptr, options.intra_op_thread_pool(),
                          options.task_runner()}) {}

    cpu::BufferAllocations allocations;
    Thunk::ExecuteParams execute_params;
  };

  // Only do a heap allocation if we're running with a thread pool, this allows
  // us to keep the execution context alive as long as we need it.
  if (options.intra_op_thread_pool()) {
    auto execution_context = std::make_unique<ExecutionContext>(
        std::move(buffers), executable->function_library(), options);

    auto execute_event =
        executable->thunks().Execute(execution_context->execute_params);

    execute_event.AndThen(
        [execution_context = std::move(execution_context)] {});

    return execute_event;
  } else {
    cpu::BufferAllocations allocations(std::move(buffers));
    Thunk::ExecuteParams execute_params{
        executable->function_library(), &allocations, /*xfeed=*/nullptr,
        options.intra_op_thread_pool(), options.task_runner()};
    return executable->thunks().Execute(execute_params);
  }
}

size_t NanoRtExecutable::temp_buffer_size() const {
  if (temp_allocation_index_.has_value()) {
    return allocation_sizes_[*temp_allocation_index_];
  }
  return 0;
}

}  // namespace xla::cpu
