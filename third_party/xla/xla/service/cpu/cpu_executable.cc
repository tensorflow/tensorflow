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
#include <cfenv>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/dynamic_annotations.h"
#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/constant_allocation.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/thread_pool_task_runner.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_executor.h"
#include "xla/backends/cpu/runtime/xfeed_manager.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_status_internal.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_execution_profile.h"
#include "xla/service/hlo_profile_printer_data.pb.h"
#include "xla/service/hlo_value.h"
#include "xla/service/maybe_owning_device_address.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/xla_debug_info_manager.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/denormal.h"
#include "tsl/platform/setround.h"
#include "tsl/profiler/lib/traceme.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla {
namespace cpu {

absl::StatusOr<std::unique_ptr<CpuExecutable>> CpuExecutable::Create(
    std::unique_ptr<FunctionLibrary> function_library,
    std::unique_ptr<BufferAssignment> assignment,
    std::unique_ptr<HloModule> hlo_module, ThunkSequence thunks,
    std::vector<ConstantAllocation> constants,
    TargetMachineOptions target_machine_options) {
  VLOG(2) << "Create CpuExecutable from a thunk sequence; module="
          << hlo_module->name() << ", constants=" << constants.size();

  std::unique_ptr<CpuExecutable> executable(
      new CpuExecutable(std::move(hlo_module), std::move(assignment),
                        std::move(target_machine_options)));
  executable->function_library_ = std::move(function_library);

  ThunkExecutor::Options thunk_executor_options;
  thunk_executor_options.is_nested_executor = false;
  TF_ASSIGN_OR_RETURN(
      executable->thunks_,
      ThunkExecutor::Create(std::move(thunks), thunk_executor_options));

  // Find if the thunk sequence contains any YNN fusion thunks. If we do have
  // any, we will prepare the YNNPACK thread pool for them at run time.
  executable->thunks_->thunk_sequence().ForEach([&](const Thunk& thunk) {
    executable->has_ynn_fusions_ |= thunk.kind() == Thunk::Kind::kYnnFusion;
  });

  // Re-index constants by their allocation index to allow efficient lookup.
  for (auto& constant : constants) {
    if (executable->constants_.size() <= constant.index) {
      executable->constants_.resize(constant.index + 1);
    }
    executable->constants_[constant.index] = std::move(constant);
  }

  return executable;
}

CpuExecutable::CpuExecutable(std::unique_ptr<HloModule> hlo_module,
                             std::unique_ptr<BufferAssignment> assignment,
                             TargetMachineOptions target_machine_options)
    : Executable(std::move(hlo_module)),
      assignment_(std::move(assignment)),
      target_machine_options_(std::move(target_machine_options)) {
  if (assignment_ && has_module()) {
    XlaDebugInfoManager::Get()->RegisterModule(shared_module(), assignment_);
  }

  if (assignment_) {
    alloc_ptrs_.reserve(assignment_->Allocations().size());
    for (const BufferAllocation& alloc : assignment_->Allocations()) {
      alloc_ptrs_.push_back(&alloc);
    }
  }

  // Once we compiled HLO module to CPU executable, we don't need to keep the
  // HLO module metadata around.
  if (has_module()) {
    *shared_module()->metadata() = HloModuleMetadata(tsl::Env::Default());
  }
}

CpuExecutable::~CpuExecutable() {
  if (has_module()) {
    XlaDebugInfoManager::Get()->UnregisterModule(module().unique_id());
  }
}

static absl::StatusOr<MaybeOwningDeviceAddress> MemoryForAllocation(
    const BufferAllocation& allocation,
    absl::Span<const ExecutionInput> arguments,
    absl::Span<const ConstantAllocation> constants,
    se::DeviceAddressAllocator* memory_allocator, int device_ordinal) {
  VLOG(3) << allocation.ToString();
  if (allocation.is_entry_computation_parameter()) {
    se::DeviceAddressBase out = arguments[allocation.parameter_number()]
                                    .Buffer(allocation.param_shape_index())
                                    .AsDeviceAddress();
    CHECK_LE(allocation.size(), out.size())
        << "Size mismatch on param " << allocation.parameter_number()
        << " at shape index " << allocation.param_shape_index().ToString();
    VLOG(3) << "allocation is a parameter";
    return MaybeOwningDeviceAddress{out};
  } else if (allocation.is_constant()) {
    VLOG(3) << "allocation is a constant";
    if (allocation.index() < constants.size()) {
      return MaybeOwningDeviceAddress(
          constants[allocation.index()].AsDeviceAddress());
    }
    return MaybeOwningDeviceAddress{se::DeviceAddressBase{}};
  } else if (allocation.is_thread_local()) {
    VLOG(3) << "buffer is thread-local";
    return MaybeOwningDeviceAddress{se::DeviceAddressBase{}};
  }

  int64_t buffer_size = allocation.size();
  TF_ASSIGN_OR_RETURN(se::ScopedDeviceAddress<uint8_t> out,
                      memory_allocator->Allocate(device_ordinal, buffer_size));
  VLOG(3) << "buffer allocated " << buffer_size << " bytes [" << out->opaque()
          << "]";

  // Since the output buffer and all the temporary buffers were written into
  // by the JITed code, memory sanitizer has no way of knowing their memory was
  // initialized. Mark them initialized so that memory sanitizer doesn't flag
  // loads from these buffers.
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(out->opaque(), buffer_size);
  return MaybeOwningDeviceAddress{std::move(out)};
}

absl::StatusOr<std::vector<MaybeOwningDeviceAddress>>
CpuExecutable::CreateBufferTable(se::DeviceAddressAllocator* memory_allocator,
                                 int device_ordinal,
                                 absl::Span<ExecutionInput const> arguments) {
  std::vector<MaybeOwningDeviceAddress> buffers(
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

static int32_t GetDeviceOrdinal(const ExecutableRunOptions* run_options) {
  if (!run_options) {
    return 0;
  }
  if (run_options->device_ordinal() != -1) {
    return run_options->device_ordinal();
  }
  return run_options->stream()->parent()->device_ordinal();
}

absl::Status CpuExecutable::ExecuteThunks(
    const ExecutableRunOptions* run_options,
    absl::Span<MaybeOwningDeviceAddress const> buffers) {
  uint64_t start_ns = tsl::Env::Default()->NowNanos();

  size_t profile_counters_size = 0;
  int64_t* profile_counters = nullptr;

  BufferAllocations allocations(buffers);

  VLOG(3) << "Executing XLA:CPU thunks:";
  VLOG(3) << absl::StrFormat("  Number of buffer allocations: %u",
                             buffers.size());
  auto mem_printer = [](std::string* out, const MaybeOwningDeviceAddress& mem) {
    absl::StrAppend(out, absl::StrFormat("%p", mem.AsDeviceAddress().opaque()));
  };
  VLOG(3) << absl::StrFormat("  Buffer allocations: [%s]",
                             absl::StrJoin(buffers, ", ", mem_printer));
  VLOG(3) << absl::StrFormat("  Number of profile counters: %u",
                             profile_counters_size);
  VLOG(3) << absl::StrFormat("  Profile counters: %p", profile_counters);

  // Prepare for executing XLA program collectively.
  TF_ASSIGN_OR_RETURN(Thunk::CollectiveExecuteParams collective_execute_params,
                      Thunk::CollectiveExecuteParams::Create(run_options));

  // Prepare for executing XLA custom calls.
  TF_ASSIGN_OR_RETURN(Thunk::CustomCallExecuteParams custom_call_execute_params,
                      Thunk::CustomCallExecuteParams::Create(run_options));

  // Prepare for executing YNNPACK fusions.
  std::optional<Thunk::YnnParams> ynn_params;
  if (has_ynn_fusions()) {
    TF_ASSIGN_OR_RETURN(ynn_params, Thunk::YnnParams::Create(run_options));
  }

  // Use the intra-op thread pool to offload thunk executor tasks.
  auto* intra_op_thread_pool = run_options->intra_op_thread_pool();
  ThreadPoolTaskRunner task_runner(
      intra_op_thread_pool ? intra_op_thread_pool->getPool() : nullptr);

  Thunk::ExecuteParams execute_params = {
      &*function_library_,
      &allocations,
      GetXfeedManager(GetDeviceOrdinal(run_options)),
      intra_op_thread_pool,
      &task_runner,
      &collective_execute_params,
      &custom_call_execute_params,
      ynn_params ? &*ynn_params : nullptr};

  auto executed_event = thunks_->Execute(execute_params);

  tsl::profiler::TraceMe trace("BlockUntilReady");
  tsl::BlockUntilReady(executed_event);

  if (run_options->execution_profile()) {
    uint64_t end_ns = tsl::Env::Default()->NowNanos();
    run_options->execution_profile()->set_compute_time_ns(
        std::max<int64_t>(end_ns - start_ns, 1));
  }

  return ABSL_PREDICT_FALSE(executed_event.IsError())
             ? executed_event.GetError()
             : absl::OkStatus();
}

absl::StatusOr<ExecutionOutput> CpuExecutable::CreateResultShapedBuffer(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<MaybeOwningDeviceAddress> buffers,
    absl::Span<ExecutionInput> arguments) {
  se::Stream* stream = run_options->stream();
  ExecutionOutput result(/*on_device_shape=*/result_shape(),
                         run_options->allocator(),
                         stream->parent()->device_ordinal());
  const HloInputOutputAliasConfig& input_output_alias =
      module().input_output_alias_config();
  HloInstruction* root = module().entry_computation()->root_instruction();
  const Shape& root_shape = root->shape();

  // Move se::ScopedDeviceAddress<uint8_t> values which contain the array(s) of
  // the result into the respective location in ScopedShapedBuffer which is
  // returned to the caller.
  for (auto& p : result.MutableResult()->buffers()) {
    const ShapeIndex& index = p.first;
    se::DeviceAddressBase& result_buffer = p.second;
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
      MaybeOwningDeviceAddress* maybe_owning_memory =
          input.MutableBuffer(alias->parameter_index);
      if (alias->must_alias() && !maybe_owning_memory->HasOwnership()) {
        return InvalidArgument(
            "An input was configured to be must-alias at "
            "compile time but not donated at runtime: %s",
            alias->ToString());
      }
      if (std::optional<se::ScopedDeviceAddress<uint8_t>> owning =
              maybe_owning_memory->Release()) {
        // If the caller passes the ownership of the device memory, reuse it
        // as the output buffer. It is up to the caller whether or not to
        // donate a buffer; the aliasing information describes which buffers
        // may alias, not buffers that must alias.
        se::DeviceAddressBase argument_buffer = owning->Release();
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
            se::ScopedDeviceAddress<uint8_t> allocated_buffer,
            run_options->allocator()->Allocate(
                stream->parent()->device_ordinal(), allocation_size));
        result_buffer = allocated_buffer.Release();
        MaybeOwningDeviceAddress& registered_buffer = buffers[buffer_index];
        CHECK_EQ(result_buffer.size(),
                 registered_buffer.AsDeviceAddress().size());
        std::memcpy(/*dest=*/result_buffer.opaque(),
                    /*src=*/registered_buffer.AsDeviceAddress().opaque(),
                    /*n=*/result_buffer.size());
        registered_buffer = result_buffer;
      }
    }

    if (result_buffer.is_null()) {
      MaybeOwningDeviceAddress& buffer = buffers[buffer_index];
      if (std::optional<se::ScopedDeviceAddress<uint8_t>> owned_buffer =
              buffer.Release()) {
        result_buffer = owned_buffer->Release();
        buffer = result_buffer;
      } else {
        result_buffer = buffer.AsDeviceAddress();
        result.AddAliasedIndex(index);
      }
    }
  }
  return std::move(result);
}

absl::StatusOr<ExecutionOutput> CpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments) {
  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode("CpuExecutable::ExecuteAsyncOnStream",
                                        {{"module_name", module_name_}});
  });

  if (GetRootValueSet().IsAmbiguous()) {
    return Unimplemented("Points-to set of root instruction is ambiguous");
  }

  if (has_module()) {
    const HloComputation* entry_comp = module().entry_computation();
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

  se::Stream* stream = run_options->stream();
  se::DeviceAddressAllocator* memory_allocator = run_options->allocator();
  TF_ASSIGN_OR_RETURN(
      std::vector<MaybeOwningDeviceAddress> buffers,
      CreateBufferTable(memory_allocator, stream->parent()->device_ordinal(),
                        arguments));

  TF_ASSIGN_OR_RETURN(
      ExecutionOutput result,
      CreateResultShapedBuffer(run_options, absl::MakeSpan(buffers),
                               absl::MakeSpan(arguments)));

  // IMPORTANT: State of the world as of June 2025 by ezhulenev@.
  //
  // Although the function is called ExecuteAsyncOnStream, we invoke compiled
  // executable on the caller thread, because the concept of device stream and
  // implicit ordering of operations does not make much sense on CPU. We use
  // stream semantics on GPU because host can run ahead of the device (which is
  // impossible on CPU because host and device are the same), and because of
  // stream-ordered memory allocation via BFC allocator (on the host we use
  // regular host allocator).
  //
  // Furthermore, this execution path is deprecated, and nearly all users
  // (certainly all important ones) go via the PjRtCpuClient route, which is not
  // affected by this code. This code is used mostly in legacy tests (not yet
  // migrated to PjRt) and in Tensorflow/XLA integration.
  //
  // By using the caller thread to kick off the execution, we avoid the
  // overhead of thread hopping for small executables, and it allows Tensorflow
  // to execute multiple XLA executable in parallel.

  // Because we do not control the caller thread, we need to explicitly set
  // flags to be consistent with compute thread pools used by TF and XLA.
  tsl::port::ScopedFlushDenormal flush;
  tsl::port::ScopedSetRound round(FE_TONEAREST);

  DCHECK(has_thunks());
  TF_RETURN_IF_ERROR(ExecuteThunks(&run_options->run_options(), buffers));

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
  int64_t metadata_size = sizeof(int32_t) * shape.dimensions().size();
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*)) + metadata_size;
}

const InstructionValueSet& CpuExecutable::GetRootValueSet() const {
  return assignment_->dataflow_analysis().GetInstructionValueSet(
      module().entry_computation()->root_instruction());
}

int64_t CpuExecutable::SizeOfGeneratedCodeInBytes() const {
  // TODO(ezhulenev): Delete this function, it's not really used anywhere.
  return 0;
}

void CpuExecutable::Finalize() {
  if (has_module()) {
    shared_module()->Finalize();
  }
  assignment_->Finalize();
}

}  // namespace cpu
}  // namespace xla
