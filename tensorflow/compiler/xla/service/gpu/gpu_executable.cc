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

#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"

#include <set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_constants.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_debug_info_manager.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_types.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/llvm_ir/buffer_assignment_util.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/stream_executor/platform.h"

namespace xla {
namespace gpu {
namespace {

using ::tensorflow::profiler::ScopedAnnotation;

}  // namespace

// Implementation note: HLO profiling is always enabled for GPU executables,
// since we can use timers around thunks.
GpuExecutable::GpuExecutable(
    const string& text, const std::vector<uint8>& binary,
    GpuVersion gpu_version, std::unique_ptr<const ThunkSchedule> thunk_schedule,
    std::shared_ptr<HloModule> hlo_module,
    std::shared_ptr<const BufferAssignment> assignment,
    std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
    std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map)
    : Executable(std::move(hlo_module), std::move(hlo_profile_printer_data),
                 std::move(hlo_profile_index_map)),
      text_(text),
      binary_(binary),
      gpu_version_(gpu_version),
      thunk_schedule_(std::move(thunk_schedule)),
      assignment_(std::move(assignment)) {
  CHECK(has_module() && assignment_);
  GpuDebugInfoManager::Get()->RegisterModule(module().name(), shared_module(),
                                             assignment_);
  ComputeThunkAnnotations();
}

GpuExecutable::~GpuExecutable() {
  CHECK(has_module() && assignment_);
  GpuDebugInfoManager::Get()->UnregisterModule(module().name(), shared_module(),
                                               assignment_);

  {
    // We could have issued host->device mem copies in ResolveConstantGlobals.
    // Wait for those to finish so that we can safely deallocate the backing HLO
    // module.
    //
    // We need for the host->device memcpies to finish they are concurrently
    // reading memory (xla::Literal's) owned by the HLO module.
    tensorflow::mutex_lock lock(module_handle_mutex_);
    for (const auto& pair : module_globals_) {
      CHECK(pair.first->SynchronizeAllActivity());
    }
  }
}

void GpuExecutable::ComputeThunkAnnotations() {
  for (Thunk* thunk : thunk_schedule_->TotalOrder()) {
    thunk->ComputeAnnotations();
  }
}

Status GpuExecutable::CheckCompatibilityWithServiceExecutableRunOptions(
    const ServiceExecutableRunOptions* run_options) {
  se::Stream* main_stream = run_options->stream();

  stream_executor::PlatformKind platform_kind =
      main_stream->parent()->platform_kind();
  if (platform_kind == stream_executor::PlatformKind::kROCm) {
    int stream_isa_version;
    main_stream->parent()->GetDeviceDescription().rocm_amdgpu_isa_version(
        &stream_isa_version);
    GpuVersion amd_isa_version = stream_isa_version;
    TF_RET_CHECK(amd_isa_version == gpu_version_)
        << "AMDGPU GCN ISA version mismatch; expected {"
        << absl::get<int>(gpu_version_) << ", but was " << stream_isa_version;
  } else if (platform_kind == stream_executor::PlatformKind::kCuda) {
    std::pair<int, int> stream_compute_compatibility;
    main_stream->parent()->GetDeviceDescription().cuda_compute_capability(
        &stream_compute_compatibility.first,
        &stream_compute_compatibility.second);
    GpuVersion nvidia_compute_compatibility = stream_compute_compatibility;
    TF_RET_CHECK(nvidia_compute_compatibility == gpu_version_)
        << "Compute capability mismatch; expected {"
        << absl::get<std::pair<int, int>>(gpu_version_).first << ", "
        << absl::get<std::pair<int, int>>(gpu_version_).second << "}, but was {"
        << stream_compute_compatibility.first << ", "
        << stream_compute_compatibility.second << "}";
  } else {
    return InternalError("Unknown platform: %d", platform_kind);
  }

  return Status::OK();
}

Status GpuExecutable::ExecuteThunks(
    const ServiceExecutableRunOptions* run_options,
    const BufferAllocations& buffer_allocations, bool block_host_until_done,
    HloExecutionProfile* hlo_execution_profile) {
  TF_RETURN_IF_ERROR(
      CheckCompatibilityWithServiceExecutableRunOptions(run_options));
  GpuDebugInfoManager::Get()->OnModuleStart(module().name());
  auto cleanup = MakeCleanup(
      [&]() { GpuDebugInfoManager::Get()->OnModuleStop(module().name()); });

  se::Stream* main_stream = run_options->stream();
  se::StreamExecutor* executor = main_stream->parent();

  bool do_profile = hlo_execution_profile != nullptr;
  if (do_profile) {
    LOG(WARNING) << "PROFILING: profiling is enabled";
  }

  // Stream 0 indicates `main_stream` and substreams start from stream 1.
  std::vector<StreamPool::Ptr> sub_streams;
  sub_streams.reserve(thunk_schedule_->StreamCount() - 1);
  while (sub_streams.size() + 1 < thunk_schedule_->StreamCount()) {
    sub_streams.emplace_back();
    TF_ASSIGN_OR_RETURN(sub_streams.back(),
                        run_options->BorrowStream(executor->device_ordinal()));
    // Require substreams to wait for the main stream, otherwise substreams may
    // execute before the program is scheduled to start on the main stream.
    sub_streams.back()->ThenWaitFor(main_stream);
  }

  HloExecutionProfiler profiler(do_profile, hlo_execution_profile, main_stream,
                                sub_streams, hlo_module_->entry_computation());
  uint64 start_micros = tensorflow::Env::Default()->NowMicros();

  tensorflow::profiler::TraceMe hlo_module_activity(
      [&] { return absl::StrCat(hlo_module_->name(), ":XLA GPU module"); },
      tensorflow::profiler::TraceMeLevel::kInfo);

  std::map<const Thunk*, std::unique_ptr<se::Event>> thunk_to_finish_event;
  std::vector<std::function<void()>> deferred_host_callbacks;
  for (Thunk* thunk : thunk_schedule_->TotalOrder()) {
    CHECK(thunk->hlo_instruction());
    // Annotate execution of this op if tracing was enabled when we started
    // running this module.  If tracing is enabled *while* we're running the
    // module, we won't get any data, but that's probably an OK trade-off.
    ScopedAnnotation annotation([&] { return thunk->profile_annotation(); });

    int32 stream_no =
        thunk_schedule_->StreamNumberForHlo(*thunk->hlo_instruction());
    se::Stream* stream =
        (stream_no == 0 ? main_stream : sub_streams[stream_no - 1].get());

    for (const Thunk* dependency : thunk_schedule_->DependsOn(thunk)) {
      stream->ThenWaitFor(FindOrDie(thunk_to_finish_event, dependency).get());
    }

    VLOG(2) << "Executing the thunk for "
            << thunk->hlo_instruction()->ToString() << " on stream "
            << stream_no;
    const GpuExecutableRunOptions* gpu_options =
        run_options->run_options().gpu_executable_run_options();
    Thunk::ExecuteParams thunk_params{
        &buffer_allocations,
        stream,
        run_options->run_options().run_id(),
        &profiler,
        run_options->run_options().device_assignment(),
        &deferred_host_callbacks,
        gpu_options && gpu_options->gpu_global_device_ids()
            ? &*gpu_options->gpu_global_device_ids()
            : nullptr,
        gpu_options && gpu_options->nccl_unique_id_callback()
            ? &gpu_options->nccl_unique_id_callback()
            : nullptr};
    TF_RETURN_IF_ERROR(thunk->ExecuteOnStream(thunk_params));
    if (thunk_schedule_->Depended(thunk)) {
      auto finish_event = absl::make_unique<se::Event>(main_stream->parent());
      finish_event->Init();
      stream->ThenRecordEvent(finish_event.get());
      thunk_to_finish_event[thunk] = std::move(finish_event);
    }
  }

  main_stream->ThenWaitFor(&sub_streams);
  if (!deferred_host_callbacks.empty()) {
    auto fn = [deferred_host_callbacks{std::move(deferred_host_callbacks)}]() {
      for (auto& callback : deferred_host_callbacks) {
        callback();
      }
    };
    if (run_options->run_options().then_execute_function()) {
      (*run_options->run_options().then_execute_function())(main_stream,
                                                            std::move(fn));
    } else {
      main_stream->ThenDoHostCallback(std::move(fn));
    }
  }
  // Make sure kernels are completed before deallocating temporary buffers or
  // the profiler state.
  // TODO(b/30100571): we could potentially postpone deallocating the temp
  // buffers until a different computation is executed.
  if (do_profile || block_host_until_done) {
    Status block_status = main_stream->BlockHostUntilDone();
    if (!block_status.ok()) {
      return InternalError(
          "Failed to complete all kernels launched on stream %p: %s",
          main_stream, block_status.error_message());
    }
  }

  // FinishExecution() blocks until main_stream has completed if profiling is
  // enabled; we therefore do not need to defer profile collection onto a
  // stream.
  profiler.FinishExecution();
  uint64 end_micros = tensorflow::Env::Default()->NowMicros();

  if (run_options->run_options().execution_profile()) {
    ExecutionProfile* profile = run_options->run_options().execution_profile();
    const double nanoseconds = (end_micros - start_micros) * 1000.0;
    profile->set_compute_time_ns(std::max(nanoseconds, 1.0));

    // If hlo profiling was disabled then the cycle count is left empty.
    if (do_profile) {
      profile->set_compute_cycle_count(
          hlo_execution_profile->total_cycles_executed(
              *module().entry_computation()));
    }
  }

  return Status::OK();
}

StatusOr<const GpuExecutable::BufferAllocToDeviceMemoryMap*>
GpuExecutable::ResolveConstantGlobals(se::Stream* stream) {
  se::StreamExecutor* executor = stream->parent();

  tensorflow::mutex_lock lock(module_handle_mutex_);
  auto it = module_globals_.find(executor);
  if (it != module_globals_.end()) {
    return &it->second;
  }

  se::MultiModuleLoaderSpec module_spec;
  if (!binary().empty()) {
    module_spec.AddCudaCubinInMemory(binary());
  }
  module_spec.AddCudaPtxInMemory(text().c_str());

  absl::flat_hash_map<int64, se::DeviceMemoryBase> globals;
  if (executor->platform_kind() == se::PlatformKind::kCuda &&
      module_spec.cuda_ptx_in_memory() == nullptr) {
    // No custom PTX => no globals.
    return &module_globals_.emplace(executor, std::move(globals)).first->second;
  }

  se::ModuleHandle module_handle;
  TF_RETURN_IF_ERROR(executor->LoadModule(module_spec, &module_handle));

  for (BufferAllocation::Index i = 0; i < assignment_->Allocations().size();
       ++i) {
    const BufferAllocation& allocation = assignment_->GetAllocation(i);
    if (allocation.is_constant()) {
      TF_ASSIGN_OR_RETURN(
          se::DeviceMemoryBase global,
          executor->GetUntypedSymbol(
              llvm_ir::ConstantBufferAllocationToGlobalName(allocation),
              module_handle));
      VLOG(3) << "Resolved global "
              << llvm_ir::ConstantBufferAllocationToGlobalName(allocation)
              << " to " << global.opaque();
      InsertOrDie(&globals, i, global);

      const Literal& literal =
          llvm_ir::LiteralForConstantAllocation(allocation);
      CHECK(literal.shape().IsArray());
      if (!ShouldEmitLiteralInLlvmIr(literal)) {
        VLOG(3) << "H2D memcpy for constant with shape "
                << ShapeUtil::HumanString(literal.shape());
        stream->ThenMemcpy(&global, literal.untyped_data(), allocation.size());
      }
    }
  }

  module_handles_.emplace(executor,
                          se::ScopedModuleHandle(executor, module_handle));
  return &module_globals_.emplace(executor, std::move(globals)).first->second;
}

StatusOr<se::DeviceMemoryBase> GpuExecutable::BufferForAllocation(
    absl::Span<ExecutionInput const> arguments,
    const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
    const BufferAllocation& allocation,
    se::DeviceMemoryAllocator* const memory_allocator, int device_ordinal,
    int64 arg_idx) {
  if (allocation.is_thread_local()) {
    return se::DeviceMemoryBase{};
  } else if (allocation.is_entry_computation_parameter()) {
    int64 param_no = allocation.parameter_number();
    se::DeviceMemoryBase registered_buffer =
        arguments[param_no]
            .Buffer(allocation.param_shape_index())
            .AsDeviceMemoryBase();
    if (registered_buffer.is_null() && registered_buffer.size() > 0) {
      return FailedPrecondition(
          "Cannot run XLA computation because pointer to (sub-)buffer at "
          "index %s of parameter %d was null.  All pointers to "
          "(sub-)buffers must not be null, unless the (sub-)buffer has "
          "zero elements.",
          allocation.param_shape_index().ToString(), param_no);
    }
    return registered_buffer;
  } else if (allocation.is_constant()) {
    return FindOrDie(*globals, arg_idx);
  } else {
    // Allocate each allocation that might escape, or is the temp buffer.
    CHECK(allocation.maybe_live_out() || allocation.IsPreallocatedTempBuffer());
    const int64 buffer_size = allocation.size();
    se::DeviceMemoryBase buffer_address;
    if (buffer_size > 0) {
      TF_ASSIGN_OR_RETURN(
          se::OwningDeviceMemory buffer,
          memory_allocator->Allocate(device_ordinal, buffer_size));
      buffer_address = buffer.Release();
    }
    return buffer_address;
  }
}

static Status CheckAlignment(const BufferAllocation& allocation,
                             se::DeviceMemoryBase buffer, int arg_idx) {
  const int64 expected_alignment = [&] {
    if (allocation.is_entry_computation_parameter()) {
      return kEntryParameterAlignBytes;
    } else if (allocation.is_constant()) {
      return kConstantBufferAlignBytes;
    } else {
      return kXlaAllocatedBufferAlignBytes;
    }
  }();
  if (!buffer.is_null() &&
      reinterpret_cast<uintptr_t>(buffer.opaque()) % expected_alignment != 0) {
    return InternalError(
        "Address of buffer %d must be a multiple of %x, but "
        "was %p",
        arg_idx, expected_alignment, buffer.opaque());
  }
  return Status::OK();
}

StatusOr<BufferAllocations> GpuExecutable::GenerateBufferAllocations(
    absl::Span<ExecutionInput const> arguments,
    const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
    se::DeviceMemoryAllocator* const memory_allocator,
    se::StreamExecutor* executor) {
  tensorflow::profiler::TraceMe hlo_module_activity(
      [&] { return std::string("Build buffer allocations"); },
      tensorflow::profiler::TraceMeLevel::kInfo);

  const int64 num_buffers = assignment_->Allocations().size();
  std::vector<se::DeviceMemoryBase> buffers;
  buffers.reserve(num_buffers);
  for (int64 i = 0; i < num_buffers; ++i) {
    const BufferAllocation& allocation = assignment_->GetAllocation(i);
    TF_ASSIGN_OR_RETURN(
        se::DeviceMemoryBase buffer,
        BufferForAllocation(arguments, globals, allocation, memory_allocator,
                            executor->device_ordinal(), i));
    buffers.push_back(buffer);
    TF_RETURN_IF_ERROR(CheckAlignment(allocation, buffer, i));
  }
  return {{buffers, executor->device_ordinal(), memory_allocator}};
}

// Returns `true` if the entire tuple contents is aliased.
static bool EntireTupleContentsAliased(
    const Shape& output_shape, const ShapeIndex& index,
    const HloInputOutputAliasConfig& alias_config) {
  const Shape& indexed_shape = ShapeUtil::GetSubshape(output_shape, index);
  if (!indexed_shape.IsTuple()) {
    return false;
  }
  bool all_aliased = true;
  ShapeUtil::ForEachSubshape(
      indexed_shape, [&](const Shape& subshape, const ShapeIndex& subindex) {
        if (subindex.empty()) {
          return;
        }
        std::vector<int64> full_index;
        absl::c_copy(index, std::back_inserter(full_index));
        absl::c_copy(subindex, std::back_inserter(full_index));
        if (!alias_config.OutputHasAlias(
                ShapeIndex(full_index.begin(), full_index.end()))) {
          all_aliased = false;
        }
      });
  return all_aliased;
}

StatusOr<ExecutionOutput> GpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  XLA_SCOPED_LOGGING_TIMER(absl::StrCat("GpuExecutable::ExecuteAsyncOnStream(",
                                        module().name(), ")"));
  se::DeviceMemoryAllocator* const memory_allocator = run_options->allocator();
  // Force synchronous execution if the allocator requires it.
  const bool block_host_until_done =
      !memory_allocator->AllowsAsynchronousDeallocation();

  if (GetRootValueSet().IsAmbiguous()) {
    return Unimplemented("Points-to set of root instruction is ambiguous");
  }

  const GpuExecutable::BufferAllocToDeviceMemoryMap* globals;
  {
    tensorflow::profiler::TraceMe hlo_module_activity(
        [&] { return std::string("Resolve constant globals"); },
        tensorflow::profiler::TraceMeLevel::kInfo);

    TF_ASSIGN_OR_RETURN(globals, ResolveConstantGlobals(run_options->stream()));
  }

  se::StreamExecutor* executor = run_options->stream()->parent();

  HloInstruction* root = hlo_module_->entry_computation()->root_instruction();
  const Shape& root_shape = root->shape();
  auto device_ordinal = executor->device_ordinal();
  ExecutionOutput result(/*on_host_shape=*/root->shape(),
                         /*on_device_shape=*/root->shape(), memory_allocator,
                         device_ordinal);

  TF_ASSIGN_OR_RETURN(BufferAllocations buffer_allocations,
                      GenerateBufferAllocations(arguments, globals,
                                                memory_allocator, executor));
  VLOG(2) << buffer_allocations.ToString();
  std::set<se::DeviceMemoryBase> buffers_in_result;
  for (auto& p : result.MutableResult()->buffers()) {
    const ShapeIndex& index = p.first;
    se::DeviceMemoryBase& result_buffer = p.second;
    const auto& sources = GetRootValueSet().element(index);
    // The points-to set is unambiguous so the set should be a
    // singleton. That is, we know exactly which instruction
    // produced the array at this element.
    CHECK_EQ(1, sources.values().size());
    HloInstruction* src_hlo = sources.values()[0]->instruction();

    VLOG(4) << "Looking at: " << src_hlo->ToString()
            << "@ index: " << index.ToString();

    const HloInputOutputAliasConfig& input_output_alias =
        module().input_output_alias_config();
    absl::optional<HloInputOutputAliasConfig::Alias> alias =
        input_output_alias.GetAliasedParameter(index);
    if (alias) {
      CHECK_LT(alias->parameter_number, arguments.size());
      ExecutionInput& input = arguments[alias->parameter_number];
      MaybeOwningDeviceMemory* maybe_owning_memory =
          input.MutableBuffer(alias->parameter_index);
      if (absl::optional<se::OwningDeviceMemory> owning =
              maybe_owning_memory->Release()) {
        // If the caller passes the ownership of the device memory, reuse it
        // as the output buffer. It is up to the caller whether or not to
        // donate a buffer; the aliasing information describes which buffers
        // may alias, not buffers that must alias.
        se::DeviceMemoryBase argument_buffer = owning->Release();
        *maybe_owning_memory = argument_buffer;
        result_buffer = argument_buffer;
        if (alias->kind == HloInputOutputAliasConfig::kUserAlias) {
          // This is a user alias, so a must alias. The caller is giving us the
          // input buffer, but in case of error from the execute call, we should
          // not be releasing it as it contains valid data (for example, it is a
          // parameter which the user wants us to alias, in a gradient update
          // computation). So we store the index into the result in the aliased
          // vector, which will be fed to the ExecutionOutput, which will use
          // the indices to drop the addresses from its own ScopedShapedBuffer
          // result, if the ExecutionOutput is not committed.
          result.AddAliasedIndex(index);
        }
      } else if (src_hlo->opcode() != HloOpcode::kParameter) {
        // The guard is above is not to insert copy-protection when aliasing
        // pass-through params, as we do not need to write into the output
        // buffer.
        VLOG(3) << "Using copy-protection: aliasing is specified, but the "
                   "buffer is not donated; allocating a fresh buffer";
        int64 allocation_size =
            ShapeUtil::ByteSizeOf(ShapeUtil::GetSubshape(root_shape, index));
        TF_ASSIGN_OR_RETURN(
            se::OwningDeviceMemory allocated_buffer,
            memory_allocator->Allocate(device_ordinal, allocation_size));
        result_buffer = allocated_buffer.Release();
        TF_ASSIGN_OR_RETURN(
            const BufferAllocation::Slice slice,
            assignment_->GetUniqueSlice(src_hlo, sources.values()[0]->index()));
        CHECK_EQ(slice.offset(), 0) << "Parameter should get its own slice";
        se::DeviceMemoryBase& aliased_buffer =
            buffer_allocations.GetMutableDeviceAddress(slice.index());
        CHECK_EQ(aliased_buffer.size(), result_buffer.size());
        run_options->stream()->ThenMemcpyD2D(&result_buffer, aliased_buffer,
                                             aliased_buffer.size());
        aliased_buffer = result_buffer;
      }
    }

    if (result_buffer.is_null()) {
      // The source instruction should have a non-parameter buffer
      // assigned.
      TF_ASSIGN_OR_RETURN(
          const BufferAllocation::Slice slice,
          assignment_->GetUniqueSlice(src_hlo, sources.values()[0]->index()));
      result_buffer = buffer_allocations.GetDeviceAddress(slice.index());

      // If the entire tuple contents is aliased, the copy insertion will *not*
      // materialize a new tuple, so we mark it as aliased as well.
      if (EntireTupleContentsAliased(root->shape(), index,
                                     input_output_alias)) {
        result.AddAliasedIndex(index);
      }
    }
    buffers_in_result.insert(result_buffer);
  }

  for (Thunk* thunk : thunk_schedule_->TotalOrder()) {
    TF_RETURN_IF_ERROR(thunk->Initialize(*this, executor));
  }
  TF_RETURN_IF_ERROR(ExecuteThunks(run_options, buffer_allocations,
                                   block_host_until_done,
                                   hlo_execution_profile));

  // Free all temporary allocations.
  TF_RETURN_IF_ERROR(
      buffer_allocations.TearDown(buffers_in_result, assignment_.get()));

  // Free allocations for arguments.
  MarkToBeReleasedArguments(absl::MakeSpan(arguments), result);
  return std::move(result);
}

const InstructionValueSet& GpuExecutable::GetRootValueSet() const {
  return assignment_->dataflow_analysis().GetInstructionValueSet(
      module().entry_computation()->root_instruction());
}

int64 GpuExecutable::SizeOfGeneratedCodeInBytes() const {
  // Non-empty PTX but empty cubin: compilation must have failed, return
  // "unknown".
  if (binary().empty() && !text_.empty()) {
    return -1;
  }
  int64 size = binary().size();
  for (BufferAllocation::Index i = 0; i < assignment_->Allocations().size();
       ++i) {
    const BufferAllocation& allocation = assignment_->GetAllocation(i);
    if (allocation.is_constant()) {
      size += allocation.size();
    }
  }
  return size;
}

}  // namespace gpu
}  // namespace xla
