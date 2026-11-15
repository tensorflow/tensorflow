/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/autotuner/gpu_profiler.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/backends/gpu/runtime/buffer_comparator.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/autotuning/redzone_buffers.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/maybe_owning_device_address.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/redzone_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/util/buffer_slice_merge.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace gpu {

namespace {

// Minimum size of the buffer used to flush the caches, in case the device does
// not report its L2 cache size.
constexpr int64_t kMinCacheFlushBytes = int64_t{64} << 20;  // 64 MiB

std::vector<ExecutionInput> CreateExecutionInputsFromBuffers(
    absl::Span<se::DeviceAddressBase const> buffers,
    absl::Span<Shape const> shapes) {
  CHECK_EQ(buffers.size(), shapes.size());
  std::vector<ExecutionInput> inputs;
  for (int i = 0; i < buffers.size(); ++i) {
    inputs.emplace_back(shapes.at(i));
    // Our executable doesn't have input-output aliasing, so we can pass
    // unowned input buffers.
    inputs.back().SetUnownedBuffer(
        /*index=*/{}, MaybeOwningDeviceAddress(/*unowned=*/buffers.at(i)));
  }
  return inputs;
}

int GetScratchBytes(const GpuExecutable& executable) {
  // Group scratch slices by allocation.
  absl::flat_hash_map<const BufferAllocation*,
                      std::vector<BufferAllocation::Slice>>
      alloc_to_slices;

  CHECK_OK(executable.thunk_executor().thunks().WalkNested(
      [&alloc_to_slices](const Thunk* thunk) {
        for (const auto& buffer_use : thunk->buffer_uses()) {
          // ContentValidity::kUndefined means the buffer is a scratch buffer.
          if (buffer_use.content_validity() ==
              BufferUse::ContentValidity::kUndefined) {
            const auto& slice = buffer_use.slice();
            alloc_to_slices[slice.allocation()].push_back(slice);
          }
        }

        return absl::OkStatus();
      }));

  int64_t scratch_bytes = 0;
  for (auto& [alloc, slices] : alloc_to_slices) {
    // We might be re-using the same physical buffer for different slices (e.g.
    // we need to use the same scratch buffer for two sequential operations), so
    // we need to merge overlapping slices.
    for (const auto& merged_slice : MergeOverlappingSlices(slices)) {
      scratch_bytes += merged_slice.size();
    }
  }

  return static_cast<int>(scratch_bytes);
}

// Initialize a specific input buffer with custom values.
// This is useful for initializing constant parameters like group sizes
// in group-gemm operations after buffers are created.
static absl::Status InitializeInputBuffer(GpuInputBuffers& gpu_buffers,
                                          se::Stream* stream, int buffer_index,
                                          const void* values,
                                          size_t size_bytes) {
  RedzoneBuffers& rz_buffers = gpu_buffers.redzone_buffers;

  if (buffer_index < 0 || buffer_index >= rz_buffers.input_buffers().size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid buffer_index %d, must be in range [0, %d)",
                        buffer_index, rz_buffers.input_buffers().size()));
  }

  se::DeviceAddressBase buffer = rz_buffers.input_buffers()[buffer_index];
  ABSL_RETURN_IF_ERROR(stream->Memcpy(const_cast<se::DeviceAddressBase*>(&buffer),
                                 values, size_bytes));
  ABSL_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  return absl::OkStatus();
}

// Initialize input buffers based on the operation type.
// This function examines the HLO instruction and initializes
// specific input buffers based on the operation's requirements.
static absl::Status InitializeBuffersIfRequiredByOpcode(
    const HloInstruction* instr, GpuInputBuffers& gpu_buffers,
    se::Stream* stream) {
  if (instr == nullptr) {
    return absl::OkStatus();
  }

  // Handle group-gemm operations
  if (instr->opcode() == HloOpcode::kCustomCall &&
      instr->custom_call_target() == "__cublas$lt$groupedMatmul") {
    // Get the backend config to extract ragged dimension information
    ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                     instr->backend_config<GpuBackendConfig>());
    const GroupedGemmBackendConfig& grouped_config =
        gpu_config.grouped_gemm_backend_config();
    const RaggedDotDimensionNumbers& ragged_dims =
        grouped_config.ragged_dot_dimension_numbers();

    // Get ragged dimension index from the config
    int64_t ragged_dim_index = ragged_dims.lhs_ragged_dimensions(0);

    // Get ragged dimension size from LHS operand
    const Shape& lhs_shape = instr->operand(0)->shape();
    int64_t ragged_dim_size = lhs_shape.dimensions(ragged_dim_index);

    // Get number of groups from the group sizes parameter shape
    // The shape can be 1D [num_groups] or 2D [batch_size, groups_per_batch]
    const Shape& group_sizes_shape =
        instr->operand(instr->operand_count() - 1)->shape();
    int64_t groups_per_batch =
        group_sizes_shape.dimensions(group_sizes_shape.dimensions().size() - 1);
    int64_t total_elements = ShapeUtil::ElementsIn(group_sizes_shape);

    // Calculate group sizes based on groups_per_batch (not total elements)
    int64_t base_group_size = ragged_dim_size / groups_per_batch;
    int64_t last_group_size =
        ragged_dim_size - ((groups_per_batch - 1) * base_group_size);

    VLOG(3) << "  Ragged dim index: " << ragged_dim_index;
    VLOG(3) << "  Ragged dim size: " << ragged_dim_size;
    VLOG(3) << "  Group sizes shape: " << group_sizes_shape.ToString();
    VLOG(3) << "  Groups per batch: " << groups_per_batch;
    VLOG(3) << "  Total elements in group sizes buffer: " << total_elements;
    VLOG(3) << "  Base group size (first n-1 groups): " << base_group_size;
    VLOG(3) << "  Last group size: " << last_group_size;

    // Determine the type of the group sizes parameter
    PrimitiveType group_sizes_type = group_sizes_shape.element_type();

    if (group_sizes_type == S32) {
      std::vector<int32_t> group_sizes(total_elements);
      // Fill with the pattern: [base_size, base_size, ..., last_size]
      // repeated for each batch
      for (int64_t i = 0; i < total_elements; ++i) {
        int64_t group_idx_in_batch = i % groups_per_batch;
        if (group_idx_in_batch == groups_per_batch - 1) {
          group_sizes[i] = static_cast<int32_t>(last_group_size);
        } else {
          group_sizes[i] = static_cast<int32_t>(base_group_size);
        }
      }
      ABSL_RETURN_IF_ERROR(InitializeInputBuffer(
          gpu_buffers, stream,
          instr->operand_count() - 1,  // Last parameter is group sizes
          group_sizes.data(), total_elements * sizeof(int32_t)));
    } else if (group_sizes_type == S64) {
      std::vector<int64_t> group_sizes(total_elements);
      // Fill with the pattern: [base_size, base_size, ..., last_size]
      // repeated for each batch
      for (int64_t i = 0; i < total_elements; ++i) {
        int64_t group_idx_in_batch = i % groups_per_batch;
        if (group_idx_in_batch == groups_per_batch - 1) {
          group_sizes[i] = last_group_size;
        } else {
          group_sizes[i] = base_group_size;
        }
      }
      ABSL_RETURN_IF_ERROR(InitializeInputBuffer(
          gpu_buffers, stream,
          instr->operand_count() - 1,  // Last parameter is group sizes
          group_sizes.data(), total_elements * sizeof(int64_t)));
    }
  }

  return absl::OkStatus();
}

}  // namespace

std::unique_ptr<GpuProfiler> GpuProfiler::Create(
    se::StreamExecutor* stream_executor, ProfileOptions options,
    se::DeviceAddressAllocator* external_allocator) {
  std::unique_ptr<se::DeviceAddressAllocator> owned_allocator;
  se::DeviceAddressAllocator* active_allocator = external_allocator;

  if (active_allocator == nullptr) {
    VLOG(1) << "No external allocator provided, creating a new allocator.";
    owned_allocator =
        std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
            stream_executor);
    active_allocator = owned_allocator.get();
  }

  // TODO(b/442997461): Create a new stream using
  // `stream_executor->CreateStream()` instead of reusing the allocator stream
  // once we can handle cuBLAS using multiple streams.
  auto stream = active_allocator->GetStream(stream_executor->device_ordinal());
  if (!stream.ok()) {
    LOG(ERROR) << "Failed to create stream: " << stream.status();
    return nullptr;
  }

  // Overwriting twice the L2 cache size evicts all previously cached lines.
  const int64_t cache_flush_bytes =
      options.flush_caches
          ? std::max(
                2 * stream_executor->GetDeviceDescription().l2_cache_size(),
                kMinCacheFlushBytes)
          : 0;
  return absl::WrapUnique(new GpuProfiler(
      stream_executor, active_allocator, std::move(owned_allocator),
      stream.value(), options, cache_flush_bytes));
}

GpuProfiler::~GpuProfiler() {
  if (!cache_flush_buffer_.is_null()) {
    absl::Status status = stream_->BlockHostUntilDone();
    if (!status.ok()) {
      LOG(ERROR) << "Failed to synchronize stream before freeing cache flush "
                    "buffer: "
                 << status;
    }
  }
}

absl::StatusOr<std::unique_ptr<InputBuffers>> GpuProfiler::CreateInputBuffers(
    const Executable* executable, const HloInstruction* instr) {
  if (!cache_flush_buffer_.is_null()) {
    ABSL_RETURN_IF_ERROR(stream_->BlockHostUntilDone());
    ABSL_RETURN_IF_ERROR(cache_flush_buffer_.Free());
  }
  cache_flush_alloc_failed_ = false;
  ABSL_ASSIGN_OR_RETURN(
      RedzoneBuffers buffers,
      RedzoneBuffers::FromProgramShape(
          executable->compute_computation_layout().ComputeProgramShape(),
          RedzoneBuffers::BuffersToCreate::kAllInputs,
          options_.should_init_buffers,
          /*should_check_correctness=*/true, options_.redzone_padding_bytes,
          allocator_, stream_));
  auto gpu_buffers = std::make_unique<GpuInputBuffers>();
  gpu_buffers->redzone_buffers = std::move(buffers);

  // Initialize buffers based on operation type
  ABSL_RETURN_IF_ERROR(
      InitializeBuffersIfRequiredByOpcode(instr, *gpu_buffers, stream_));

  return gpu_buffers;
}

absl::StatusOr<ProfileResult> GpuProfiler::Profile(
    Executable* executable, const InputBuffers& buffers) {
  const GpuInputBuffers& gpu_buffers =
      absl::down_cast<const GpuInputBuffers&>(buffers);
  const RedzoneBuffers& rz_buffers = gpu_buffers.redzone_buffers;

  ProfileResult result;
  if (auto* gpu_executable = dynamic_cast<const GpuExecutable*>(executable);
      gpu_executable != nullptr) {
    result.scratch_bytes = GetScratchBytes(*gpu_executable);
  }
  {
    // Warm-up run: route every buffer the executable allocates on-demand
    // (result/output buffers and workspace/scratch buffers alike) through a
    // RedzoneDeviceAddressAllocator, so that any out-of-bounds write lands in
    // a mapped post-redzone rather than causing a GPU VM fault.
    std::optional<se::RedzoneDeviceAddressAllocator> warmup_rz;
    se::DeviceAddressAllocator* warmup_alloc = allocator_;
    if (options_.redzone_padding_bytes > 0) {
      warmup_rz.emplace(stream_, allocator_, options_.redzone_padding_bytes);
      warmup_alloc = &warmup_rz.value();
    }
    std::vector<ExecutionInput> execution_inputs =
        CreateExecutionInputsFromBuffers(rz_buffers.input_buffers(),
                                         rz_buffers.input_shapes());
    ABSL_RETURN_IF_ERROR(Execute(executable, std::move(execution_inputs),
                            /*profile=*/nullptr, warmup_alloc)
                        .status());
    ABSL_RETURN_IF_ERROR(stream_->BlockHostUntilDone());
    if (warmup_rz.has_value()) {
      absl::ReaderMutexLock lock(GetGpuMutex(stream_executor_));
      ABSL_ASSIGN_OR_RETURN(se::RedzoneAllocator::RedzoneCheckStatus rz_check,
                       warmup_rz->CheckRedzones());
      if (!rz_check.ok()) {
        std::string redzone_failure_msg = rz_check.RedzoneFailureMsg();
        VLOG(1) << "Autotuning candidate discarded: out-of-bounds write "
                   "detected past an allocated buffer. "
                << redzone_failure_msg;
        return absl::InternalError(absl::StrCat(
            "Autotuning candidate rejected: kernel wrote past its allocated "
            "buffer. ",
            redzone_failure_msg));
      }
    }
  }

  // Average over several timed runs. With a single run, the device timer's
  // quantization and run-to-run noise are comparable to the differences
  // between the best candidates, which makes the selection nondeterministic.
  const int num_timed_runs = std::max(1, options_.num_timed_runs);
  absl::Duration total_duration = absl::ZeroDuration();
  ExecutionProfile profile;
  for (int run = 0; run < num_timed_runs; ++run) {
    ABSL_RETURN_IF_ERROR(FlushCaches());
    profile.Clear();
    profile.set_warmup_run_executed(true);
    std::vector<ExecutionInput> execution_inputs =
        CreateExecutionInputsFromBuffers(rz_buffers.input_buffers(),
                                         rz_buffers.input_shapes());

    ABSL_ASSIGN_OR_RETURN(
        ExecutionOutput execution_output,
        Execute(executable, std::move(execution_inputs), &profile, allocator_));

    total_duration += absl::Nanoseconds(profile.compute_time_ns());
    if (run == num_timed_runs - 1) {
      result.output_buffer = execution_output.Commit().ConsumeResult();
    }
  }
  result.duration = total_duration / num_timed_runs;
  return result;
}

absl::Status GpuProfiler::FlushCaches() {
  if (cache_flush_bytes_ <= 0 || cache_flush_alloc_failed_) {
    return absl::OkStatus();
  }
  auto allocate_flush_buffer = [&]() {
    absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> buffer =
        allocator_->Allocate(stream_executor_->device_ordinal(),
                             cache_flush_bytes_,
                             /*retry_on_failure=*/false);
    if (!buffer.ok()) {
      cache_flush_alloc_failed_ = true;
      if (!logged_cache_flush_alloc_failure_) {
        logged_cache_flush_alloc_failure_ = true;
        LOG(WARNING) << "Failed to allocate " << cache_flush_bytes_
                     << " bytes to flush caches between autotuning runs, "
                        "profiling with warm caches: "
                     << buffer.status();
      }
    }
    return buffer;
  };

  if (allocator_->AllowsAsynchronousDeallocation()) {
    // Pool and VMM allocators handle stream-ordered reuse in Deallocate(), so
    // release the scratch buffer immediately after scheduling MemZero() rather
    // than holding it across input-buffer creation, warm-up redzones, or
    // timed executions.
    absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> buffer =
        allocate_flush_buffer();
    if (!buffer.ok()) {
      return absl::OkStatus();
    }
    return stream_->MemZero(buffer->ptr(), (*buffer)->size());
  }

  if (cache_flush_buffer_.is_null()) {
    absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> buffer =
        allocate_flush_buffer();
    if (!buffer.ok()) {
      return absl::OkStatus();
    }
    cache_flush_buffer_ = *std::move(buffer);
  }
  return stream_->MemZero(cache_flush_buffer_.ptr(),
                          cache_flush_buffer_->size());
}

absl::StatusOr<ExecutionOutput> GpuProfiler::Execute(
    Executable* executable, std::vector<ExecutionInput> inputs,
    ExecutionProfile* profile, se::DeviceAddressAllocator* allocator) {
  // Require exclusive GPU lock to prevent other runs during autotuning.
  GpuExecutableRunOptions gpu_opts;
  gpu_opts.set_requires_exclusive_lock_on_gpu();

  ExecutableRunOptions run_options;
  run_options.set_device_ordinal(stream_executor_->device_ordinal());
  run_options.set_stream(stream_);
  run_options.set_allocator(allocator);
  run_options.set_gpu_executable_run_options(&gpu_opts);
  run_options.set_execution_profile(profile);
  ServiceExecutableRunOptions service_run_options(run_options);
  return executable->ExecuteAsyncOnStreamWrapper(&service_run_options,
                                                 std::move(inputs));
}

absl::Status GpuProfiler::CheckInputBuffers(InputBuffers& buffers) {
  if (options_.redzone_padding_bytes == 0) {
    return absl::OkStatus();
  }
  absl::ReaderMutexLock gpu_lock(GetGpuMutex(stream_executor_));
  const GpuInputBuffers& gpu_buffers =
      absl::down_cast<const GpuInputBuffers&>(buffers);
  const RedzoneBuffers& rz_buffers = gpu_buffers.redzone_buffers;
  ABSL_ASSIGN_OR_RETURN(se::RedzoneAllocator::RedzoneCheckStatus rz_check_status,
                   rz_buffers.RedzoneAllocator().CheckRedzones());
  if (rz_check_status.ok()) {
    return absl::OkStatus();
  }
  LOG(ERROR) << "Red zone modified";
  return absl::InternalError(rz_check_status.RedzoneFailureMsg());
}

absl::Status GpuProfiler::CheckOutputBuffer(ScopedShapedBuffer& output,
                                            ScopedShapedBuffer& reference,
                                            float rtol) {
  absl::ReaderMutexLock gpu_lock(GetGpuMutex(stream_executor_));
  return ShapeUtil::ForEachLeafShapeWithStatus(
      reference.on_device_shape(),
      [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
        std::string diff_report;
        BufferComparator comparator(subshape, rtol,
                                    /*verbose=*/false);

        ABSL_ASSIGN_OR_RETURN(
            bool outputs_match,
            comparator.CompareEqual(stream_, output.buffer(index),
                                    reference.buffer(index), &diff_report));
        if (outputs_match) {
          return absl::OkStatus();
        }
        if (!diff_report.empty()) {
          return absl::InternalError(absl::StrFormat(
              "Output buffer does not match reference buffer at leaf index "
              "%s:\n%s",
              index.ToString(), diff_report));
        }
        return absl::InternalError(absl::StrFormat(
            "Output buffer does not match reference buffer at leaf index %s.",
            index.ToString()));
      });
}

}  // namespace gpu

}  // namespace xla
