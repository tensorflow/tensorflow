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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/backends/gpu/runtime/buffer_comparator.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/autotuning/redzone_buffers.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/maybe_owning_device_address.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/gpu/redzone_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

namespace xla {

namespace gpu {

namespace {

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

int GetScratchBytes(const Executable* executable) {
  int scratch_bytes = 0;
  for (const auto* allocation : executable->GetAllocations()) {
    if (allocation->IsPreallocatedTempBuffer()) {
      for (const auto& [buffer, offset] : allocation->assigned_buffers()) {
        // Scratch space is allocated as the second element in the output tuple
        // of the instruction.
        const auto& shape_index = buffer->positions().front().index;
        bool is_second_element_in_output_tuple =
            !shape_index.empty() && shape_index[0] == 1;
        if (is_second_element_in_output_tuple) {
          scratch_bytes += offset.size;
        }
      }
    }
  }
  return scratch_bytes;
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
  TF_RETURN_IF_ERROR(stream->Memcpy(const_cast<se::DeviceAddressBase*>(&buffer),
                                    values, size_bytes));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

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
    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
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
      TF_RETURN_IF_ERROR(InitializeInputBuffer(
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
      TF_RETURN_IF_ERROR(InitializeInputBuffer(
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
  return absl::WrapUnique(new GpuProfiler(stream_executor, active_allocator,
                                          std::move(owned_allocator),
                                          stream.value(), options));
}

absl::StatusOr<std::unique_ptr<InputBuffers>> GpuProfiler::CreateInputBuffers(
    const Executable* executable, const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(
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
  TF_RETURN_IF_ERROR(
      InitializeBuffersIfRequiredByOpcode(instr, *gpu_buffers, stream_));

  return gpu_buffers;
}

absl::StatusOr<ProfileResult> GpuProfiler::Profile(
    Executable* executable, const InputBuffers& buffers) {
  const GpuInputBuffers& gpu_buffers =
      tsl::down_cast<const GpuInputBuffers&>(buffers);
  const RedzoneBuffers& rz_buffers = gpu_buffers.redzone_buffers;
  ProfileResult result;
  result.scratch_bytes = GetScratchBytes(executable);
  {
    // Warm up run.
    std::vector<ExecutionInput> execution_inputs =
        CreateExecutionInputsFromBuffers(rz_buffers.input_buffers(),
                                         rz_buffers.input_shapes());
    TF_RETURN_IF_ERROR(Execute(executable, std::move(execution_inputs),
                               /*profile=*/nullptr)
                           .status());

    TF_RETURN_IF_ERROR(stream_->BlockHostUntilDone());
  }

  ExecutionProfile profile;
  profile.set_warmup_run_executed(true);
  std::vector<ExecutionInput> execution_inputs =
      CreateExecutionInputsFromBuffers(rz_buffers.input_buffers(),
                                       rz_buffers.input_shapes());

  TF_ASSIGN_OR_RETURN(
      ExecutionOutput execution_output,
      Execute(executable, std::move(execution_inputs), &profile));

  result.duration = absl::Nanoseconds(profile.compute_time_ns());
  result.output_buffer = execution_output.Commit().ConsumeResult();
  return result;
}

absl::StatusOr<ExecutionOutput> GpuProfiler::Execute(
    Executable* executable, std::vector<ExecutionInput> inputs,
    ExecutionProfile* profile) {
  // Require exclusive GPU lock to prevent other runs during autotuning.
  GpuExecutableRunOptions gpu_opts;
  gpu_opts.set_requires_exclusive_lock_on_gpu();

  ExecutableRunOptions run_options;
  run_options.set_device_ordinal(stream_executor_->device_ordinal());
  run_options.set_stream(stream_);
  run_options.set_allocator(allocator_);
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
  const GpuInputBuffers& gpu_buffers =
      tsl::down_cast<const GpuInputBuffers&>(buffers);
  const RedzoneBuffers& rz_buffers = gpu_buffers.redzone_buffers;
  TF_ASSIGN_OR_RETURN(se::RedzoneAllocator::RedzoneCheckStatus rz_check_status,
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
  return ShapeUtil::ForEachLeafShapeWithStatus(
      reference.on_device_shape(),
      [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
        BufferComparator comparator(subshape, rtol,
                                    /*verbose=*/false);

        TF_ASSIGN_OR_RETURN(
            bool outputs_match,
            comparator.CompareEqual(stream_, output.buffer(index),
                                    reference.buffer(index)));
        if (outputs_match) {
          return absl::OkStatus();
        }
        return absl::InternalError(
            "Output buffer does not match reference buffer.");
      });
}

}  // namespace gpu

}  // namespace xla
