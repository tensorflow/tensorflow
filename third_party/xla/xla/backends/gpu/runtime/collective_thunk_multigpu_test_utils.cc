/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/collective_thunk_multigpu_test_utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/scratch_memory_requests.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"

namespace xla::gpu {

se::StreamExecutor* GetGpuExecutor(int ordinal) {
  auto* platform =
      se::PlatformManager::PlatformWithName(se::GpuPlatformName()).value();
  return platform->ExecutorForDevice(ordinal).value();
}

bool IsAtLeastCuda12900(const se::StreamExecutor* executor) {
  const auto& desc = executor->GetDeviceDescription();
  const auto* cuda_cc = desc.gpu_compute_capability().cuda_compute_capability();
  if (cuda_cc == nullptr) {
    return false;
  }
  return std::min(desc.driver_version(), desc.compile_time_toolkit_version()) >=
         se::SemanticVersion(12, 9, 0);
}

bool HasEnoughGpus(int num_devices) {
  auto platform = se::PlatformManager::PlatformWithName(se::GpuPlatformName());
  if (!platform.ok()) {
    return false;
  }
  return (*platform)->VisibleDeviceCount() >= num_devices;
}

DeviceAssignment MakeDeviceAssignment(int num_devices) {
  DeviceAssignment device_assignment(num_devices, /*computation_count=*/1);
  for (int i = 0; i < num_devices; ++i) {
    device_assignment(i, 0) = i;
  }
  return device_assignment;
}

absl::Status SetupCollectiveThunkDevice(
    int device_ordinal, int num_devices, absl::Span<const int64_t> buffer_sizes,
    CollectiveThunk& thunk, const DeviceAssignment& device_assignment,
    CollectiveThunkMultiGpuTestState& state) {
  return SetupCollectiveThunksDevice(device_ordinal, num_devices, buffer_sizes,
                                     {&thunk}, device_assignment, state);
}

absl::Status SetupCollectiveThunksDevice(
    int device_ordinal, int num_devices, absl::Span<const int64_t> buffer_sizes,
    std::initializer_list<CollectiveThunk*> thunks,
    const DeviceAssignment& device_assignment,
    CollectiveThunkMultiGpuTestState& state) {
  state.device_ordinal = device_ordinal;
  state.executor = GetGpuExecutor(device_ordinal);
  ASSIGN_OR_RETURN(state.stream, state.executor->CreateStream());
  state.allocator =
      std::make_unique<se::StreamExecutorAddressAllocator>(state.executor);

  state.create_buffers.clear();
  state.update_buffers.clear();
  state.create_buffers.reserve(buffer_sizes.size());
  state.update_buffers.reserve(buffer_sizes.size());
  for (int64_t size : buffer_sizes) {
    se::DeviceAddressBase create_buffer =
        state.executor->Allocate(size, /*memory_space=*/0);
    if (create_buffer.is_null()) {
      return absl::InternalError(absl::StrFormat(
          "failed to allocate %d bytes on device %d", size, device_ordinal));
    }
    state.create_buffers.push_back(create_buffer);

    se::DeviceAddressBase update_buffer =
        state.executor->Allocate(size, /*memory_space=*/0);
    if (update_buffer.is_null()) {
      return absl::InternalError(absl::StrFormat(
          "failed to allocate %d bytes on device %d", size, device_ordinal));
    }
    state.update_buffers.push_back(update_buffer);
  }

  GpuExecutableRunOptions::DeviceIdMap id_map;
  for (int i = 0; i < num_devices; ++i) {
    id_map[LocalDeviceId(i)] = GlobalDeviceId(i);
  }
  state.gpu_run_options.set_gpu_global_device_ids(std::move(id_map));
  state.run_options.mutable_run_options()->set_stream(state.stream.get());
  state.run_options.mutable_run_options()->set_device_assignment(
      &device_assignment);
  state.run_options.mutable_run_options()->set_gpu_executable_run_options(
      &state.gpu_run_options);
  state.run_options.mutable_run_options()->set_local_device_count(num_devices);

  ASSIGN_OR_RETURN(
      CollectiveParams params,
      CollectiveParams::Create(state.run_options, /*async_streams=*/{},
                               LocalDeviceId(device_ordinal)));

  BufferAllocations allocations =
      MakeBufferAllocations(state, state.create_buffers);
  CollectiveCliqueRequests clique_requests;
  CollectiveMemoryRequests memory_requests(allocations);
  ScratchMemoryRequests scratch_requests;
  Thunk::PrepareParams prepare_params{&params,          &clique_requests,
                                      &memory_requests, &scratch_requests,
                                      state.executor,   &allocations};
  for (CollectiveThunk* thunk : thunks) {
    RETURN_IF_ERROR(thunk->Prepare(prepare_params));
  }

  ASSIGN_OR_RETURN(state.collective_cliques,
                   AcquireCollectiveCliques(params, clique_requests));

  Thunk::InitializeParams init_params;
  init_params.executor = state.executor;
  init_params.stream = state.stream.get();
  init_params.command_buffer_trace_stream = state.stream.get();
  init_params.buffer_allocations = &allocations;
  init_params.collective_params = &params;
  init_params.collective_cliques = &state.collective_cliques;
  init_params.local_device_count = num_devices;
  for (CollectiveThunk* thunk : thunks) {
    RETURN_IF_ERROR(thunk->Initialize(init_params));
  }

  state.collective_params = std::move(params);
  return absl::OkStatus();
}

BufferAllocations MakeBufferAllocations(
    const CollectiveThunkMultiGpuTestState& state,
    absl::Span<const se::DeviceAddressBase> buffers) {
  return BufferAllocations(buffers, state.device_ordinal,
                           state.allocator.get());
}

Thunk::ExecuteParams MakeExecuteParams(
    CollectiveThunkMultiGpuTestState& state,
    const BufferAllocations& buffer_allocations) {
  return Thunk::ExecuteParams::Create(
      state.run_options, buffer_allocations, state.stream.get(),
      /*command_buffer_trace_stream=*/state.stream.get(),
      &*state.collective_params, &state.collective_cliques,
      /*collective_memory=*/nullptr);
}

absl::Status ExecuteOnStreamAndBlock(CollectiveThunk& thunk,
                                     const Thunk::ExecuteParams& params) {
  RETURN_IF_ERROR(thunk.ExecuteOnStream(params));
  return params.stream->BlockHostUntilDone();
}

absl::Status RecordCommandBufferCreate(
    CollectiveThunkMultiGpuTestState& state, CollectiveThunk& thunk,
    const Thunk::ExecuteParams& execute_params) {
  ASSIGN_OR_RETURN(
      state.command_buffer,
      state.executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));

  Command::RecordParams record_params = {state.state_manager};
  ASSIGN_OR_RETURN(state.command,
                   thunk.Record(execute_params, record_params,
                                Command::RecordCreate{/*dependencies=*/{}},
                                state.command_buffer.get()));
  if (state.command == nullptr) {
    return absl::InternalError("Record(create) returned null command node");
  }

  return state.command_buffer->Finalize();
}

absl::Status RecordCommandBufferUpdate(
    CollectiveThunkMultiGpuTestState& state, CollectiveThunk& thunk,
    const Thunk::ExecuteParams& execute_params,
    std::vector<BufferAllocation::Index> updated_allocations) {
  Command::RecordParams record_params = {state.state_manager,
                                         std::move(updated_allocations)};

  RETURN_IF_ERROR(state.command_buffer->Update());
  ASSIGN_OR_RETURN(const se::CommandBuffer::Command* updated_command,
                   thunk.Record(execute_params, record_params,
                                Command::RecordUpdate{state.command},
                                state.command_buffer.get()));

  if (updated_command != state.command) {
    return absl::InternalError(
        "Update returned a different command node; expected the original to be "
        "reused");
  }

  return state.command_buffer->Finalize();
}

absl::Status SubmitCommandBuffer(CollectiveThunkMultiGpuTestState& state) {
  RETURN_IF_ERROR(state.command_buffer->Submit(state.stream.get()));
  return state.stream->BlockHostUntilDone();
}

absl::Status FillDeviceBuffer(se::Stream& stream, se::DeviceAddressBase buffer,
                              absl::Span<const float> data) {
  RETURN_IF_ERROR(
      stream.Memcpy(&buffer, data.data(), sizeof(float) * data.size()));
  return stream.BlockHostUntilDone();
}

absl::StatusOr<std::vector<float>> ReadDeviceBuffer(
    se::Stream& stream, se::DeviceAddressBase buffer, int64_t length) {
  std::vector<float> data(length);
  RETURN_IF_ERROR(stream.Memcpy(data.data(), buffer, sizeof(float) * length));
  RETURN_IF_ERROR(stream.BlockHostUntilDone());
  return data;
}

absl::Status VerifyDeviceBuffer(se::Stream& stream,
                                se::DeviceAddressBase buffer,
                                absl::Span<const float> expected_values) {
  ASSIGN_OR_RETURN(
      std::vector<float> output,
      ReadDeviceBuffer(stream, buffer,
                       static_cast<int64_t>(expected_values.size())));
  for (size_t i = 0; i < expected_values.size(); ++i) {
    if (output[i] != expected_values[i]) {
      return absl::InternalError(absl::StrFormat("output[%d] = %g, expected %g",
                                                 static_cast<int>(i), output[i],
                                                 expected_values[i]));
    }
  }
  return absl::OkStatus();
}

std::vector<float> SentinelValues(int64_t length) {
  return std::vector<float>(length, -7.0f);
}

}  // namespace xla::gpu
