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

#include "xla/pjrt/gpu/tfrt/utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "google/protobuf/text_format.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/local_client.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/executable_run_options.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/maybe_owning.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/distributed/protocol.pb.h"
#include "xla/pjrt/distributed/topology_util.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/gpu/se_gpu_topology_description.h"
#include "xla/pjrt/gpu/tfrt/gpu_event.h"
#include "xla/pjrt/gpu/tfrt/tfrt_gpu_client.h"
#include "xla/pjrt/gpu/tfrt/tfrt_gpu_device.h"
#include "xla/pjrt/gpu/tfrt/thread_checker.h"
#include "xla/pjrt/gpu/tfrt/tracked_gpu_device_buffer.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/runtime/device_id.h"
#include "xla/runtime/process_id.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/gpu_topology.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/profiler/lib/traceme.h"

#if defined(PLATFORM_WINDOWS)
// Required to build successfully with Mingw
#undef CreateEvent
#endif

namespace xla {

std::unique_ptr<se::Stream> MaybeCreateStream(se::StreamExecutor* executor) {
  if (executor == nullptr) {
    return nullptr;
  }
  return executor->CreateStream().value();
}

absl::Status WaitForEventOnStream(se::Stream* stream, se::Event* event) {
  if (!event) {
    return absl::OkStatus();
  }
  return stream->WaitFor(event);
}

absl::StatusOr<std::shared_ptr<se::Event>> CreateCudaEvent(
    TfrtGpuDevice* device) {
  TF_ASSIGN_OR_RETURN(auto cuda_event, device->executor()->CreateEvent());
  return absl::ShareUniquePtr(std::move(cuda_event));
}

Future<> CreateFutureForEvent(tsl::AsyncValueRef<xla::GpuEvent> event) {
  auto [promise, future] = MakePromise<>();
  auto done_fn = [promise = std::move(promise), event]() mutable {
    if (const absl::Status* error = event.GetErrorIfPresent()) {
      VLOG(3) << "Setting future: " << *error;
      promise.Set(*error);
    } else {
      VLOG(3) << "Setting future to OK";
      promise.Set();
    }
  };
  if (event.IsAvailable()) {
    // If the event is available, we can set the promise immediately.
    done_fn();
  } else {
    event.AndThen(std::move(done_fn));
  }
  return future;
}

absl::StatusOr<Shape> GetDestinationDeviceShape(const Shape& host_shape,
                                                TfrtGpuDevice* device,
                                                TfrtGpuClient* client,
                                                PjRtMemorySpace* memory_space) {
  if (host_shape.IsTuple()) {
    return InvalidArgument(
        "Cannot allocate a PjRtStreamExecutorBuffer for a tuple.");
  }

  PjRtMemorySpace* default_memory_space =
      device->default_memory_space().value_or(nullptr);
  if (!memory_space) {
    memory_space = default_memory_space;
  }
  bool is_pinned_host_memory =
      memory_space && (memory_space->kind() == PinnedHostMemorySpace::kKind);
  // Only allow pinned host memory or device memory.
  if (memory_space != default_memory_space && !is_pinned_host_memory) {
    return InvalidArgument("Buffer allocation: invalid memory space");
  }

  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(host_shape));
  TransferManager* transfer_manager =
      client->xla_client()->backend().transfer_manager();
  Shape device_shape = transfer_manager->HostShapeToDeviceShape(host_shape);
  if (is_pinned_host_memory) {
    device_shape.mutable_layout()->set_memory_space(Layout::kHostMemorySpace);
  }
  TF_RET_CHECK(LayoutUtil::HasLayout(device_shape));
  return device_shape;
}

absl::StatusOr<std::unique_ptr<TfrtGpuBuffer>> AllocateTfrtGpuDestinationBuffer(
    const Shape& on_host_shape, tsl::AsyncValueRef<GpuEvent> definition_event,
    TfrtGpuDevice* device, TfrtGpuClient* client, PjRtMemorySpace* memory_space,
    int64_t pack_size) {
  if (on_host_shape.IsTuple()) {
    return Unimplemented(
        "tuple case not implemented for AllocateTfrtGpuDestinationBuffer");
  }
  TF_ASSIGN_OR_RETURN(
      Shape on_device_shape,
      GetDestinationDeviceShape(on_host_shape, device, client, memory_space));
  size_t byte_size =
      pack_size > 0 ? pack_size : ShapeUtil::ByteSizeOf(on_device_shape);
  TF_ASSIGN_OR_RETURN(
      auto device_buffer,
      GpuDeviceMemory::Allocate(client->allocator(),
                                device->local_device_id().value(), byte_size,
                                LayoutUtil::MemorySpace(on_device_shape)));
  auto buffer_async_value_ref =
      tsl::MakeAvailableAsyncValueRef<GpuDeviceMemory>(
          std::move(device_buffer));

  // TODO: Use the right ready event instead of the definition event.
  tsl::AsyncValueRef<GpuEvent> ready_event = definition_event.CopyRef();
  return std::make_unique<TfrtGpuBuffer>(
      on_device_shape,
      std::make_unique<TrackedGpuDeviceBuffer>(buffer_async_value_ref,
                                               std::move(definition_event),
                                               std::move(ready_event)),
      client, device, memory_space);
}

bool IsAllZeros(const DeviceAssignment& assignment) {
  return std::all_of(
      assignment.begin(), assignment.end(),
      [](const DeviceAssignment::value_type& v) { return v == 0; });
}

std::vector<tsl::RCReference<tsl::AsyncValue>> CopyAsyncValues(
    absl::Span<const tsl::RCReference<tsl::AsyncValue>> events) {
  std::vector<tsl::RCReference<tsl::AsyncValue>> avs;
  avs.reserve(events.size());
  for (const auto& ev : events) {
    avs.push_back(ev);
  }
  return avs;
}

// Checks that the input buffers passed in by the user have the correct size
// on device for the compiled program.
absl::Status CheckBufferCompatibilities(
    absl::Span<int64_t const> input_buffer_sizes_in_bytes,
    absl::Span<TrackedGpuDeviceBuffer* const> input_buffers) {
  if (input_buffers.size() != input_buffer_sizes_in_bytes.size()) {
    return InvalidArgument(
        "Execution supplied %lld buffers but compiled program expected %lld "
        "buffers",
        input_buffers.size(), input_buffer_sizes_in_bytes.size());
  }
  for (int i = 0; i < input_buffers.size(); ++i) {
    const auto& buffer = input_buffers[i];
    if (input_buffer_sizes_in_bytes[i] != buffer->buffer()->size_bytes()) {
      return InvalidArgument(
          "Executable expected parameter %d of size %lld but got buffer with"
          " incompatible size %lld ",
          i, input_buffer_sizes_in_bytes[i], buffer->buffer()->size_bytes());
    }
  }
  return absl::OkStatus();
}

std::optional<stream_executor::GpuTargetConfigProto> GetTargetConfigForDevices(
    absl::Span<PjRtDevice* const> devices) {
  if (devices.empty()) {
    return std::nullopt;
  }
  // Temporary ability to disable TargetConfig via env var until
  // internal tests can be fixed.
  const char* disable_target_config_str =
      std::getenv("PJRT_GPU_SE_DISABLE_TARGET_CONFIG");
  int disable_target_config = 0;
  if (disable_target_config_str &&
      absl::SimpleAtoi(disable_target_config_str, &disable_target_config)) {
    if (disable_target_config == 1) {
      return std::nullopt;
    }
  }
  for (const PjRtDevice* device : devices) {
    se::StreamExecutor* executor =
        absl::down_cast<const TfrtGpuDevice*>(device)->executor();
    if (executor != nullptr) {
      return xla::Compiler::GpuTargetConfig(executor).ToProto();
    }
  }
  return std::nullopt;
}

absl::flat_hash_map<std::string, PjRtDeviceAttribute> GetAttrsForDevices(
    std::optional<stream_executor::GpuTargetConfigProto> target_config) {
  absl::flat_hash_map<std::string, PjRtDeviceAttribute> attrs;
  if (target_config.has_value()) {
    std::string attr;
    if (tsl::protobuf::TextFormat::PrintToString(*target_config, &attr)) {
      attrs["target_config"] = std::move(attr);
    }
  }
  return attrs;
}

class TfrtGpuCopyToDeviceStream : public CopyToDeviceStream {
 public:
  TfrtGpuCopyToDeviceStream(int64_t channel_id, se::Stream* stream,
                            se::DeviceAddressBase dst,
                            tsl::AsyncValueRef<std::unique_ptr<se::Event>> done)
      : CopyToDeviceStream(dst.size(), /*granule_bytes=*/1),
        channel_id_(channel_id),
        stream_(stream),
        dst_(dst),
        done_(std::move(done)) {}

  Future<> AddChunk(PjRtChunk chunk) final {
    tsl::profiler::TraceMe trace([&] {
      return tsl::profiler::TraceMeEncode("TfrtGpuCopyToDeviceStream::AddChunk",
                                          {{"channel_id", channel_id_}});
    });

    absl::ReleasableMutexLock lock(mu_);

    VLOG(4) << "Add chunk to a H2D channel #" << channel_id_ << ": "
            << "size=" << chunk.size() << ", "
            << "current_bytes=" << current_bytes_ << ", "
            << "total_bytes=" << total_bytes_;

    if (chunk.size() % granule_size_in_bytes() != 0) {
      done_.SetError(absl::InvalidArgumentError(absl::StrFormat(
          "Chunk size (%d) was not a multiple of the granule size (%d)",
          chunk.size(), granule_size_in_bytes())));
      return Future<>(done_.GetError());
    }

    if (current_bytes_ + chunk.size() > total_bytes_) {
      done_.SetError(absl::InvalidArgumentError(
          absl::StrFormat("Adding chunk of size %d would overflow buffer of "
                          "size %d (%d already transferred)",
                          chunk.size(), total_bytes_, current_bytes_)));
      return Future<>(done_.GetError());
    }

    se::DeviceAddressBase dst(
        reinterpret_cast<std::byte*>(dst_.opaque()) + current_bytes_,
        dst_.size() - current_bytes_);

    current_bytes_ += chunk.size();
    bool complete = IsCompleteLocked();
    lock.Release();

    VLOG(3) << "H2D copy: " << chunk.data() << " -> " << dst.opaque() << " ("
            << chunk.size() << " bytes)";
    auto copied = stream_->Memcpy(&dst, chunk.data(), chunk.size());
    if (!copied.ok()) {
      done_.SetError(copied);
      return Future<>(done_.GetError());
    }

    // Delete chunk once the memcpy operation completes.
    auto deleted = stream_->DoHostCallback(
        [chunk = std::move(chunk), buffer_opaque = dst.opaque()]() {
          VLOG(3) << "H2D copy done. " << buffer_opaque;
        });
    if (!deleted.ok()) {
      done_.SetError(deleted);
      return Future<>(done_.GetError());
    }

    // Record done event once processed the last chunk. It is the caller
    // responsibility to synchronize with this event before submitting any new
    // computations to the stream.
    if (complete) {
      auto recorded = stream_->RecordEvent(done_.get().get());
      if (!recorded.ok()) {
        done_.SetError(recorded);
        return Future<>(done_.GetError());
      }
      done_.SetStateConcrete();
    }

    return Future<>(absl::OkStatus());
  }

 private:
  int64_t channel_id_;
  se::Stream* stream_;
  se::DeviceAddressBase dst_;

  // Async value will become available after we'll submit the last memcpy
  // operation, and the event will be recorded on the stream.
  tsl::AsyncValueRef<std::unique_ptr<se::Event>> done_;
};

// Converts PjRt SendCallbacks to an XLA StreamExecutor send function.
SendDeviceMemoryFunction ConvertSendCallbacksToSendFunction(
    int replica, const ExecuteOptions& options, AsyncWorkRunner* runner) {
  // Check if we have callbacks registered for the given replica.
  if (replica >= options.send_callbacks.size()) {
    return [replica](int64_t channel_id, se::Stream*, const Shape&,
                     const se::DeviceAddressBase&,
                     const absl::flat_hash_map<std::string, std::string>&) {
      return Internal(
          "Don't send a buffer to the channel_id=%d, there was no send "
          "callbacks registered for the replica=%d",
          channel_id, replica);
    };
  }

  // SendCallbacks registered for a device ordinal. Can be empty.
  absl::Span<const SendCallback> callbacks = options.send_callbacks[replica];

  return [callbacks, runner](
             int64_t channel_id, se::Stream* stream, const Shape& shape,
             const se::DeviceAddressBase& src,
             const absl::flat_hash_map<std::string, std::string>&)
             -> absl::StatusOr<tsl::AsyncValueRef<std::unique_ptr<se::Event>>> {
    VLOG(4) << "Send " << src.size() << " bytes to channel #" << channel_id
            << " (shape=" << shape.ToString() << ")";

    const SendCallback* send = FindCallback(channel_id, callbacks);
    if (!send) {
      return InvalidArgument(
          "Failed to send a buffer to the channel_id=%d, callback not found",
          channel_id);
    }

    // Allocate event that will signal completion of send operation. We do not
    // actually track the completion of the send callback, we only have to keep
    // the device memory long enough to complete the memcpy command.
    TF_ASSIGN_OR_RETURN(auto se_event, stream->parent()->CreateEvent());
    auto done_event =
        tsl::MakeConstructedAsyncValueRef<std::unique_ptr<se::Event>>(
            std::move(se_event));

    runner->Schedule([done_event, stream, src, channel_id, shape, send] {
      tsl::profiler::TraceMe trace([&] {
        return tsl::profiler::TraceMeEncode("TfrtGpuExecutable::Send",
                                            {{"channel_id", channel_id}});
      });

      // Allocate chunk on the host for copying data from device.
      PjRtChunk chunk = PjRtChunk::AllocateDefault(src.size());

      VLOG(3) << "D2H copy: " << src.opaque() << " -> " << chunk.data() << " ("
              << src.size() << " bytes)";
      auto status = stream->Memcpy(chunk.data(), src, src.size());
      if (!status.ok()) {
        LOG(ERROR) << "Failed to copy data from device to host: " << status;
        done_event.SetError(status);
        return;
      }
      status = stream->RecordEvent(done_event.get().get());
      if (!status.ok()) {
        done_event.SetError(status);
        return;
      }

      // Wait for the data to be available on the host.
      status = BlockHostUntilDoneWithHostCallback(stream);
      VLOG(3) << "D2H copy done. " << status;
      if (!status.ok()) {
        done_event.SetError(absl::InternalError(absl::StrFormat(
            "failed to synchronize send operation with a stream: %s",
            status.message())));
        return;
      }

      // Pass chunk to the registered callback.
      auto sent = send->callback({shape}, std::move(chunk),
                                 /*total_size_in_bytes=*/src.size(),
                                 /*done=*/true);

      if (!sent.ok()) {
        done_event.SetError(sent);
      } else {
        done_event.SetStateConcrete();
      }
    });

    return std::move(done_event);
  };
}

RecvDeviceMemoryFunction ConvertRecvCallbacksToRecvFunction(
    int replica, const ExecuteOptions& options) {
  // Check if we have callbacks registered for the given replica.
  if (replica >= options.recv_callbacks.size()) {
    return [replica](int64_t channel_id, se::Stream*, const Shape&,
                     se::DeviceAddressBase*,
                     const absl::flat_hash_map<std::string, std::string>&) {
      return InvalidArgument(
          "Failed to receive a buffer from the channel_id=%d, there was no "
          "recv callbacks registered for the replica=%d",
          channel_id, replica);
    };
  }

  // RecvCallbacks registered for a device ordinal. Can be empty.
  absl::Span<const RecvCallback> callbacks = options.recv_callbacks[replica];

  return [callbacks](int64_t channel_id, se::Stream* stream, const Shape& shape,
                     se::DeviceAddressBase* dst,
                     const absl::flat_hash_map<std::string, std::string>&)
             -> absl::StatusOr<tsl::AsyncValueRef<std::unique_ptr<se::Event>>> {
    VLOG(4) << "Recv from channel #" << channel_id
            << " (shape=" << shape.ToString() << ")";

    tsl::profiler::TraceMe trace([&] {
      return tsl::profiler::TraceMeEncode("TfrtGpuExecutable::Recv",
                                          {{"channel_id", channel_id}});
    });

    const RecvCallback* recv = FindCallback(channel_id, callbacks);
    if (!recv) {
      return InvalidArgument(
          "Failed to recv a buffer from the channel_id=%d, callback not found",
          channel_id);
    }

    // Allocate event that will signal completion of recv operation. We record
    // it on a stream after submitting the memcpy for the last chunk (see
    // `TfrtGpuCopyToDeviceStream` implementation above).
    TF_ASSIGN_OR_RETURN(auto event, stream->parent()->CreateEvent());
    auto done_event =
        tsl::MakeConstructedAsyncValueRef<std::unique_ptr<se::Event>>(
            std::move(event));

    recv->callback({shape}, std::make_unique<TfrtGpuCopyToDeviceStream>(
                                channel_id, stream, *dst, done_event));

    return std::move(done_event);
  };
}

std::vector<PjRtMemorySpace*> GetMemorySpacePointers(
    const std::vector<std::unique_ptr<PjRtMemorySpace>>& memory_spaces) {
  std::vector<PjRtMemorySpace*> memory_spaces_ptrs;
  memory_spaces_ptrs.reserve(memory_spaces.size());
  for (const std::unique_ptr<PjRtMemorySpace>& memory_space : memory_spaces) {
    memory_spaces_ptrs.push_back(memory_space.get());
  }
  return memory_spaces_ptrs;
}

std::vector<PjRtDevice*> InitializeDevices(
    TfrtGpuClient* client,
    const std::vector<std::unique_ptr<TfrtGpuDevice>>& owned_devices) {
  std::vector<PjRtDevice*> devices;
  devices.reserve(owned_devices.size());
  for (const std::unique_ptr<TfrtGpuDevice>& device : owned_devices) {
    device->SetClient(client);
    devices.push_back(device.get());
  }
  return devices;
}

absl::flat_hash_map<GlobalDeviceId, TfrtGpuDevice*> GetIdToDeviceMap(
    absl::Span<const std::unique_ptr<TfrtGpuDevice>> devices) {
  absl::flat_hash_map<GlobalDeviceId, TfrtGpuDevice*> id_to_device;
  for (const std::unique_ptr<TfrtGpuDevice>& device : devices) {
    CHECK(id_to_device.emplace(device->global_device_id(), device.get()).second)
        << "Duplicate device id: " << device->id();
  }
  return id_to_device;
}

std::vector<PjRtDevice*> GetAddressableDevicePointers(
    absl::Span<const std::unique_ptr<TfrtGpuDevice>> devices) {
  std::vector<PjRtDevice*> addressable_devices;
  for (const std::unique_ptr<TfrtGpuDevice>& device : devices) {
    if (device->IsAddressable()) {
      addressable_devices.push_back(device.get());
    }
  }
  // TODO(phawkins): we don't really promise anything about the order of
  // these devices, but users may be depending on the current order. Sort into
  // device ordinal order, which is the historical order these values have
  // appeared.
  absl::c_sort(addressable_devices,
               [](const PjRtDevice* a, const PjRtDevice* b) {
                 return a->local_device_id() < b->local_device_id();
               });
  return addressable_devices;
}

StreamExecutorGpuTopologyDescription GetTopology(
    absl::string_view platform_name,
    std::shared_ptr<const GpuTopology> gpu_topology,
    absl::Span<PjRtDevice* const> devices) {
  auto target_config = GetTargetConfigForDevices(devices);
  return StreamExecutorGpuTopologyDescription(
      tsl::Fingerprint64(platform_name), platform_name, std::move(gpu_topology),
      GetAttrsForDevices(target_config), target_config);
}

std::vector<std::unique_ptr<PjRtMemorySpace>> InitializeMemorySpaces(
    int global_device_count,
    absl::Span<PjRtDevice* const> addressable_devices) {
  std::vector<std::unique_ptr<PjRtMemorySpace>> memory_spaces;
  for (auto* device : addressable_devices) {
    // Use the device id to construct a globally unique memory space id. We do
    // not promise that memory space ids and device ids are the same.
    TfrtGpuDevice* gpu_device = absl::down_cast<TfrtGpuDevice*>(device);
    // Initialize the default memory space.
    const int global_device_id = gpu_device->global_device_id().value();
    auto memory_space =
        std::make_unique<TfrtGpuDeviceMemorySpace>(global_device_id, device);
    gpu_device->AttachMemorySpace(memory_space.get(), /*is_default=*/true);
    memory_spaces.push_back(std::move(memory_space));
  }
  const int basePinnedId = global_device_count;
  for (auto* device : addressable_devices) {
    TfrtGpuDevice* gpu_device = absl::down_cast<TfrtGpuDevice*>(device);
    const int global_device_id = gpu_device->global_device_id().value();
    auto pinned = std::make_unique<PinnedHostMemorySpace>(
        basePinnedId + global_device_id, device);
    gpu_device->AttachMemorySpace(pinned.get());
    memory_spaces.push_back(std::move(pinned));
  }
  // We don't promise anything about the order of memory spaces, but this
  // sorting is done for consistency with the device list that's sorted above.
  absl::c_sort(memory_spaces, [](const std::unique_ptr<PjRtMemorySpace>& a,
                                 const std::unique_ptr<PjRtMemorySpace>& b) {
    return a->id() < b->id();
  });
  return memory_spaces;
}

absl::StatusOr<std::unique_ptr<tsl::Allocator>> CreateAllocatorForDevice(
    se::StreamExecutor* executor, const GpuAllocatorConfig& allocator_config) {
  switch (allocator_config.kind) {
    case GpuAllocatorConfig::Kind::kCudaAsync:
      return absl::UnimplementedError(
          "CudaAsync allocator is not supported in TfrtGpuClient.");
    case GpuAllocatorConfig::Kind::kDefault:
    case GpuAllocatorConfig::Kind::kBFC:
      LOG_FIRST_N(INFO, 1) << "Using BFC allocator.";
      return CreateBFCAllocator(executor, allocator_config.memory_fraction,
                                allocator_config.preallocate,
                                allocator_config.gpu_system_memory_size,
                                allocator_config.sub_allocator_alloc_visitors,
                                allocator_config.sub_allocator_free_visitors);
    case GpuAllocatorConfig::Kind::kPlatform:
      LOG(FATAL) << "Platform allocator should be handled before calling this "
                    "function.";
  }
}

absl::StatusOr<MaybeOwning<se::DeviceAddressAllocator>> CreateDeviceAllocator(
    LocalClient* xla_client, const GpuAllocatorConfig& allocator_config,
    const std::vector<std::unique_ptr<TfrtGpuDevice>>& devices) {
  if (allocator_config.kind == GpuAllocatorConfig::Kind::kPlatform) {
    LOG(INFO) << "Using platform allocator.";
    if (allocator_config.collective_memory_size != 0) {
      LOG(WARNING)
          << "collective_memory_size is non-zero, but allocator kind is set "
             "to \"platform\". Collective memory will not be allocated.";
    }
    return MaybeOwning<se::DeviceAddressAllocator>(
        xla_client->backend().memory_allocator());
  }

  std::vector<se::MultiDeviceAdapter::AllocatorInfo> allocators;
  for (const auto& device : devices) {
    se::StreamExecutor* executor = device->executor();
    if (executor == nullptr) {
      // Skips remote devices.
      continue;
    }

    // The stream in the allocator will be used during compilation.
    se::Stream* stream = device->stream();

    std::unique_ptr<tsl::Allocator> allocator;
    if ((allocator_config.kind == GpuAllocatorConfig::Kind::kDefault ||
         allocator_config.kind == GpuAllocatorConfig::Kind::kBFC) &&
        allocator_config.preallocate) {
      GpuAllocatorConfig device_allocator_config = allocator_config;
      // Assert that CUDA alloc/free calls are not made on the caller thread.
      auto visitor = [](void*, int index, size_t) {
        TfrtGpuThreadChecker::AssertCudaCallAllowedOnThisThread();
      };
      device_allocator_config.sub_allocator_alloc_visitors.push_back(visitor);
      device_allocator_config.sub_allocator_free_visitors.push_back(visitor);

      TF_ASSIGN_OR_RETURN(allocator, CreateAllocatorForDevice(
                                         executor, device_allocator_config));

      // Immediately expand the allocator instead of on the first allocation so
      // that we can control the thread on which the expansion happens. This
      // works because the BFC allocator with preallocation expands its pool to
      // the configured size on first allocation.
      allocator->DeallocateRaw(
          allocator->AllocateRaw(tsl::Allocator::kAllocatorAlignment, 1));
    } else {
#ifdef PLATFORM_GOOGLE
      LOG(WARNING)
          << "TFRT GPU is running without preallocation; this may cause CUDA "
             "calls to happen inline on the calling thread any time the "
             "allocator runs out of memory and has to expand synchronously, "
             "which is problematic if the calling thread is a fiber";
#endif
      TF_ASSIGN_OR_RETURN(allocator,
                          CreateAllocatorForDevice(executor, allocator_config));
    }
    allocators.emplace_back(
        std::move(allocator), stream,
        /*memory_space=*/
        static_cast<int>(stream_executor::MemorySpace::kDevice),
        executor->device_ordinal(), executor->GetPlatform());

    TF_ASSIGN_OR_RETURN(
        auto collective_bfc_allocator,
        CreateCollectiveBFCAllocator(
            executor,
            /*memory_fraction=*/1.0 - allocator_config.memory_fraction,
            allocator_config.collective_memory_size));
    allocators.emplace_back(std::move(collective_bfc_allocator), stream,
                            /*memory_space=*/1, executor->device_ordinal(),
                            executor->GetPlatform());

    TF_ASSIGN_OR_RETURN(auto host_allocator, GetGpuHostAllocator(executor));
    allocators.emplace_back(
        std::move(host_allocator), stream,
        /*memory_space=*/static_cast<int>(stream_executor::MemorySpace::kHost),
        executor->device_ordinal(), executor->GetPlatform());
  }
  return MaybeOwning<se::DeviceAddressAllocator>(
      std::make_unique<se::MultiDeviceAdapter>(xla_client->platform(),
                                               std::move(allocators)));
}

using DeviceTopologyPair =
    std::pair<std::vector<std::unique_ptr<TfrtGpuDevice>>, GpuTopologyProto>;

absl::StatusOr<DeviceTopologyPair> BuildDistributedDevices(
    absl::string_view platform_name, LocalClient* xla_client, int process_id,
    int max_inflight_computations, int num_nodes,
    gpu::GpuExecutableRunOptions* gpu_executable_run_options,
    std::shared_ptr<KeyValueStoreInterface> kv_store, bool enable_mock_nccl,
    std::optional<absl::string_view> mock_gpu_topology,
    std::optional<int> partition_index,
    absl::Duration get_local_topology_timeout,
    absl::Duration get_global_topology_timeout) {
  std::vector<std::unique_ptr<TfrtGpuDevice>> devices;
  LocalTopologyProto local_topology;
  local_topology.set_process_id(process_id);
  auto boot_id_str_or_status = GetBootIdString();
  if (!boot_id_str_or_status.ok()) {
    LOG(INFO) << boot_id_str_or_status.status();
  } else {
    local_topology.set_boot_id(boot_id_str_or_status.value());
  }
  if (partition_index.has_value()) {
    local_topology.set_partition_index(*partition_index);
  }

  for (se::StreamExecutor* executor :
       xla_client->backend().stream_executors()) {
    const se::Platform* platform = executor->GetPlatform();

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::se::DeviceDescription> desc,
        platform->DescriptionForDevice(executor->device_ordinal()));
    DeviceProto* device_proto = local_topology.add_devices();
    device_proto->set_local_device_ordinal(executor->device_ordinal());
    device_proto->set_name(desc->name());
    device_proto->set_vendor(desc->device_vendor());
    device_proto->set_compute_capability(
        stream_executor::MakeComputeCapabilityAttributeString(*desc));
    device_proto->set_core_count(desc->core_count());

    // TODO: hhb
    // const se::GpuComputeCapability& compute_capability =
    //     desc->gpu_compute_capability();
    // if (std::holds_alternative<se::CudaComputeCapability>(compute_capability)
    // &&
    //     std::get<se::CudaComputeCapability>(compute_capability).major >= 9) {
    //   auto fabric_info = GetDeviceFabricInfo(executor->device_ordinal());
    //   if (fabric_info.ok()) {
    //     device_proto->set_fabric_uuid(*fabric_info);
    //   }
    // }
  }

  GlobalTopologyProto global_topology;
  if (enable_mock_nccl) {
    TopologySizes sizes;
    if (mock_gpu_topology.has_value()) {
      TF_ASSIGN_OR_RETURN(sizes, TopologySizes::FromString(*mock_gpu_topology));
    } else {
      // If there is no topology spec, we assume that each node is a partition,
      // there is one process (host) on each partition and each host
      // has all the local devices.
      sizes.num_partitions = num_nodes;
      sizes.num_hosts_per_partition = 1;
      sizes.num_devices_per_host = local_topology.devices().size();
    }

    if (sizes.num_devices_per_host != local_topology.devices().size()) {
      return absl::InternalError(
          "The number of devices per host in 'mock_gpu_topology' "
          "must be the same as the number of devices in the local topology");
    }

    if (sizes.num_partitions * sizes.num_hosts_per_partition != num_nodes) {
      return absl::InternalError(
          "The number of hosts in 'mock_gpu_topology' "
          "must be the same as 'num_nodes'");
    }

    std::vector<LocalTopologyProto> local_topologies(num_nodes, local_topology);
    for (int i = 0; i < sizes.num_partitions; ++i) {
      for (int j = 0; j < sizes.num_hosts_per_partition; j++) {
        int process_id = i * sizes.num_hosts_per_partition + j;
        local_topologies[process_id].set_process_id(process_id);
        local_topologies[process_id].set_boot_id(absl::StrCat(i));
      }
    }
    TF_ASSIGN_OR_RETURN(global_topology,
                        BuildGlobalTopology(absl::MakeSpan(local_topologies),
                                            /*assign_global_device_ids=*/true));
  } else {
    TF_RETURN_IF_ERROR(ExchangeTopologies(
        platform_name, process_id, num_nodes, get_local_topology_timeout,
        get_global_topology_timeout, kv_store.get(), local_topology,
        &global_topology, /*assign_global_device_ids=*/true));
  }

  absl::btree_map<LocalDeviceId, GlobalDeviceId> gpu_device_ids;
  absl::flat_hash_map<GlobalDeviceId, ProcessId> device_to_process;
  int curr_partition_index = -1;
  int curr_process_index = -1;
  int curr_process_index_in_partition = 0;
  for (const LocalTopologyProto& node : global_topology.processes()) {
    for (const DeviceProto& device_proto : node.devices()) {
      // The devices in the global topology are ordered by process_id,
      // partition_index. This is guaranteed by the `BuildGlobalTopology`
      // function and the `ExchangeTopologies` function.
      if (curr_partition_index != device_proto.partition_index()) {
        curr_partition_index = device_proto.partition_index();
        curr_process_index = node.process_id();
        curr_process_index_in_partition = 0;
      }
      if (curr_process_index != node.process_id()) {
        curr_process_index = node.process_id();
        curr_process_index_in_partition++;
      }

      GlobalDeviceId global_device_id(device_proto.global_device_id());
      device_to_process[global_device_id] = ProcessId(node.process_id());
      TfrtGpuDevice::Options options;
      if (node.process_id() == process_id) {
        gpu_device_ids[LocalDeviceId(device_proto.local_device_ordinal())] =
            global_device_id;
        // Assign some descriptive names for profiling tools.
        // TODO: hhb
        // NameDeviceAndLauncherThread(node, device_proto,
        //                             local_device->execute_thread());

        TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                            xla_client->backend().stream_executor(
                                device_proto.local_device_ordinal()));
        options.local_device_id = executor->device_ordinal();
        options.local_hardware_id = executor->device_ordinal();
        options.executor = executor;
      } else {
        options.local_device_id = device_proto.local_device_ordinal();
        options.local_hardware_id = -1;
        options.executor = nullptr;
      }
      options.id = device_proto.global_device_id();
      options.process_index = node.process_id();
      options.process_index_in_partition = curr_process_index_in_partition;
      options.partition_index = device_proto.partition_index();
      options.max_inflight_computations = max_inflight_computations;
      options.platform_version = device_proto.name();
      options.device_vendor = device_proto.vendor();
      options.compute_capability = device_proto.compute_capability();
      options.core_count = device_proto.core_count();

      auto device = std::make_unique<TfrtGpuDevice>(std::move(options));
      devices.push_back(std::move(device));
    }
  }
  for (se::StreamExecutor* executor :
       xla_client->backend().stream_executors()) {
    TF_RET_CHECK(gpu_device_ids.find(LocalDeviceId(
                     executor->device_ordinal())) != gpu_device_ids.end());
  }
  gpu_executable_run_options->set_gpu_global_device_ids(
      std::move(gpu_device_ids));

  TF_ASSIGN_OR_RETURN(xla::Collectives * collectives,
                      xla::CollectivesRegistry::Default("gpu"));
  xla::gpu::GpuCollectives* gpu_collectives =
      absl::down_cast<gpu::GpuCollectives*>(collectives);

  if (gpu_collectives == nullptr) {
    return absl::InternalError("Failed to get GPU collectives");
  }

  size_t num_processes = global_topology.processes().size();
  TF_ASSIGN_OR_RETURN(auto clique_id_callback,
                      gpu_collectives->InitializeTopology(
                          {ProcessId(process_id), num_processes,
                           xla_client->backend().stream_executors().size(),
                           kv_store, device_to_process}));
  gpu_executable_run_options->set_clique_id_callback(
      std::move(clique_id_callback));

  TF_ASSIGN_OR_RETURN(GpuTopologyProto gpu_topology,
                      BuildGpuTopology(global_topology));
  return std::make_pair(std::move(devices), gpu_topology);
}

absl::StatusOr<absl::string_view> MemoryKindFromSimpleShape(
    const Shape& shape, absl::string_view default_memory_kind) {
  if (!shape.has_layout()) {
    return default_memory_kind;
  }
  switch (shape.layout().memory_space()) {
    case Layout::kHostMemorySpace:
      return PinnedHostMemorySpace::kKind;
    case Layout::kGenericFastMemorySpace:
    case Layout::kDefaultMemorySpace:
      return default_memory_kind;
    default:
      return InvalidArgument("Unexpected memory space %d in output layout",
                             shape.layout().memory_space());
  }
}

absl::StatusOr<std::vector<absl::string_view>> MemoryKindsFromShape(
    const Shape& shape, absl::string_view default_memory_kind) {
  if (!shape.IsTuple()) {
    TF_ASSIGN_OR_RETURN(absl::string_view memory_kind,
                        MemoryKindFromSimpleShape(shape, default_memory_kind));
    return {{memory_kind}};
  }
  std::vector<absl::string_view> result;
  result.reserve(shape.tuple_shapes().size());
  for (const auto& element_shape : shape.tuple_shapes()) {
    TF_ASSIGN_OR_RETURN(
        absl::string_view element_memory_kind,
        MemoryKindFromSimpleShape(element_shape, default_memory_kind));
    result.push_back(element_memory_kind);
  }
  return result;
}

absl::flat_hash_map<GlobalDeviceId, IncarnationId> GetLatestIncarnations(
    absl::Span<PjRtDevice* const> devices,
    const absl::flat_hash_map<int, IncarnationId>& incarnations) {
  // Map every device to its incarnation.
  absl::flat_hash_map<GlobalDeviceId, IncarnationId> device_incarnations;
  for (const PjRtDevice* device : devices) {
    int task_id = device->process_index();
    auto it = incarnations.find(task_id);
    if (it == incarnations.end()) {
      // The task might be dead.
      LOG(WARNING) << "Incarnation for task " << task_id << " not found";
      continue;
    }
    GlobalDeviceId device_id(device->global_device_id().value());
    device_incarnations[device_id] = it->second;
  }
  return device_incarnations;
}

absl::Status BlockHostUntilDoneWithHostCallback(se::Stream* stream) {
  absl::Notification event;

  tsl::profiler::TraceMe traceme("BlockHostUntilDoneWithHostCallback");
  auto status = stream->DoHostCallback([&event]() {
    tsl::profiler::TraceMe traceme(
        "BlockHostUntilDoneWithHostCallback::Callback");
    event.Notify();
  });

  event.WaitForNotification();

  return status;
}

}  // namespace xla
