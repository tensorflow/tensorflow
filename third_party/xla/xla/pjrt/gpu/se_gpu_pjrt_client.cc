/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"

#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/time/time.h"
#include "xla/client/local_client.h"
#include "xla/client/xla_computation.h"
#include "xla/pjrt/distributed/topology_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/stream_executor_executable.h"
#include "xla/pjrt/tracked_device_buffer.h"
#include "xla/pjrt/utils.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream_executor_internal.h"
#include "tsl/framework/allocator.h"
#include "tsl/framework/bfc_allocator.h"
#include "tsl/lib/strings/proto_serialization.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/threadpool.h"
#include "tsl/profiler/lib/connected_traceme.h"

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
#include "xla/pjrt/compile_options.pb.h"
#include "xla/pjrt/gpu/nccl_id_store.h"
#include "xla/pjrt/metrics.h"
#include "xla/pjrt/stream_executor_executable.pb.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/xla.pb.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "xla/stream_executor/gpu/gpu_cudamallocasync_allocator.h"
#elif TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#endif

#include "xla/client/client_library.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/platform_util.h"
#include "xla/statusor.h"
#include "xla/stream_executor/integrations/device_host_allocator.h"
#include "xla/stream_executor/integrations/device_mem_allocator.h"
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"
#include "xla/util.h"
#include "tsl/framework/device_id.h"
#include "tsl/util/env_var.h"

namespace xla {
class AsyncHostToDeviceTransferManager
    : public xla::PjRtClient::AsyncHostToDeviceTransferManager {
 public:
  static StatusOr<std::unique_ptr<AsyncHostToDeviceTransferManager>> Create(
      absl::Span<const Shape> shapes, PjRtStreamExecutorDevice* device,
      PjRtStreamExecutorClient* client) {
    absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4> buffers;
    absl::InlinedVector<std::shared_ptr<TrackedDeviceBuffer>, 4> buffer_ptrs;
    absl::InlinedVector<std::shared_ptr<BufferSequencingEvent>, 4>
        definition_events;
    buffers.reserve(shapes.size());
    buffer_ptrs.reserve(shapes.size());
    definition_events.reserve(shapes.size());
    for (const auto& shape : shapes) {
      if (shape.IsTuple()) {
        return Unimplemented(
            "Async buffer transfer of tuples not implemented.");
      }
      // Initialize a definition event for each async buffer. The definition
      // event will block the buffer usage until the transfer is done.
      definition_events.push_back(
          std::make_shared<BufferSequencingEvent>(client->thread_pool()));
      TF_ASSIGN_OR_RETURN(auto buffer,
                          client->CreateUninitializedBuffer(
                              shape, device, definition_events.back()));
      // Get a temporary hold just so we can fish out a shared_ptr to the
      // TrackedDeviceBuffer. It's ok to drop the hold before return the
      // buffers, because the invariants of this class ensure that the buffer
      // definition event will not fire until after all of this class' uses of
      // the TrackedDeviceBuffer have completed.
      auto* se_buffer =
          tensorflow::down_cast<PjRtStreamExecutorBuffer*>(buffer.get());
      DCHECK(se_buffer);
      auto hold = se_buffer->GetBufferWithUsageHold();
      buffer_ptrs.push_back(hold.buffer());
      buffers.push_back(std::move(buffer));
    }

    return std::make_unique<AsyncHostToDeviceTransferManager>(
        std::move(buffers), std::move(buffer_ptrs),
        std::move(definition_events), device);
  }

  AsyncHostToDeviceTransferManager(
      absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4> buffers,
      absl::InlinedVector<std::shared_ptr<TrackedDeviceBuffer>, 4> buffer_ptrs,
      absl::InlinedVector<std::shared_ptr<BufferSequencingEvent>, 4>
          definition_events,
      PjRtStreamExecutorDevice* device)
      : buffers_(std::move(buffers)),
        buffer_ptrs_(std::move(buffer_ptrs)),
        definition_events_(std::move(definition_events)),
        remaining_buffer_count_(buffer_ptrs_.size()),
        transfers_in_flight_(0),
        device_(device) {
    buffer_sizes_.reserve(buffer_ptrs_.size());
    for (const auto& ptr : buffer_ptrs_) {
      DCHECK_EQ(ptr->device_memory().size(), 1);
      buffer_sizes_.push_back(ptr->device_memory()[0].size());
    }
    last_transfer_started_.resize(buffer_ptrs_.size(), false);
  }

  ~AsyncHostToDeviceTransferManager() override {
    auto transfers_finished = [this]() {
      mu_.AssertHeld();
      return transfers_in_flight_ == 0;
    };
    {
      absl::MutexLock l(&mu_);
      // Make sure we don't leave dangling pointers in cleanup routines even
      // if the client lets the object go out of scope.
      mu_.Await(absl::Condition(&transfers_finished));
    }
  }

  size_t buffer_count() const override { return buffers_.size(); };

  size_t buffer_size(int buffer_index) const override {
    DCHECK_LT(buffer_index, buffer_sizes_.size());
    return buffer_sizes_[buffer_index];
  }

  PjRtDevice* device() const override { return device_; }

  std::unique_ptr<PjRtBuffer> RetrieveBuffer(int buffer_index) override {
    DCHECK_LT(buffer_index, buffers_.size());
    return std::move(buffers_[buffer_index]);
  };

  Status TransferLiteralToBuffer(
      int buffer_index, const LiteralSlice& literal,
      absl::AnyInvocable<void() &&> on_done) override {
    tsl::profiler::TraceMe traceme(
        "AsyncHostToDeviceTransferManager::TransferLiteralToBuffer");
    auto* stream = device_->local_device_state()->host_to_device_stream();
    auto* se_client =
        tensorflow::down_cast<PjRtStreamExecutorClient*>(device_->client());
    DCHECK(se_client);

    TransferManager* transfer_manager =
        se_client->client()->backend().transfer_manager();
    TF_ASSIGN_OR_RETURN(
        Shape compact_shape,
        transfer_manager->ChooseCompactLayoutForShape(literal.shape()));

    std::shared_ptr<TrackedDeviceBuffer> buffer;
    {
      absl::MutexLock l(&mu_);

      DCHECK_LT(buffer_index, buffer_ptrs_.size());
      if (last_transfer_started_[buffer_index]) {
        return InvalidArgument(
            "TransferLiteralToBuffer requested for buffer index %d which has "
            "already been fully transferred",
            buffer_index);
      }
      last_transfer_started_[buffer_index] = true;
      buffer = buffer_ptrs_[buffer_index];
      DCHECK(buffer);
      if (buffer->device_memory().empty()) {
        return InvalidArgument(
            "TransferLiteralToBuffer requested for buffer index %d which has "
            "been donated. Async transfer of donated buffers is not supported "
            "in SE:GPU",
            buffer_index);
      }
      DCHECK_EQ(buffer->device_memory().size(), 1);

      auto& buffer_memory = buffer->device_memory()[0];
      if (transfer_manager->GetByteSizeRequirement(compact_shape) !=
          buffer_memory.size()) {
        return InvalidArgument(
            "TransferLiteralToBuffer shape %s has size %lld "
            "but buffer has size %lld",
            ShapeUtil::HumanStringWithLayout(compact_shape),
            transfer_manager->GetByteSizeRequirement(compact_shape),
            buffer_memory.size());
      }
      ++transfers_in_flight_;
    }

    // The host to device transfer is performed on a thread pool, mostly because
    // it includes linearization that may be slow.
    // TODO(misard) assess if it would be preferable to introduce a heuristic to
    // put the transfer into the calling thread for small literals.
    auto transfer_h2d = [this, buffer_index, stream, transfer_manager, literal,
                         device_buffer = buffer.get(), compact_shape,
                         local_device =
                             std::move(device_->local_device_state()),
                         on_done = std::move(on_done)]() mutable {
      tsl::profiler::TraceMe traceme(
          "AsyncHostToDeviceTransferManager::TransferLiteralToBuffer::transfer_"
          "h2d");

      auto event = local_device->event_pool().AllocateEvent(stream->parent());

      // Initiate linearization and transfer of the buffer on the stream.
      ShapedBuffer buffer = device_buffer->AsShapedBuffer(compact_shape);
      TF_CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(
          stream, literal, buffer));
      local_device->event_pool().ThenRecordEvent(stream, event.value());

      // Call cleanup once the transfer has finished on the stream.
      auto cleanup = [this, buffer_index, stream, on_done = std::move(on_done),
                      event = std::move(event).value()]() mutable {
        CleanUp(buffer_index, std::move(event), stream,
                /*is_last_transfer=*/true, std::move(on_done));
      };
      stream->ThenDoHostCallback(std::move(cleanup));
    };
    se_client->thread_pool()->Schedule(
        ([ptr = new absl::AnyInvocable<void()>(std::move(transfer_h2d))]() {
          (*ptr)();
          delete ptr;
        }));
    return OkStatus();
  }

  Status TransferRawDataToBuffer(
      int buffer_index, absl::string_view data,
      absl::AnyInvocable<void() &&> on_done) override {
    return TransferRawDataToSubBuffer(buffer_index, data.data(),
                                      /*offset=*/0, data.size(),
                                      /*is_last_transfer=*/true,
                                      std::move(on_done));
  }

  Status TransferRawDataToSubBuffer(
      int buffer_index, const void* data, int64_t offset, int64_t transfer_size,
      bool is_last_transfer, absl::AnyInvocable<void() &&> on_done) override {
    auto* stream = device_->local_device_state()->host_to_device_stream();

    absl::ReleasableMutexLock l(&mu_);
    DCHECK_LT(buffer_index, buffer_ptrs_.size());
    if (last_transfer_started_[buffer_index]) {
      return InvalidArgument(
          "TransferRawData requested for buffer index %d which has "
          "already been fully transferred",
          buffer_index);
    }
    if (is_last_transfer) {
      last_transfer_started_[buffer_index] = true;
    }
    DCHECK(buffer_ptrs_[buffer_index]);
    if (buffer_ptrs_[buffer_index]->device_memory().empty()) {
      return InvalidArgument(
          "TransferRawDataToSubBuffer requested for buffer index %d which has "
          "been donated. Async transfer of donated buffers is not supported "
          "in SE:GPU",
          buffer_index);
    }
    DCHECK_EQ(buffer_ptrs_[buffer_index]->device_memory().size(), 1);
    auto& buffer_memory = buffer_ptrs_[buffer_index]->device_memory()[0];
    se::DeviceMemoryBase sub_buffer;
    CHECK_LE(offset, buffer_memory.size());
    CHECK_LE(transfer_size, buffer_memory.size() - offset);
    if (transfer_size < buffer_memory.size()) {
      sub_buffer = se::DeviceMemoryBase(
          reinterpret_cast<char*>(buffer_memory.opaque()) + offset,
          transfer_size);
    } else {
      sub_buffer = buffer_memory;
    }

    ++transfers_in_flight_;
    auto event = device_->local_device_state()->event_pool().AllocateEvent(
        stream->parent());
    if (transfer_size != 0) {
      stream->ThenMemcpy(&sub_buffer, data, transfer_size);
    }
    device_->local_device_state()->event_pool().ThenRecordEvent(stream,
                                                                event.value());
    // Release the lock before calling ThenDoHostCallback in case cleanup
    // could be called on this thread, to avoid deadlock.
    l.Release();

    auto cleanup = [this, buffer_index, event = std::move(event).value(),
                    stream, is_last_transfer,
                    on_done = std::move(on_done)]() mutable {
      CleanUp(buffer_index, std::move(event), stream, is_last_transfer,
              std::move(on_done));
    };
    stream->ThenDoHostCallback(std::move(cleanup));
    return OkStatus();
  }

  void SetBufferError(int buffer_index, Status error) override {
    {
      absl::MutexLock l(&mu_);
      // For a given buffer_index, SetBufferError can't be called twice, or
      // called after the last transfer has been enqueued.
      CHECK(!definition_events_[buffer_index]->IsDefined());
      definition_events_[buffer_index]->SetDefinedStatus(error);
    }
    VLOG(1) << "SetBufferError sets the " << buffer_index
            << "th buffer error: " << error;
  }

  void AddTransferMetadata(const TransferMetadata& meta) override {}

 private:
  absl::Mutex mu_;
  // The newly created buffers, which will be returned to the caller via
  // Retrieve.
  absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4> buffers_;
  // Cached versions of the sizes of all the buffers, so we can return them
  // without acquiring mu_.
  absl::InlinedVector<size_t, 4> buffer_sizes_;
  // References to the underlying storage for all the buffers, which ensures
  // that the buffers can't be freed before all transfers complete.
  absl::InlinedVector<std::shared_ptr<TrackedDeviceBuffer>, 4> buffer_ptrs_
      ABSL_GUARDED_BY(mu_);
  // True if the last transfer for a buffer has been initiated. Used to prevent
  // a client initiating another transfer after the last transfer has already
  // been initiated.
  absl::InlinedVector<bool, 4> last_transfer_started_ ABSL_GUARDED_BY(mu_);
  // The buffer definition events on all the buffers, unblocked once the
  // corresponding buffer transfer has completed.
  absl::InlinedVector<std::shared_ptr<BufferSequencingEvent>, 4>
      definition_events_ ABSL_GUARDED_BY(mu_);
  // Count of buffers that have not yet been fully transferred.
  size_t remaining_buffer_count_ ABSL_GUARDED_BY(mu_);
  // Count of transfers that have been started but have not yet called cleanup.
  // Used to block in the destructor to avoid dangling pointers in cleanup.
  int transfers_in_flight_ ABSL_GUARDED_BY(mu_);

  PjRtStreamExecutorDevice* device_;  // not owned.

  void CleanUp(int buffer_index, EventPool::Handle event, se::Stream* stream,
               bool is_last_transfer, absl::AnyInvocable<void() &&> on_done) {
    {
      absl::MutexLock l(&mu_);

      CHECK_GT(transfers_in_flight_, 0);
      --transfers_in_flight_;
      if (is_last_transfer) {
        // Drop our reference to the TrackedDeviceBuffer for this buffer.
        CHECK(buffer_ptrs_[buffer_index]);
        buffer_ptrs_[buffer_index] = nullptr;
        CHECK_GT(remaining_buffer_count_, 0);
        --remaining_buffer_count_;
        definition_events_[buffer_index]->SetSequencingEvent(std::move(event),
                                                             stream);
        if (remaining_buffer_count_ == 0) {
          VLOG(1) << "TransferLiteralToBuffer for all buffers is done.";
        }
      }
    }

    // Call on_done after finishing all housekeeping and releasing the lock.
    std::move(on_done)();
  }
};

absl::string_view StreamExecutorGpuClient::platform_version() const {
#define STRINGIFY2(X) #X
#define STRINGIFY(X) STRINGIFY2(X)
#if TENSORFLOW_USE_ROCM && defined(TF_ROCM_VERSION)  // rocm
  // TF_ROCM_VERSION format may change in future. Use it
  // cautiously
  return "rocm " STRINGIFY(TF_ROCM_VERSION);
#elif GOOGLE_CUDA && defined(CUDART_VERSION)  // cuda
  return "cuda " STRINGIFY(CUDART_VERSION);
#else
  return "<unknown>";
#endif  // TENSORFLOW_USE_ROCM && defined(TF_ROCM_VERSION)
}

StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
StreamExecutorGpuClient::CreateBuffersForAsyncHostToDevice(
    absl::Span<const Shape> shapes, PjRtDevice* device) {
  auto* stream_executor_device =
      tensorflow::down_cast<PjRtStreamExecutorDevice*>(device);
  return xla::AsyncHostToDeviceTransferManager::Create(
      shapes, stream_executor_device, this);
}

xla::StatusOr<xla::DeviceAssignment>
StreamExecutorGpuClient::GetDefaultDeviceAssignment(int num_replicas,
                                                    int num_partitions) const {
  if (num_partitions == 1 && num_replicas <= addressable_devices().size()) {
    xla::DeviceAssignment assignment(num_replicas, 1);
    for (int i = 0; i < num_replicas; ++i) {
      assignment(i, 0) = addressable_devices().at(i)->id();
    }
    return assignment;
  }
  // Fallback to default global device assignment if we can't run locally.
  return PjRtStreamExecutorClient::GetDefaultDeviceAssignment(num_replicas,
                                                              num_partitions);
}

PjRtFuture<absl::Status> StreamExecutorGpuClient::CopyRawSubBufferToHost(
    PjRtBuffer* pjrt_buffer, void* dst, int64_t offset, int64_t transfer_size) {
  auto* buffer = tensorflow::down_cast<PjRtStreamExecutorBuffer*>(pjrt_buffer);
  DCHECK(buffer);
  PjRtStreamExecutorDevice* device = buffer->device();
  LocalDeviceState* local_device = device->local_device_state();
  // Always borrow a stream to avoid potential deadlocks enqueueing transfers
  // that might be required in order to compute the inputs for computations
  // that have already been enqueued. Such cycles can occur when there are
  // cross-host data dependencies.
  auto stream = local_device->BorrowStreamFromPool();

  PjRtStreamExecutorBuffer::ScopedHold hold(buffer->GetBufferWithUsageHold());
  if (!hold.ok()) {
    return PjRtFuture<absl::Status>(hold.status());
  }
  auto device_buffer = hold.buffer();
  if (device_buffer->device_memory().size() != 1) {
    return PjRtFuture<absl::Status>(
        InvalidArgument("Copy raw buffer called on tuple"));
  }
  auto& device_memory = device_buffer->device_memory()[0];
  if (offset < 0 || offset > device_memory.size() ||
      device_memory.size() - offset < transfer_size) {
    return PjRtFuture<absl::Status>(
        InvalidArgument("Copy raw buffer called on buffer size %lld with "
                        "invalid offset %lld, transfer size %lld",
                        device_memory.size(), offset, transfer_size));
  }
  WaitForBufferDefinitionEventsOnStream(*device_buffer, stream.get());
  absl::StatusOr<EventPool::Handle> event_or =
      local_device->event_pool().AllocateEvent(stream->parent());
  if (!event_or.ok()) {
    return PjRtFuture<absl::Status>(event_or.status());
  }

  std::unique_ptr<se::DeviceMemoryBase> sub_buffer;
  if (transfer_size < device_memory.size()) {
    sub_buffer = std::make_unique<se::DeviceMemoryBase>(
        reinterpret_cast<char*>(device_memory.opaque()) + offset,
        transfer_size);
  } else {
    sub_buffer = std::make_unique<se::DeviceMemoryBase>(device_memory);
  }

  if (transfer_size != 0) {
    // D2H request holds a non-owned pointer into sub_buffer base address
    // that needs to outlive the transfer until the stream callback is invoked.
    stream->ThenMemcpy(dst, *sub_buffer, transfer_size);
  }

  auto usage_event =
      std::make_shared<BufferSequencingEvent>(this->thread_pool());
  local_device->event_pool().ThenRecordEvent(stream.get(), event_or.value());
  usage_event->SetSequencingEvent(std::move(event_or).value(), stream.get());
  // This usage hold will prevent device_buffer from being deleted before
  // the transfer is complete.
  hold.ConvertUsageHold(stream.get(), std::move(usage_event),
                        /*reference_held=*/false);

  auto promise = PjRtFuture<absl::Status>::CreatePromise();
  local_device->ThenExecuteCallback(
      stream.get(), [promise, free_sub_range = sub_buffer.release(),
                     free_stream = stream.release(), local_device]() mutable {
        auto stream = std::unique_ptr<se::Stream>(free_stream);
        auto sub_range = std::unique_ptr<se::DeviceMemoryBase>(free_sub_range);
        local_device->ReturnStreamToPool(std::move(stream));
        promise.Set(OkStatus());
      });

  return PjRtFuture<Status>(
      std::move(promise),
      /*on_block_start=*/
      []() {
        tsl::profiler::TraceMeProducer traceme(
            "StreamExecutorGpuClient::CopyRawSubBufferToHost");
        VLOG(1) << "StreamExecutorGpuClient::CopyRawSubBufferToHost";
        return PjRtFutureHelpers::ProfilingKeys(
            {/*traceme_context_id =*/traceme.GetContextId()});
      },
      /*on_block_end=*/
      [](PjRtFutureHelpers::ProfilingKeys keys) {
        tsl::profiler::TraceMeConsumer traceme(
            "StreamExecutorGpuClient::CopyRawSubBufferToHost",
            keys.traceme_context_id);
      });
}

StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
StreamExecutorGpuClient::Compile(const XlaComputation& computation,
                                 CompileOptions options) {
  auto executable = PjRtStreamExecutorClient::Compile(computation, options);

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
  metrics::RecordFreeGpuSystemMemory();
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return executable;
}

namespace {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
StatusOr<std::unique_ptr<StreamExecutorExecutable>> FromProto(
    const StreamExecutorExecutableProto& proto) {
  TF_ASSIGN_OR_RETURN(CompileOptions compile_options,
                      CompileOptions::FromProto(proto.compile_options()));
  std::vector<std::unique_ptr<xla::AotCompilationResult>>
      deserialized_aot_executables;
  for (const auto& executable : proto.executables()) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::AotCompilationResult> deserialized,
        gpu::GpuXlaRuntimeAotCompilationResult::FromString(executable));
    deserialized_aot_executables.push_back(std::move(deserialized));
  }
  return std::make_unique<StreamExecutorExecutable>(
      compile_options, std::move(deserialized_aot_executables),
      proto.num_replicas(), proto.num_partitions(), proto.name());
}
#endif
}  // namespace

StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
StreamExecutorGpuClient::LoadSerialized(absl::string_view serialized,
                                        std::optional<CompileOptions> options,
                                        const LoadOptions& load_options) {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  StreamExecutorExecutableProto proto;
  if (serialized.size() > std::numeric_limits<int>::max()) {
    return Internal(
        "PjRtStreamExecutorClient::DeserializeExecutable proto too large "
        "(>2GB)");
  }
  if (!proto.ParseFromArray(serialized.data(), serialized.size())) {
    return Internal(
        "StreamExecutorGpuClient::DeserializeExecutable proto deserialization "
        "failed");
  }
  TF_ASSIGN_OR_RETURN(auto se_executable, FromProto(proto));
  // TODO(b/296466237): Unify the `Load` method.
  return Load(std::move(se_executable));
#endif
  return absl::InternalError("LoadSerialized only works with cuda or rocm.");
}

std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> BuildLocalDevices(
    std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states,
    int node_id) {
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  for (auto& ordinal_and_device : local_device_states) {
    const se::DeviceDescription& description =
        ordinal_and_device.second->executor()->GetDeviceDescription();
    auto device = std::make_unique<StreamExecutorGpuDevice>(
        ordinal_and_device.first, std::move(ordinal_and_device.second),
        description.name(), description.device_vendor(), node_id);
    devices.push_back(std::move(device));
  }
  return devices;
}

StatusOr<std::unique_ptr<PjRtLoadedExecutable>> StreamExecutorGpuClient::Load(
    std::unique_ptr<PjRtExecutable> executable) {
  auto se_executable = absl::WrapUnique(
      tensorflow::down_cast<StreamExecutorExecutable*>(executable.release()));

  CompileOptions compile_options = se_executable->compile_options();
  CompileOptions input_options = compile_options;
  TF_RETURN_IF_ERROR(compile_options.ApplyAllOptionOverrides());
  TF_ASSIGN_OR_RETURN(ExecutableExtras extras,
                      GetExecutableExtras(&compile_options));

  // Load Executable from AOT compilation result.
  std::vector<std::unique_ptr<LocalExecutable>> local_executables;
  local_executables.reserve(se_executable->aot_executables().size());
  for (std::unique_ptr<xla::AotCompilationResult>& aot_executable :
       se_executable->aot_executables()) {
    TF_ASSIGN_OR_RETURN(std::string serialized,
                        aot_executable->SerializeAsString());
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<LocalExecutable> local_executable,
        client()->Load(serialized, compile_options.executable_build_options));
    local_executables.push_back(std::move(local_executable));
  }
  bool parameter_is_tupled_arguments =
      compile_options.parameter_is_tupled_arguments;
  auto ret = std::make_unique<PjRtStreamExecutorLoadedExecutable>(
      std::move(local_executables), parameter_is_tupled_arguments,
      std::move(extras.device_assignment), std::move(input_options),
      std::move(extras.addressable_device_logical_ids),
      std::move(extras.addressable_devices), this);
  TF_RETURN_IF_ERROR(ret->SetUpDonation(parameter_is_tupled_arguments));
  return std::unique_ptr<PjRtLoadedExecutable>(std::move(ret));
}

namespace {

#if defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020

StatusOr<std::unique_ptr<se::MultiDeviceAdapter>> CreateCudaAsyncAllocator(
    se::Platform* platform,
    const std::map<int, std::unique_ptr<LocalDeviceState>>& addressable_devices,
    double memory_fraction, bool preallocate) {
  CHECK_GT(addressable_devices.size(), 0);
  std::vector<se::MultiDeviceAdapter::AllocatorWithStream> allocators;

  for (auto& ordinal_and_device : addressable_devices) {
    se::StreamExecutor* executor = ordinal_and_device.second->executor();
    int device_ordinal = executor->device_ordinal();

    int64_t free_memory;
    int64_t total_memory;
    if (!executor->DeviceMemoryUsage(&free_memory, &total_memory)) {
      return Unavailable("Failed to query available memory from device %i",
                         device_ordinal);
    }
    // To allow full GPU memory to be visible to the Cuda Async allocator
    // if using unified memory.
    // When unified memory is enabled, allow GPU memory oversubscription by
    // setting memory_fraction > 1.
    size_t allocator_memory = total_memory * memory_fraction;
    if (preallocate) {
      LOG(INFO) << "XLA backend allocating " << allocator_memory
                << " bytes on device " << device_ordinal
                << " for CudaAsyncAllocator.";
    } else {
      LOG(INFO) << "XLA backend will use up to " << allocator_memory
                << " bytes on device " << device_ordinal
                << " for CudaAsyncAllocator.";
    }

    auto allocator = std::make_unique<se::GpuCudaMallocAsyncAllocator>(
        tsl::PlatformDeviceId(device_ordinal), allocator_memory, preallocate);
    allocator->SetStreamAndPreallocateMemory(
        ordinal_and_device.second->compute_stream()
            ->platform_specific_handle()
            .stream);
    allocators.emplace_back(std::move(allocator),
                            ordinal_and_device.second->compute_stream());
  }
  return std::make_unique<se::MultiDeviceAdapter>(platform,
                                                  std::move(allocators));
}

#else  // defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020

StatusOr<std::unique_ptr<se::MultiDeviceAdapter>> CreateCudaAsyncAllocator(
    se::Platform* platform,
    const std::map<int, std::unique_ptr<LocalDeviceState>>& addressable_devices,
    double memory_fraction, bool preallocate) {
  return FailedPrecondition("CUDA async allocator requires CUDA >= 11.2");
}

#endif  // defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020

// Builds a LocalDeviceState for each GPU present.
StatusOr<std::map<int, std::unique_ptr<LocalDeviceState>>>
BuildLocalDeviceStates(LocalClient* xla_client) {
  std::map<int, std::unique_ptr<LocalDeviceState>> addressable_devices;
  for (se::StreamExecutor* executor :
       xla_client->backend().stream_executors()) {
    addressable_devices.emplace(
        executor->device_ordinal(),
        std::make_unique<LocalDeviceState>(
            executor, xla_client, LocalDeviceState::kComputeSynchronized,
            /*max_inflight_computations=*/32,
            /*allow_event_reuse=*/true, /*use_callback_stream=*/true));
  }
  return std::move(addressable_devices);
}

// Constructs a GPU device memory allocator to use, according to the allocator
// configuration the client requested.
StatusOr<std::unique_ptr<se::DeviceMemoryAllocator>>
GetStreamExecutorGpuDeviceAllocator(
    se::Platform* platform, const GpuAllocatorConfig& allocator_config,
    const std::map<int, std::unique_ptr<LocalDeviceState>>&
        addressable_devices) {
  std::unique_ptr<se::DeviceMemoryAllocator> allocator;
  switch (allocator_config.kind) {
    case GpuAllocatorConfig::Kind::kCudaAsync: {
      auto allocator_or = CreateCudaAsyncAllocator(
          platform, addressable_devices, allocator_config.memory_fraction,
          allocator_config.preallocate);
      if (allocator_or.ok()) {
        LOG(INFO) << "Using CUDA async allocator.";
        allocator = std::move(allocator_or.value());
        break;
      }
      LOG(ERROR) << "Failed to initialize CUDA async allocator: "
                 << allocator_or.status() << "; falling back to BFC.";
      [[fallthrough]];
    }

    case GpuAllocatorConfig::Kind::kDefault:
    case GpuAllocatorConfig::Kind::kBFC: {
      LOG(INFO) << "Using BFC allocator.";
      std::vector<se::StreamExecutor*> executors;
      executors.reserve(addressable_devices.size());
      std::vector<se::MultiDeviceAdapter::AllocatorWithStream>
          allocators_and_streams;
      for (const auto& ordinal_and_device : addressable_devices) {
        TF_ASSIGN_OR_RETURN(
            auto bfc_allocator,
            CreateBFCAllocator(ordinal_and_device.second->executor(),
                               allocator_config.memory_fraction,
                               allocator_config.preallocate));
        allocators_and_streams.emplace_back(
            std::move(bfc_allocator),
            ordinal_and_device.second->compute_stream());
      }
      allocator = std::make_unique<se::MultiDeviceAdapter>(
          platform, std::move(allocators_and_streams));
      break;
    }

    case GpuAllocatorConfig::Kind::kPlatform:
      LOG(INFO) << "Using platform allocator.";
      break;
  }
  return std::move(allocator);
}

// Exists on Linux systems. Unique per OS kernel restart.
static constexpr char kBootIdPath[] = "/proc/sys/kernel/random/boot_id";

// Retrieve content of /proc/sys/kernel/random/boot_id as a string.
// Note that procfs file may have file size 0 which throws off generic file
// readers such as tsl::ReadFileToString.
StatusOr<std::string> GetBootIdString() {
  std::string boot_id_str;
#ifdef __linux__
  std::ifstream file(kBootIdPath);
  if (!file) {
    return NotFound("%s not found.", kBootIdPath);
  }
  std::string line;
  while (std::getline(file, line)) {
    absl::StripAsciiWhitespace(&line);
    absl::StrAppend(&boot_id_str, line);
  }
#endif
  return boot_id_str;
}

static std::string GetLocalTopologyKey(int node_id) {
  return absl::StrCat("local_topology:", node_id);
}

static std::string GetGlobalTopologyKey() { return "global_topology"; }

static StatusOr<std::vector<LocalTopologyProto>> GetAllLocalTopologies(
    int num_nodes, const PjRtClient::KeyValueGetCallback& kv_get,
    absl::Duration timeout) {
  std::vector<StatusOr<std::string>> local_topology_strs(num_nodes);

  // TODO(ezhulenev): Should a thread pool become a function argument?
  tsl::thread::ThreadPool thread_pool(
      tsl::Env::Default(), "GetAllLocalTopologies", DefaultThreadPoolSize());

  absl::BlockingCounter blocking_counter(num_nodes);
  absl::Mutex mu;
  for (int i = 0; i < num_nodes; i++) {
    thread_pool.Schedule([&, i] {
      StatusOr<std::string> local_topology_str =
          kv_get(GetLocalTopologyKey(i), timeout);
      {
        absl::MutexLock lock(&mu);
        local_topology_strs[i] = local_topology_str;
      }
      blocking_counter.DecrementCount();
    });
  }
  blocking_counter.Wait();

  std::vector<std::string> error_messages;
  std::vector<LocalTopologyProto> local_topologies;
  int max_num_failed_message = 10;
  int failed_count = 0;
  for (const StatusOr<std::string>& str : local_topology_strs) {
    if (str.ok()) {
      LocalTopologyProto local;
      local.ParseFromString(*str);
      local_topologies.push_back(local);
    } else {
      error_messages.push_back(
          absl::StrCat("Error ", ++failed_count, ": ", str.status().message()));
      if (failed_count > max_num_failed_message) {
        break;
      }
    }
  }
  if (error_messages.empty()) {
    return local_topologies;
  }
  return absl::InternalError(
      absl::StrCat("Getting local topologies failed: ",
                   absl::StrJoin(error_messages, "\n\n")));
}

Status BuildDistributedDevices(
    std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states,
    int node_id, int num_nodes,
    std::vector<std::unique_ptr<PjRtStreamExecutorDevice>>* devices,
    gpu::GpuExecutableRunOptions* gpu_executable_run_options,
    PjRtClient::KeyValueGetCallback kv_get,
    PjRtClient::KeyValuePutCallback kv_put,
    absl::Duration get_local_topology_timeout = absl::Minutes(2),
    absl::Duration get_global_topology_timeout = absl::Minutes(5)) {
  LocalTopologyProto local_topology;
  local_topology.set_node_id(node_id);
  std::string boot_id_str;
  auto boot_id_str_or_status = GetBootIdString();
  if (!boot_id_str_or_status.ok()) {
    LOG(INFO) << boot_id_str_or_status.status();
  } else {
    boot_id_str = boot_id_str_or_status.value();
  }
  local_topology.set_boot_id(boot_id_str);
  for (const auto& ordinal_and_device : local_device_states) {
    const se::Platform* platform =
        ordinal_and_device.second->executor()->platform();
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::se::DeviceDescription> desc,
        platform->DescriptionForDevice(ordinal_and_device.first));
    DeviceProto* device_proto = local_topology.add_devices();
    device_proto->set_local_device_ordinal(ordinal_and_device.first);
    device_proto->set_name(desc->name());
    device_proto->set_vendor(desc->device_vendor());
  }
  VLOG(3) << "GPU Local Topology:\n" << local_topology.DebugString();
  TF_RETURN_IF_ERROR(
      kv_put(GetLocalTopologyKey(node_id), local_topology.SerializeAsString()));

  GlobalTopologyProto global_topology;
  // The lead node gets all local topologies, builds the global topology and
  // puts it to the key-value store.
  if (node_id == 0) {
    TF_ASSIGN_OR_RETURN(
        std::vector<LocalTopologyProto> local_topologies,
        GetAllLocalTopologies(num_nodes, kv_get, get_local_topology_timeout));
    global_topology =
        BuildGlobalTopology(absl::Span<LocalTopologyProto>(local_topologies));
    TF_RETURN_IF_ERROR(
        kv_put(GetGlobalTopologyKey(), global_topology.SerializeAsString()));
  } else {
    TF_ASSIGN_OR_RETURN(
        std::string global_topology_str,
        kv_get(GetGlobalTopologyKey(), get_global_topology_timeout));
    global_topology.ParseFromString(global_topology_str);
  }
  VLOG(3) << "GPU Global Topology:\n" << global_topology.DebugString();

  std::map<int, GlobalDeviceId> gpu_device_ids;
  absl::flat_hash_map<GlobalDeviceId, int> device_to_node;
  for (const LocalTopologyProto& node : global_topology.nodes()) {
    for (const DeviceProto& device_proto : node.devices()) {
      GlobalDeviceId global_device_id(device_proto.global_device_id());
      device_to_node[global_device_id] = node.node_id();
      std::unique_ptr<LocalDeviceState> local_device;
      if (node.node_id() == node_id) {
        auto it = local_device_states.find(device_proto.local_device_ordinal());
        TF_RET_CHECK(it != local_device_states.end())
            << device_proto.local_device_ordinal();
        TF_RET_CHECK(it->second != nullptr);
        local_device = std::move(it->second);
        gpu_device_ids[device_proto.local_device_ordinal()] = global_device_id;
      }
      auto device = std::make_unique<StreamExecutorGpuDevice>(
          device_proto.global_device_id(), std::move(local_device),
          device_proto.name(), device_proto.vendor(), node.node_id(),
          device_proto.slice_index());
      devices->push_back(std::move(device));
    }
  }
  for (const auto& device : local_device_states) {
    TF_RET_CHECK(device.second == nullptr);
  }
  gpu_executable_run_options->set_gpu_global_device_ids(
      std::move(gpu_device_ids));
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  auto nccl_id_store =
      std::make_shared<NcclIdStore>(node_id, device_to_node, kv_get, kv_put);
  gpu_executable_run_options->set_nccl_unique_id_callback(
      [nccl_id_store](const gpu::NcclCliqueKey& key) {
        return nccl_id_store->GetNcclUniqueId(key);
      });
#endif  // GOOGLE_CUDA
  return OkStatus();
}

}  // namespace

StreamExecutorGpuDevice::StreamExecutorGpuDevice(
    int id, std::unique_ptr<LocalDeviceState> local_device_state,
    std::string device_kind, std::string device_vendor, int node_id,
    int slice_index)
    : PjRtStreamExecutorDevice(id, std::move(local_device_state),
                               std::move(device_kind), node_id),
      device_vendor_(std::move(device_vendor)),
      slice_index_(slice_index) {
  description().SetAttributes({
      {"device_vendor", device_vendor_},
      {"slice_index", static_cast<int64_t>(slice_index)},
  });
  description().SetToString(absl::StrFormat(
      "StreamExecutorGpuDevice(id=%i, process_index=%i, slice_index=%i)", id,
      process_index(), slice_index));
}

int StreamExecutorGpuDevice::slice_index() const { return slice_index_; }

absl::string_view StreamExecutorGpuDevice::device_vendor() const {
  return device_vendor_;
}

absl::StatusOr<tsl::AllocatorStats> StreamExecutorGpuDevice::GetAllocatorStats()
    const {
  if (!IsAddressable()) {
    return FailedPrecondition(
        "GetAllocatorStats() is allowed only for addressable devices");
  }

  TF_ASSIGN_OR_RETURN(
      auto allocator,
      tensorflow::down_cast<se::MultiDeviceAdapter*>(
          tensorflow::down_cast<PjRtStreamExecutorClient*>(client())
              ->allocator())
          ->GetAllocator(local_hardware_id()));

  auto stats = allocator->GetStats();
  TF_RET_CHECK(stats.has_value());
  return stats.value();
}

StatusOr<std::unique_ptr<PjRtClient>> GetStreamExecutorGpuClient(
    bool asynchronous, const GpuAllocatorConfig& allocator_config, int node_id,
    int num_nodes, const std::optional<std::set<int>>& allowed_devices,
    std::optional<std::string> platform_name,
    bool should_stage_host_to_device_transfers,
    PjRtClient::KeyValueGetCallback kv_get,
    PjRtClient::KeyValuePutCallback kv_put, bool enable_mock_nccl) {
  TF_ASSIGN_OR_RETURN(LocalClient * xla_client,
                      GetGpuXlaClient(platform_name, allowed_devices));
  std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states;
  TF_ASSIGN_OR_RETURN(local_device_states, BuildLocalDeviceStates(xla_client));
  EnablePeerAccess(xla_client->backend().stream_executors());
  TF_ASSIGN_OR_RETURN(
      auto allocator,
      GetStreamExecutorGpuDeviceAllocator(
          xla_client->platform(), allocator_config, local_device_states));
  auto host_memory_allocator =
      GetGpuHostAllocator(local_device_states.begin()->second->executor());

  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  auto gpu_run_options = std::make_unique<gpu::GpuExecutableRunOptions>();
  if (enable_mock_nccl) {
    gpu_run_options->set_enable_mock_nccl_collectives();
  }
  if (num_nodes > 1) {
    absl::flat_hash_map<std::string, std::string> device_maps;
    absl::Mutex mu;
    if (enable_mock_nccl) {
      kv_get = [&device_maps, &mu, &num_nodes](
                   const std::string& k,
                   absl::Duration timeout) -> xla::StatusOr<std::string> {
        std::string result;
        {
          absl::MutexLock lock(&mu);
          if (device_maps.contains(k)) {
            result = device_maps[k];
          } else {
            int device_id;
            std::vector<std::string> tokens = absl::StrSplit(k, ':');
            if (tokens.size() != 2 ||
                !absl::SimpleAtoi(tokens[1], &device_id)) {
              device_id = num_nodes - 1;
            }
            // Return fake local topology with device_id info back.
            xla::LocalTopologyProto local;
            local.set_boot_id("fake_boot_id");
            local.set_node_id(device_id);
            xla::DeviceProto* device = local.add_devices();
            device->set_global_device_id(device_id);
            device->set_name("fake_device");
            device->set_vendor("fake_vendor");
            result = local.SerializeAsString();
          }
        }
        return result;
      };
      kv_put = [&device_maps, &mu](const std::string& k,
                                   const std::string& v) -> xla::Status {
        {
          absl::MutexLock lock(&mu);
          device_maps[k] = v;
        }
        return xla::OkStatus();
      };
    }
    TF_RET_CHECK(kv_get != nullptr);
    TF_RET_CHECK(kv_put != nullptr);
    TF_RETURN_IF_ERROR(BuildDistributedDevices(
        std::move(local_device_states), node_id, num_nodes, &devices,
        gpu_run_options.get(), kv_get, kv_put));
  } else {
    devices = BuildLocalDevices(std::move(local_device_states), node_id);
  }

#if TENSORFLOW_USE_ROCM
  auto pjrt_platform_name = xla::RocmName();
#else   // TENSORFLOW_USE_ROCM
  auto pjrt_platform_name = xla::CudaName();
#endif  // TENSORFLOW_USE_ROCM

  return std::unique_ptr<PjRtClient>(std::make_unique<StreamExecutorGpuClient>(
      pjrt_platform_name, xla_client, std::move(devices),
      /*node_id=*/node_id, std::move(allocator),
      std::move(host_memory_allocator), should_stage_host_to_device_transfers,
      /*gpu_run_options=*/std::move(gpu_run_options)));
}

absl::StatusOr<std::string> StreamExecutorGpuTopologyDescription::Serialize()
    const {
  std::string result;
  if (!tsl::SerializeToStringDeterministic(gpu_topology_.ToProto(), &result)) {
    return absl::InternalError("Failed to serialize gpu_topology");
  }
  return result;
}

}  // namespace xla
