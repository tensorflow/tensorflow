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

#include "xla/pjrt/gpu/tfrt/tfrt_gpu_client.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "mlir/IR/BuiltinOps.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/gpu/gpu_topology.h"
#include "xla/pjrt/gpu/gpu_topology.pb.h"
#include "xla/pjrt/gpu/tfrt/gpu_event.h"
#include "xla/pjrt/gpu/tfrt/host_memory_allocator.h"
#include "xla/pjrt/gpu/tfrt/stream_pool.h"
#include "xla/pjrt/gpu/tfrt/tracked_tfrt_gpu_device_buffer.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/semaphore.h"
#include "xla/pjrt/transpose.h"
#include "xla/pjrt/utils.h"
#include "xla/pjrt/worker_thread.h"
#include "xla/primitive_util.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/service/generic_transfer_manager.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/context_types.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace {

void EnqueueWork(tsl::thread::ThreadPool* pool,
                 absl::AnyInvocable<void()> callee) {
  // TSL TheadPool expects std::function that must be copyable, so we are
  // forced to do a little bit of manual memory management here.
  pool->Schedule([ptr = new absl::AnyInvocable<void()>(std::move(callee))]() {
    (*ptr)();
    delete ptr;
  });
}

// Enqueue to a thread pool when all `values` are ready.
void EnqueueWorkWhenReady(
    tsl::thread::ThreadPool* pool,
    absl::Span<const tsl::RCReference<tsl::AsyncValue>> values,
    absl::AnyInvocable<void()> callee) {
  tsl::RunWhenReady(values, [pool, callee = std::move(callee)]() mutable {
    VLOG(2) << "EnqueueWork: pool: " << pool;
    EnqueueWork(pool, std::move(callee));
  });
}

std::string get_platform_version(xla::LocalClient* xla_client) {
  const stream_executor::DeviceDescription& device =
      xla_client->backend().default_stream_executor()->GetDeviceDescription();
  if (std::holds_alternative<stream_executor::RocmComputeCapability>(
          device.gpu_compute_capability())) {
    return absl::StrCat("rocm ", device.runtime_version());
  }
  if (std::holds_alternative<stream_executor::CudaComputeCapability>(
          device.gpu_compute_capability())) {
    return absl::StrCat("cuda ", device.runtime_version());
  }
  return "<unknown>";
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
    absl::Span<TrackedTfrtGpuDeviceBuffer* const> input_buffers) {
  if (input_buffers.size() != input_buffer_sizes_in_bytes.size()) {
    return InvalidArgument(
        "Execution supplied %lld buffers but compiled program expected %lld "
        "buffers",
        input_buffers.size(), input_buffer_sizes_in_bytes.size());
  }
  for (int i = 0; i < input_buffers.size(); ++i) {
    const auto& buffer = input_buffers[i];
    if (input_buffer_sizes_in_bytes[i] != buffer->buffer()->size()) {
      return InvalidArgument(
          "Executable expected parameter %d of size %lld but got buffer with"
          " incompatible size %lld ",
          i, input_buffer_sizes_in_bytes[i], buffer->buffer()->size());
    }
  }
  return absl::OkStatus();
}

class TfrtGpuAsyncHostToDeviceTransferManager
    : public PjRtClient::AsyncHostToDeviceTransferManager {
 public:
  static absl::StatusOr<
      std::unique_ptr<TfrtGpuAsyncHostToDeviceTransferManager>>
  Create(absl::Span<const PjRtClient::ShapeSpec> shape_specs,
         std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
         TfrtGpuDevice* device, TfrtGpuClient* client,
         PjRtMemorySpace* memory_space) {
    if (device_layouts.has_value() &&
        device_layouts->size() != shape_specs.size()) {
      return InvalidArgument(
          "Number of layouts %d does not match the number of shapes %d",
          device_layouts->size(), shape_specs.size());
    }
    absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4> buffers;
    absl::InlinedVector<tsl::AsyncValueRef<MaybeOwningGpuMemory>, 4>
        buffer_ptrs;
    absl::InlinedVector<absl::InlinedVector<tsl::AsyncValueRef<GpuEvent>, 4>, 4>
        definition_events;
    absl::InlinedVector<Shape, 4> device_shapes;
    buffers.reserve(shape_specs.size());
    definition_events.reserve(shape_specs.size());
    device_shapes.reserve(shape_specs.size());
    for (int i = 0; i < shape_specs.size(); ++i) {
      const PjRtClient::ShapeSpec& shape_spec = shape_specs[i];
      if (shape_spec.element_type == TUPLE) {
        return Unimplemented(
            "Async buffer transfer of tuples not implemented.");
      }
      // Initialize a definition event for each async buffer. The definition
      // event will block the buffer usage until the transfer is done.
      tsl::AsyncValueRef<GpuEvent> copy_event =
          tsl::MakeConstructedAsyncValueRef<GpuEvent>();
      // Since transfer of tuples are not supported, we can use a single event
      // for each buffer.
      definition_events.push_back({copy_event.CopyRef()});
      Shape& device_shape = device_shapes.emplace_back(
          ShapeUtil::MakeShape(shape_spec.element_type, shape_spec.dims));
      if (device_layouts.has_value() && (*device_layouts)[i].has_value()) {
        *device_shape.mutable_layout() = *(*device_layouts)[i];
      } else {
        TF_ASSIGN_OR_RETURN(device_shape,
                            client->xla_client()
                                ->backend()
                                .transfer_manager()
                                ->ChooseCompactLayoutForShape(device_shape));
      }
      int64_t byte_size = ShapeUtil::ByteSizeOf(device_shape);

      buffer_ptrs.push_back(
          tsl::MakeUnconstructedAsyncValueRef<MaybeOwningGpuMemory>());
      auto buffer_allocated =
          MaybeOwningGpuMemory::AllocateShared(device->allocator(), byte_size);
      if (!buffer_allocated.ok()) {
        copy_event.SetError(buffer_allocated.status());
        return absl::InternalError("Failed to allocate buffer.");
      } else {
        buffer_ptrs.back().emplace(std::move(buffer_allocated.value()));
      }
      auto tracked_device_buffer = std::make_unique<TrackedTfrtGpuDeviceBuffer>(
          buffer_ptrs.back(), definition_events.back());
      buffers.push_back(std::make_unique<TfrtGpuBuffer>(
          device_shape, std::move(tracked_device_buffer), client, device,
          memory_space));
    }

    return std::make_unique<TfrtGpuAsyncHostToDeviceTransferManager>(
        std::move(buffers), std::move(buffer_ptrs),
        std::move(definition_events), std::move(device_shapes), device);
  }

  TfrtGpuAsyncHostToDeviceTransferManager(
      absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4> buffers,
      absl::InlinedVector<tsl::AsyncValueRef<MaybeOwningGpuMemory>, 4>
          buffer_ptrs,
      absl::InlinedVector<absl::InlinedVector<tsl::AsyncValueRef<GpuEvent>, 4>,
                          4>
          definition_events,
      absl::InlinedVector<Shape, 4> device_shapes, TfrtGpuDevice* device)
      : buffers_(std::move(buffers)),
        h2d_thread_(std::make_unique<WorkerThread>(
            tsl::Env::Default(),
            "TfrtGpuAsyncHostToDeviceTransferManager_h2d_thread")),
        buffer_ptrs_(std::move(buffer_ptrs)),
        definition_events_(std::move(definition_events)),
        device_shapes_(std::move(device_shapes)),
        remaining_buffer_count_(buffers_.size()),
        transfers_in_flight_(0),
        device_(device),
        client_(tsl::down_cast<TfrtGpuClient*>(device_->client())) {
    VLOG(2) << "TfrtGpuAsyncHostToDeviceTransferManager::"
               "TfrtGpuAsyncHostToDeviceTransferManager: this="
            << this << " buffers_.size()=" << buffers_.size();

    buffer_sizes_.reserve(buffers_.size());
    for (const auto& buffer : buffers_) {
      buffer_sizes_.push_back(buffer->GetOnDeviceSizeInBytes().value());
    }
    last_transfer_started_.resize(buffer_ptrs_.size(), false);
  }

  ~TfrtGpuAsyncHostToDeviceTransferManager() override {
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

  absl::Status TransferLiteralToBuffer(
      int buffer_index, const LiteralSlice& literal,
      absl::AnyInvocable<void() &&> on_done) override {
    tsl::profiler::TraceMe traceme(
        "AsyncHostToDeviceTransferManager::TransferLiteralToBuffer");
    VLOG(2) << "TfrtGpuAsyncHostToDeviceTransferManager::"
               "TransferLiteralToBuffer: this="
            << this << " buffer_index=" << buffer_index;
    auto* client = tsl::down_cast<TfrtGpuClient*>(device_->client());
    DCHECK(client);

    TransferManager* transfer_manager =
        client->xla_client()->backend().transfer_manager();

    tsl::AsyncValueRef<MaybeOwningGpuMemory> buffer;
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

      ++transfers_in_flight_;
    }

    // The host to device transfer is performed on a thread pool, mostly
    // because it includes linearization that may be slow.
    // TODO(misard) assess if it would be preferable to introduce a heuristic
    // to put the transfer into the calling thread for small literals.
    auto transfer_h2d = [this, buffer_index, transfer_manager, literal, buffer,
                         on_done = std::move(on_done)]() mutable {
      tsl::profiler::TraceMe traceme(
          "TfrtGpuAsyncHostToDeviceTransferManager::TransferLiteralToBuffer::"
          "transfer_h2d");

      // Initiate linearization and transfer of the buffer on the stream.
      ShapedBuffer shaped_buffer =
          buffer->AsShapedBuffer(device_shapes_[buffer_index], device_);

      absl::StatusOr<BoundedStreamPool::Handle> handle_or =
          device_->stream_pool().Borrow();
      CHECK_OK(handle_or.status());
      BoundedStreamPool::Handle stream = std::move(handle_or.value());
      CHECK_NE(stream.get(), nullptr);

      GenericTransferManager::LiteralFromDeviceMetadata transfer_metadata;
      // We never call device functions from the `done` callback.
      transfer_metadata.callback_is_host_callback_safe = true;
      TransferManager::TransferMetadata* transfer_metadata_ptr =
          (dynamic_cast<GenericTransferManager*>(transfer_manager) != nullptr)
              ? &transfer_metadata
              : nullptr;

      TF_CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(
          stream.get(), literal, shaped_buffer, transfer_metadata_ptr));

      auto status = stream->BlockHostUntilDone();
      CHECK_OK(status) << "Failed to block host until done";

      CleanUp(buffer_index, /*is_last_transfer=*/true, std::move(on_done));
    };
    // Enqueue the transfer to the h2d thread.
    h2d_thread_->Schedule(std::move(transfer_h2d));
    return absl::OkStatus();
  }

  absl::Status TransferRawDataToBuffer(
      int buffer_index, absl::string_view data,
      absl::AnyInvocable<void() &&> on_done) override {
    return TransferRawDataToSubBuffer(buffer_index, data.data(),
                                      /*offset=*/0, data.size(),
                                      /*is_last_transfer=*/true,
                                      std::move(on_done));
  }

  absl::Status TransferRawDataToSubBuffer(
      int buffer_index, const void* data, int64_t offset, int64_t transfer_size,
      bool is_last_transfer, absl::AnyInvocable<void() &&> on_done) override {
    VLOG(2) << "TfrtGpuAsyncHostToDeviceTransferManager::"
               "TransferRawDataToSubBuffer: this="
            << this << " buffer_index=" << buffer_index << " offset=" << offset
            << " transfer_size=" << transfer_size
            << " is_last_transfer=" << is_last_transfer;

    auto* client = tsl::down_cast<TfrtGpuClient*>(device_->client());
    DCHECK(client);
    auto* host_memory_allocator = client->host_memory_allocator();
    if (host_memory_allocator == nullptr) {
      return InvalidArgument(
          "host_memory_allocator should be initialized for staging buffer "
          "transfer.");
    }

    std::unique_ptr<void, std::function<void(void*)>> staging_buffer =
        host_memory_allocator->Allocate(transfer_size);

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
    auto& buffer_memory = buffer_ptrs_[buffer_index];
    se::DeviceMemoryBase sub_buffer;
    CHECK_LE(offset, buffer_memory->size());
    CHECK_LE(transfer_size, buffer_memory->size() - offset);
    if (transfer_size < buffer_memory->size()) {
      sub_buffer = buffer_memory->buffer().GetByteSlice(offset, transfer_size);
    } else {
      sub_buffer = buffer_memory->buffer();
    }

    ++transfers_in_flight_;
    // Release the lock before transfer in case transfer or cleanup could be
    // called on this thread, to avoid deadlock.
    l.Release();

    auto copy_to_gpu =
        [transfer_size, staging_buffer = std::move(staging_buffer), data,
         sub_buffer = std::move(sub_buffer), buffer_index, is_last_transfer,
         on_done = std::move(on_done), this]() mutable {
          if (transfer_size != 0) {
            std::memcpy(staging_buffer.get(), data, transfer_size);

            absl::StatusOr<BoundedStreamPool::Handle> handle_or =
                device_->stream_pool().Borrow();
            CHECK_OK(handle_or.status())
                << "Failed to borrow a stream from the pool";
            BoundedStreamPool::Handle stream = std::move(handle_or.value());
            CHECK_NE(stream.get(), nullptr);

            if (auto status = stream->Memcpy(&sub_buffer, staging_buffer.get(),
                                             transfer_size);
                !status.ok()) {
              CHECK_OK(status) << "Failed to copy data to GPU";
            }
            auto status = stream->BlockHostUntilDone();
            CHECK_OK(status) << "Failed to block host until done";
          }
          CleanUp(buffer_index, is_last_transfer, std::move(on_done));
        };
    // Enqueue the transfer to the h2d thread.
    h2d_thread_->Schedule(std::move(copy_to_gpu));
    return absl::OkStatus();
  }

  void SetBufferError(int buffer_index, absl::Status error) override {
    {
      absl::MutexLock l(&mu_);
      // For a given buffer_index, SetBufferError can't be called twice, or
      // called after the last transfer has been enqueued.
      CHECK(!definition_events_[buffer_index].back().IsConcrete());
      definition_events_[buffer_index].back().SetError(error);
    }
    VLOG(1) << "SetBufferError sets the " << buffer_index
            << "th buffer error: " << error;
  }

  void AddTransferMetadata(const TransferMetadata& meta) override {}

 private:
  void CleanUp(int buffer_index, bool is_last_transfer,
               absl::AnyInvocable<void() &&> on_done) {
    {
      absl::MutexLock l(&mu_);

      CHECK_GT(transfers_in_flight_, 0);
      --transfers_in_flight_;
      VLOG(2) << "CleanUp for buffer_index=" << buffer_index
              << " is_last_transfer=" << is_last_transfer
              << " remaining_buffer_count_=" << remaining_buffer_count_
              << "; this: " << this;

      if (is_last_transfer) {
        // Drop our reference to the TrackedDeviceBuffer for this buffer.
        CHECK(buffer_ptrs_[buffer_index]);
        buffer_ptrs_[buffer_index] = nullptr;
        CHECK_GT(remaining_buffer_count_, 0);
        --remaining_buffer_count_;
        definition_events_[buffer_index].back().SetStateConcrete();
        if (remaining_buffer_count_ == 0) {
          VLOG(2) << "TransferLiteralToBuffer for all buffers is done. this: "
                  << this;
        }
      }
    }

    // Call on_done after finishing all housekeeping and releasing the lock.
    std::move(on_done)();
  }

  absl::Mutex mu_;
  // The newly created buffers, which will be returned to the caller via
  // Retrieve.
  absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4> buffers_;

  // Just a single thread, to ensure transfers are ordered.
  std::unique_ptr<WorkerThread> h2d_thread_;

  absl::InlinedVector<tsl::AsyncValueRef<MaybeOwningGpuMemory>, 4> buffer_ptrs_;
  // Cached versions of the sizes of all the buffers, so we can return them
  // without acquiring mu_.
  absl::InlinedVector<size_t, 4> buffer_sizes_;
  // True if the last transfer for a buffer has been initiated. Used to
  // prevent a client initiating another transfer after the last transfer has
  // already been initiated.
  absl::InlinedVector<bool, 4> last_transfer_started_ ABSL_GUARDED_BY(mu_);
  // The buffer definition events on all the buffers, unblocked once the
  // corresponding buffer transfer has completed.
  absl::InlinedVector<absl::InlinedVector<tsl::AsyncValueRef<GpuEvent>, 4>, 4>
      definition_events_ ABSL_GUARDED_BY(mu_);
  // Device shapes for all buffers with either compact or custom layout.
  const absl::InlinedVector<Shape, 4> device_shapes_;
  // Count of buffers that have not yet been fully transferred.
  size_t remaining_buffer_count_ ABSL_GUARDED_BY(mu_);
  // Count of transfers that have been started but have not yet called
  // cleanup. Used to block in the destructor to avoid dangling pointers in
  // cleanup.
  int transfers_in_flight_ ABSL_GUARDED_BY(mu_);

  TfrtGpuDevice* device_;  // not owned.
  TfrtGpuClient* client_;  // not owned.
};

}  // namespace

TfrtGpuMemorySpace::TfrtGpuMemorySpace(int id, PjRtDevice* device,
                                       absl::string_view kind, int kind_id)
    : id_(id), device_(device), kind_(kind), kind_id_(kind_id) {
  DCHECK(device_ != nullptr && device_->client() != nullptr);
  to_string_ = absl::StrCat("MEMORY_SPACE_", id_);
  debug_string_ = absl::StrFormat("TfrtGpuMemory(id=%i, device=%s)", id_,
                                  device_->DebugString());
}

const int TfrtGpuDeviceMemorySpace::kKindId = []() {
  uint32_t kind_id = tsl::Fingerprint32(TfrtGpuDeviceMemorySpace::kKind);
  return static_cast<int>(kind_id);
}();

TfrtGpuDeviceMemorySpace::TfrtGpuDeviceMemorySpace(int id, PjRtDevice* device)
    : TfrtGpuMemorySpace(id, device, kKind, kKindId) {}

TfrtGpuDevice::TfrtGpuDevice(Options&& options)
    : id_(options.id),
      local_device_id_(options.local_device_id),
      local_hardware_id_(options.local_hardware_id),
      executor_(options.executor),
      stream_pool_(options.executor, options.stream_capacity),
      compute_stream_pool_(options.executor, options.max_inflight_computations),
      allocator_(std::move(options.allocator)),
      prng_seed_generator_(prng_seed_device_()),
      prng_seed_distribution_(std::numeric_limits<int>::min(),
                              std::numeric_limits<int>::max()),
      last_collective_launch_event_(
          tsl::MakeAvailableAsyncValueRef<GpuEvent>()),
      description_(options.id, options.platform_version),
      max_inflight_computations_semaphore_(
          /*capacity=*/options.max_inflight_computations) {
  description_.SetDebugString(absl::StrCat("TFRT_GPU_", id_));
  description_.SetToString(absl::StrCat("GpuDevice(id=", id_, ")"));

  CHECK_OK(executor_->CreateStream().status()) << "Failed to create stream";

  se_allocator_ = std::make_unique<se::TfAllocatorAdapter>(
      allocator_.get(), const_cast<se::Platform*>(executor_->GetPlatform()));
}

void TfrtGpuDevice::SetClient(PjRtClient* client) {
  CHECK(client_ == nullptr);
  client_ = client;

  // We have to define debug_string_ and to_string_ here, because
  // platform_name() requires client_ to be set.
  std::string device_name =
      absl::StrCat(MakeAsciiTitlecase(client_->platform_name()), "Device");
  description_.SetDebugString(
      absl::StrCat(client_->platform_name(), ":", id()));
  description_.SetToString(absl::StrCat(device_name, "(id=", id(), ")"));
}

absl::Status TfrtGpuDevice::TransferToInfeed(const LiteralSlice& literal) {
  return Unimplemented("TransferToInfeed");
}

absl::Status TfrtGpuDevice::TransferFromOutfeed(
    MutableBorrowingLiteral literal) {
  return Unimplemented("TransferFromOutfeed");
}

int TfrtGpuDevice::GetNewPrngSeed() {
  absl::MutexLock lock(&mu_);
  int x = 0;
  do {
    x = prng_seed_distribution_(prng_seed_generator_);
  } while (x == 0);
  return x;
}

void TfrtGpuDevice::AttachMemorySpace(PjRtMemorySpace* memory_space,
                                      bool is_default) {
  CHECK(memory_space != nullptr);
  CHECK(client_ == memory_space->client()) << absl::StrFormat(
      "Could not attach a TfrtGpuExecutable to a PjRtMemorySpace owned "
      "by a different client, the device's client: %s, the memory space's "
      "client: %s.",
      client_->platform_name(), memory_space->client()->platform_name());

  memory_spaces_.push_back(memory_space);
  memory_spaces_by_kind_id_.emplace(memory_space->kind_id(), memory_space);
  if (is_default) {
    CHECK(default_memory_space_ == nullptr)
        << "Default memory space already set to "
        << default_memory_space_->DebugString() << ".";
    default_memory_space_ = memory_space;
  }
}

absl::Span<PjRtMemorySpace* const> TfrtGpuDevice::memory_spaces() const {
  return memory_spaces_;
}

absl::StatusOr<PjRtMemorySpace*> TfrtGpuDevice::memory_space_by_kind_id(
    int id) const {
  auto it = memory_spaces_by_kind_id_.find(id);
  if (it == memory_spaces_by_kind_id_.end()) {
    return absl::InternalError(
        absl::StrCat("No memory space found (kind_id: ", id, ")"));
  }
  return it->second;
}

absl::StatusOr<PjRtMemorySpace*> TfrtGpuDevice::memory_space_by_kind(
    absl::string_view kind) const {
  auto it = absl::c_find_if(memory_spaces_, [kind](PjRtMemorySpace* ms) {
    return ms->kind() == kind;
  });
  if (it != memory_spaces_.end()) {
    return *it;
  }
  return absl::InternalError(
      absl::StrCat("No memory space found (kind: ", kind, ")"));
}

absl::StatusOr<PjRtMemorySpace*> TfrtGpuDevice::default_memory_space() const {
  if (default_memory_space_ == nullptr) {
    return absl::InternalError(
        "No default memory space is set for this device.");
  }
  return default_memory_space_;
}

void TfrtGpuDevice::SetLastCollectiveLaunchEvent(
    tsl::AsyncValueRef<GpuEvent> event) {
  absl::MutexLock lock(&mu_);
  VLOG(2) << "SetLastCollectiveLaunchEvent: IsAvailable: "
          << event.IsAvailable() << "; pointer: " << event.GetAsyncValue();
  last_collective_launch_event_ = std::move(event);
}

tsl::AsyncValueRef<GpuEvent> TfrtGpuDevice::GetLastCollectiveLaunchEvent() {
  absl::MutexLock lock(&mu_);
  VLOG(2) << "GetLastCollectiveLaunchEvent: IsAvailable: "
          << last_collective_launch_event_.IsAvailable()
          << "; pointer: " << last_collective_launch_event_.GetAsyncValue();
  return last_collective_launch_event_.CopyRef();
}

TfrtGpuClient::TfrtGpuClient(
    int process_index, xla::LocalClient* xla_client,
    std::vector<std::unique_ptr<TfrtGpuDevice>> devices,
    bool should_stage_host_to_device_transfers,
    std::unique_ptr<tsl::Allocator> host_memory_allocator,
    std::shared_ptr<const GpuTopology> gpu_topology)
    : process_index_(process_index),
      xla_client_(CHECK_NOTNULL(xla_client)),
      platform_version_(get_platform_version(xla_client)),
      should_stage_host_to_device_transfers_(
          should_stage_host_to_device_transfers),
      host_memory_allocator_(std::make_unique<HostMemoryAllocator>(
          std::move(host_memory_allocator))),
      owned_devices_(std::move(devices)),
      computation_placer_(std::make_unique<ComputationPlacer>()),
      compile_thread_pool_(std::make_unique<tsl::thread::ThreadPool>(
          tsl::Env::Default(), "TfrtGpuClient_compile_thread_pool",
          std::max<int>(DefaultThreadPoolSize(), xla_client->device_count()))),
      blocking_thread_pool_(std::make_unique<tsl::thread::ThreadPool>(
          tsl::Env::Default(), "TfrtGpuClient_blocking_thread_pool",
          std::max<int>(DefaultThreadPoolSize(), xla_client->device_count()))),
      non_blocking_thread_pool_(std::make_unique<tsl::thread::ThreadPool>(
          tsl::Env::Default(), "TfrtGpuClient_non_blocking_thread_pool",
          std::max<int>(DefaultThreadPoolSize(), xla_client->device_count()))),
      transpose_cache_(1024) {
  for (const std::unique_ptr<TfrtGpuDevice>& device : owned_devices_) {
    devices_.push_back(device.get());
    CHECK(
        id_to_device_.emplace(device->global_device_id(), device.get()).second)
        << "Duplicate device id: " << device->id();

    device->SetClient(this);
    if (device->IsAddressable()) {
      int idx = device->local_hardware_id().value();
      if (idx >= addressable_devices_.size()) {
        addressable_devices_.resize(idx + 1);
      }
      CHECK(addressable_devices_[idx] == nullptr) << idx;
      addressable_devices_[idx] = device.get();
    }
  }
  for (int idx = 0; idx < addressable_devices_.size(); ++idx) {
    CHECK(addressable_devices_[idx] != nullptr) << idx;
  }

  for (auto* device : addressable_devices()) {
    // Use the device id to construct a globally unique memory space id. We do
    // not promise that memory space ids and device ids are the same.
    TfrtGpuDevice* gpu_device = tensorflow::down_cast<TfrtGpuDevice*>(device);
    // Initialize the default memory space.
    const int global_device_id = gpu_device->global_device_id().value();
    auto memory_space =
        std::make_unique<TfrtGpuDeviceMemorySpace>(global_device_id, device);
    gpu_device->AttachMemorySpace(memory_space.get(), /*is_default=*/true);
    memory_spaces_.push_back(memory_space.get());
    owned_memory_spaces_.push_back(std::move(memory_space));
  }
  const int basePinnedId = device_count();
  for (auto* device : addressable_devices()) {
    TfrtGpuDevice* gpu_device = tensorflow::down_cast<TfrtGpuDevice*>(device);
    const int global_device_id = gpu_device->global_device_id().value();
    auto pinned = std::make_unique<PinnedHostMemorySpace>(
        basePinnedId + global_device_id, device);
    gpu_device->AttachMemorySpace(pinned.get());
    memory_spaces_.push_back(pinned.get());
    owned_memory_spaces_.push_back(std::move(pinned));
  }

  LOG(INFO) << "TfrtGpuClient created.";
}

TfrtGpuClient::~TfrtGpuClient() { LOG(INFO) << "TfrtGpuClient destroyed."; }

absl::StatusOr<PjRtDevice*> TfrtGpuClient::LookupDevice(
    PjRtGlobalDeviceId global_device_id) const {
  auto it = id_to_device_.find(global_device_id);
  if (it != id_to_device_.end()) {
    return it->second;
  }
  return InvalidArgument("No matching device found for device_id %d",
                         global_device_id.value());
}

absl::StatusOr<PjRtDevice*> TfrtGpuClient::LookupAddressableDevice(
    PjRtLocalDeviceId local_device_id) const {
  for (auto* device : addressable_devices_) {
    if (local_device_id == device->local_device_id()) {
      return device;
    }
  }
  return InvalidArgument("No matching device found for local_hardware_id %d",
                         local_device_id.value());
}

absl::StatusOr<Layout> TfrtGpuClient::GetDefaultLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dims) {
  Shape shape = ShapeUtil::MakeShape(element_type, dims);
  TF_ASSIGN_OR_RETURN(
      shape,
      xla_client_->backend().transfer_manager()->ChooseCompactLayoutForShape(
          shape));
  return shape.layout();
}

absl::StatusOr<std::unique_ptr<HloCostAnalysis>>
TfrtGpuClient::GetHloCostAnalysis() const {
  return std::make_unique<HloCostAnalysis>(
      xla_client_->backend().compiler()->ShapeSizeBytesFunction());
}

absl::Span<PjRtMemorySpace* const> TfrtGpuClient::memory_spaces() const {
  return memory_spaces_;
}

absl::StatusOr<DeviceAssignment> TfrtGpuClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  return computation_placer_->AssignDevices(num_replicas, num_partitions);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
TfrtGpuClient::CompileAndLoad(const XlaComputation& computation,
                              CompileOptions options) {
  std::vector<const Shape*> argument_layout_pointers;
  const ExecutableBuildOptions& build_options =
      options.executable_build_options;
  const bool allow_auto_layout =
      build_options.has_debug_options() &&
      build_options.debug_options().xla_pjrt_allow_auto_layout_in_hlo();
  TF_RETURN_IF_ERROR(DetermineArgumentLayoutsFromCompileOptions(
      computation,
      [local_client = xla_client_,
       allow_auto_layout](Shape shape) -> absl::StatusOr<Shape> {
        if (allow_auto_layout && !shape.has_layout()) {
          return shape;
        }
        return local_client->backend()
            .transfer_manager()
            ->ChooseCompactLayoutForShape(shape);
      },
      options.argument_layouts, &options.executable_build_options,
      &argument_layout_pointers));
  return CompileInternal(computation, argument_layout_pointers,
                         /* layout_canonicalization_callback = */ nullptr,
                         options);

  // TODO: b/382117736 - Record free gpu memory.
  // Ref:
  // https://github.com/openxla/xla/blob/b729ae319d85d5ec1ec11c488092c2d6683a63f2/xla/pjrt/gpu/se_gpu_pjrt_client.cc#L792-L809
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
TfrtGpuClient::CompileInternal(
    const XlaComputation& computation,
    const std::vector<const Shape*>& argument_layout_pointers,
    LayoutCanonicalizationCallback layout_canonicalization_callback,
    CompileOptions options) {
  tsl::profiler::TraceMe traceme("TfrtGpuClient::Compile");
  VLOG(1) << "TfrtGpuClient::Compile";
  if (key_value_store().has_value() &&
      !options.executable_build_options.key_value_store()) {
    options.executable_build_options.set_key_value_store(*key_value_store());
  }
  auto input_options = options;

  TF_RETURN_IF_ERROR(options.ApplyAllOptionOverrides());

  TF_ASSIGN_OR_RETURN(ExecutableExtras extras, GetExecutableExtras(&options));
  std::shared_ptr<DeviceAssignment>& device_assignment =
      extras.device_assignment;
  std::vector<TfrtGpuExecutable::LogicalDeviceIds>&
      addressable_device_logical_ids = extras.addressable_device_logical_ids;
  std::vector<PjRtDevice*>& addressable_devices = extras.addressable_devices;

  // It is important to set the canonicalization callback after creating
  // a copy of the options so that the executable's options remain without
  // the callback - the callback would break the executable's serializability.
  if (layout_canonicalization_callback) {
    options.executable_build_options.set_layout_canonicalization_callback(
        layout_canonicalization_callback);
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<LocalExecutable>> local_executables,
      xla_client_->Compile(computation, argument_layout_pointers,
                           options.executable_build_options));

  auto executable = std::make_unique<TfrtGpuExecutable>(
      std::move(local_executables), options.parameter_is_tupled_arguments,
      std::move(device_assignment), std::move(input_options),
      std::move(addressable_device_logical_ids), std::move(addressable_devices),
      this);

  TF_RETURN_IF_ERROR(
      executable->SetUpDonation(options.parameter_is_tupled_arguments));
  const auto& ex_options = options.executable_build_options;
  if (ex_options.has_debug_options() &&
      ex_options.debug_options().xla_gpu_dump_hlo_unoptimized_snapshots()) {
    executable->SetInputHloSnapshotBits(
        computation.proto(), options.executable_build_options.debug_options());
  }
  return std::unique_ptr<PjRtLoadedExecutable>(std::move(executable));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
TfrtGpuClient::CompileAndLoad(mlir::ModuleOp module, CompileOptions options) {
  XlaComputation xla_computation;
  const ExecutableBuildOptions& exec_build_options =
      options.executable_build_options;
  TF_RETURN_IF_ERROR(MlirToXlaComputation(
      module, xla_computation,
      /*use_tuple_args=*/options.parameter_is_tupled_arguments,
      /*return_tuple=*/false, exec_build_options.use_shardy_partitioner()));

  // TODO: b/382117736 - Add support for LayoutModesToXlaShapes
  // Ref:
  // https://github.com/openxla/xla/blob/b729ae319d85d5ec1ec11c488092c2d6683a63f2/xla/pjrt/pjrt_stream_executor_client.cc#L3538-L3586
  return CompileAndLoad(xla_computation, options);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
TfrtGpuClient::CreateViewOfDeviceBuffer(
    void* device_ptr, const Shape& shape, PjRtMemorySpace* memory_space,
    std::function<void()> on_delete_callback,
    std::optional<std::intptr_t> stream) {
  CHECK_EQ(memory_space->devices().size(), 1);
  auto* device = memory_space->devices().front();
  size_t byte_size = ShapeUtil::ByteSizeOf(shape);
  se::DeviceMemoryBase device_memory(device_ptr, byte_size);
  auto non_owning_buffer = MaybeOwningGpuMemory(device_memory);
  auto buffer_async_value_ref =
      tsl::MakeAvailableAsyncValueRef<MaybeOwningGpuMemory>(
          std::move(non_owning_buffer));
  auto tracked_device_buffer = std::make_unique<TrackedTfrtGpuDeviceBuffer>(
      std::move(buffer_async_value_ref),
      /*definition_event=*/tsl::MakeAvailableAsyncValueRef<GpuEvent>(),
      std::move(on_delete_callback));
  return std::make_unique<TfrtGpuBuffer>(
      shape, std::move(tracked_device_buffer), this,
      tsl::down_cast<TfrtGpuDevice*>(device), memory_space);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> TfrtGpuClient::CreateErrorBuffer(
    absl::Status error, const Shape& shape, PjRtMemorySpace* memory_space) {
  CHECK_EQ(memory_space->devices().size(), 1);
  if (memory_space->client() != this) {
    return absl::InvalidArgumentError(
        "Memory space is not attached to this client");
  }

  TfrtGpuDevice* device =
      tensorflow::down_cast<TfrtGpuDevice*>(memory_space->devices().front());
  if (VLOG_IS_ON(1)) {
    LOG(INFO) << "TfrtGpuClient::CreateErrorBuffer: shape: " << shape.ToString()
              << " device: " << device->DebugString() << " error: " << error;
  }

  // Create a dummy buffer because the rest of the code expects a buffer
  // regardless of whether the definition event is an error.
  int64_t byte_size = ShapeUtil::ByteSizeOf(shape);
  void* device_ptr = device->allocator()->AllocateRaw(
      tsl::Allocator::kAllocatorAlignment, byte_size);
  se::DeviceMemoryBase device_memory(device_ptr, byte_size);
  auto gpu_buffer =
      MaybeOwningGpuMemory(device->allocator(), std::move(device_memory));
  auto buffer_async_value_ref =
      tsl::MakeAvailableAsyncValueRef<MaybeOwningGpuMemory>(
          std::move(gpu_buffer));
  auto tracked_device_buffer = std::make_unique<TrackedTfrtGpuDeviceBuffer>(
      std::move(buffer_async_value_ref),
      /*definition_event=*/tsl::MakeErrorAsyncValueRef(std::move(error)));
  return std::make_unique<TfrtGpuBuffer>(
      shape, std::move(tracked_device_buffer), this,
      tsl::down_cast<TfrtGpuDevice*>(device), memory_space);
}

absl::StatusOr<TfrtGpuClient::ExecutableExtras>
TfrtGpuClient::GetExecutableExtras(CompileOptions* options) {
  ExecutableExtras extras;
  std::shared_ptr<DeviceAssignment>& device_assignment =
      extras.device_assignment;
  std::vector<TfrtGpuExecutable::LogicalDeviceIds>&
      addressable_device_logical_ids = extras.addressable_device_logical_ids;
  std::vector<PjRtDevice*>& addressable_devices = extras.addressable_devices;

  ExecutableBuildOptions& build_options = options->executable_build_options;
  if (!build_options.compile_thread_pool()) {
    build_options.set_compile_thread_pool(compile_thread_pool_.get());
  }

  if (!build_options.device_allocator()) {
    build_options.set_device_allocator(
        xla_client_->backend().memory_allocator());
  }

  auto layout_callback = [local_client = xla_client_,
                          options](const HloModule& module)
      -> absl::StatusOr<std::pair<std::vector<Shape>, Shape>> {
    ExecutableBuildOptions build_options = options->executable_build_options;
    std::vector<const Shape*> argument_layout_pointers;
    std::optional<std::vector<Shape>> argument_layouts =
        options->argument_layouts;
    Shape result_layout;
    const bool allow_auto_layout =
        build_options.has_debug_options() &&
        build_options.debug_options().xla_pjrt_allow_auto_layout_in_hlo();
    TF_RETURN_IF_ERROR(DetermineArgumentLayoutsFromCompileOptions(
        XlaComputation(module.ToProto()),
        [local_client,
         allow_auto_layout](Shape shape) -> absl::StatusOr<Shape> {
          if (allow_auto_layout && !shape.has_layout()) {
            return shape;
          }
          return local_client->backend()
              .transfer_manager()
              ->ChooseCompactLayoutForShape(shape);
        },
        argument_layouts, &build_options, &argument_layout_pointers));
    result_layout = *build_options.result_layout();
    return std::make_pair(*argument_layouts, result_layout);
  };

  build_options.set_layout_canonicalization_callback(layout_callback);

  int num_replicas;
  int num_partitions;
  TF_RETURN_IF_ERROR(ParseDeviceAssignmentCompileOptions(
      options->compile_portable_executable, &options->executable_build_options,
      [this](int num_replicas, int num_partitions) {
        return this->GetDefaultDeviceAssignment(num_replicas, num_partitions);
      },
      &num_replicas, &num_partitions, &device_assignment));

  // Find devices that are addressable by this client/task.
  if (device_assignment != nullptr) {
    addressable_device_logical_ids.reserve(num_replicas * num_partitions);
    addressable_devices.reserve(num_replicas * num_partitions);
    absl::flat_hash_set<int> all_process_indices;
    std::optional<int> this_process_index;
    for (int replica = 0; replica < num_replicas; ++replica) {
      for (int partition = 0; partition < num_partitions; ++partition) {
        int64_t device_id = (*device_assignment)(replica, partition);
        PjRtGlobalDeviceId global_device_id(device_id);

        TF_ASSIGN_OR_RETURN(PjRtDevice * device,
                            LookupDevice(global_device_id));
        all_process_indices.insert(device->process_index());
        if (device->process_index() != process_index()) {
          VLOG(3) << "Non-local device: " << device_id;
          continue;
        }
        if (!this_process_index.has_value()) {
          this_process_index = all_process_indices.size() - 1;
        }
        PjRtLoadedExecutable::LogicalDeviceIds logica_device_ids;
        logica_device_ids.replica = replica;
        logica_device_ids.partition = partition;
        addressable_device_logical_ids.push_back(std::move(logica_device_ids));
        addressable_devices.push_back(device);
      }
    }
    if (addressable_devices.empty()) {
      return InvalidArgument(
          "Device assignment (%s) does not have any local devices.",
          device_assignment->ToString());
    }

    if (build_options.device_ordinal() < 0) {
      build_options.set_device_ordinal(
          addressable_devices.front()->local_hardware_id().value());
    }

    build_options.set_process_index(*this_process_index);
    build_options.set_process_count(all_process_indices.size());
  }
  return extras;
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> TfrtGpuClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    PjRtMemorySpace* memory_space, const Layout* device_layout) {
  // TODO: b/382117736 - support device_layout
  // TODO: b/382117736 - support non-default memory_space (e.g. pinned)
  PjRtDevice* device = memory_space->devices()[0];
  tsl::profiler::TraceMe traceme("TfrtGpuClient::BufferFromHostBuffer");
  Shape device_shape = ShapeUtil::MakeShape(type, dims);
  if (VLOG_IS_ON(2)) {
    LOG(INFO) << "TfrtGpuClient::BufferFromHostBuffer: shape: "
              << device_shape.ToString()
              << " device: " << device->DebugString();
  }
  absl::InlinedVector<int64_t, 4> tmp_strides;
  if (!byte_strides) {
    tmp_strides.resize(dims.size());
    TF_RETURN_IF_ERROR(
        ShapeUtil::ByteStrides(device_shape, absl::MakeSpan(tmp_strides)));
    byte_strides = tmp_strides;
  }
  int64_t byte_size = ShapeUtil::ByteSizeOf(device_shape);

  TransferManager* transfer_manager = xla_client_->backend().transfer_manager();
  TF_ASSIGN_OR_RETURN(
      device_shape,
      transfer_manager->ChooseCompactLayoutForShape(device_shape));

  absl::InlinedVector<int64_t, 4> shape_strides(device_shape.dimensions_size());
  TF_RETURN_IF_ERROR(
      ShapeUtil::ByteStrides(device_shape, absl::MakeSpan(shape_strides)));
  bool host_and_device_strides_equal =
      (byte_size == 0 || *byte_strides == shape_strides);

  std::shared_ptr<TransposePlan> transpose;
  if (!host_and_device_strides_equal) {
    absl::InlinedVector<int64_t, 4> permutation(dims.size());
    absl::c_reverse_copy(device_shape.layout().minor_to_major(),
                         permutation.begin());
    TransposePlan::Options options;
    options.elem_size_in_bytes = primitive_util::ByteWidth(type);
    options.dims = dims;
    options.permutation = permutation;
    options.input_layout = TransposePlan::Striding{*byte_strides};
    absl::MutexLock lock(&transpose_mu_);
    TF_ASSIGN_OR_RETURN(transpose, transpose_cache_.GetOrCreate(options));
  }

  // TODO: b/382117736 - support SubByteNonPredType
  if (primitive_util::IsSubByteNonPredType(type)) {
    return absl::UnimplementedError(
        "SubByteNonPredType is not supported in TfrtGpuClient.");
  }

  auto* gpu_device = tsl::down_cast<TfrtGpuDevice*>(device);

  auto gpu_buffer = tsl::MakeUnconstructedAsyncValueRef<MaybeOwningGpuMemory>();
  absl::InlinedVector<tsl::AsyncValueRef<GpuEvent>, 4> definition_events;
  tsl::AsyncValueRef<GpuEvent> copy_event =
      tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  definition_events.push_back(copy_event.CopyRef());

  // TODO: b/382117736 - support DmaMapping
  const bool should_sync_copy =
      (host_buffer_semantics ==
       HostBufferSemantics::kImmutableOnlyDuringCall) ||
      should_stage_host_to_device_transfers();

  // Define H2D copy lambda. First, copy host data to staging buffer, then copy
  // staging buffer to GPU device.
  auto h2d_copy =
      [this, data, byte_size, gpu_device, transpose(std::move(transpose)),
       copy_event(std::move(copy_event)), gpu_buffer{gpu_buffer.CopyRef()},
       on_done_with_host_buffer = std::move(on_done_with_host_buffer),
       host_memory_allocator = host_memory_allocator_.get()]() mutable {
        tsl::profiler::TraceMe traceme("H2D staging copy");
        HostMemoryAllocator::OwnedPtr staging_buffer =
            host_memory_allocator->Allocate(byte_size);
        if (transpose) {
          transpose->Execute(data, staging_buffer.get());
        } else {
          std::memcpy(staging_buffer.get(), data, byte_size);
        }
        if (on_done_with_host_buffer) {
          std::move(on_done_with_host_buffer)();
        }

        auto copy_to_gpu = [gpu_device, byte_size,
                            copy_event(std::move(copy_event)),
                            gpu_buffer{gpu_buffer.CopyRef()},
                            staging_buffer(std::move(staging_buffer))]() {
          tsl::profiler::TraceMe traceme("H2D GPU copy");
          auto gpu_buffer_or = MaybeOwningGpuMemory::AllocateShared(
              gpu_device->allocator(), byte_size);
          if (gpu_buffer_or.ok()) {
            gpu_buffer.emplace(std::move(gpu_buffer_or.value()));
          } else {
            gpu_buffer.SetError(gpu_buffer_or.status());
            copy_event.SetError(gpu_buffer_or.status());
            return;
          }

          absl::StatusOr<BoundedStreamPool::Handle> handle_or =
              gpu_device->stream_pool().Borrow();
          if (!handle_or.ok()) {
            copy_event.SetError(handle_or.status());
            return;
          }
          BoundedStreamPool::Handle stream = std::move(handle_or.value());

          se::DeviceMemoryBase dest = gpu_buffer->buffer();
          absl::Status status =
              stream->Memcpy(&dest, staging_buffer.get(), byte_size);
          if (!status.ok()) {
            copy_event.SetError(status);
            return;
          }
          status = stream->BlockHostUntilDone();
          if (status.ok()) {
            copy_event.SetStateConcrete();
          } else {
            copy_event.SetError(status);
          }
        };

        EnqueueWork(blocking_thread_pool_.get(), std::move(copy_to_gpu));
      };

  if (should_sync_copy) {
    h2d_copy();
  } else {
    EnqueueWork(non_blocking_thread_pool_.get(), std::move(h2d_copy));
  }

  std::function<void()> on_delete_callback;
  auto tracked_device_buffer = std::make_unique<TrackedTfrtGpuDeviceBuffer>(
      std::move(gpu_buffer), std::move(definition_events),
      std::move(on_delete_callback));

  return std::unique_ptr<PjRtBuffer>(std::make_unique<TfrtGpuBuffer>(
      device_shape, std::move(tracked_device_buffer), this, gpu_device,
      memory_space));
}

absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
TfrtGpuClient::CreateBuffersForAsyncHostToDevice(
    absl::Span<const ShapeSpec> shape_specs,
    std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
    PjRtMemorySpace* memory_space) {
  CHECK_EQ(memory_space->devices().size(), 1);
  PjRtDevice* device = memory_space->devices()[0];
  auto* tfrt_gpu_device = tensorflow::down_cast<TfrtGpuDevice*>(device);
  return TfrtGpuAsyncHostToDeviceTransferManager::Create(
      shape_specs, device_layouts, tfrt_gpu_device, this, memory_space);
}

static absl::StatusOr<std::vector<std::unique_ptr<TfrtGpuDevice>>>
GetTfrtGpuDevices(LocalClient* xla_client) {
  std::vector<std::unique_ptr<TfrtGpuDevice>> devices;
  int i = 0;
  for (se::StreamExecutor* executor :
       xla_client->backend().stream_executors()) {
    // TODO: b/382117736 - allow GPU allocator parameters to be configurable.
    TF_ASSIGN_OR_RETURN(auto allocator,
                        CreateBFCAllocator(executor, /*memory_fraction=*/0.9,
                                           /*preallocate=*/true, std::nullopt));

    TfrtGpuDevice::Options options;
    options.id = i;
    options.local_device_id = PjRtLocalDeviceId(i);
    options.local_hardware_id = PjRtLocalHardwareId(i);
    options.executor = executor;
    options.allocator = std::move(allocator);
    options.stream_capacity = 4;
    options.max_inflight_computations = 1;
    const se::Platform* platform = executor->GetPlatform();
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::se::DeviceDescription> desc,
        platform->DescriptionForDevice(options.local_hardware_id.value()));
    options.platform_version = desc->name();

    auto device = std::make_unique<TfrtGpuDevice>(std::move(options));
    devices.push_back(std::move(device));
    ++i;
  }
  return std::move(devices);
}

absl::StatusOr<std::unique_ptr<PjRtClient>> GetTfrtGpuClient(
    const GpuClientOptions& options) {
  TF_ASSIGN_OR_RETURN(
      LocalClient * xla_client,
      GetGpuXlaClient(options.platform_name, options.allowed_devices));
  EnablePeerAccess(xla_client->backend().stream_executors());
  std::unique_ptr<tsl::Allocator> host_memory_allocator;
  if (!xla_client->backend().stream_executors().empty()) {
    TF_ASSIGN_OR_RETURN(
        host_memory_allocator,
        GetGpuHostAllocator(xla_client->backend().stream_executors().front()));
  }
  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<TfrtGpuDevice>> devices,
                      GetTfrtGpuDevices(xla_client));

  GpuTopologyProto gpu_topology_proto;
  for (const auto& device : devices) {
    if (gpu_topology_proto.platform_version().empty()) {
      gpu_topology_proto.set_platform_version(
          std::string(device->device_kind()));
    }
    gpu_topology_proto.add_device_ids(device->id());
  }

  // TODO: b/382117736 - Support multi-host
  gpu_topology_proto.set_num_slices(1);
  gpu_topology_proto.set_num_hosts_per_slice(1);
  gpu_topology_proto.set_num_devices_per_host(devices.size());

  auto gpu_topology = std::shared_ptr<const GpuTopology>(
      GpuTopology::FromProto(gpu_topology_proto));

  return std::unique_ptr<PjRtClient>(std::make_unique<TfrtGpuClient>(
      /*process_index=*/0, xla_client, std::move(devices),
      options.should_stage_host_to_device_transfers,
      std::move(host_memory_allocator), gpu_topology));
}

TfrtGpuBuffer::TfrtGpuBuffer(
    Shape on_device_shape,
    std::unique_ptr<TrackedTfrtGpuDeviceBuffer> tracked_device_buffer,
    TfrtGpuClient* client, TfrtGpuDevice* device, PjRtMemorySpace* memory_space)
    : client_(client),
      on_device_shape_(std::move(on_device_shape)),
      device_(device),
      memory_space_(CHECK_NOTNULL(memory_space)),
      tracked_device_buffer_(std::move(tracked_device_buffer)),
      donation_event_(tsl::MakeAvailableAsyncValueRef<bool>(false)),
      external_references_dropped_event_(
          tsl::MakeConstructedAsyncValueRef<GpuEvent>()) {}

TfrtGpuBuffer::~TfrtGpuBuffer() { Delete(); }

absl::StatusOr<size_t> TfrtGpuBuffer::GetOnDeviceSizeInBytes() const {
  return ShapeUtil::ByteSizeOf(on_device_shape_);
}

TrackedTfrtGpuDeviceBuffer* TfrtGpuBuffer::AcquireUsage(
    tsl::AsyncValueRef<GpuEvent> usage_event) {
  absl::MutexLock lock(&mu_);
  if (!tracked_device_buffer_) {
    return nullptr;
  }

  tracked_device_buffer_->AddUsageEvents(absl::MakeSpan(&usage_event, 1));
  return tracked_device_buffer_.get();
}

absl::StatusOr<Shape> TfrtGpuBuffer::logical_on_device_shape() {
  if (on_device_shape_.is_static()) {
    return on_device_shape_;
  }

  auto usage_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto* device_buffer = AcquireUsage(usage_event);
  if (device_buffer == nullptr) {
    return InvalidArgument(
        "logical_on_device_shape() called on deleted or donated buffer");
  }
  MarkGpuEventReadyOnExit ready_on_exit(std::move(usage_event));

  // Wait for the definition event.
  const auto& av = device_buffer->definition_event();
  tsl::BlockUntilReady(av);
  if (auto* error = av.GetErrorIfPresent()) {
    return absl::InternalError(
        absl::StrFormat("Error Execute: %s", error->message()));
  }

  const auto& buffer = device_buffer->buffer();

  ShapedBuffer shaped_buffer =
      buffer->AsShapedBuffer(on_device_shape_, device_);
  Shape ret_shape = on_device_shape_;
  TransferManager* transfer_manager =
      client_->xla_client()->backend().transfer_manager();

  TF_ASSIGN_OR_RETURN(BoundedStreamPool::Handle stream,
                      device_->stream_pool_.Borrow());
  TF_RETURN_IF_ERROR(transfer_manager->ReadDynamicShapes(
      stream.get(), &shaped_buffer, &ret_shape));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  return ret_shape;
}

PjRtFuture<> TfrtGpuBuffer::GetReadyFuture() {
  tsl::AsyncValueRef<GpuEvent> definition_event;
  absl::MutexLock lock(&mu_);
  if (!tracked_device_buffer_) {
    return PjRtFuture<>(InvalidArgument(
        "GetReadyFuture() called on deleted or donated buffer"));
  }
  definition_event = tracked_device_buffer_->definition_event();
  DCHECK(definition_event);
  if (!definition_promise_) {
    definition_promise_ = PjRtFuture<>::CreatePromise();
    if (definition_event.IsAvailable()) {
      if (definition_event.IsError()) {
        return PjRtFuture<>(
            FailedPrecondition("Buffer Definition Event: %s",
                               definition_event.GetError().message()));
      }
      definition_promise_.Set(absl::OkStatus());
    } else {
      definition_event.AndThen(
          [definition_event,
           definition_promise = definition_promise_]() mutable {
            if (definition_event.IsError()) {
              VLOG(2) << "definition_event.GetError(): "
                      << definition_event.GetError();
              definition_promise.Set(definition_event.GetError());
            } else {
              definition_promise.Set(absl::OkStatus());
            }
          });
    }
  }

  return PjRtFuture<>(
      definition_promise_,
      /*on_block_start=*/
      []() {
        tsl::profiler::TraceMeProducer traceme("TfrtGpuBuffer::Await");
        VLOG(1) << "TfrtGpuBuffer::Await";
        return PjRtFutureHelpers::ProfilingKeys(
            {/*traceme_context_id=*/traceme.GetContextId()});
      },
      /*on_block_end=*/
      [](PjRtFutureHelpers::ProfilingKeys keys) {
        tsl::profiler::TraceMeConsumer traceme("TfrtGpuBuffer::Await",
                                               keys.traceme_context_id);
      });
}

absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
TfrtGpuBuffer::AcquireExternalReference() {
  class ScopedExternalReference : public PjRtBuffer::ExternalReference {
   public:
    explicit ScopedExternalReference(
        TfrtGpuBuffer* buffer, tsl::AsyncValueRef<MaybeOwningGpuMemory> data)
        : buffer_(buffer), data_(std::move(data)) {
      DCHECK(data_);
      data_ptr_ = data_->buffer().opaque();
    }

    ~ScopedExternalReference() override { buffer_->DropExternalReference(); }

   private:
    TfrtGpuBuffer* buffer_ = nullptr;
    // Keep a reference to the underlying data used. Note that it is still
    // users' responsibility to synchronize reads and writes to the data.
    tsl::AsyncValueRef<MaybeOwningGpuMemory> data_;
  };

  absl::MutexLock lock(&mu_);
  if (tracked_device_buffer_ == nullptr) {
    return InvalidArgument("Buffer has been deleted or donated.");
  }

  ++external_reference_counter_;

  tsl::BlockUntilReady(tracked_device_buffer_->definition_event());
  if (tracked_device_buffer_->definition_event().IsError()) {
    return tracked_device_buffer_->definition_event().GetError();
  }
  return {std::make_unique<ScopedExternalReference>(
      this, tracked_device_buffer_->buffer())};
}

class TrackedGpuDeviceBufferExternalReference
    : public PjRtBuffer::ExternalReference {
 public:
  explicit TrackedGpuDeviceBufferExternalReference(
      std::unique_ptr<TrackedTfrtGpuDeviceBuffer> tracked_device_buffer)
      : tracked_device_buffer_(std::move(tracked_device_buffer)) {
    data_ptr_ = tracked_device_buffer_->buffer()->buffer().opaque();
  }

  ~TrackedGpuDeviceBufferExternalReference() override = default;

 private:
  std::unique_ptr<TrackedTfrtGpuDeviceBuffer> tracked_device_buffer_;
};

absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
TfrtGpuBuffer::ReleaseDeviceMemoryOwnership(
    bool wait_for_operations_to_complete) {
  if (on_device_shape_.IsTuple()) {
    return InvalidArgument(
        "ReleaseDeviceMemoryOwnership allowed only for non-tuple");
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TrackedTfrtGpuDeviceBuffer> tracked_device_buffer,
      Release(wait_for_operations_to_complete));

  std::unique_ptr<PjRtBuffer::ExternalReference> ref;
  if (tracked_device_buffer) {
    ref = std::make_unique<TrackedGpuDeviceBufferExternalReference>(
        std::move(tracked_device_buffer));
  }
  return ref;
}

PjRtFuture<> TfrtGpuBuffer::ToLiteral(MutableLiteralBase* literal) {
  tsl::profiler::TraceMe traceme("TfrtGpuBuffer::ToLiteral");
  auto promise = PjRtFuture<>::CreatePromise();
  auto usage_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto* device_buffer = AcquireUsage(usage_event);
  if (device_buffer == nullptr) {
    promise.Set(
        InvalidArgument("ToLiteral() called on deleted or donated buffer"));
    return PjRtFuture<>(promise);
  }
  EnqueueWorkWhenReady(
      client_->blocking_thread_pool(),
      {device_buffer->definition_event().CopyRCRef()},
      [device(device_), device_buffer, usage_event(std::move(usage_event)),
       literal, promise, client = client_]() mutable {
        tsl::profiler::TraceMe traceme("D2H copy");
        if (device_buffer->definition_event().IsError()) {
          usage_event.SetStateConcrete();
          VLOG(2) << "device_buffer->definition_event().GetError(): "
                  << device_buffer->definition_event().GetError();

          promise.Set(device_buffer->definition_event().GetError());
          return;
        }
        size_t byte_size = device_buffer->buffer()->buffer().size();
        HostMemoryAllocator::OwnedPtr staging_buffer =
            client->host_memory_allocator()->Allocate(byte_size);

        {
          tsl::profiler::TraceMe traceme2("D2H GPU copy");
          MarkGpuEventReadyOnExit ready_on_exit(std::move(usage_event));

          auto stream_or = device->stream_pool().Borrow();
          if (!stream_or.ok()) {
            promise.Set(stream_or.status());
            return;
          }
          BoundedStreamPool::Handle stream = std::move(stream_or.value());

          CHECK_OK(stream->Memcpy(staging_buffer.get(),
                                  device_buffer->buffer()->buffer(), byte_size))
              << "stream->Memcpy failed copying from GPU to host";
          absl::Status status = stream->BlockHostUntilDone();
          if (!status.ok()) {
            VLOG(2) << "stream->BlockHostUntilDone failed: " << status;
            promise.Set(status);
            return;
          }
        }
        tsl::profiler::TraceMe traceme3("D2H staging copy");
        std::memcpy(literal->untyped_data(), staging_buffer.get(), byte_size);
        VLOG(2) << "D2H staging copy done";
        promise.Set(absl::OkStatus());
      });

  return PjRtFuture<>(
      std::move(promise),
      /*on_block_start=*/
      []() {
        tsl::profiler::TraceMeProducer traceme("TfrtGpuBuffer::ToLiteral");
        VLOG(1) << "TfrtGpuBuffer::ToLiteral";
        return PjRtFutureHelpers::ProfilingKeys(
            {/*traceme_context_id =*/traceme.GetContextId()});
      },
      /*on_block_end=*/
      [](PjRtFutureHelpers::ProfilingKeys keys) {
        tsl::profiler::TraceMeConsumer traceme("TfrtGpuBuffer::ToLiteral",
                                               keys.traceme_context_id);
      });
}

PjRtFuture<> TfrtGpuBuffer::LazyToLiteral(
    absl::AnyInvocable<absl::StatusOr<MutableLiteralBase*>() &&> generator) {
  auto buffer = std::move(generator)();
  if (!buffer.ok()) {
    return PjRtFuture<>(buffer.status());
  }
  return ToLiteral(buffer.value());
}

void TfrtGpuBuffer::Delete() {
  tsl::profiler::TraceMe traceme("Gpu buffer delete");
  std::unique_ptr<TrackedTfrtGpuDeviceBuffer> device_buffer;
  tsl::AsyncValueRef<GpuEvent> external_references_dropped_event;
  {
    absl::MutexLock lock(&mu_);
    device_buffer = ReleaseBufferLocked();
    if (device_buffer == nullptr) return;

    if (external_reference_counter_ > 0) {
      external_references_dropped_event =
          external_references_dropped_event_.CopyRef();
    } else {
      external_references_dropped_event =
          tsl::MakeAvailableAsyncValueRef<GpuEvent>();
    }
  }
  if (device_buffer == nullptr) return;

  tsl::AsyncValueRef<bool> donation_event = GetDonationEvent();

  // Now that all holds have completed and no more can be added, we can get
  // the final set of usage events.
  tsl::AsyncValueRef<GpuEvent> usage_event =
      device_buffer->LockUseAndTransferUsageEvents();

  std::array event_avs{
      usage_event.GetAsyncValue(),
      // We should also wait for the definition event.
      device_buffer->definition_event().GetAsyncValue(),
      donation_event.GetAsyncValue(),
      external_references_dropped_event.GetAsyncValue(),
  };

  tsl::RunWhenReady(
      event_avs, [device_buffer = std::move(device_buffer),
                  usage_event(std::move(usage_event)),
                  donation_event(std::move(donation_event))]() mutable {
        VLOG(3) << "device_buffer is being deleted: " << device_buffer.get();
        device_buffer.reset();
      });
}

bool TfrtGpuBuffer::IsDeleted() {
  absl::MutexLock lock(&mu_);
  return tracked_device_buffer_ == nullptr;
}

void TfrtGpuBuffer::DropExternalReference() {
  absl::MutexLock lock(&mu_);
  CHECK_GT(external_reference_counter_, 0);
  --external_reference_counter_;
  if (external_reference_counter_ == 0) {
    external_references_dropped_event_.SetStateConcrete();
  }
}

absl::StatusOr<std::unique_ptr<TrackedTfrtGpuDeviceBuffer>>
TfrtGpuBuffer::Release(bool wait_for_operations_to_complete) {
  auto donation_event = GetDonationEvent();
  tsl::BlockUntilReady(donation_event);
  std::unique_ptr<TrackedTfrtGpuDeviceBuffer> device_buffer;
  {
    absl::MutexLock lock(&mu_);
    device_buffer = ReleaseBufferLocked();
  }
  if (device_buffer == nullptr) return {nullptr};

  std::array events{
      // Now that all holds have completed and no more can be added, we can get
      // the final set of usage events.
      device_buffer->LockUseAndTransferUsageEvents(),
      device_buffer->definition_event().CopyRef(),
  };

  if (wait_for_operations_to_complete) {
    // Block the host until all usage events have completed. Usage events
    // dominate definition events, so this also waits for the buffer to be
    // defined. Return the first error encountered.
    absl::Status first_error;
    for (const auto& av : events) {
      tsl::BlockUntilReady(av);
      if (auto* error = av.GetErrorIfPresent()) {
        first_error.Update(*error);
      }
    }
    if (!first_error.ok()) return std::move(first_error);
  }

  return device_buffer;
}

std::unique_ptr<TrackedTfrtGpuDeviceBuffer>
TfrtGpuBuffer::ReleaseBufferLocked() {
  tsl::profiler::TraceMe traceme("TfrtGpuBuffer::ReleaseBufferLocked");
  return std::move(tracked_device_buffer_);
}

TfrtGpuExecutable::TfrtGpuExecutable(
    std::vector<std::unique_ptr<LocalExecutable>> executables,
    bool parameter_is_tupled_arguments,
    std::shared_ptr<DeviceAssignment> device_assignment,
    CompileOptions compile_options,
    std::vector<LogicalDeviceIds> addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices, TfrtGpuClient* client)
    : client_(client),
      device_assignment_(std::move(device_assignment)),
      compile_options_(std::move(compile_options)),
      parameter_is_tupled_arguments_(parameter_is_tupled_arguments),
      addressable_device_logical_ids_(
          std::move(addressable_device_logical_ids)),
      addressable_devices_(std::move(addressable_devices)) {
  executables_.reserve(executables.size());
  tsl::Fprint128 fingerprint = tsl::Fingerprint128(fingerprint_);
  for (auto& executable : executables) {
    const auto& computation_layout =
        executable->executable()->module().entry_computation_layout();
    std::vector<Shape> parameter_shapes;
    parameter_shapes.reserve(computation_layout.parameter_count());
    for (int i = 0; i < computation_layout.parameter_count(); ++i) {
      // TODO: b/400541410 - Convert to device shape when we have transfer
      // manager.
      // parameter_shapes.push_back(transfer_manager->HostShapeToDeviceShape(
      // computation_layout.parameter_shape(i)));
      parameter_shapes.push_back(computation_layout.parameter_shape(i));
    }
    on_device_executable_parameter_shapes_.push_back(
        std::make_shared<std::vector<Shape>>(std::move(parameter_shapes)));

    auto input_buffer_sizes_in_bytes = std::make_shared<std::vector<int64_t>>();

    // Assume compiled program expects either many non-tupled arguments or a
    // singled tupled argument, or no arguments. Nested tuple is not yet
    // supported.
    if (computation_layout.parameter_count() == 0) {
      // No arguments. Do nothing.
    } else if (computation_layout.parameter_count() == 1 &&
               computation_layout.parameter_shape(0).IsTuple()) {
      const std::vector<Shape>& tuple_shapes =
          computation_layout.parameter_shape(0).tuple_shapes();
      input_buffer_sizes_in_bytes->reserve(tuple_shapes.size());
      for (const Shape& shape : tuple_shapes) {
        input_buffer_sizes_in_bytes->push_back(ShapeUtil::ByteSizeOf(shape));
      }
    } else {
      const std::vector<ShapeLayout>& parameter_layouts =
          computation_layout.parameter_layouts();
      input_buffer_sizes_in_bytes->reserve(parameter_layouts.size());
      for (const ShapeLayout& layout : parameter_layouts) {
        input_buffer_sizes_in_bytes->push_back(
            ShapeUtil::ByteSizeOf(layout.shape()));
      }
    }
    input_buffer_sizes_in_bytes_.push_back(
        std::move(input_buffer_sizes_in_bytes));

    fingerprint = tsl::FingerprintCat128(
        fingerprint,
        tsl::Fingerprint128(executable->executable()->module().ToString()));
    executables_.emplace_back(std::move(executable));
  }
  fingerprint_ = absl::StrCat(fingerprint.low64, fingerprint.high64);

  int num_partitions;
  if (device_assignment_ == nullptr) {
    // This must go after `executables_` is initialized.
    VLOG(3) << "TfrtGpuExecutable portable single-core";
    num_partitions = 1;
    CHECK(addressable_devices_.empty());
  } else {
    // This must go after `executables_` is initialized.
    if (VLOG_IS_ON(3)) {
      VLOG(3) << "TfrtGpuExecutable device_assignment:\n"
              << device_assignment_->ToString();
    }
    CHECK_GE(addressable_devices_.size(), 1) << device_assignment_->ToString();

    if ((device_assignment_->replica_count() > 1 ||
         device_assignment_->computation_count() > 1) &&
        IsAllZeros(*device_assignment_)) {
      // This code path should only be triggered when we intentionally compile
      // an HLO without having enough devices to actually run it. See the
      // "--run=false" option in
      // tensorflow/compiler/xla/tools/multihost_hlo_runner/hlo_runner_main.cc.
      // That will help us debug the XLA compiler locally.
      LOG(INFO) << "A workaround is in effect to allow compiling multi-device "
                   "HLOs on machines with fewer devices. Don't run this "
                   "executable.";
    } else {
      CHECK_LE(addressable_devices_.size(), client_->addressable_device_count())
          << "Inconsistent local device count.";
    }

    num_partitions = device_assignment_->computation_count();
  }

  // SPMD sharding produces a single executable for multiple partitions.
  if (executables_.size() > 1) {
    CHECK_EQ(num_partitions, executables_.size())
        << "Number of executables " << executables_.size()
        << " did not match number of partitions " << num_partitions;
  }
}

absl::StatusOr<PjRtLoadedExecutable::Result> TfrtGpuExecutable::ExecuteHelper(
    absl::Span<PjRtBuffer* const> argument_handles, int replica, int partition,
    const RunId& run_id, const ExecuteOptions& options,
    tsl::AsyncValueRef<GpuEvent> last_collective_launch_event, bool fill_future,
    TfrtGpuDevice* device) {
  tsl::profiler::TraceMe traceme("TfrtGpuExecutable::ExecuteHelper");
  VLOG(2) << "ExecuteHelper " << name() << ": " << options.launch_id
          << "; replica: " << replica;

  std::shared_ptr<DeviceAssignment> device_assignment;
  if (device == nullptr) {
    CHECK(device_assignment_ != nullptr);
    const int device_id = (*device_assignment_)(replica, partition);
    VLOG(2) << "device_id: " << device_id;
    TF_ASSIGN_OR_RETURN(PjRtDevice * pjrt_device,
                        client_->LookupDevice(PjRtGlobalDeviceId(device_id)));
    device = tsl::down_cast<TfrtGpuDevice*>(pjrt_device);
    device_assignment = device_assignment_;
  } else {
    CHECK(device_assignment_ == nullptr);
    CHECK_EQ(replica, 0);
    CHECK_EQ(partition, 0);
    CHECK(addressable_devices_.empty());
    device_assignment = std::make_shared<DeviceAssignment>(1, 1);
    (*device_assignment)(0, 0) = device->id();
  }
  CHECK_EQ(device->process_index(), client_->process_index());

  // Handle inputs.
  if (options.arguments_are_tupled) {
    if (!parameter_is_tupled_arguments_) {
      return InvalidArgument(
          "Arguments may only be supplied as a tuple when the executable was"
          "compiled with a single tupled parameter");
    }
    if (argument_handles.size() != 1) {
      return InvalidArgument(
          "Option arguments_are_tupled was true but %d buffers were passed to"
          "execution",
          argument_handles.size());
    }
  }

  // `execute_event` indicates whether gpu computation is complete and whether
  // there was an error.
  auto execute_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();

  absl::InlinedVector<TfrtGpuBuffer::DonationTransaction, 4>
      donation_transactions;

  absl::InlinedVector<TrackedTfrtGpuDeviceBuffer*, 4> tracked_buffers;
  absl::InlinedVector<bool, 4> buffer_is_donated;
  tracked_buffers.reserve(argument_handles.size());
  buffer_is_donated.reserve(argument_handles.size());
  // To avoid clobbering inputs, we must ensure that
  //   `extra_deps` = inputs' definition events + donated inputs' usage events.
  // This also ensures that the returned `execute_event` dominates all inputs'
  // events, and thus output buffer only need to contain `execute_event` as
  // the single definition event.
  std::vector<tsl::RCReference<tsl::AsyncValue>> prepare_input_deps;
  std::vector<tsl::RCReference<tsl::AsyncValue>> input_deps;
  input_deps.reserve(argument_handles.size() + 1);

  // TODO(b/382117736): Support multiple devices.
  CHECK_EQ(parameters_that_must_be_donated_.size(), 1);
  auto donate_it = parameters_that_must_be_donated_[0].begin();

  for (int i = 0; i < argument_handles.size(); ++i) {
    PjRtBuffer* handle = argument_handles[i];
    auto* tfrt_buffer = tsl::down_cast<TfrtGpuBuffer*>(handle);
    if (tfrt_buffer->device() != device) {
      return InvalidArgument(
          "Buffer passed to Execute() as argument %d to replica %d is on "
          "device %s, but replica is assigned to device %s.",
          i, replica, tfrt_buffer->device()->DebugString(),
          device->DebugString());
    }

    bool must_donate = donate_it != parameters_that_must_be_donated_[0].end() &&
                       *donate_it == i;
    TrackedTfrtGpuDeviceBuffer* tracked_buffer = nullptr;
    if (must_donate) {
      ++donate_it;
      TF_ASSIGN_OR_RETURN(auto donation_transaction,
                          tfrt_buffer->AcquireDonation());

      // After acquiring the buffer for donation, we retrieve the dependent
      // usage events. Note that we don't need any locking here as
      // AcquireDonation() is supposed to synchronize with other usages.
      input_deps.push_back(
          donation_transaction.device_buffer()->AfterAllUsageEvents());
      tracked_buffer = donation_transaction.device_buffer();
      tracked_buffers.push_back(tracked_buffer);
      buffer_is_donated.push_back(true);
      donation_transactions.push_back(std::move(donation_transaction));
    } else {
      tracked_buffer = tfrt_buffer->AcquireUsage(execute_event);
      if (!tracked_buffer) {
        return InvalidArgument(
            "Invalid buffer passed: buffer has been deleted or donated.");
      }
      tracked_buffers.push_back(tracked_buffer);
      buffer_is_donated.push_back(false);
    }
    prepare_input_deps.push_back(tracked_buffer->buffer().CopyRCRef());

    // Definition events are never modified after buffer construction. If they
    // are available and have no error, they can be skipped in input deps.
    // In contrast, already known errors in the input are taken as deps so that
    // they can poison output buffers.
    const auto& definition_event = tracked_buffer->definition_event();
    if (!definition_event.IsAvailable() || definition_event.IsError()) {
      VLOG(2) << "definition_event is not available: AsyncValue pointer: "
              << definition_event.GetAsyncValue();
      input_deps.push_back(definition_event.CopyRCRef());
    }
  }

  // Schedule only one collective at a time.
  bool is_a_collective_launch = !!last_collective_launch_event;
  if (is_a_collective_launch) {
    VLOG(2) << "last_collective_launch_event: AsyncValue pointer: "
            << last_collective_launch_event.GetAsyncValue()
            << "; IsAvailable: " << last_collective_launch_event.IsAvailable();
    input_deps.push_back(std::move(last_collective_launch_event));
    device->SetLastCollectiveLaunchEvent(execute_event);
  }

  std::vector<tsl::AsyncValueRef<MaybeOwningGpuMemory>> output_buffers;
  std::vector<std::unique_ptr<PjRtBuffer>> outputs;
  Executable* executable = executables_[0]->executable();
  const Shape& result_shape = executable->result_shape();
  bool untuple_result = options.untuple_result;
  bool result_is_tuple = result_shape.IsTuple();
  if (options.untuple_result && result_shape.IsTuple()) {
    output_buffers.reserve(result_shape.tuple_shapes_size());
    outputs.reserve(output_buffers.size());
    for (int i = 0; i < result_shape.tuple_shapes_size(); ++i) {
      output_buffers.push_back(
          tsl::MakeUnconstructedAsyncValueRef<MaybeOwningGpuMemory>());
      // Program execution writes to output buffers so it's a definition
      // event.
      auto leaf_tracked_device_buffer =
          std::make_unique<TrackedTfrtGpuDeviceBuffer>(
              output_buffers.back().CopyRef(), execute_event.CopyRef());
      VLOG(4) << "created leaf_tracked_device_buffer: "
              << leaf_tracked_device_buffer.get();

      const Shape& shape = result_shape.tuple_shapes(i);
      PjRtMemorySpace* memory_space =
          device->default_memory_space().value_or(nullptr);
      if (shape.has_layout() &&
          shape.layout().memory_space() == Layout::kHostMemorySpace) {
        TF_ASSIGN_OR_RETURN(memory_space, device->memory_space_by_kind_id(
                                              PinnedHostMemorySpace::kKindId));
      }

      auto output = std::make_unique<TfrtGpuBuffer>(
          result_shape.tuple_shapes(i), std::move(leaf_tracked_device_buffer),
          client_, device, memory_space);
      outputs.push_back(std::move(output));
    }
  } else {
    output_buffers.push_back(
        tsl::MakeUnconstructedAsyncValueRef<MaybeOwningGpuMemory>());
    // Program execution writes to output buffers so it's a definition event.
    auto tracked_device_buffer = std::make_unique<TrackedTfrtGpuDeviceBuffer>(
        output_buffers.back().CopyRef(),
        /*definition_event=*/execute_event.CopyRef());
    VLOG(4) << "created tracked_device_buffer: " << tracked_device_buffer.get();

    const Shape& shape = result_shape;
    PjRtMemorySpace* memory_space =
        device->default_memory_space().value_or(nullptr);
    if (shape.has_layout() &&
        shape.layout().memory_space() == Layout::kHostMemorySpace) {
      TF_ASSIGN_OR_RETURN(memory_space, device->memory_space_by_kind_id(
                                            PinnedHostMemorySpace::kKindId));
    }

    auto tfrt_output_buffer = std::make_unique<TfrtGpuBuffer>(
        result_shape, std::move(tracked_device_buffer), client_, device,
        memory_space);
    outputs.push_back(std::move(tfrt_output_buffer));
  }

  // The choice of where we wait is arbitrary; the reason for the wait is
  // pacing to avoid problems such as memory fragmentation and running ahead
  // too far, not for correctness. Placing it before the executable launch
  // allows the inputs for the next executable to be fetched even if the
  // launch is delayed.
  VLOG(0) << "Going to get compute reservation for " << name() << ": "
          << options.launch_id << "; replica: " << replica;
  auto compute_reservation = std::make_unique<Semaphore::ScopedReservation>(
      device->max_inflight_computations_semaphore().ScopedAcquire(1));
  VLOG(0) << "Got compute reservation for " << name() << ": "
          << options.launch_id << "; replica: " << replica;
  auto ffi_context =
      options.context != nullptr ? &options.context->ffi_context() : nullptr;
  auto execute_fn =
      [replica, partition, device, launch_id(options.launch_id), run_id(run_id),
       output_buffers(output_buffers), execute_event(execute_event.CopyRef()),
       untuple_result(untuple_result), result_is_tuple(result_is_tuple),
       compute_reservation(std::move(compute_reservation)),
       donation_transactions(std::move(donation_transactions)),
       parameter_shapes(on_device_executable_parameter_shapes_[0]),
       gpu_executable(executables_[0]), device_assignment(device_assignment),
       executable_name(name()), ffi_context(ffi_context),
       inputs_avs(CopyAsyncValues(input_deps)),
       client = client_](std::vector<ExecutionInput> execution_inputs) mutable {
        VLOG(0) << "execute_fn for " << executable_name << ": " << launch_id
                << "; replica: " << replica;
        tsl::profiler::TraceMe traceme("execute_fn");
        auto set_error = [&](absl::Status status) {
          for (auto& output_buffer : output_buffers) {
            output_buffer.SetError(status);
          }
          execute_event.SetError(status);
        };

        for (const auto& av : inputs_avs) {
          if (auto* error = av->GetErrorIfPresent()) {
            set_error(*error);
            return;
          }
        }

        absl::StatusOr<BoundedStreamPool::Handle> stream_or =
            device->compute_stream_pool().Borrow();
        if (!stream_or.ok()) {
          set_error(stream_or.status());
          return;
        }

        BoundedStreamPool::Handle stream = std::move(stream_or.value());
        ExecutableRunOptions run_options;
        run_options.set_stream(stream.get());
        run_options.set_host_to_device_stream(stream.get());
        run_options.set_device_to_host_stream(stream.get());
        run_options.set_allocator(device->se_allocator_.get());
        run_options.set_device_assignment(device_assignment.get());

        if (launch_id != 0) {
          run_options.set_run_id(RunId(launch_id));
        } else {
          run_options.set_run_id(run_id);
        }
        run_options.set_rng_seed(device->GetNewPrngSeed());
        run_options.set_gpu_executable_run_options(client->gpu_run_options());
        run_options.set_launch_id(launch_id);
        run_options.set_local_device_count(client->device_count());
        run_options.set_device_ordinal(device->local_device_id().value());
        run_options.set_physical_device_ordinal(
            device->local_hardware_id().value());
        run_options.set_ffi_execution_context(ffi_context);
        VLOG(2) << "launch id for " << executable_name << ": "
                << run_options.launch_id();

        // TODO(phawkins): *technically* this should probably happen after
        // calling RunAsync(). But that causes a large performance problem: it
        // prevents the main thread from freeing the buffer objects.
        for (auto& donation_transaction : donation_transactions) {
          VLOG(2) << "Committing donation transaction: "
                  << donation_transaction.device_buffer();
          std::move(donation_transaction).Commit();
        }

        absl::StatusOr<ExecutionOutput> result_buffer_or_status =
            gpu_executable->RunAsync(std::move(execution_inputs), run_options);
        VLOG(0) << "Replica " << replica << " partition " << partition
                << " completed; ok=" << result_buffer_or_status.ok();

        if (!result_buffer_or_status.ok()) {
          set_error(result_buffer_or_status.status());
          return;
        }

        ExecutionOutput& execution_output = result_buffer_or_status.value();
        ScopedShapedBuffer output = execution_output.ConsumeResult();
        if (untuple_result && result_is_tuple) {
          for (int i = 0; i < output_buffers.size(); ++i) {
            ScopedShapedBuffer tuple_buffer = output.TakeSubTree({i});
            auto* elem = tuple_buffer.buffers().mutable_element({});
            VLOG(4) << "untuple: output_buffers[" << i
                    << "].emplace: " << elem->opaque();
            output_buffers[i].emplace(device->allocator(), *elem);
            *elem = se::DeviceMemoryBase();
          }
        } else {
          CHECK_EQ(output_buffers.size(), 1);
          auto* elem = output.buffers().mutable_element({});
          VLOG(4) << "output_buffers[0].emplace: " << elem->opaque();
          output_buffers.front().emplace(device->allocator(), *elem);
          *elem = se::DeviceMemoryBase();
        }

        absl::Status status = stream->BlockHostUntilDone();
        if (!status.ok()) {
          execute_event.SetError(status);
          return;
        }
        execute_event.SetStateConcrete();
      };

  auto prepare_inputs =
      [blocking_thread_pool = client_->blocking_thread_pool(), device,
       tracked_buffers(std::move(tracked_buffers)),
       buffer_is_donated(std::move(buffer_is_donated)),
       prepare_inputs_avs(CopyAsyncValues(prepare_input_deps)),
       execute_event(execute_event.CopyRef()),
       output_buffers(std::move(output_buffers)),
       execute_fn(std::move(execute_fn)), input_deps(std::move(input_deps)),
       parameter_shapes(on_device_executable_parameter_shapes_[0]),
       parameter_is_tupled_arguments(parameter_is_tupled_arguments_),
       arguments_are_tupled(options.arguments_are_tupled),
       input_buffer_sizes_in_bytes(input_buffer_sizes_in_bytes_[0])]() mutable {
        tsl::profiler::TraceMe traceme("prepare_inputs");

        VLOG(2) << "prepare_inputs";
        DCHECK_EQ(tracked_buffers.size(), buffer_is_donated.size());

        auto set_error = [&](absl::Status status) {
          execute_event.SetError(status);
          for (auto& output_buffer : output_buffers) {
            output_buffer.SetError(status);
          }
        };

        for (const auto& av : prepare_inputs_avs) {
          if (auto* error = av->GetErrorIfPresent()) {
            set_error(*error);
            return;
          }
        }

        absl::Status status = CheckBufferCompatibilities(
            *input_buffer_sizes_in_bytes, tracked_buffers);
        if (!status.ok()) {
          set_error(status);
          return;
        }

        std::vector<ExecutionInput> inputs;
        if (parameter_is_tupled_arguments && !arguments_are_tupled) {
          inputs.emplace_back(
              ShapeTree<MaybeOwningDeviceMemory>(&parameter_shapes->front()));
          ExecutionInput& input = inputs.back();
          for (int i = 0; i < tracked_buffers.size(); ++i) {
            if (buffer_is_donated[i]) {
              input.SetUnownedBuffer(
                  {i}, MaybeOwningDeviceMemory(se::OwningDeviceMemory(
                           tracked_buffers[i]->buffer()->buffer(),
                           device->local_hardware_id().value(),
                           device->se_allocator_.get())));
            } else {
              input.SetBuffer({i}, MaybeOwningDeviceMemory(
                                       tracked_buffers[i]->buffer()->buffer()));
            }
          }
        } else {
          inputs.reserve(tracked_buffers.size());
          for (int i = 0; i < tracked_buffers.size(); ++i) {
            inputs.emplace_back(
                ShapeTree<MaybeOwningDeviceMemory>(&(*parameter_shapes)[i]));
            ExecutionInput& input = inputs.back();
            if (buffer_is_donated[i]) {
              input.SetUnownedBuffer(
                  {}, MaybeOwningDeviceMemory(se::OwningDeviceMemory(
                          tracked_buffers[i]->buffer()->buffer(),
                          device->local_hardware_id().value(),
                          device->se_allocator_.get())));
            } else {
              input.SetBuffer({}, MaybeOwningDeviceMemory(
                                      tracked_buffers[i]->buffer()->buffer()));
            }
          }
        }

        VLOG(2) << "prepare_inputs done";
        VLOG(2) << "EnqueueWorkWhenReady; blocking_thread_pool: "
                << blocking_thread_pool;

        EnqueueWorkWhenReady(blocking_thread_pool, input_deps,
                             [execute_fn(std::move(execute_fn)),
                              inputs(std::move(inputs))]() mutable {
                               execute_fn(std::move(inputs));
                             });
      };
  EnqueueWorkWhenReady(client_->non_blocking_thread_pool(), prepare_input_deps,
                       std::move(prepare_inputs));

  // Create output TFRT buffers.
  std::optional<PjRtFuture<>> future;
  if (fill_future) {
    auto promise = PjRtFuture<>::CreatePromise();
    future = PjRtFuture<>(promise);
    execute_event.AndThen([promise = std::move(promise),
                           event = execute_event.CopyRef()]() mutable {
      absl::Status s;
      if (auto* error = event.GetErrorIfPresent()) {
        s = absl::InternalError(
            absl::StrFormat("Compute error: %s", error->message()));
      }
      VLOG(1) << "Setting future: " << s;
      promise.Set(s);
    });
  }
  return Result({/*future=*/std::move(future),
                 /*buffers=*/std::move(outputs)});
}

absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
TfrtGpuExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<PjRtFuture<>>>& returned_futures) {
  RunId run_id(options.launch_id);
  tsl::profiler::TraceMeProducer activity("TfrtGpuExecutable::Execute",
                                          tsl::profiler::ContextType::kPjRt,
                                          run_id.ToInt());
  if (device_assignment_ == nullptr) {
    return InvalidArgument("Execute expects a non-null device_assignment");
  }
  const int num_addressable_devices = addressable_devices_.size();

  if (argument_handles.size() != num_addressable_devices) {
    return InvalidArgument(
        "Attempted to execute with %d argument lists when local device "
        "count is %d (total replica count: %d, partition count: %d)",
        argument_handles.size(), num_addressable_devices, num_replicas(),
        num_partitions());
  }

  VLOG(1) << "Executing computation " << name()
          << "; num_replicas=" << num_replicas()
          << " num_partitions=" << num_partitions()
          << " num_addressable_devices=" << num_addressable_devices;

  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> wrapped_results(
      num_addressable_devices);
  if (returned_futures.has_value()) {
    returned_futures->resize(num_addressable_devices);
  }
  if (num_addressable_devices == 1) {
    // Fast-path if there is only one device — run the computation on the
    // current thread.
    const int replica = addressable_device_logical_ids_[0].replica;
    const int partition = addressable_device_logical_ids_[0].partition;

    // TODO(b/382117736): Dump HLO snapshot.
    // Dump once before running, in case there's a crash.
    // MaybeDumpHloSnapshot(gpu_executable_->module(), run_id,
    //                      argument_handles[0], {});

    auto statusor = ExecuteHelper(
        argument_handles[0], replica, partition, run_id, options,
        /*last_collective_launch_event=*/tsl::AsyncValueRef<GpuEvent>(),
        returned_futures.has_value());

    if (!statusor.ok()) {
      return std::move(statusor).status();
    }

    wrapped_results[0] = std::move(statusor->buffers);
    if (returned_futures.has_value()) {
      (*returned_futures)[0] = std::move(*statusor->future);
    }

    // TODO(b/382117736): Dump HLO snapshot.
    // MaybeDumpHloSnapshot(cpu_executable_->module(), run_id,
    //                      argument_handles[0], wrapped_results[0]);
  } else {
    absl::Mutex mu;
    int running = num_addressable_devices;
    int failed = 0;
    absl::Status first_failure_status;

    for (int i = 0; i < num_addressable_devices; ++i) {
      const int replica = addressable_device_logical_ids_[i].replica;
      const int partition = addressable_device_logical_ids_[i].partition;
      const int device_id = (*device_assignment_)(replica, partition);
      TF_ASSIGN_OR_RETURN(PjRtDevice * pjrt_device,
                          client_->LookupDevice(PjRtGlobalDeviceId(device_id)));
      TfrtGpuDevice* gpu_device =
          tensorflow::down_cast<TfrtGpuDevice*>(pjrt_device);

      VLOG(2) << "ExecuteHelper: " << i << " " << replica << " " << partition
              << " " << device_id << " "
              << gpu_device->local_hardware_id().value();

      // Gang schedule collectives to ensure that collectives with the same
      // RunId are run at the same time. We conservatively run only one
      // collective at a time, because we may not have enough threads to run
      // arbitrary number of collectives concurrently.
      EnqueueWork(
          client_->non_blocking_thread_pool(),
          [this, gpu_device, replica, partition, i, &argument_handles, &run_id,
           &options, &returned_futures, &wrapped_results, &mu, &running,
           &failed, &first_failure_status] {
            auto statusor = ExecuteHelper(
                argument_handles[i], replica, partition, run_id, options,
                gpu_device->GetLastCollectiveLaunchEvent(),
                returned_futures.has_value());
            if (statusor.ok()) {
              wrapped_results[i] = std::move(statusor->buffers);
              if (returned_futures.has_value()) {
                (*returned_futures)[i] = std::move(*statusor->future);
              }
            }

            absl::MutexLock lock(&mu);
            --running;
            if (!statusor.ok()) {
              if (failed == 0) {
                first_failure_status = AppendStatus(
                    std::move(statusor).status(),
                    absl::StrFormat(
                        "while running replica %d and partition %d of a "
                        "replicated computation (other "
                        "replicas may have failed as well).",
                        replica, partition));
              }
              ++failed;
            }
          });
    }

    {
      auto done_running = [&]() {
        mu.AssertHeld();
        return running == 0;
      };
      absl::MutexLock lock(&mu);
      mu.Await(absl::Condition(&done_running));
    }

    if (!first_failure_status.ok()) return first_failure_status;
  }
  VLOG(1) << "Replicated execution complete.";

  return wrapped_results;
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
TfrtGpuExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<PjRtFuture<>>& returned_future,
    bool fill_future) {
  RunId run_id(options.launch_id);
  tsl::profiler::TraceMeProducer activity("TfrtGpuExecutable::ExecuteSharded",
                                          tsl::profiler::ContextType::kPjRt,
                                          run_id.ToInt());
  if (device_assignment_ == nullptr) {
    return InvalidArgument("ExecuteShard expects a non-null device_assignment");
  }
  for (int i = 0; i < addressable_devices_.size(); ++i) {
    if (addressable_devices_[i] == device) {
      VLOG(1) << "ExecuteShard executes computation " << name()
              << " on assigned replica/partition on device "
              << device->DebugString();
      TF_ASSIGN_OR_RETURN(
          auto result,
          ExecuteHelper(
              argument_handles, addressable_device_logical_ids_[i].replica,
              addressable_device_logical_ids_[i].partition, run_id, options,
              tsl::down_cast<TfrtGpuDevice*>(device)
                  ->GetLastCollectiveLaunchEvent(),
              fill_future));
      returned_future = std::move(result.future);
      return std::move(result.buffers);
    }
  }
  return InvalidArgument(
      "ExecuteShard attempted to execute on device id %d which is not "
      "addressable by this client",
      device->global_device_id().value());
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
TfrtGpuExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<PjRtFuture<>>& returned_future,
    bool fill_future) {
  RunId run_id(options.launch_id);
  tsl::profiler::TraceMeProducer activity("TfrtGpuExecutable::ExecutePortable",
                                          tsl::profiler::ContextType::kPjRt,
                                          run_id.ToInt());
  if (device_assignment_ != nullptr) {
    return InvalidArgument("ExecutePortable gets a non-portable executable");
  }
  if (num_replicas() != 1 || num_partitions() != 1) {
    return InvalidArgument(
        "ExecutePortable expects a single-core executable but gets "
        "one with %d replica %d partition",
        num_replicas(), num_partitions());
  }
  if (device == nullptr) {
    return InvalidArgument("ExecutePortable expects a device to be specified");
  }
  VLOG(1) << "ExecutePortable executes single-core portable executable "
          << name();
  TF_ASSIGN_OR_RETURN(
      auto result,
      ExecuteHelper(
          argument_handles,
          /*replica=*/0,
          /*partition=*/0, run_id, options,
          /*last_collective_launch_event=*/tsl::AsyncValueRef<GpuEvent>(),
          fill_future, tsl::down_cast<TfrtGpuDevice*>(device)));
  returned_future = std::move(result.future);
  return std::move(result.buffers);
}

absl::string_view TfrtGpuExecutable::name() const {
  Executable* executable = executables_[0]->executable();
  if (executable->has_module()) {
    return executable->module().name();
  } else {
    return "<unknown executable>";
  }
}

absl::StatusOr<std::vector<std::shared_ptr<HloModule>>>
TfrtGpuExecutable::GetHloModules() const {
  std::vector<std::shared_ptr<HloModule>> modules;
  modules.reserve(executables_.size());
  for (const auto& local_exec : executables_) {
    if (!local_exec->executable()->has_module()) {
      return InvalidArgument("Executable does not have HLO modules.");
    }
    modules.push_back(local_exec->executable()->shared_module());
  }
  return std::move(modules);
}

namespace {

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
  result.reserve(shape.tuple_shapes_size());
  for (const auto& element_shape : shape.tuple_shapes()) {
    TF_ASSIGN_OR_RETURN(
        absl::string_view element_memory_kind,
        MemoryKindFromSimpleShape(element_shape, default_memory_kind));
    result.push_back(element_memory_kind);
  }
  return result;
}

}  // namespace

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
TfrtGpuExecutable::GetOutputMemoryKinds() const {
  TF_ASSIGN_OR_RETURN(auto shapes, GetOutputShapes());
  if (addressable_devices().empty()) {
    return Unimplemented(
        "GetOutputMemoryKinds is not supported when there are no addressable "
        "devices in TfrtGpuExecutable.");
  }
  TF_ASSIGN_OR_RETURN(PjRtMemorySpace * default_memory_space,
                      addressable_devices()[0]->default_memory_space());
  std::vector<std::vector<absl::string_view>> out;
  out.reserve(shapes.size());
  for (const auto& shape : shapes) {
    TF_ASSIGN_OR_RETURN(
        std::vector<absl::string_view> memory_kind,
        MemoryKindsFromShape(shape, default_memory_space->kind()));
    out.push_back(memory_kind);
  }
  return out;
}

absl::Status TfrtGpuExecutable::SetUpDonation(bool tuple_inputs) {
  parameters_that_must_be_donated_.reserve(executables_.size());
  for (auto& executable : executables_) {
    TF_ASSIGN_OR_RETURN(std::vector<int> parameters_to_donate,
                        ComputeParametersThatMustBeDonated(
                            executable->executable()->module(), tuple_inputs));
    parameters_that_must_be_donated_.emplace_back(
        std::move(parameters_to_donate));
  }
  return absl::OkStatus();
}

absl::StatusOr<TfrtGpuBuffer::DonationTransaction>
TfrtGpuBuffer::AcquireDonation() {
  absl::MutexLock lock(&mu_);

  if (tracked_device_buffer_ == nullptr) {
    return InvalidArgument("Donation requested for invalid buffer");
  }

  if (external_reference_counter_ > 0) {
    return InvalidArgument(
        "Donation requested for buffer with external reference");
  }

  CHECK(donation_event_.IsAvailable());
  CHECK(!donation_event_.get());
  donation_event_ = tsl::MakeUnconstructedAsyncValueRef<bool>();

  // Swap out `tracked_device_buffer_` so that no one can acquire a usage
  // event after this point.
  VLOG(1) << "TfrtGpuBuffer::AcquireDonation: " << tracked_device_buffer_.get();
  return DonationTransaction(donation_event_,
                             std::move(tracked_device_buffer_));
}

}  // namespace xla
