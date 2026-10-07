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

#include "xla/pjrt/gpu/tfrt_gpu_client.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
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
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/gpu/gpu_event.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/gpu/gpu_topology.h"
#include "xla/pjrt/gpu/gpu_topology.pb.h"
#include "xla/pjrt/gpu/host_memory_allocator.h"
#include "xla/pjrt/gpu/stream_pool.h"
#include "xla/pjrt/gpu/tracked_tfrt_gpu_device_buffer.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/semaphore.h"
#include "xla/pjrt/transpose.h"
#include "xla/pjrt/utils.h"
#include "xla/primitive_util.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/context_types.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace {

// A RAII helper class used to set an AsyncValueRef<GpuEvent> to a ready state
// upon destruction. In many cases in PjRt implementation, there will be
// multiple return statements in the function, all of which require setting
// some AsyncValueRef<GpuEvent> to be ready. This class could make such code
// more robust by using setting the AsyncValue in the destructor.
class MarkEventReadyOnExit {
 public:
  explicit MarkEventReadyOnExit(tsl::AsyncValueRef<GpuEvent> event)
      : event_(std::move(event)) {}

  MarkEventReadyOnExit(const MarkEventReadyOnExit&) = delete;
  MarkEventReadyOnExit& operator=(const MarkEventReadyOnExit&) = delete;
  MarkEventReadyOnExit(MarkEventReadyOnExit&&) = default;
  MarkEventReadyOnExit& operator=(MarkEventReadyOnExit&&) = default;

  ~MarkEventReadyOnExit() {
    if (event_) event_.SetStateConcrete();
  }

  tsl::AsyncValueRef<GpuEvent> Release() && { return std::move(event_); }

 private:
  tsl::AsyncValueRef<GpuEvent> event_;
};

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
    EnqueueWork(pool, std::move(callee));
  });
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
        buffer_ptrs_(std::move(buffer_ptrs)),
        definition_events_(std::move(definition_events)),
        device_shapes_(std::move(device_shapes)),
        remaining_buffer_count_(buffers_.size()),
        transfers_in_flight_(0),
        device_(device),
        client_(tsl::down_cast<TfrtGpuClient*>(device_->client())) {
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
      CHECK_OK(handle_or.status()) << "Failed to borrow a stream from the pool";
      BoundedStreamPool::Handle stream = std::move(handle_or.value());
      CHECK_NE(stream.get(), nullptr);

      TF_CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(
          stream.get(), literal, shaped_buffer));

      auto status = stream->BlockHostUntilDone();
      CHECK_OK(status) << "Failed to block host until done";

      CleanUp(buffer_index, /*is_last_transfer=*/true, std::move(on_done));
    };
    EnqueueWork(client_->blocking_thread_pool(), std::move(transfer_h2d));
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
    EnqueueWork(client_->blocking_thread_pool(), std::move(copy_to_gpu));
    return absl::OkStatus();
  }

  void SetBufferError(int buffer_index, absl::Status error) override {
    LOG(FATAL) << "SetBufferError not implemented";
  }

  void AddTransferMetadata(const TransferMetadata& meta) override {}

 private:
  void CleanUp(int buffer_index, bool is_last_transfer,
               absl::AnyInvocable<void() &&> on_done) {
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
        definition_events_[buffer_index].back().SetStateConcrete();
        if (remaining_buffer_count_ == 0) {
          VLOG(1) << "TransferLiteralToBuffer for all buffers is done.";
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
  auto* client = device_->client();
  to_string_ = absl::StrFormat("MEMORY_SPACE_%i", id_);
  debug_string_ =
      absl::StrFormat("TfrtGpuMemory(id=%i, process_index=%i, client=%s)", id_,
                      client->process_index(), client->platform_name());
}

const int TfrtGpuHbmMemorySpace::kKindId = []() {
  uint32_t kind_id = tsl::Fingerprint32(TfrtGpuHbmMemorySpace::kKind);
  return static_cast<int>(kind_id);
}();

TfrtGpuHbmMemorySpace::TfrtGpuHbmMemorySpace(int id, PjRtDevice* device)
    : TfrtGpuMemorySpace(id, device, kKind, kKindId) {}

TfrtGpuDevice::TfrtGpuDevice(Options&& options)
    : id_(options.id),
      local_device_id_(options.local_device_id),
      local_hardware_id_(options.local_hardware_id),
      executor_(options.executor),
      stream_pool_(options.executor, options.stream_capacity),
      allocator_(std::move(options.allocator)),
      prng_seed_generator_(prng_seed_device_()),
      prng_seed_distribution_(std::numeric_limits<int>::min(),
                              std::numeric_limits<int>::max()),
      description_(options.id, options.platform_version),
      max_inflight_computations_semaphore_(
          /*capacity=*/options.max_inflight_computations) {
  debug_string_ = absl::StrCat("TFRT_GPU_", id_);
  to_string_ = absl::StrCat("GpuDevice(id=", id_, ")");
  auto stream_or = executor_->CreateStream();
  CHECK_OK(stream_or.status()) << "Failed to create stream";
  compute_stream_ = std::move(stream_or.value());
  se_allocator_ = std::make_unique<se::TfAllocatorAdapter>(
      allocator_.get(), compute_stream_.get());
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
  return memory_space_by_kind_id(TfrtGpuHbmMemorySpace::kKindId);
}

absl::string_view TfrtGpuDevice::DebugString() const { return debug_string_; }

absl::string_view TfrtGpuDevice::ToString() const { return to_string_; }

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

tsl::AsyncValueRef<GpuEvent> TfrtGpuDevice::GetAllocationSequencingEvent() {
  absl::MutexLock lock(&mu_);
  return deallocation_events_
      .AfterAll();  // TODO(b/382117736): This might cause slowness.
}

static absl::StatusOr<std::vector<std::unique_ptr<TfrtGpuDevice>>>
GetTfrtGpuDevices(LocalClient* xla_client) {
  std::vector<std::unique_ptr<TfrtGpuDevice>> devices;
  int i = 0;
  for (se::StreamExecutor* executor :
       xla_client->backend().stream_executors()) {
    // TODO(phawkins): allow GPU allocator parameters to be configurable.
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
    TfrtGpuClient::Options options) {
  LOG(INFO) << "GetTfrtGpuClient";

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
  // TODO(b/382117736): Support multi-host
  gpu_topology_proto.set_num_slices(1);
  gpu_topology_proto.set_num_hosts_per_slice(1);
  gpu_topology_proto.set_num_devices_per_host(devices.size());

  auto gpu_topology = std::shared_ptr<const GpuTopology>(
      GpuTopology::FromProto(gpu_topology_proto));

  return std::unique_ptr<PjRtClient>(std::make_unique<TfrtGpuClient>(
      /*process_index=*/0, xla_client, std::move(devices),
      std::move(host_memory_allocator), gpu_topology));
}

namespace {
static std::optional<stream_executor::GpuTargetConfigProto>
GetTargetConfigForDevices(
    const std::vector<std::unique_ptr<TfrtGpuDevice>>& devices) {
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
  if (devices.empty()) {
    return std::nullopt;
  }
  return xla::Compiler::TargetConfig(devices.front()->executor()).ToProto();
}

static absl::flat_hash_map<std::string, PjRtDeviceAttribute> GetAttrsForDevices(
    const std::vector<std::unique_ptr<TfrtGpuDevice>>& devices) {
  absl::flat_hash_map<std::string, PjRtDeviceAttribute> attrs;
  auto target_config = GetTargetConfigForDevices(devices);
  if (target_config.has_value()) {
    std::string attr;
    if (tsl::protobuf::TextFormat::PrintToString(*target_config, &attr)) {
      attrs["target_config"] = std::move(attr);
    }
  }
  return attrs;
}
}  // namespace

TfrtGpuClient::TfrtGpuClient(
    int process_index, xla::LocalClient* xla_client,
    std::vector<std::unique_ptr<TfrtGpuDevice>> devices,
    std::unique_ptr<tsl::Allocator> host_memory_allocator,
    std::shared_ptr<const GpuTopology> gpu_topology)
    : process_index_(process_index),
      platform_(xla_client->backend().platform()),
      xla_client_(xla_client),
      host_memory_allocator_(std::make_unique<HostMemoryAllocator>(
          std::move(host_memory_allocator))),
      owned_devices_(std::move(devices)),
      computation_placer_(std::make_unique<ComputationPlacer>()),
      compile_thread_pool_(std::make_unique<tsl::thread::ThreadPool>(
          tsl::Env::Default(), "TfrtGpuClient_compile_thread_pool",
          DefaultThreadPoolSize())),
      blocking_thread_pool_(std::make_unique<tsl::thread::ThreadPool>(
          tsl::Env::Default(), "TfrtGpuClient_blocking_thread_pool",
          DefaultThreadPoolSize())),
      non_blocking_thread_pool_(std::make_unique<tsl::thread::ThreadPool>(
          tsl::Env::Default(), "TfrtGpuClient_non_blocking_thread_pool",
          DefaultThreadPoolSize())),
      last_collective_launch_event_(
          tsl::MakeAvailableAsyncValueRef<GpuEvent>()),
      transpose_cache_(1024),
      topology_(tsl::Fingerprint64(xla::CudaName()), xla::CudaName(),
                std::move(gpu_topology), GetAttrsForDevices(owned_devices_),
                GetTargetConfigForDevices(owned_devices_)) {
  for (const std::unique_ptr<TfrtGpuDevice>& device : owned_devices_) {
    devices_.push_back(device.get());
    CHECK(
        id_to_device_.insert({device->global_device_id(), device.get()}).second)
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

    // Initialize the default memory space.
    const int global_device_id = device->global_device_id().value();
    auto memory_space =
        std::make_unique<TfrtGpuHbmMemorySpace>(global_device_id, device.get());
    device->memory_spaces_.push_back(memory_space.get());
    device->memory_spaces_by_kind_id_[memory_space->kind_id()] =
        memory_space.get();
    memory_spaces_.push_back(memory_space.get());
    owned_memory_spaces_.push_back(std::move(memory_space));
  }
  for (int idx = 0; idx < addressable_devices_.size(); ++idx) {
    CHECK(addressable_devices_[idx] != nullptr) << idx;
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

absl::Span<PjRtMemorySpace* const> TfrtGpuClient::memory_spaces() const {
  return memory_spaces_;
}

absl::string_view TfrtGpuClient::platform_version() const {
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

absl::StatusOr<DeviceAssignment> TfrtGpuClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  return computation_placer_->AssignDevices(num_replicas, num_partitions);
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

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> TfrtGpuClient::Compile(
    const XlaComputation& computation, CompileOptions options) {
  tsl::profiler::TraceMe traceme("TfrtGpuClient::Compile");
  auto input_options = options;

  TF_ASSIGN_OR_RETURN(ExecutableExtras extras, GetExecutableExtras(&options));
  std::shared_ptr<DeviceAssignment>& device_assignment =
      extras.device_assignment;
  std::vector<TfrtGpuExecutable::LogicalDeviceIds>&
      addressable_device_logical_ids = extras.addressable_device_logical_ids;
  std::vector<PjRtDevice*>& addressable_devices = extras.addressable_devices;

  // TfrtGpuDevice* device =
  //     tsl::down_cast<TfrtGpuDevice*>(addressable_devices.front());
  // TF_ASSIGN_OR_RETURN(BoundedStreamPool::Handle stream,
  //                     device->stream_pool_.Borrow());

  std::vector<const Shape*> argument_layout_pointers;
  TF_RETURN_IF_ERROR(DetermineArgumentLayoutsFromCompileOptions(
      computation,
      [local_client = xla_client_](Shape shape) {
        return local_client->backend()
            .transfer_manager()
            ->ChooseCompactLayoutForShape(shape);
      },
      options.argument_layouts, &options.executable_build_options,
      &argument_layout_pointers));

  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<LocalExecutable>> local_executables,
      xla_client_->Compile(computation, argument_layout_pointers,
                           options.executable_build_options));

  TF_RET_CHECK(local_executables.size() == 1) << local_executables.size();
  auto executable = std::make_unique<TfrtGpuExecutable>(
      std::move(local_executables.front()),
      options.parameter_is_tupled_arguments, std::move(device_assignment),
      std::move(input_options), std::move(addressable_device_logical_ids),
      std::move(addressable_devices), this);

  TF_RETURN_IF_ERROR(
      executable->SetUpDonation(options.parameter_is_tupled_arguments));

  return std::unique_ptr<PjRtLoadedExecutable>(std::move(executable));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> TfrtGpuClient::Compile(
    mlir::ModuleOp module, CompileOptions options) {
  XlaComputation xla_computation;
  // TODO(b/382117736): Add support for LayoutModesToXlaShapes
  TF_RETURN_IF_ERROR(MlirToXlaComputation(
      module, xla_computation,
      /*use_tuple_args=*/options.parameter_is_tupled_arguments,
      /*return_tuple=*/false,
      options.executable_build_options.use_shardy_partitioner()));
  return Compile(xla_computation, options);
}

absl::StatusOr<std::unique_ptr<TfrtGpuBuffer>> AllocateTfrtGpuDestinationBuffer(
    const Shape& on_device_shape,
    absl::InlinedVector<tsl::AsyncValueRef<GpuEvent>, 4> definition_events,
    TfrtGpuDevice* device, TfrtGpuClient* client,
    PjRtMemorySpace* memory_space) {
  if (!on_device_shape.IsTuple()) {
    size_t byte_size = ShapeUtil::ByteSizeOf(on_device_shape);
    TF_ASSIGN_OR_RETURN(
        auto device_buffer,
        MaybeOwningGpuMemory::AllocateShared(device->allocator(), byte_size));
    auto buffer_async_value_ref =
        tsl::MakeAvailableAsyncValueRef<MaybeOwningGpuMemory>(
            std::move(device_buffer));
    return std::make_unique<TfrtGpuBuffer>(
        on_device_shape,
        std::make_unique<TrackedTfrtGpuDeviceBuffer>(
            buffer_async_value_ref, std::move(definition_events)),
        client, device, memory_space);
  }
  // Tuple case.
  // absl::InlinedVector<std::shared_ptr<MaybeOwningGpuMemory>, 4> buffers;
  // buffers.reserve(on_device_shape.tuple_shapes().size());
  // for (const auto& leaf_shape : on_device_shape.tuple_shapes()) {
  //   size_t byte_size = ShapeUtil::ByteSizeOf(leaf_shape);
  //   TF_ASSIGN_OR_RETURN(
  //       auto device_buffer,
  //       MaybeOwningGpuMemory::AllocateShared(device->allocator(),
  //       byte_size));
  //   auto tracked_device_buffer =
  //   std::make_unique<TrackedTfrtGpuDeviceBuffer>(
  //       std::move(buffers), std::move(definition_events));
  //   buffers.push_back(std::move(device_buffer));
  // }
  // return std::make_unique<TfrtGpuBuffer>(
  //     on_device_shape,
  //     std::make_unique<TrackedTfrtGpuDeviceBuffer>(
  //         /*is_tuple=*/true, std::move(buffers),
  //         std::move(definition_events)),
  //     client, device);
  return Unimplemented(
      "tuple case not implemented for AllocateTfrtGpuDestinationBuffer");
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
TfrtGpuClient::CreateViewOfDeviceBuffer(
    void* device_ptr, const Shape& shape, PjRtMemorySpace* memory_space,
    std::function<void()> on_delete_callback,
    std::optional<std::intptr_t> stream) {
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
      tsl::down_cast<TfrtGpuDevice*>(memory_space->devices()[0]), memory_space);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
TfrtGpuClient::CreateUninitializedBuffer(const Shape& shape,
                                         PjRtMemorySpace* memory_space) {
  tsl::profiler::TraceMe traceme("TfrtGpuClient::CreateUninitializedBuffer");
  VLOG(1) << "TfrtGpuClient::CreateUninitializedBuffer: shape: "
          << shape.DebugString()
          << " memory_space: " << memory_space->DebugString();
  return AllocateTfrtGpuDestinationBuffer(
      shape, /*definition_events=*/{},
      tsl::down_cast<TfrtGpuDevice*>(memory_space->devices()[0]), this,
      memory_space);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
TfrtGpuClient::DeserializeExecutable(absl::string_view serialized,
                                     std::optional<CompileOptions> options) {
  ExecutableAndOptionsProto proto;
  if (serialized.size() > std::numeric_limits<int>::max()) {
    return Internal(
        "TfrtGpuClient::DeserializeExecutable proto too large "
        "(>2GB)");
  }
  if (!proto.ParseFromArray(serialized.data(), serialized.size())) {
    return Internal(
        "TfrtGpuClient::DeserializeExecutable proto "
        "deserialization "
        "failed");
  }

  VLOG(1) << "TfrtGpuClient::DeserializeExecutable";
  VLOG(3) << "TfrtGpuClient::DeserializeExecutable "
             "proto.compile_options(): "
          << proto.compile_options().DebugString();
  VLOG(4) << "TfrtGpuClient::DeserializeExecutable proto: "
          << proto.DebugString();

  CompileOptions compile_options;
  if (options.has_value()) {
    compile_options = *std::move(options);
  } else {
    TF_ASSIGN_OR_RETURN(compile_options,
                        CompileOptions::FromProto(proto.compile_options()));
  }
  auto input_options = compile_options;

  tsl::profiler::TraceMe traceme("TfrtGpuClient::DeserializeExecutable");

  TF_ASSIGN_OR_RETURN(ExecutableExtras extras,
                      GetExecutableExtras(&compile_options));
  std::shared_ptr<DeviceAssignment>& device_assignment =
      extras.device_assignment;
  std::vector<TfrtGpuExecutable::LogicalDeviceIds>&
      addressable_device_logical_ids = extras.addressable_device_logical_ids;
  std::vector<PjRtDevice*>& addressable_devices = extras.addressable_devices;

  std::string str = std::move(*proto.mutable_serialized_executable());
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<LocalExecutable> loaded,
      xla_client_->Load(str, compile_options.executable_build_options));

  auto executable = std::make_unique<TfrtGpuExecutable>(
      std::move(loaded), compile_options.parameter_is_tupled_arguments,
      std::move(device_assignment), std::move(input_options),
      std::move(addressable_device_logical_ids), std::move(addressable_devices),
      this);

  TF_RETURN_IF_ERROR(
      executable->SetUpDonation(compile_options.parameter_is_tupled_arguments));
  return std::unique_ptr<PjRtLoadedExecutable>(std::move(executable));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
TfrtGpuClient::LoadSerializedExecutable(absl::string_view serialized,
                                        std::optional<CompileOptions> options,
                                        const LoadOptions& load_options) {
  return DeserializeExecutable(serialized, options);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> TfrtGpuClient::CreateErrorBuffer(
    absl::Status error, const Shape& shape, PjRtMemorySpace* memory) {
  if (memory->client() != this) {
    return absl::InvalidArgumentError(
        "Memory space is not attached to this client");
  }
  auto* device = memory->devices()[0];
  VLOG(1) << "PjRtStreamExecutorClient::CreateErrorBuffer: shape: "
          << shape.ToString() << " device: " << device->DebugString()
          << " error: " << error;
  auto dummy_gpu_buffer = MaybeOwningGpuMemory(se::DeviceMemoryBase());
  auto buffer_async_value_ref =
      tsl::MakeAvailableAsyncValueRef<MaybeOwningGpuMemory>(
          std::move(dummy_gpu_buffer));
  auto tracked_device_buffer = std::make_unique<TrackedTfrtGpuDeviceBuffer>(
      std::move(buffer_async_value_ref),
      /*definition_event=*/tsl::MakeAvailableAsyncValueRef<GpuEvent>());
  return std::make_unique<TfrtGpuBuffer>(
      shape, std::move(tracked_device_buffer), this,
      tsl::down_cast<TfrtGpuDevice*>(device), memory);
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
    if (!build_options.device_allocator()) {
      build_options.set_device_allocator(
          xla_client_->backend().memory_allocator());
    }
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

absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
TfrtGpuClient::CreateBuffersForAsyncHostToDevice(
    absl::Span<const Shape> shapes, PjRtMemorySpace* memory_space) {
  absl::InlinedVector<PjRtClient::ShapeSpec, 4> shape_specs;
  shape_specs.reserve(shapes.size());
  for (const auto& shape : shapes) {
    shape_specs.emplace_back(PjRtClient::ShapeSpec{
        shape.element_type(),
        DimensionVector(shape.dimensions().begin(), shape.dimensions().end())});
  }
  return TfrtGpuAsyncHostToDeviceTransferManager::Create(
      shape_specs, /*device_layouts=*/std::nullopt,
      tsl::down_cast<TfrtGpuDevice*>(memory_space->devices()[0]), this,
      memory_space);
}

absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
TfrtGpuClient::CreateBuffersForAsyncHostToDevice(
    absl::Span<const PjRtClient::ShapeSpec> shape_specs,
    std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
    PjRtMemorySpace* memory_space) {
  return TfrtGpuAsyncHostToDeviceTransferManager::Create(
      shape_specs, device_layouts,
      tsl::down_cast<TfrtGpuDevice*>(memory_space->devices()[0]), this,
      memory_space);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> TfrtGpuClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    PjRtMemorySpace* memory_space, const Layout* device_layout) {
  // TODO(b/382117736): support device_layout
  // TODO(b/382117736): support non-default memory_space
  PjRtDevice* device = memory_space->devices()[0];
  tsl::profiler::TraceMe traceme("TfrtGpuClient::BufferFromHostBuffer");
  VLOG(2) << "TfrtGpuClient::BufferFromHostBuffer: "
          << " device: " << device->DebugString();
  Shape device_shape = ShapeUtil::MakeShape(type, dims);
  VLOG(1) << "PjRtStreamExecutorClient::BufferFromHostBuffer: shape: "
          << device_shape.ToString() << " device: " << device->DebugString();

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

  // TODO(b/382117736): support SubByteNonPredType

  auto* gpu_device = tsl::down_cast<TfrtGpuDevice*>(device);

  auto gpu_buffer = tsl::MakeUnconstructedAsyncValueRef<MaybeOwningGpuMemory>();
  absl::InlinedVector<tsl::AsyncValueRef<GpuEvent>, 4> definition_events;
  tsl::AsyncValueRef<GpuEvent> copy_event =
      tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  definition_events.push_back(copy_event.CopyRef());
  auto sequencing_event = gpu_device->GetAllocationSequencingEvent();

  const bool should_sync_copy =
      host_buffer_semantics == HostBufferSemantics::kImmutableOnlyDuringCall;

  auto copy_to_staging_buffer =
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
          on_done_with_host_buffer = nullptr;
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
    tsl::BlockUntilReady(sequencing_event);
    if (sequencing_event.IsError()) {
      return sequencing_event.GetError();
    }
    copy_to_staging_buffer();
  } else {
    EnqueueWorkWhenReady(non_blocking_thread_pool_.get(),
                         {sequencing_event.CopyRCRef()},
                         std::move(copy_to_staging_buffer));
  }

  std::function<void()> on_delete_callback;
  auto tracked_device_buffer = std::make_unique<TrackedTfrtGpuDeviceBuffer>(
      std::move(gpu_buffer), std::move(definition_events),
      std::move(on_delete_callback));

  return std::unique_ptr<PjRtBuffer>(std::make_unique<TfrtGpuBuffer>(
      device_shape, std::move(tracked_device_buffer), this, gpu_device,
      memory_space));
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
TfrtGpuClient::BufferFromHostLiteral(const LiteralSlice& literal,
                                     PjRtMemorySpace* memory_space) {
  PjRtDevice* device = memory_space->devices()[0];
  tsl::profiler::TraceMe traceme("TfrtGpuClient::BufferFromHostLiteral");
  VLOG(1) << "TfrtGpuClient::BufferFromHostLiteral: shape: "
          << literal.shape().DebugString()
          << " device: " << device->DebugString();
  const Shape& shape = literal.shape();

  // Add a placeholder definition event for each leaf buffer when creating the
  // buffer. They are set only after h2d dispatch.
  absl::InlinedVector<tsl::AsyncValueRef<GpuEvent>, 4> definition_events;
  absl::InlinedVector<tsl::RCReference<tsl::AsyncValue>, 4> avs;
  int num_leaf_buffers = shape.IsTuple() ? shape.tuple_shapes_size() : 1;
  for (int i = 0; i < num_leaf_buffers; ++i) {
    tsl::AsyncValueRef<GpuEvent> definition_event =
        tsl::MakeConstructedAsyncValueRef<GpuEvent>();
    definition_events.push_back(definition_event.CopyRef());
    avs.push_back(std::move(definition_event));
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TfrtGpuBuffer> output_buffer,
      AllocateTfrtGpuDestinationBuffer(shape, std::move(definition_events),
                                       tsl::down_cast<TfrtGpuDevice*>(device),
                                       this, memory_space));

  auto usage_event = tsl::MakeAvailableAsyncValueRef<GpuEvent>();
  auto* device_buffer = output_buffer->AcquireUsage(std::move(usage_event));
  CHECK(device_buffer);
  if (!shape.IsTuple()) {
    // It is OK to capture `buffer` pointer because the `output_buffer` can't
    // be deleted until all the usage holds have gone away.
    EnqueueWork(non_blocking_thread_pool_.get(),
                [literal, av = avs[0], device_buffer, shape, this,
                 device = tsl::down_cast<TfrtGpuDevice*>(device)]() mutable {
                  tsl::profiler::TraceMe traceme("H2D Dispatch");
                  TransferManager* transfer_manager =
                      xla_client()->backend().transfer_manager();

                  absl::StatusOr<BoundedStreamPool::Handle> handle_or =
                      device->stream_pool().Borrow();
                  CHECK_OK(handle_or.status())
                      << "Failed to borrow a stream from the pool";
                  BoundedStreamPool::Handle stream =
                      std::move(handle_or.value());

                  const auto& b = device_buffer->buffer();
                  CHECK_EQ(literal.size_bytes(), b->size());

                  ShapedBuffer shaped_buffer = b->AsShapedBuffer(shape, device);

                  CHECK_NE(stream.get(), nullptr);
                  TF_CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(
                      stream.get(), literal, shaped_buffer));

                  auto status = stream->BlockHostUntilDone();
                  CHECK_OK(status) << "Failed to block host until done";

                  av->SetStateConcrete();
                });
  } else {
    return Unimplemented(
        "Tuple case is not supported in TfrtGpuClient::BufferFromHostLiteral");
  }
  return std::unique_ptr<PjRtBuffer>(std::move(output_buffer));
}

TfrtGpuBuffer::TfrtGpuBuffer(
    Shape on_device_shape,
    std::unique_ptr<TrackedTfrtGpuDeviceBuffer> tracked_device_buffer,
    TfrtGpuClient* client, TfrtGpuDevice* device, PjRtMemorySpace* memory_space)
    : client_(client),
      on_device_shape_(std::move(on_device_shape)),
      device_(device),
      memory_space_(memory_space),
      tracked_device_buffer_(std::move(tracked_device_buffer)) {
  if (memory_space_ == nullptr) {
    LOG(FATAL) << "memory_space_ is null when initializing TfrtGpuBuffer";
  }
}

TfrtGpuBuffer::~TfrtGpuBuffer() {
  Delete();
  CHECK_EQ(external_reference_counter_, 0);
}

absl::StatusOr<size_t> TfrtGpuBuffer::GetOnDeviceSizeInBytes() const {
  return ShapeUtil::ByteSizeOf(on_device_shape_);
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

void TfrtGpuBuffer::CommitDonation() {
  absl::MutexLock lock(&mu_);
  CHECK(pending_donation_);
  CHECK(!tracked_device_buffer_);
  pending_donation_ = false;
}

void TfrtGpuBuffer::AbortDonation(
    std::unique_ptr<TrackedTfrtGpuDeviceBuffer> device_buffer) {
  absl::MutexLock lock(&mu_);
  CHECK(pending_donation_);
  CHECK(!tracked_device_buffer_);
  pending_donation_ = false;
  tracked_device_buffer_ = std::move(device_buffer);
}

void TfrtGpuBuffer::Delete() {
  tsl::profiler::TraceMe traceme("Gpu buffer delete");
  auto device_buffer = ReleaseBufferLocked();
  if (device_buffer == nullptr) return;

  // Now that all holds have completed and no more can be added, we can get
  // the final set of usage events.
  tsl::AsyncValueRef<GpuEvent> usage_event =
      device_buffer->LockUseAndTransferUsageEvents();

  std::vector<tsl::AsyncValue*> event_avs;
  event_avs.reserve(2);
  event_avs.push_back(usage_event.GetAsyncValue());
  // We should also wait for the definition event.
  event_avs.push_back(device_buffer->definition_event().GetAsyncValue());

  {
    tsl::profiler::TraceMe traceme2("add deallocation event");
    absl::MutexLock lock(&device_->mu_);
    device_->deallocation_events_.Add(device_buffer->deallocation_event());
  }

  tsl::profiler::TraceMe traceme3("runwhenready");
  tsl::RunWhenReady(event_avs, [device_buffer = std::move(device_buffer),
                                usage_event(std::move(usage_event))]() mutable {
    VLOG(3) << "device_buffer is being deleted: " << device_buffer.get();
    device_buffer.reset();
  });
}

bool TfrtGpuBuffer::IsDeleted() {
  absl::MutexLock lock(&mu_);
  return tracked_device_buffer_ == nullptr;
}

std::unique_ptr<TrackedTfrtGpuDeviceBuffer>
TfrtGpuBuffer::ReleaseBufferLocked() {
  absl::MutexLock lock(&mu_);
  auto condition = [this]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    return !pending_donation_;
  };
  mu_.Await(absl::Condition(&condition));
  return std::move(tracked_device_buffer_);
}

absl::StatusOr<std::unique_ptr<TrackedTfrtGpuDeviceBuffer>>
TfrtGpuBuffer::Release(bool wait_for_operations_to_complete) {
  std::unique_ptr<TrackedTfrtGpuDeviceBuffer> device_buffer =
      ReleaseBufferLocked();
  if (device_buffer == nullptr) return {nullptr};

  absl::InlinedVector<tsl::AsyncValueRef<GpuEvent>, 4> events;
  // Now that all holds have completed and no more can be added, we can get
  // the final set of usage events.
  events.push_back(device_buffer->LockUseAndTransferUsageEvents());
  events.push_back(device_buffer->definition_event().CopyRef());

  if (wait_for_operations_to_complete) {
    // Block the host until all usage events have completed. Usage events
    // dominate definition events, so this also waits for the buffer to be
    // defined. Return the first error encountered.
    absl::Status first_error;
    for (const auto& av : events) {
      tsl::BlockUntilReady(av);
      if (auto* error = av.GetErrorIfPresent()) {
        first_error.Update(absl::InternalError(
            absl::StrFormat("Error Execute: %s", error->message())));
      }
    }
    if (!first_error.ok()) return std::move(first_error);
  }

  return device_buffer;
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

  CHECK(!pending_donation_);
  pending_donation_ = true;

  // Swap out `tracked_device_buffer_` so that no one can acquire a usage
  // event after this point.
  VLOG(1) << "TfrtGpuBuffer::AcquireDonation: " << tracked_device_buffer_.get();
  return DonationTransaction(this, std::move(tracked_device_buffer_));
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
  MarkEventReadyOnExit ready_on_exit(std::move(usage_event));

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
  return ret_shape;
}

static std::vector<tsl::RCReference<tsl::AsyncValue>> CopyAsyncValues(
    absl::Span<const tsl::RCReference<tsl::AsyncValue>> events) {
  std::vector<tsl::RCReference<tsl::AsyncValue>> avs;
  avs.reserve(events.size());
  for (const auto& ev : events) {
    avs.push_back(ev);
  }
  return avs;
}

PjRtFuture<> TfrtGpuBuffer::ToLiteral(MutableLiteralBase* literal) {
  tsl::profiler::TraceMe traceme("TfrtGpuBuffer::ToLiteral");
  auto promise = PjRtFuture<>::CreatePromise();
  auto usage_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto* device_buffer = AcquireUsage(usage_event);
  if (device_buffer == nullptr) {
    return PjRtFuture<>(InvalidArgument(
        "CopyToHostAsync() called on deleted or donated buffer"));
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
            client->host_memory_allocator_->Allocate(byte_size);

        {
          tsl::profiler::TraceMe traceme2("D2H GPU copy");
          MarkEventReadyOnExit ready_on_exit(std::move(usage_event));
          absl::StatusOr<BoundedStreamPool::Handle> handle_or =
              device->stream_pool_.Borrow();
          if (!handle_or.ok()) {
            promise.Set(handle_or.status());
            return;
          }

          BoundedStreamPool::Handle stream = std::move(handle_or.value());
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

absl::StatusOr<std::unique_ptr<PjRtBuffer>> TfrtGpuBuffer::CopyToMemorySpace(
    PjRtMemorySpace* dst_memory_space) {
  // TODO(b/382117736): Support non-default memory spaces.
  tsl::profiler::TraceMe traceme("TfrtGpuBuffer::CopyToMemorySpace");
  PjRtDevice* dst_device = dst_memory_space->devices()[0];
  // TODO(zhangqiaorjc): Remove this restriction after removing the test that
  // explicitly asserts this.
  if (dst_device == device_) {
    return InvalidArgument(
        "CopyToMemorySpace cannot accept the same source and destination "
        "devices");
  }

  // Copying across PjRtClients involves a copy through the host.
  if (dst_device->client() != client_) {
    TF_ASSIGN_OR_RETURN(std::shared_ptr<Literal> literal, ToLiteralSync());
    // Avoid use-after-free on `literal` due to unsequenced move and use.
    Literal* literal_pointer = literal.get();
    absl::InlinedVector<int64_t, 4> byte_strides(
        literal->shape().dimensions_size());
    TF_RETURN_IF_ERROR(
        ShapeUtil::ByteStrides(literal->shape(), absl::MakeSpan(byte_strides)));
    return dst_device->client()->BufferFromHostBuffer(
        literal_pointer->untyped_data(),
        literal_pointer->shape().element_type(),
        literal_pointer->shape().dimensions(), byte_strides,
        TfrtGpuClient::HostBufferSemantics::kImmutableZeroCopy,
        [literal{std::move(literal)}]() { /* frees literal */ },
        dst_memory_space,
        /*device_layout=*/nullptr);
  }

  // Copy each leaf buffer to a destination buffer.
  auto usage_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto* src_device_buffer = AcquireUsage(usage_event);
  if (src_device_buffer == nullptr) {
    return InvalidArgument(
        "CopyToMemorySpace called on deleted or donated buffer");
  }

  TfrtGpuDevice* gpu_dst_device = tsl::down_cast<TfrtGpuDevice*>(dst_device);
  auto sequencing_event = gpu_dst_device->GetAllocationSequencingEvent();
  tsl::AsyncValueRef<MaybeOwningGpuMemory> src_buffer =
      src_device_buffer->buffer();
  auto dst_buffer = tsl::MakeUnconstructedAsyncValueRef<MaybeOwningGpuMemory>();
  auto dst_definition_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();

  auto copy_task = [src_buffer(src_buffer.CopyRef()),
                    dst_buffer(dst_buffer.CopyRef()),
                    dst_definition_event(dst_definition_event.CopyRef()),
                    dst_device(gpu_dst_device),
                    usage_event(usage_event.CopyRef())]() {
    tsl::profiler::TraceMe traceme("D2D copy");
    if (auto* error = dst_definition_event.GetErrorIfPresent()) {
      dst_buffer.SetError(*error);
      dst_definition_event.SetError(*error);
      usage_event.SetStateConcrete();
      return;
    }

    MarkEventReadyOnExit ready_on_exit(std::move(usage_event));
    auto dst_buffer_or = MaybeOwningGpuMemory::AllocateShared(
        dst_device->allocator(), src_buffer->buffer().size());
    if (!dst_buffer_or.ok()) {
      dst_buffer.SetError(dst_buffer_or.status());
      dst_definition_event.SetError(dst_buffer_or.status());
      return;
    }
    dst_buffer.emplace(std::move(dst_buffer_or.value()));

    absl::StatusOr<BoundedStreamPool::Handle> handle_or =
        dst_device->stream_pool_.Borrow();
    if (!handle_or.ok()) {
      dst_definition_event.SetError(handle_or.status());
      return;
    }
    BoundedStreamPool::Handle stream = std::move(handle_or.value());
    se::DeviceMemoryBase dst(dst_buffer->buffer());
    absl::Status status =
        stream->Memcpy(&dst, src_buffer->buffer(), src_buffer->buffer().size());
    if (!status.ok()) {
      dst_definition_event.SetError(status);
      return;
    }
    status = stream->BlockHostUntilDone();
    if (status.ok()) {
      dst_definition_event.SetStateConcrete();
    } else {
      dst_definition_event.SetError(status);
    }
  };

  EnqueueWorkWhenReady(client_->blocking_thread_pool(),
                       {src_device_buffer->definition_event().CopyRCRef(),
                        sequencing_event.CopyRCRef()},
                       std::move(copy_task));
  return std::unique_ptr<PjRtBuffer>(std::make_unique<TfrtGpuBuffer>(
      on_device_shape_,
      std::make_unique<TrackedTfrtGpuDeviceBuffer>(
          std::move(dst_buffer), std::move(dst_definition_event)),
      client(), tsl::down_cast<TfrtGpuDevice*>(dst_device), dst_memory_space));
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
                      << definition_event.GetError().message();
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

TfrtGpuExecutable::TfrtGpuExecutable(
    std::unique_ptr<LocalExecutable> gpu_executable,
    bool parameter_is_tupled_arguments,
    std::shared_ptr<DeviceAssignment> device_assignment,
    CompileOptions compile_options,
    std::vector<LogicalDeviceIds> addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices, TfrtGpuClient* client)
    : client_(client),
      gpu_executable_(std::move(gpu_executable)),
      device_assignment_(std::move(device_assignment)),
      parameter_is_tupled_arguments_(parameter_is_tupled_arguments),
      addressable_device_logical_ids_(
          std::move(addressable_device_logical_ids)),
      addressable_devices_(std::move(addressable_devices)),
      compile_options_(std::move(compile_options)) {
  const auto& computation_layout =
      gpu_executable_->executable()->module().entry_computation_layout();
  input_buffer_sizes_in_bytes_ = std::make_shared<std::vector<int64_t>>();
  // Assume compiled program expects either many non-tupled arguments or a
  // singled tupled argument, or no arguments. Nested tuple is not yet
  // supported.
  if (computation_layout.parameter_count() > 0) {
    if (computation_layout.parameter_count() > 1 ||
        !computation_layout.parameter_shape(0).IsTuple()) {
      input_buffer_sizes_in_bytes_->reserve(
          computation_layout.parameter_count());
      for (int i = 0; i < computation_layout.parameter_count(); ++i) {
        input_buffer_sizes_in_bytes_->push_back(
            ShapeUtil::ByteSizeOf(computation_layout.parameter_shape(i)));
      }
    } else {
      input_buffer_sizes_in_bytes_->reserve(
          computation_layout.parameter_shape(0).tuple_shapes_size());
      for (int i = 0;
           i < computation_layout.parameter_shape(0).tuple_shapes_size(); ++i) {
        input_buffer_sizes_in_bytes_->push_back(ShapeUtil::ByteSizeOf(
            computation_layout.parameter_shape(0).tuple_shapes(i)));
      }
    }
  }
  tsl::Fprint128 fingerprint = tsl::Fingerprint128(fingerprint_);
  fingerprint = tsl::FingerprintCat128(
      fingerprint,
      tsl::Fingerprint128(gpu_executable_->executable()->module().ToString()));
  fingerprint_ = absl::StrCat(fingerprint.low64, fingerprint.high64);

  parameter_shapes_ = std::make_shared<std::vector<Shape>>();
  parameter_shapes_->reserve(computation_layout.parameter_count());
  for (int i = 0; i < computation_layout.parameter_count(); ++i) {
    parameter_shapes_->push_back(computation_layout.parameter_shape(i));
  }
}

void TfrtGpuExecutable::Delete() {}

bool TfrtGpuExecutable::IsDeleted() { return false; }

absl::StatusOr<std::string> TfrtGpuExecutable::SerializeExecutable() const {
  if (gpu_executable_ == nullptr) {
    return absl::FailedPreconditionError("gpu_executable_ is null");
  }
  Executable* built_executable = gpu_executable_->executable();
  Compiler* compiler = client_->xla_client()->backend().compiler();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<AotCompilationResult> aot_result,
                      compiler->Export(built_executable));
  TF_ASSIGN_OR_RETURN(std::string serialized, aot_result->SerializeAsString());
  if (serialized.empty()) {
    return Internal(
        "TfrtGpuExecutable::SerializeExecutable proto serialization "
        "failed");
  }
  ExecutableAndOptionsProto proto;
  *proto.mutable_serialized_executable() = std::move(serialized);
  TF_ASSIGN_OR_RETURN(*proto.mutable_compile_options(),
                      compile_options_.ToProto());
  return proto.SerializeAsString();
}

absl::Status TfrtGpuExecutable::SetUpDonation(bool tuple_inputs) {
  TF_ASSIGN_OR_RETURN(
      parameters_that_must_be_donated_,
      ComputeParametersThatMustBeDonated(
          *gpu_executable_->executable()->shared_module(), tuple_inputs));
  return absl::OkStatus();
}

// Checks that the input buffers passed in by the user have the correct size
// on device for the compiled program.
static absl::Status CheckBufferCompatibilities(
    absl::Span<int64_t const> input_buffer_sizes_in_bytes,
    absl::Span<TrackedTfrtGpuDeviceBuffer* const> input_buffers) {
  if (input_buffers.size() != input_buffer_sizes_in_bytes.size()) {
    LOG(FATAL) << "Execution supplied " << input_buffers.size()
               << " buffers but compiled program "
               << "expected: " << input_buffer_sizes_in_bytes.size()
               << " buffers";
    return InvalidArgument(
        "Execution supplied %lld buffers but compiled program expected %lld "
        "buffers",
        input_buffers.size(), input_buffer_sizes_in_bytes.size());
  }
  for (int i = 0; i < input_buffers.size(); ++i) {
    const auto& buffer = input_buffers[i];
    if (input_buffer_sizes_in_bytes[i] != buffer->buffer()->size()) {
      LOG(FATAL) << "Executable expected parameter " << i << " of size "
                 << input_buffer_sizes_in_bytes[i] << " but got buffer with"
                 << " incompatible size " << buffer->buffer()->size()
                 << "; TrackedTfrtGpuDeviceBuffer*: " << buffer;
      return InvalidArgument(
          "Executable expected parameter %d of size %lld but got buffer with"
          " incompatible size %lld ",
          i, input_buffer_sizes_in_bytes[i], buffer->buffer()->size());
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<PjRtLoadedExecutable::Result> TfrtGpuExecutable::ExecuteHelper(
    absl::Span<PjRtBuffer* const> argument_handles, int replica, int partition,
    const RunId& run_id, const ExecuteOptions& options,
    tsl::AsyncValueRef<GpuEvent> last_collective_launch_event, bool fill_future,
    TfrtGpuDevice* device) {
  tsl::profiler::TraceMe traceme("TfrtGpuExecutable::ExecuteHelper");

  std::shared_ptr<DeviceAssignment> device_assignment;
  if (device == nullptr) {
    CHECK(device_assignment_ != nullptr);
    const int device_id = (*device_assignment_)(replica, partition);
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
  //   `extra_deps` = inputs' definition events + donated inputs' usage
  //   events.
  // This also ensures that the returned `execute_event` dominates all inputs'
  // events, and thus output buffer only need to contain `execute_event` as
  // the single definition event.
  std::vector<tsl::RCReference<tsl::AsyncValue>> prepare_input_deps;
  std::vector<tsl::RCReference<tsl::AsyncValue>> input_deps;
  input_deps.reserve(argument_handles.size() + 2);

  input_deps.push_back(device->GetAllocationSequencingEvent().CopyRCRef());

  auto donate_it = parameters_that_must_be_donated_.begin();

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

    bool must_donate =
        donate_it != parameters_that_must_be_donated_.end() && *donate_it == i;
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

    // Definition events are never modified after buffer construction.
    const auto& definition_event = tracked_buffer->definition_event();
    if (!definition_event.IsAvailable()) {
      input_deps.push_back(definition_event.CopyRCRef());
    }
  }

  // Schedule only one collective at a time.
  bool is_a_collective_launch = !!last_collective_launch_event;
  if (is_a_collective_launch) {
    input_deps.push_back(std::move(last_collective_launch_event));
  }

  std::vector<tsl::AsyncValueRef<MaybeOwningGpuMemory>> output_buffers;
  std::vector<std::unique_ptr<PjRtBuffer>> outputs;
  Executable* executable = gpu_executable_->executable();
  gpu::GpuExecutable* gpu_executable =
      tsl::down_cast<gpu::GpuExecutable*>(executable);
  const Shape& result_shape = gpu_executable->output_shape();
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
      if (shape.layout().memory_space() == Layout::kHostMemorySpace) {
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
    if (shape.layout().memory_space() == Layout::kHostMemorySpace) {
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
  auto compute_reservation = std::make_unique<Semaphore::ScopedReservation>(
      device->max_inflight_computations_semaphore().ScopedAcquire(1));

  auto execute_fn =
      [replica, partition, device, launch_id(options.launch_id), run_id(run_id),
       output_buffers(output_buffers), execute_event(execute_event.CopyRef()),
       untuple_result(untuple_result), result_is_tuple(result_is_tuple),
       compute_reservation(std::move(compute_reservation)),
       donation_transactions(std::move(donation_transactions)),
       parameter_shapes(parameter_shapes_), gpu_executable(gpu_executable_),
       device_assignment(device_assignment), executable_name(name()),
       client = client_](std::vector<ExecutionInput> execution_inputs) mutable {
        tsl::profiler::TraceMe traceme("execute_fn");
        absl::StatusOr<BoundedStreamPool::Handle> stream_or =
            device->stream_pool_.Borrow();
        if (!stream_or.ok()) {
          for (auto& output_buffer : output_buffers) {
            output_buffer.SetError(stream_or.status());
          }
          execute_event.SetError(stream_or.status());
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
        if (run_options.launch_id() != 0) {
          VLOG(3) << "launch id for " << executable_name << ": "
                  << run_options.launch_id();
        }

        // TODO(phawkins): *technically* this should probably happen after
        // calling RunAsync(). But that causes a large performance problem: it
        // prevents the main thread from freeing the buffer objects.
        for (auto& donation_transaction : donation_transactions) {
          std::move(donation_transaction).Commit();
        }

        absl::StatusOr<ExecutionOutput> result_buffer_or_status =
            gpu_executable->RunAsync(std::move(execution_inputs), run_options);
        VLOG(1) << "Replica " << replica << " partition " << partition
                << " completed; ok=" << result_buffer_or_status.ok();

        if (!result_buffer_or_status.ok()) {
          for (auto& output_buffer : output_buffers) {
            output_buffer.SetError(result_buffer_or_status.status());
          }
          execute_event.SetError(result_buffer_or_status.status());
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

  std::vector<tsl::RCReference<tsl::AsyncValue>> prepare_inputs_avs_copy =
      CopyAsyncValues(prepare_input_deps);

  auto prepare_inputs =
      [blocking_thread_pool = client_->blocking_thread_pool(), device,
       tracked_buffers(std::move(tracked_buffers)),
       buffer_is_donated(std::move(buffer_is_donated)),
       prepare_inputs_avs(std::move(prepare_inputs_avs_copy)),
       execute_event(execute_event.CopyRef()),
       output_buffers(std::move(output_buffers)),
       execute_fn(std::move(execute_fn)), input_deps(std::move(input_deps)),
       parameter_shapes(parameter_shapes_),
       parameter_is_tupled_arguments(parameter_is_tupled_arguments_),
       input_buffer_sizes_in_bytes(input_buffer_sizes_in_bytes_)]() mutable {
        tsl::profiler::TraceMe traceme("prepare_inputs");
        DCHECK_EQ(tracked_buffers.size(), buffer_is_donated.size());

        for (const auto& av : prepare_inputs_avs) {
          if (auto* error = av->GetErrorIfPresent()) {
            absl::Status status = *error;
            //  absl::InternalError(absl::StrCat(
            //     "Error dispatching computation: ", error->message()));
            execute_event.SetError(status);
            for (auto& output_buffer : output_buffers) {
              output_buffer.SetError(status);
            }
            return;
          }
        }

        absl::Status status = CheckBufferCompatibilities(
            *input_buffer_sizes_in_bytes, tracked_buffers);
        if (!status.ok()) {
          execute_event.SetError(status);
          for (auto& output_buffer : output_buffers) {
            output_buffer.SetError(status);
          }
          return;
        }

        std::vector<ExecutionInput> inputs;
        if (parameter_is_tupled_arguments) {
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
  tsl::profiler::TraceMe traceme("TfrtGpuExecutable::Execute");
  if (device_assignment_ == nullptr) {
    return InvalidArgument("Execute expects a non-null device_assignment");
  }

  RunId run_id;
  tsl::profiler::TraceMeProducer activity("TfrtGpuExecutable::Execute",
                                          tsl::profiler::ContextType::kPjRt,
                                          run_id.ToInt());

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

  } else {
    // Gang schedule collectives to ensure that collectives with the same
    // RunId are run at the same time. We conservatively run only one
    // collective at a time, because we may not have enough threads to run
    // arbitrary number of collectives concurrently.
    tsl::AsyncValueRef<GpuEvent> last_collective_launch_event =
        client_->GetLastCollectiveLaunchEvent();

    absl::Mutex mu;
    int running = num_addressable_devices;
    int failed = 0;
    absl::Status first_failure_status;

    for (int i = 0; i < num_addressable_devices; ++i) {
      const int replica = addressable_device_logical_ids_[i].replica;
      const int partition = addressable_device_logical_ids_[i].partition;

      EnqueueWork(
          client_->non_blocking_thread_pool(), [&, replica, partition, i] {
            auto statusor =
                ExecuteHelper(argument_handles[i], replica, partition, run_id,
                              options, last_collective_launch_event.CopyRef(),
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
  tsl::profiler::TraceMe traceme("TfrtGpuExecutable::ExecuteSharded");
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
              addressable_device_logical_ids_[i].partition, RunId(), options,
              /*last_collective_launch_event=*/
              tsl::AsyncValueRef<GpuEvent>(), fill_future));
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
  tsl::profiler::TraceMe traceme("TfrtGpuExecutable::ExecutePortable");
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
          /*partition=*/0, RunId(), options,
          /*last_collective_launch_event=*/tsl::AsyncValueRef<GpuEvent>(),
          fill_future, tsl::down_cast<TfrtGpuDevice*>(device)));
  returned_future = std::move(result.future);
  return std::move(result.buffers);
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

}  // namespace xla
