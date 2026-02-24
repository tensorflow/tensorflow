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

#include "xla/pjrt/gpu/tfrt/tfrt_gpu_buffer.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/distributed/protocol.pb.h"
#include "xla/pjrt/gpu/tfrt/gpu_event.h"
#include "xla/pjrt/gpu/tfrt/tfrt_gpu_client.h"
#include "xla/pjrt/gpu/tfrt/tracked_gpu_device_buffer.h"
#include "xla/pjrt/gpu/tfrt/utils.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/transpose.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu_topology.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/mem.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {

class TrackedGpuDeviceBufferExternalReference
    : public PjRtBuffer::ExternalReference {
 public:
  explicit TrackedGpuDeviceBufferExternalReference(
      std::unique_ptr<TrackedGpuDeviceBuffer> tracked_device_buffer)
      : tracked_device_buffer_(std::move(tracked_device_buffer)) {
    data_ptr_ = tracked_device_buffer_->buffer()->buffer().opaque();
  }

  ~TrackedGpuDeviceBufferExternalReference() override = default;

 private:
  std::unique_ptr<TrackedGpuDeviceBuffer> tracked_device_buffer_;
};

TfrtGpuBuffer::TfrtGpuBuffer(
    Shape on_device_shape,
    std::unique_ptr<TrackedGpuDeviceBuffer> tracked_device_buffer,
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

TrackedGpuDeviceBuffer* TfrtGpuBuffer::AcquireUsage(
    tsl::AsyncValueRef<GpuEvent> usage_event) {
  absl::MutexLock lock(mu_);
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
  MarkGpuEventReadyOnExit ready_on_exit(usage_event);

  auto get_shape = [this, device_buffer]() -> absl::StatusOr<Shape> {
    if (auto* error = device_buffer->definition_event().GetErrorIfPresent()) {
      return *error;
    }

    const auto& buffer = device_buffer->buffer();

    ShapedBuffer shaped_buffer =
        buffer->AsShapedBuffer(on_device_shape_, device_);
    Shape ret_shape = on_device_shape_;
    TransferManager* transfer_manager =
        client_->xla_client()->backend().transfer_manager();

    auto stream = device_->stream();
    TF_RETURN_IF_ERROR(transfer_manager->ReadDynamicShapes(
        stream, &shaped_buffer, &ret_shape));
    TF_RETURN_IF_ERROR(BlockHostUntilDoneWithHostCallback(stream));
    return ret_shape;
  };

  absl::StatusOr<Shape> shape_or;
  client_->blocking_thread_pool()->ScheduleWhenReady(
      {device_buffer->definition_event().CopyRCRef()},
      [get_shape = std::move(get_shape), &shape_or,
       usage_event_holder = std::move(ready_on_exit)]() {
        shape_or = get_shape();
      });

  tsl::BlockUntilReady(usage_event);
  return shape_or;
}

Future<> TfrtGpuBuffer::GetReadyFuture() {
  VLOG(4) << "TfrtGpuBuffer::GetReadyFuture";
  absl::MutexLock lock(mu_);
  if (!tracked_device_buffer_) {
    return Future<>(InvalidArgument(
        "GetReadyFuture() called on deleted or donated buffer"));
  }
  if (!ready_future_) {
    ready_future_ = CreateFutureForEvent(tracked_device_buffer_->ready_event());
  }
  return FutureHelpers::WithProfiling(
      ready_future_,
      /*on_block_start=*/
      []() {
        tsl::profiler::TraceMeProducer traceme("TfrtGpuBuffer::Await");
        VLOG(4) << "TfrtGpuBuffer::Await";
        return FutureHelpers::ProfilingKeys(
            {/*traceme_context_id=*/traceme.GetContextId()});
      },
      /*on_block_end=*/
      [](FutureHelpers::ProfilingKeys keys) {
        tsl::profiler::TraceMeConsumer traceme("TfrtGpuBuffer::Await",
                                               keys.traceme_context_id);
      });
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
TfrtGpuBuffer::DonateWithControlDependency(Future<> dependency) {
  VLOG(4) << "TfrtGpuBuffer::DonateWithControlDependency";

  TF_ASSIGN_OR_RETURN(DonationTransaction donation_transaction,
                      AcquireDonation());

  TrackedGpuDeviceBuffer* tracked_buffer = donation_transaction.device_buffer();

  if (tracked_buffer == nullptr) {
    return InvalidArgument(
        "DonateWithControlDependency was called on a deleted or donated "
        "buffer.");
  }

  // Combine the original definition event and usage event.
  tsl::AsyncValueRef<GpuEvent> usage_definition_events =
      AfterAll({tracked_buffer->LockUseAndTransferUsageEvents(),
                tracked_buffer->definition_event()});

  // Create an event for `dependency`.
  tsl::AsyncValueRef<GpuEvent> dependency_event =
      tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  dependency.OnReady([dependency_event](absl::Status status) {
    if (status.ok()) {
      dependency_event.SetStateConcrete();
    } else {
      dependency_event.SetError(status);
    }
  });

  // Create new buffer with the combined event and underlying data from the
  // original buffer.
  tsl::AsyncValueRef<GpuEvent> new_definition_event =
      AfterAll({usage_definition_events, dependency_event});
  auto new_tracked_buffer = std::make_unique<TrackedGpuDeviceBuffer>(
      tracked_buffer->buffer(), std::move(new_definition_event),
      tracked_buffer->ready_event(),
      std::move(tracked_buffer->on_delete_callback_),
      std::move(tracked_buffer->cuda_event_));

  auto new_pjrt_buffer = std::make_unique<TfrtGpuBuffer>(
      on_device_shape_, std::move(new_tracked_buffer), client_, device_,
      memory_space_);

  // Commit will set the underlying device buffer unowned. This may break other
  // ongoing users. Only commit after all the pending definition and usage
  // events are ready.
  usage_definition_events.AndThen(
      [donation_transaction = std::move(donation_transaction)]() mutable {
        std::move(donation_transaction).Commit();
      });

  return new_pjrt_buffer;
}

PjRtDevice* TfrtGpuBuffer::device() const { return device_; }

PjRtClient* TfrtGpuBuffer::client() const { return client_; }

bool TfrtGpuBuffer::IsOnCpu() const {
  return memory_space() != nullptr &&
         memory_space()->kind() == PinnedHostMemorySpace::kKind;
}

const tsl::AsyncValueRef<GpuDeviceMemory>& TfrtGpuBuffer::GetBufferPtr() const {
  absl::MutexLock lock(mu_);
  return tracked_device_buffer_->buffer();
}

absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
TfrtGpuBuffer::AcquireExternalReference() {
  class ScopedExternalReference : public PjRtBuffer::ExternalReference {
   public:
    explicit ScopedExternalReference(TfrtGpuBuffer* buffer,
                                     tsl::AsyncValueRef<GpuDeviceMemory> data)
        : buffer_(buffer), data_(std::move(data)) {
      DCHECK(data_);
      data_ptr_ = data_->buffer().opaque();
    }

    ~ScopedExternalReference() override { buffer_->DropExternalReference(); }

   private:
    TfrtGpuBuffer* buffer_ = nullptr;
    // Keep a reference to the underlying data used. Note that it is still
    // users' responsibility to synchronize reads and writes to the data.
    tsl::AsyncValueRef<GpuDeviceMemory> data_;
  };

  absl::MutexLock lock(mu_);
  if (tracked_device_buffer_ == nullptr) {
    return InvalidArgument("Buffer has been deleted or donated.");
  }

  // If the external reference event is concrete, it means we previously dropped
  // the last external reference but want to create one again without having
  // deleted the buffer. So we need a new external_references_dropped_event_.
  if (external_references_dropped_event_.IsConcrete()) {
    external_references_dropped_event_ =
        tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  }

  tsl::BlockUntilReady(tracked_device_buffer_->definition_event());
  if (tracked_device_buffer_->definition_event().IsError()) {
    return tracked_device_buffer_->definition_event().GetError();
  }
  ++external_reference_counter_;
  return {std::make_unique<ScopedExternalReference>(
      this, tracked_device_buffer_->buffer())};
}

absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
TfrtGpuBuffer::ReleaseDeviceMemoryOwnership(
    bool wait_for_operations_to_complete) {
  if (on_device_shape_.IsTuple()) {
    return InvalidArgument(
        "ReleaseDeviceMemoryOwnership allowed only for non-tuple");
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TrackedGpuDeviceBuffer> tracked_device_buffer,
      Release(wait_for_operations_to_complete));

  std::unique_ptr<PjRtBuffer::ExternalReference> ref;
  if (tracked_device_buffer) {
    ref = std::make_unique<TrackedGpuDeviceBufferExternalReference>(
        std::move(tracked_device_buffer));
  }
  return ref;
}

Future<> TfrtGpuBuffer::ToLiteral(MutableLiteralBase* literal) {
  VLOG(3) << "TfrtGpuBuffer::ToLiteral for a tensor of shape "
          << literal->shape().ToString();
  return ToLiteralHelper(literal, nullptr);
}

Future<> TfrtGpuBuffer::ToLiteralHelper(
    MutableLiteralBase* literal,
    absl::AnyInvocable<Future<MutableLiteralBase*>() &&> generator) {
  tsl::profiler::TraceMe traceme("TfrtGpuBuffer::ToLiteral");
  auto [promise, future] = MakePromise<>();
  auto usage_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto* device_buffer = AcquireUsage(usage_event);
  if (device_buffer == nullptr) {
    promise.Set(
        InvalidArgument("ToLiteral() called on deleted or donated buffer"));
    return future;
  }

  bool unpack_subbyte_types =
      client_->xla_client()->backend().transfer_manager()->PackSubbyteTypes();

  auto d2h_copy = [device(device_), device_buffer,
                   usage_event = std::move(usage_event),
                   promise = std::move(promise), client = client_,
                   on_device_shape = on_device_shape_, unpack_subbyte_types,
                   literal, generator = std::move(generator),
                   thread_pool = client_->blocking_thread_pool()]() mutable {
    tsl::profiler::TraceMe traceme("ToLiteral::D2H_copy");
    if (device_buffer->definition_event().IsError()) {
      usage_event.SetStateConcrete();
      VLOG(3) << "device_buffer->definition_event().GetError(): "
              << device_buffer->definition_event().GetError();
      promise.Set(device_buffer->definition_event().GetError());
      return;
    }
    size_t byte_size = device_buffer->buffer()->buffer().size();

    PrimitiveType type = on_device_shape.element_type();
    bool should_unpack =
        unpack_subbyte_types && primitive_util::IsSubByteNonPredType(type);

    auto copy_to_literal =
        [device = std::move(device), device_buffer = std::move(device_buffer),
         client = std::move(client),
         on_device_shape = std::move(on_device_shape), should_unpack,
         byte_size](const absl::StatusOr<MutableLiteralBase*>& value) mutable
        -> absl::Status {
      TF_ASSIGN_OR_RETURN(MutableLiteralBase* const literal, value);

      std::shared_ptr<TransposePlan> transpose;
      if (on_device_shape.IsArray()) {
        xla::Layout literal_layout;
        if (literal->shape().has_layout()) {
          literal_layout = literal->shape().layout();
        } else {
          literal_layout = LayoutUtil::MakeDescendingLayout(
              on_device_shape.dimensions().size());
        }

        if (on_device_shape.layout() != literal_layout) {
          absl::InlinedVector<int64_t, 4> byte_strides(
              on_device_shape.dimensions().size());
          TF_RETURN_IF_ERROR(ShapeUtil::UnpackedByteStrides(
              on_device_shape, absl::MakeSpan(byte_strides)));
          absl::Span<const int64_t> dims = on_device_shape.dimensions();
          absl::InlinedVector<int64_t, 4> permutation(dims.size());
          absl::c_reverse_copy(literal_layout.minor_to_major(),
                               permutation.begin());
          TransposePlan::Options options;
          options.elem_size_in_bytes =
              primitive_util::ByteWidth(on_device_shape.element_type());
          options.dims = on_device_shape.dimensions();
          options.permutation = permutation;
          options.input_striding = TransposePlan::Striding{byte_strides};
          {
            absl::MutexLock lock(client->transpose_mu_);
            TF_ASSIGN_OR_RETURN(transpose,
                                client->transpose_cache_.GetOrCreate(options));
          }
        }
      }

      HostMemoryAllocator::OwnedPtr staging_buffer;
      void* buffer_ptr;
      if (on_device_shape.IsArray()) {
        buffer_ptr = literal->untyped_data();
        if (should_unpack || transpose != nullptr ||
            client->ShouldStageHostToDeviceTransfers(buffer_ptr, byte_size)) {
          staging_buffer =
              client->GetHostMemoryAllocator()->Allocate(byte_size);
          buffer_ptr = staging_buffer.get();
        }
      } else {
        CHECK_EQ(byte_size, 0);
        buffer_ptr = nullptr;
      }

      {
        tsl::profiler::TraceMe traceme2([&] {
          return tsl::profiler::TraceMeEncode("ToLiteral::D2H_GPU_copy",
                                              {
                                                  {"device", device->id()},
                                                  {"size", byte_size},
                                              });
        });

        auto d2h_stream = device->d2h_stream();

        // If we do not have a CudaEvent, we need to fall back to the host
        // event to check readiness.
        // TODO: Remove this once cuda events are always set/not nullptr.
        if (device_buffer->GetCudaEvent() == nullptr) {
          tsl::BlockUntilReady(device_buffer->ready_event());
        }

        absl::Status cuda_event_wait_status =
            WaitForEventOnStream(d2h_stream, device_buffer->GetCudaEvent());
        if (!cuda_event_wait_status.ok()) {
          LOG(ERROR) << "Failed to wait for cuda event: "
                     << cuda_event_wait_status;
          return cuda_event_wait_status;
        }

        VLOG(3) << "D2H copy: " << device_buffer->buffer()->buffer().opaque()
                << " -> " << buffer_ptr << " (" << byte_size << " bytes)";
        CHECK_OK(d2h_stream->Memcpy(
            buffer_ptr, device_buffer->buffer()->buffer(), byte_size))
            << "stream->Memcpy failed copying from GPU to host";

        absl::Status status = BlockHostUntilDoneWithHostCallback(d2h_stream);
        VLOG(3) << "D2H copy done. " << status;
        if (!status.ok()) {
          VLOG(3) << "stream BlockHostUntilDoneWithHostCallback failed: "
                  << status;
          return status;
        }

        tsl::BlockUntilReady(device_buffer->ready_event());
        if (device_buffer->ready_event().IsError()) {
          return device_buffer->ready_event().GetError();
        }
      }
      void* buffer;
      if (should_unpack) {
        tsl::profiler::TraceMe traceme("ToLiteral::D2H_staging_copy");
        int64_t unpacked_size = ShapeUtil::ElementsIn(on_device_shape);
        if (transpose != nullptr) {
          buffer = tsl::port::AlignedMalloc(
              unpacked_size, static_cast<std::align_val_t>(
                                 tsl::Allocator::kAllocatorAlignment));
        } else {
          buffer = literal->untyped_data();
        }
        primitive_util::UnpackIntN(
            on_device_shape.element_type(),
            absl::MakeConstSpan(static_cast<const char*>(buffer_ptr),
                                byte_size),
            absl::MakeSpan(static_cast<char*>(buffer), unpacked_size));
        VLOG(3) << "D2H staging copy done";
      } else {
        buffer = buffer_ptr;
      }
      if (transpose != nullptr) {
        tsl::profiler::TraceMe traceme("Transpose");
        transpose->Execute(buffer, static_cast<char*>(literal->untyped_data()));
        if (should_unpack) {
          tsl::port::AlignedFree(buffer);
        }
      }
      if (on_device_shape.IsArray() && staging_buffer != nullptr &&
          !should_unpack && transpose == nullptr) {
        std::memcpy(literal->untyped_data(), buffer, literal->size_bytes());
      }
      return absl::OkStatus();
    };
    auto copy_to_literal_and_set_event =
        [copy_to_literal = std::move(copy_to_literal),
         usage_event = std::move(usage_event), promise = std::move(promise)](
            const absl::StatusOr<MutableLiteralBase*>& value) mutable {
          absl::Status status = copy_to_literal(value);
          usage_event.SetStateConcrete();
          promise.Set(status);
        };

    if (literal != nullptr) {
      copy_to_literal_and_set_event(literal);
    } else {
      Future<MutableLiteralBase*> generated = std::move(generator)();
      if (generated.IsKnownReady()) {
        copy_to_literal_and_set_event(generated.Await());
      } else {
        generated.OnReady(*client->blocking_thread_pool(),
                          std::move(copy_to_literal_and_set_event));
      }
    }
  };
  client_->blocking_thread_pool()->ScheduleWhenReady(
      {device_buffer->definition_event().CopyRCRef()}, std::move(d2h_copy));

  return FutureHelpers::WithProfiling(
      std::move(future),
      /*on_block_start=*/
      []() {
        tsl::profiler::TraceMeProducer traceme("TfrtGpuBuffer::ToLiteral");
        VLOG(3) << "TfrtGpuBuffer::ToLiteral::OnBlockStart";
        return FutureHelpers::ProfilingKeys(
            {/*traceme_context_id =*/traceme.GetContextId()});
      },
      /*on_block_end=*/
      [](FutureHelpers::ProfilingKeys keys) {
        tsl::profiler::TraceMeConsumer traceme("TfrtGpuBuffer::ToLiteral",
                                               keys.traceme_context_id);
      });
}

Future<> TfrtGpuBuffer::LazyToLiteral(
    absl::AnyInvocable<Future<MutableLiteralBase*>() &&> generator) {
  VLOG(3) << "TfrtGpuBuffer::LazyToLiteral";
  return ToLiteralHelper(nullptr, std::move(generator));
}

Future<> TfrtGpuBuffer::CopyRawToHostFuture(Future<void*> dst_future,
                                            int64_t offset,
                                            int64_t transfer_size) {
  VLOG(3) << "TfrtGpuBuffer::CopyRawToHostFuture";
  tsl::profiler::TraceMe traceme("TfrtGpuBuffer::CopyRawToHostFuture");
  auto [promise, future] = MakePromise<>();
  auto usage_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto* device_buffer = AcquireUsage(usage_event);
  MarkGpuEventReadyOnExit usage_event_holder(std::move(usage_event));
  if (device_buffer == nullptr) {
    return Future<>(
        InvalidArgument("ToLiteral() called on deleted or donated buffer"));
  }
  auto d2h_copy = [device(device_), device_buffer,
                   usage_event_holder = std::move(usage_event_holder),
                   client = client_, offset,
                   transfer_size](Promise<> promise, void* dst) mutable {
    if (device_buffer->definition_event().IsError()) {
      LOG(ERROR) << "device_buffer->definition_event().GetError(): "
                 << device_buffer->definition_event().GetError();
      promise.Set(device_buffer->definition_event().GetError());
      return;
    }
    se::DeviceAddressBase device_memory = device_buffer->buffer()->buffer();
    if (offset < 0 || offset > device_memory.size() ||
        device_memory.size() - offset < transfer_size) {
      LOG(ERROR) << "Copy raw buffer called on buffer size "
                 << device_memory.size() << " with invalid offset " << offset
                 << ", transfer size " << transfer_size;
      promise.Set(
          InvalidArgument("Copy raw buffer called on buffer size %lld with "
                          "invalid offset %lld, transfer size %lld",
                          device_memory.size(), offset, transfer_size));
      return;
    }

    se::DeviceAddressBase sub_buffer;
    if (transfer_size < device_memory.size()) {
      sub_buffer = device_memory.GetByteSlice(offset, transfer_size);
    } else {
      sub_buffer = device_memory;
    }

    HostMemoryAllocator::OwnedPtr staging_buffer;
    const bool use_staging =
        client->ShouldStageHostToDeviceTransfers(dst, transfer_size);

    if (use_staging) {
      staging_buffer =
          client->GetHostMemoryAllocator()->Allocate(transfer_size);
    }

    void* host_ptr = use_staging ? staging_buffer.get() : dst;

    auto d2h_stream = device->d2h_stream();
    absl::Status cuda_event_wait_status =
        WaitForEventOnStream(d2h_stream, device_buffer->GetCudaEvent());
    if (!cuda_event_wait_status.ok()) {
      LOG(ERROR) << "Failed to wait for cuda event: " << cuda_event_wait_status;
      promise.Set(cuda_event_wait_status);
      return;
    }

    VLOG(3) << "D2H copy: " << sub_buffer.opaque() << " -> " << host_ptr << " ("
            << transfer_size << " bytes)";
    absl::Status status =
        d2h_stream->Memcpy(host_ptr, sub_buffer, transfer_size);
    if (!status.ok()) {
      LOG(ERROR) << "stream->Memcpy failed: " << status;
      promise.Set(status);
      return;
    }

    status = BlockHostUntilDoneWithHostCallback(d2h_stream);

    if (!status.ok()) {
      LOG(ERROR) << "d2h_stream BlockHostUntilDoneWithHostCallback failed: "
                 << status;
      promise.Set(status);
      return;
    }
    if (use_staging) {
      tsl::profiler::TraceMe traceme3("CopyRawToHostFuture::D2H_staging_copy");
      std::memcpy(dst, staging_buffer.get(), transfer_size);
      VLOG(3) << "D2H staging copy done: " << staging_buffer.get() << " -> "
              << dst << " (" << transfer_size << " bytes)";
    }
    promise.Set(absl::OkStatus());
  };

  dst_future.OnReady(
      [client(client_), promise = std::move(promise), device_buffer,
       d2h_copy = std::move(d2h_copy)](absl::StatusOr<void*> dst_or) mutable {
        if (!dst_or.ok()) {
          promise.Set(dst_or.status());
          LOG(ERROR) << "dst resolved to an error: " << dst_or.status();
          return;
        }
        client->blocking_thread_pool()->ScheduleWhenReady(
            {device_buffer->definition_event().CopyRCRef()},
            [dst = std::move(dst_or.value()), promise = std::move(promise),
             d2h_copy = std::move(d2h_copy)]() mutable {
              std::move(d2h_copy)(std::move(promise), dst);
            });
      });

  return FutureHelpers::WithProfiling(
      std::move(future),
      /*on_block_start=*/
      []() {
        tsl::profiler::TraceMeProducer traceme(
            "TfrtGpuBuffer::CopyRawToHostFuture");
        VLOG(3) << "TfrtGpuBuffer::CopyRawToHostFuture";
        return FutureHelpers::ProfilingKeys(
            {/*traceme_context_id =*/traceme.GetContextId()});
      },
      /*on_block_end=*/
      [](FutureHelpers::ProfilingKeys keys) {
        tsl::profiler::TraceMeConsumer traceme(
            "TfrtGpuBuffer::CopyRawToHostFuture", keys.traceme_context_id);
      });
}

void TfrtGpuBuffer::Delete() {
  tsl::profiler::TraceMe traceme("Gpu buffer delete");
  VLOG(4) << " TfrtGpuBuffer::Delete";
  std::unique_ptr<TrackedGpuDeviceBuffer> device_buffer;
  tsl::AsyncValueRef<GpuEvent> external_references_dropped_event;
  {
    absl::MutexLock lock(mu_);
    device_buffer = ReleaseBufferLocked();
    if (device_buffer == nullptr) {
      return;
    }

    if (external_reference_counter_ > 0) {
      external_references_dropped_event =
          external_references_dropped_event_.CopyRef();
    } else {
      external_references_dropped_event =
          tsl::MakeAvailableAsyncValueRef<GpuEvent>();
    }
  }
  if (device_buffer == nullptr) {
    return;
  }

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
        VLOG(4) << "device_buffer is being deleted: " << device_buffer.get();
        device_buffer.reset();
      });
}

bool TfrtGpuBuffer::IsDeleted() const {
  absl::MutexLock lock(mu_);
  return tracked_device_buffer_ == nullptr;
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> TfrtGpuBuffer::CopyToMemorySpace(
    PjRtMemorySpace* dst_memory_space) {
  tsl::profiler::TraceMe traceme("TfrtGpuBuffer::CopyToMemorySpace");
  PjRtDevice* dst_device = dst_memory_space->devices()[0];

  VLOG(1) << "TfrtGpuBuffer::CopyToMemorySpace:  dst_device: "
          << dst_device->DebugString()
          << " dst_memory_space: " << dst_memory_space->kind();

  // Copying across PjRtClients involves a copy through the host.
  if (dst_device->client() != client_) {
    TF_ASSIGN_OR_RETURN(std::shared_ptr<Literal> literal,
                        PjRtBuffer::ToLiteral().Await());
    // Avoid use-after-free on `literal` due to unsequenced move and use.
    Literal* literal_pointer = literal.get();
    absl::InlinedVector<int64_t, 4> byte_strides(
        literal->shape().dimensions().size());
    TF_RETURN_IF_ERROR(ShapeUtil::UnpackedByteStrides(
        literal->shape(), absl::MakeSpan(byte_strides)));
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
  auto src_usage_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  TrackedGpuDeviceBuffer* src_device_buffer = AcquireUsage(src_usage_event);
  if (src_device_buffer == nullptr) {
    return InvalidArgument(
        "CopyToMemorySpace called on deleted or donated buffer");
  }

  TfrtGpuDevice* gpu_src_device = tsl::down_cast<TfrtGpuDevice*>(device());
  TfrtGpuDevice* gpu_dst_device = tsl::down_cast<TfrtGpuDevice*>(dst_device);
  tsl::AsyncValueRef<GpuDeviceMemory> src_buffer = src_device_buffer->buffer();

  auto dst_definition_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  TF_ASSIGN_OR_RETURN(auto output_buffer,
                      AllocateTfrtGpuDestinationBuffer(
                          on_device_shape_, dst_definition_event.CopyRef(),
                          gpu_dst_device, client_, dst_memory_space));
  auto dst_usage_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  TrackedGpuDeviceBuffer* allocated_dst_device_buffer =
      output_buffer->AcquireUsage(dst_usage_event);
  CHECK(allocated_dst_device_buffer != nullptr);
  auto allocated_dst_buffer = allocated_dst_device_buffer->buffer();

  absl::AnyInvocable<void()> transfer_d2d =
      [src_buffer(src_buffer.CopyRef()),
       allocated_dst_buffer(allocated_dst_buffer.CopyRef()),
       dst_definition_event(dst_definition_event.CopyRef()),
       src_definition_event(src_device_buffer->definition_event().CopyRef()),
       src_device(gpu_src_device), dst_device(gpu_dst_device),
       src_usage_event(src_usage_event.CopyRef()),
       dst_usage_event(dst_usage_event.CopyRef())]() {
        MarkGpuEventReadyOnExit ready_on_exit_src(std::move(src_usage_event));
        MarkGpuEventReadyOnExit ready_on_exit_dst(std::move(dst_usage_event));

        // If the source buffer has an error, propagate it to the destination
        // buffer.
        if (const absl::Status* error =
                src_definition_event.GetErrorIfPresent()) {
          dst_definition_event.SetError(*error);
          return;
        }

        VLOG(3) << "Request to transfer D2D from "
                << src_buffer->buffer().opaque() << " on device "
                << src_device->id() << " to "
                << allocated_dst_buffer->buffer().opaque() << " on device "
                << dst_device->id();

        tsl::profiler::TraceMe trace([&] {
          return tsl::profiler::TraceMeEncode(
              "CopyToMemorySpace::D2D_copy",
              {
                  {"src_device", src_device->id()},
                  {"dst_device", dst_device->id()},
                  {"size", src_buffer->buffer().size()},
              });
        });

        auto stream = dst_device->stream();

        se::DeviceAddressBase dst(allocated_dst_buffer->buffer());
        VLOG(3) << "D2D copy: " << src_buffer->buffer().opaque() << " -> "
                << dst.opaque() << " (" << src_buffer->buffer().size()
                << " bytes)";
        absl::Status status = stream->Memcpy(&dst, src_buffer->buffer(),
                                             src_buffer->buffer().size());
        if (!status.ok()) {
          dst_definition_event.SetError(status);
          return;
        }

        status = BlockHostUntilDoneWithHostCallback(stream);
        if (status.ok()) {
          VLOG(3) << "D2D copy done. dst: " << dst.opaque();
          dst_definition_event.SetStateConcrete();
        } else {
          LOG(ERROR) << "D2D copy failed. dst: " << dst.opaque()
                     << " status: " << status;
          dst_definition_event.SetError(status);
        }
      };

  client_->blocking_thread_pool()->ScheduleWhenReady(
      {src_device_buffer->ready_event().CopyRCRef()}, std::move(transfer_d2d));
  return output_buffer;
}

void TfrtGpuBuffer::DropExternalReference() {
  absl::MutexLock lock(mu_);
  CHECK_GT(external_reference_counter_, 0);
  --external_reference_counter_;
  if (external_reference_counter_ == 0) {
    external_references_dropped_event_.SetStateConcrete();
  }
}

absl::StatusOr<std::unique_ptr<TrackedGpuDeviceBuffer>> TfrtGpuBuffer::Release(
    bool wait_for_operations_to_complete) {
  auto donation_event = GetDonationEvent();
  tsl::BlockUntilReady(donation_event);
  std::unique_ptr<TrackedGpuDeviceBuffer> device_buffer;
  {
    absl::MutexLock lock(mu_);
    device_buffer = ReleaseBufferLocked();
  }
  if (device_buffer == nullptr) {
    return {nullptr};
  }

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
    if (!first_error.ok()) {
      return std::move(first_error);
    }
  }

  return device_buffer;
}

std::unique_ptr<TrackedGpuDeviceBuffer> TfrtGpuBuffer::ReleaseBufferLocked() {
  tsl::profiler::TraceMe traceme("TfrtGpuBuffer::ReleaseBufferLocked");
  return std::move(tracked_device_buffer_);
}

absl::StatusOr<TfrtGpuBuffer::DonationTransaction>
TfrtGpuBuffer::AcquireDonation() {
  absl::MutexLock lock(mu_);

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
  VLOG(4) << "TfrtGpuBuffer::AcquireDonation: " << tracked_device_buffer_.get();
  return DonationTransaction(donation_event_,
                             std::move(tracked_device_buffer_));
}

}  // namespace xla
