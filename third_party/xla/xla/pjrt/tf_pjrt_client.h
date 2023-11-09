/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_PJRT_TF_PJRT_CLIENT_H_
#define XLA_PJRT_TF_PJRT_CLIENT_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "tsl/platform/errors.h"

namespace xla {

class TfPjRtClient;

// Wrapper for PjRtBuffer that translates the device.
class TfPjRtBuffer : public PjRtBuffer {
 public:
  TfPjRtBuffer(TfPjRtClient* client, std::unique_ptr<PjRtBuffer> wrapped);
  ~TfPjRtBuffer() override;

  PjRtBuffer* wrapped() const { return wrapped_.get(); }
  const Shape& on_device_shape() const override {
    return wrapped_->on_device_shape();
  }
  StatusOr<Shape> logical_on_device_shape() override {
    return wrapped_->logical_on_device_shape();
  }
  PjRtMemorySpace* memory_space() const override {
    return wrapped_->memory_space();
  }
  PjRtDevice* device() const override { return wrapped_->device(); }
  PjRtClient* client() const override;
  StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference()
      override {
    return wrapped_->AcquireExternalReference();
  }
  PjRtFuture<Status> ToLiteral(MutableLiteralBase* literal) override {
    return wrapped_->ToLiteral(literal);
  }
  StatusOr<size_t> GetOnDeviceSizeInBytes() const override {
    return wrapped_->GetOnDeviceSizeInBytes();
  }
  PjRtFuture<Status> CopyRawToHost(void* dst, int64_t offset,
                                   int64_t transfer_size) override {
    return wrapped_->CopyRawToHost(dst, offset, transfer_size);
  }
  void Delete() override { wrapped_->Delete(); }
  StatusOr<std::unique_ptr<ExternalReference>> ReleaseDeviceMemoryOwnership(
      bool wait_for_operations_to_complete) override {
    return wrapped_->ReleaseDeviceMemoryOwnership(
        wait_for_operations_to_complete);
  }
  bool IsDeleted() override { return wrapped_->IsDeleted(); }
  StatusOr<std::unique_ptr<PjRtBuffer>> CopyToDevice(
      PjRtDevice* dst_device) override;
  StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(
      PjRtMemorySpace* dst_memory_space) override {
    return Unimplemented("CopyToMemorySpace not implemented");
  }
  void CopyToRemoteDevice(
      PjRtFuture<StatusOr<std::string>> serialized_descriptor,
      RemoteSendCallback on_done) override {
    wrapped_->CopyToRemoteDevice(std::move(serialized_descriptor),
                                 std::move(on_done));
  }
  void CopyToRemoteDeviceScattered(
      PjRtFuture<StatusOr<std::vector<std::string>>> serialized_descriptors,
      std::vector<RemoteSendCallback> callbacks,
      const ScatterDetails& scatter_details) override {
    return wrapped_->CopyToRemoteDeviceScattered(
        std::move(serialized_descriptors), std::move(callbacks),
        scatter_details);
  }
  PjRtFuture<Status> GetReadyFuture() override {
    return wrapped_->GetReadyFuture();
  }
  bool IsOnCpu() const override { return wrapped_->IsOnCpu(); }

  // Not thread-safe. The caller should promises to have some external
  // synchronization that ensures that all uses of the buffer have completed
  // (and a thread synchronization has occurred that involves all the necessary
  // memory barriers) before this method is called.
  void DestroyWrappedBuffer() { wrapped_.reset(nullptr); }

 private:
  TfPjRtClient* client_;
  std::unique_ptr<PjRtBuffer> wrapped_;
};

// Wrapper for PjRtLoadedExecutable that wraps and unwraps argument and result
// buffers.
class TfPjRtExecutable : public PjRtLoadedExecutable {
 public:
  TfPjRtExecutable(TfPjRtClient* client,
                   std::unique_ptr<PjRtLoadedExecutable> wrapped);

  PjRtLoadedExecutable* wrapped() const { return wrapped_.get(); }

  PjRtClient* client() const override;
  absl::string_view name() const override { return wrapped_->name(); }
  int num_replicas() const override { return wrapped_->num_replicas(); }
  int num_partitions() const override { return wrapped_->num_partitions(); }
  int64_t SizeOfGeneratedCodeInBytes() const override {
    return wrapped_->SizeOfGeneratedCodeInBytes();
  }
  const DeviceAssignment& device_assignment() const override {
    return wrapped_->device_assignment();
  }
  absl::Span<const LogicalDeviceIds> addressable_device_logical_ids()
      const override {
    return wrapped_->addressable_device_logical_ids();
  }
  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return wrapped_->addressable_devices();
  }
  StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override {
    return wrapped_->GetHloModules();
  }
  StatusOr<std::vector<std::vector<absl::string_view>>> GetOutputMemoryKinds()
      const override {
    return wrapped_->GetOutputMemoryKinds();
  }
  using PjRtLoadedExecutable::Execute;
  StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options,
      std::optional<std::vector<PjRtFuture<Status>>>& returned_futures)
      override;
  using PjRtLoadedExecutable::ExecuteSharded;
  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<Status>>& returned_future,
      bool fill_future) override;
  using PjRtLoadedExecutable::ExecutePortable;
  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<Status>>& returned_future,
      bool fill_future) override;

  void Delete() override { return wrapped_->Delete(); }
  bool IsDeleted() override { return wrapped_->IsDeleted(); }
  bool IsReturnedFutureSupported() const override {
    return wrapped_->IsReturnedFutureSupported();
  }

  StatusOr<std::string> SerializeExecutable() const override {
    return wrapped_->SerializeExecutable();
  }

  StatusOr<struct CompileOptions> GetCompileOptions() const override {
    return wrapped_->GetCompileOptions();
  }

  StatusOr<std::string> FingerprintExecutable() const override {
    return wrapped_->FingerprintExecutable();
  }

 private:
  TfPjRtClient* client_;
  std::unique_ptr<PjRtLoadedExecutable> wrapped_;
};

// A thin wrapper of PjRtClient that includes management of PjRtBuffer it
// creates.
class TfPjRtClient : public PjRtClient {
 public:
  static std::unique_ptr<TfPjRtClient> CreateTfPjRtClient(
      std::unique_ptr<PjRtClient> wrapped);
  explicit TfPjRtClient(std::unique_ptr<PjRtClient> wrapped);
  ~TfPjRtClient() override;
  int process_index() const override { return wrapped_->process_index(); }
  int device_count() const override { return wrapped_->device_count(); }
  int addressable_device_count() const override {
    return wrapped_->addressable_device_count();
  }
  absl::Span<PjRtDevice* const> devices() const override {
    return wrapped_->devices();
  }
  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return wrapped_->addressable_devices();
  }
  StatusOr<PjRtDevice*> LookupDevice(int device_id) const override {
    return wrapped_->LookupDevice(device_id);
  }
  StatusOr<PjRtDevice*> LookupAddressableDevice(
      int local_hardware_id) const override {
    if (wrapped_ == nullptr) {
      return tsl::errors::Internal(
          "Wrapped PJRT client in TfPjRtClient is already destoryed.");
    }
    return wrapped_->LookupAddressableDevice(local_hardware_id);
  }
  absl::Span<PjRtMemorySpace* const> memory_spaces() const override {
    return wrapped_->memory_spaces();
  }
  PjRtPlatformId platform_id() const override {
    return wrapped_->platform_id();
  }
  absl::string_view platform_name() const override {
    return wrapped_->platform_name();
  }
  absl::string_view platform_version() const override {
    return wrapped_->platform_version();
  }
  PjRtRuntimeType runtime_type() const override {
    return wrapped_->runtime_type();
  }
  StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override {
    return wrapped_->GetDefaultDeviceAssignment(num_replicas, num_partitions);
  }
  StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis()
      const override {
    return wrapped_->GetHloCostAnalysis();
  }
  StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) override {
    return WrapExecutable(wrapped_->Compile(computation, options));
  }
  StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      mlir::ModuleOp module, CompileOptions options) override {
    return WrapExecutable(wrapped_->Compile(std::move(module), options));
  }

  StatusOr<std::unique_ptr<PjRtLoadedExecutable>> DeserializeExecutable(
      absl::string_view serialized,
      std::optional<CompileOptions> options) override {
    return WrapExecutable(wrapped_->DeserializeExecutable(serialized, options));
  }

  StatusOr<std::unique_ptr<PjRtBuffer>> CreateUninitializedBuffer(
      const Shape& shape, PjRtDevice* device) override {
    return Unimplemented(
        "CreateUninitializedBuffer not supported for TfPjRtClient.");
  }
  StatusOr<std::unique_ptr<AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(absl::Span<const Shape> shapes,
                                    PjRtDevice* device) override {
    return Unimplemented(
        "AsyncHostToDeviceTransferManager not supported for Tf.");
  }
  StatusOr<std::unique_ptr<AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(absl::Span<const Shape> shapes,
                                    PjRtMemorySpace* memory_space) override {
    return Unimplemented(
        "AsyncHostToDeviceTransferManager not supported for Tf.");
  }
  StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      std::function<void()> on_done_with_host_buffer,
      PjRtDevice* device) override {
    return WrapBuffer(wrapped_->BufferFromHostBuffer(
        data, type, dims, byte_strides, host_buffer_semantics,
        on_done_with_host_buffer, device));
  }
  StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      std::function<void()> on_done_with_host_buffer, PjRtDevice* device,
      const Layout* device_layout) override {
    return WrapBuffer(wrapped_->BufferFromHostBuffer(
        data, type, dims, byte_strides, host_buffer_semantics,
        on_done_with_host_buffer, device, device_layout));
  }
  StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtDevice* device) override {
    return WrapBuffer(wrapped_->BufferFromHostLiteral(literal, device));
  }
  StatusOr<std::unique_ptr<PjRtBuffer>> CreateViewOfDeviceBuffer(
      void* device_ptr, const Shape& shape, PjRtDevice* device,
      std::function<void()> on_delete_callback,
      std::optional<std::intptr_t> stream) override {
    return WrapBuffer(wrapped_->CreateViewOfDeviceBuffer(
        device_ptr, shape, device, on_delete_callback, stream));
  }
  StatusOr<std::uintptr_t> UnsafeBufferPointer(PjRtBuffer* buffer) override {
    return wrapped_->UnsafeBufferPointer(UnwrapBuffer(buffer));
  }
  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeCrossHostReceiveBuffers(absl::Span<const Shape> shapes,
                              PjRtDevice* device,
                              PjRtCrossHostRecvNotifier notifier) override {
    return wrapped_->MakeCrossHostReceiveBuffers(shapes, device,
                                                 std::move(notifier));
  }
  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeCrossHostReceiveBuffersForGather(
      absl::Span<const Shape> shapes, std::vector<GatherDetails> gather_details,
      PjRtDevice* device, PjRtCrossHostRecvNotifier notifier) override {
    return wrapped_->MakeCrossHostReceiveBuffersForGather(
        shapes, std::move(gather_details), device, std::move(notifier));
  }
  StatusOr<ChannelHandle> CreateChannelHandle() override {
    return wrapped_->CreateChannelHandle();
  }
  StatusOr<ChannelHandle> CreateDeviceToHostChannelHandle() override {
    return wrapped_->CreateDeviceToHostChannelHandle();
  }
  StatusOr<ChannelHandle> CreateHostToDeviceChannelHandle() override {
    return wrapped_->CreateHostToDeviceChannelHandle();
  }
  Status Defragment() override { return wrapped_->Defragment(); }

  PjRtClient* wrapped() const { return wrapped_.get(); }

  StatusOr<std::unique_ptr<PjRtBuffer>> WrapBuffer(
      StatusOr<std::unique_ptr<PjRtBuffer>> to_wrap);

  // Tracks a non-owning pointer of TfPjRtBuffer in TfPjRtClient.
  void TrackBuffer(TfPjRtBuffer* buffer);

  // Untracks a TfPjRtBuffer in TfPjRtClient.
  void UntrackBuffer(const TfPjRtBuffer* buffer);

  // Destroy all the wrapped PjRtBuffer in the tracked set of TfPjRtBuffers and
  // then destroy the wrapped PjRtClient.
  void DestroyWrappedBuffersAndClient();

 private:
  // Unwraps a TfPjRtBuffer.
  PjRtBuffer* UnwrapBuffer(PjRtBuffer* buffer) const {
    return tensorflow::down_cast<TfPjRtBuffer*>(buffer)->wrapped();
  }

  // Unwraps a TfPjRtExecutable.
  const PjRtLoadedExecutable& UnwrapExecutable(
      const PjRtLoadedExecutable& executable) const {
    return *tensorflow::down_cast<const TfPjRtExecutable*>(&executable)
                ->wrapped();
  }

  StatusOr<std::unique_ptr<PjRtLoadedExecutable>> WrapExecutable(
      StatusOr<std::unique_ptr<PjRtLoadedExecutable>> to_wrap);

  std::unique_ptr<PjRtClient> wrapped_;

  absl::flat_hash_map<int, int> mutex_id_from_device_id_;

  // Depending on `sizeof(absl::flat_hash_set<TfPjRtBuffer*>)`, might need to
  // add some padding to the struct.
  struct DeviceBuffers {
    absl::Mutex mu;
    absl::flat_hash_set<TfPjRtBuffer*> alive_buffers ABSL_GUARDED_BY(mu);
  };
  std::vector<DeviceBuffers> alive_buffers_;
};

}  // namespace xla

#endif  // XLA_PJRT_TF_PJRT_CLIENT_H_
