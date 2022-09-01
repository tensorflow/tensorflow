/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_PJRT_C_API_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_PJRT_C_API_CLIENT_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_helpers.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"

namespace xla {

class PjRtCApiClient;

class PjRtCApiDevice : public PjRtDevice {
 public:
  explicit PjRtCApiDevice(PJRT_Device* device, PjRtCApiClient* client);

  PjRtClient* client() const override;

  bool IsAddressable() const override;

  int id() const override;

  int process_index() const override;

  int local_hardware_id() const override;

  absl::string_view device_kind() const override;

  absl::string_view DebugString() const override;

  absl::string_view ToString() const override;

  Status TransferToInfeed(const LiteralSlice& literal) override {
#ifdef PJRT_C_API_BYPASS
    return wrapped_->TransferToInfeed(literal);
#endif  // PJRT_C_API_BYPASS
    return Unimplemented("PJRT C API does not support TransferToInfeed");
  }

  Status TransferFromOutfeed(MutableBorrowingLiteral literal) override {
#ifdef PJRT_C_API_BYPASS
    return wrapped_->TransferFromOutfeed(std::move(literal));
#endif  // PJRT_C_API_BYPASS
    return Unimplemented("PJRT C API does not support TransferFromOutfeed");
  }

  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override {
#ifdef PJRT_C_API_BYPASS
    return wrapped_->CreateAsyncTrackingEvent(description);
#endif  // PJRT_C_API_BYPASS
    LOG(WARNING) << "PJRT C API does not support CreateAsyncTrackingEvent";
    return nullptr;
  }

  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override;

  PJRT_Device* c_device() const { return device_; }

  PjRtDevice* wrapped() const { return wrapped_; }

  static PjRtDevice* GetWrapped(PjRtDevice* c_api_device) {
    return tensorflow::down_cast<PjRtCApiDevice*>(c_api_device)->wrapped();
  }

 private:
  PjRtCApiClient* client_ = nullptr;
  // `device_` is owned by the `PJRT_Client` wrapped by `client_`
  PJRT_Device* device_;
  // TODO(shahrokhi): wrapped_ is a non-C API pointer that was used to bypass
  // the C API calls until all the C API's got implemented. Remove it when it's
  // usage is reduced to zero.
  PjRtDevice* wrapped_;
  // Device specific attributes with corresponding values.
  absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute> attributes_;

  // Initializes device specific attributes.
  void InitAttributes();
};

class PjRtCApiClient : public PjRtClient {
 public:
  PjRtCApiClient(const PJRT_Api* c_api, PJRT_Client* c_client);

  int process_index() const override;

  int device_count() const override;
  int addressable_device_count() const override;

  absl::Span<PjRtDevice* const> devices() const override;
  absl::Span<PjRtDevice* const> addressable_devices() const override;

  StatusOr<PjRtDevice*> LookupDevice(int device_id) const override;

  StatusOr<PjRtDevice*> LookupAddressableDevice(
      int local_hardware_id) const override {
#ifdef PJRT_C_API_BYPASS
    TF_ASSIGN_OR_RETURN(PjRtDevice * wrapped_device,
                        wrapped_->LookupAddressableDevice(local_hardware_id));
    return GetCApiDevice(wrapped_device);
#endif  // PJRT_C_API_BYPASS
    return Unimplemented("PJRT C API does not support LookupAddressableDevice");
  }

  PjRtPlatformId platform_id() const override {
#ifdef PJRT_C_API_BYPASS
    return wrapped_->platform_id();
#endif  // PJRT_C_API_BYPASS
    CHECK(false) << "PJRT C API does not support platform_id.";
  }

  absl::string_view platform_name() const override;

  absl::string_view platform_version() const override;

  PjRtRuntimeType runtime_type() const override {
#ifdef PJRT_C_API_BYPASS
    return wrapped_->runtime_type();
#endif  // PJRT_C_API_BYPASS
    CHECK(false) << "PJRT C API does not support runtime_type.";
  }

  StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis() override {
#ifdef PJRT_C_API_BYPASS
    return wrapped_->GetHloCostAnalysis();
#endif  // PJRT_C_API_BYPASS
    return Unimplemented("PJRT C API does not support GetHloCostAnalysis");
  }

  StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) override;

  StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      mlir::ModuleOp module, CompileOptions options) override;

  StatusOr<std::optional<std::string>> ExecutableFingerprint(
      const PjRtLoadedExecutable& executable) const override;

  StatusOr<std::string> SerializeExecutable(
      const PjRtLoadedExecutable& executable) const override;

  StatusOr<std::unique_ptr<PjRtLoadedExecutable>> DeserializeExecutable(
      absl::string_view serialized, CompileOptions options) override;

  StatusOr<std::unique_ptr<PjRtBuffer>> CreateUninitializedBuffer(
      const Shape& shape, PjRtDevice* device) override {
    return Unimplemented(
        "PJRT C API does not support CreateUninitializedBuffer");
  }

  StatusOr<std::unique_ptr<AsyncBufferTransferManager>>
  CreateBuffersForAsyncTransfer(absl::Span<const Shape> shapes,
                                PjRtDevice* device) override {
    return Unimplemented(
        "PJRT C API does not support CreateBuffersForAsyncTransfer");
  }

  StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      std::function<void()> on_done_with_host_buffer,
      PjRtDevice* device) override;

  StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtDevice* device) override {
#ifdef PJRT_C_API_BYPASS
    return WrapBuffer(wrapped_->BufferFromHostLiteral(
        literal, PjRtCApiDevice::GetWrapped(device)));
#endif  // PJRT_C_API_BYPASS
    return Unimplemented("PJRT C API does not support BufferFromHostLiteral");
  }

  StatusOr<std::unique_ptr<PjRtBuffer>> CreateViewOfDeviceBuffer(
      void* device_ptr, const Shape& shape, PjRtDevice* device,
      std::function<void()> on_delete_callback) override {
#ifdef PJRT_C_API_BYPASS
    return WrapBuffer(wrapped_->CreateViewOfDeviceBuffer(
        device_ptr, shape, PjRtCApiDevice::GetWrapped(device),
        on_delete_callback));
#endif  // PJRT_C_API_BYPASS
    return Unimplemented(
        "PJRT C API does not support CreateViewOfDeviceBuffer");
  }

  StatusOr<std::uintptr_t> UnsafeBufferPointer(PjRtBuffer* buffer) override;

  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeCrossHostReceiveBuffers(absl::Span<const Shape> shapes,
                              PjRtDevice* device,
                              PjRtCrossHostRecvNotifier notifier) override {
    return Unimplemented(
        "PJRT C API does not support MakeCrossHostReceiveBuffers");
  }

  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeCrossHostReceiveBuffersForGather(
      absl::Span<const Shape> shapes, std::vector<GatherDetails> gather_details,
      PjRtDevice* device, PjRtCrossHostRecvNotifier notifier) override {
    return Unimplemented(
        "PJRT C API does not support MakeCrossHostReceiveBuffers");
  }

  StatusOr<ChannelHandle> CreateChannelHandle() override {
    return Unimplemented("PJRT C API does not support CreateChannelHandle");
  }

  StatusOr<ChannelHandle> CreateDeviceToHostChannelHandle() override {
    return Unimplemented(
        "PJRT C API does not support CreateDeviceToHostChannelHandle");
  }

  StatusOr<ChannelHandle> CreateHostToDeviceChannelHandle() override {
    return Unimplemented(
        "PJRT C API does not support CreateHostToDeviceChannelHandle");
  }

  Status Defragment() override { return wrapped_->Defragment(); }

  PjRtDevice* GetCApiDevice(PjRtDevice* wrapped_device) const {
    auto it = wrapped_device_map_.find(wrapped_device);
    CHECK(it != wrapped_device_map_.end());
    return it->second;
  }

  StatusOr<std::unique_ptr<PjRtLoadedExecutable>> WrapExecutable(
      StatusOr<std::unique_ptr<PjRtLoadedExecutable>> to_wrap);

  StatusOr<std::unique_ptr<PjRtBuffer>> WrapBuffer(
      StatusOr<std::unique_ptr<PjRtBuffer>> to_wrap);

  const PJRT_Api* pjrt_c_api() const;

  PJRT_Client* pjrt_c_client() { return c_client_.get(); }

  PjRtCApiDevice* GetCppDevice(PJRT_Device* c_device) const {
    auto it = c_to_cpp_device_map_.find(c_device);
    CHECK(it != c_to_cpp_device_map_.end());
    return it->second;
  }

 private:
  void InitDevices();

  const PJRT_Api* c_api_;
  std::unique_ptr<PJRT_Client, ::pjrt::PJRT_ClientDeleter> c_client_;

  std::vector<std::unique_ptr<PjRtCApiDevice>> owned_devices_;
  std::vector<PjRtDevice*> devices_;
  std::vector<PjRtDevice*> addressable_devices_;
  absl::flat_hash_map<PJRT_Device*, PjRtCApiDevice*> c_to_cpp_device_map_;

  // TODO(skyewm): this is a shim so we can run PjRtCApiClient code without the
  // C API being fully implemented. All methods using wrapped_ should either be
  // marked unimplemented or implemented in terms of the C API, at which point
  // wrapped_ and related functionality should be removed.
  PjRtClient* wrapped_;
  absl::flat_hash_map<PjRtDevice*, PjRtCApiDevice*> wrapped_device_map_;
};

class PjRtCApiBuffer : public PjRtBuffer {
 public:
  PjRtCApiBuffer(PjRtCApiClient* client, PJRT_Buffer* buffer);

  const Shape& on_device_shape() const override;

  StatusOr<Shape> logical_on_device_shape() override {
#ifdef PJRT_C_API_BYPASS
    return wrapped_->logical_on_device_shape();
#endif  // PJRT_C_API_BYPASS
    return Unimplemented("PJRT C API does not support logical_on_device_shape");
  }

  PjRtDevice* device() const override;

  PjRtClient* client() const override { return client_; }

  StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference()
      override {
#ifdef PJRT_C_API_BYPASS
    return wrapped_->AcquireExternalReference();
#endif  // PJRT_C_API_BYPASS
    return Unimplemented(
        "PJRT C API does not support AcquireExternalReference");
  }

  PjRtFuture<Status> ToLiteral(MutableLiteralBase* literal) override;

  StatusOr<size_t> GetOnDeviceSizeInBytes() const override;

  PjRtFuture<Status> CopyRawToHost(void* dst, int64_t offset,
                                   int64_t transfer_size) override {
#ifdef PJRT_C_API_BYPASS
    return wrapped_->CopyRawToHost(dst, offset, transfer_size);
#endif  // PJRT_C_API_BYPASS
    return PjRtFuture<Status>(
        Unimplemented("PJRT C API does not support CopyRawToHost"));
  }

  void Delete() override;

  StatusOr<std::unique_ptr<ExternalReference>> ReleaseDeviceMemoryOwnership(
      bool wait_for_operations_to_complete) override {
#ifdef PJRT_C_API_BYPASS
    return wrapped_->ReleaseDeviceMemoryOwnership(
        wait_for_operations_to_complete);
#endif  // PJRT_C_API_BYPASS
    return Unimplemented(
        "PJRT C API does not support ReleaseDeviceMemoryOwnership");
  }

  bool IsDeleted() override;

  StatusOr<std::unique_ptr<PjRtBuffer>> CopyToDevice(
      PjRtDevice* dst_device) override;

  void CopyToRemoteDevice(absl::string_view serialized_descriptor,
                          RemoteSendCallback on_done) override {
    LOG(ERROR) << "PJRT C API does not support CopyToRemoteDevice";
  }

  void CopyToRemoteDeviceScattered(
      absl::Span<const std::pair<std::string, RemoteSendCallback>>
          serialized_descriptors_and_callbacks,
      const ScatterDetails& scatter_details) override {
    LOG(ERROR) << "PJRT C API does not support CopyToRemoteDeviceScattered";
  }

  PjRtFuture<Status> GetReadyFuture() override;

  bool IsOnCpu() const override;

  PjRtBuffer* wrapped() const { return wrapped_; }

  PJRT_Buffer* c_buffer() const { return buffer_.get(); }

  static PjRtBuffer* GetWrapped(PjRtBuffer* c_api_buffer) {
    return tensorflow::down_cast<PjRtCApiBuffer*>(c_api_buffer)->wrapped();
  }

  static std::vector<PjRtBuffer*> GetWrappedVector(
      absl::Span<PjRtBuffer* const> c_api_buffers) {
    std::vector<PjRtBuffer*> wrapped;
    wrapped.reserve(c_api_buffers.size());
    for (PjRtBuffer* c_api_buf : c_api_buffers) {
      wrapped.push_back(GetWrapped(c_api_buf));
    }
    return wrapped;
  }

  const PJRT_Api* pjrt_c_api() const { return client_->pjrt_c_api(); }

 private:
  // TODO(b/238999986): Refactor or Remove.
  void set_shape();

  // Gets the raw pointer to `readiness_event_`. If `readiness_event_` has not
  // yet been initialized, this function does so before returning the pointer.
  PJRT_Event* GetReadyEvent();

  // `MakePromiseTrackEvent` sets `readiness_promise_` up to track
  // `readiness_event_`. This is used to implement `GetReadyFuture()`.
  // `readiness_promise_` should be created before calling this function.
  void MakePromiseTrackEvent();

  PjRtCApiClient* client_;
  std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter> buffer_;
  std::optional<xla::Shape> shape_;
  std::unique_ptr<PJRT_Event, ::pjrt::PJRT_EventDeleter> readiness_event_;
  // This is a shared_ptr to keep the underlying future alive even if
  // `readiness_promise` is destroyed before `readiness_event`, and the callback
  // we set on `readiness_event` modifies `readiness_promise_`.
  std::shared_ptr<PjRtFuture<Status>::Promise> readiness_promise_;

  // TODO(amangu): _wrapped is a non-C API pointer that was used to bypass the
  // C API calls until all the C API's got implemented. Remove it when it's
  // usage is reduced to zero.
  PjRtBuffer* wrapped_;
};

class PjRtCApiExecutable : public PjRtLoadedExecutable {
 public:
  PjRtCApiExecutable(PjRtCApiClient* client,
                     std::unique_ptr<PjRtLoadedExecutable> wrapped);

  PjRtCApiExecutable(PjRtCApiClient* client, PJRT_Executable* executable);

  PjRtClient* client() const override { return client_; }
  absl::string_view name() const override;
  int num_replicas() const override { return wrapped()->num_replicas(); }
  int num_partitions() const override { return wrapped()->num_partitions(); }

  int64_t SizeOfGeneratedCodeInBytes() const override {
#ifdef PJRT_C_API_BYPASS
    return wrapped()->SizeOfGeneratedCodeInBytes();
#endif  // PJRT_C_API_BYPASS
    CHECK(false) << "PJRT C API does not support SizeOfGeneratedCodeInBytes";
  }

  const DeviceAssignment& device_assignment() const override {
#ifdef PJRT_C_API_BYPASS
    return wrapped()->device_assignment();
#endif  // PJRT_C_API_BYPASS
    CHECK(false) << "PJRT C API does not support device_assignment";
  }

  absl::Span<const LogicalDeviceIds> addressable_device_logical_ids()
      const override {
#ifdef PJRT_C_API_BYPASS
    return wrapped()->addressable_device_logical_ids();
#endif  // PJRT_C_API_BYPASS
    CHECK(false)
        << "PJRT C API does not support addressable_device_logical_ids";
  }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return addressable_devices_;
  }

  StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override {
#ifdef PJRT_C_API_BYPASS
    return wrapped()->GetHloModules();
#endif  // PJRT_C_API_BYPASS
    return Unimplemented("PJRT C API does not support GetHloModules");
  }

  StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options,
      std::optional<std::vector<PjRtFuture<Status>>>& returned_futures)
      override;

  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<Status>>& returned_future,
      bool fill_future) override;

  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<Status>>& returned_future,
      bool fill_future) override;

  void Delete() override;
  bool IsDeleted() override;

  PjRtLoadedExecutable* wrapped() const;

  static PjRtLoadedExecutable* GetWrapped(
      const PjRtLoadedExecutable* c_api_executable) {
    return tensorflow::down_cast<const PjRtCApiExecutable*>(c_api_executable)
        ->wrapped();
  }

  const PJRT_Api* pjrt_c_api() const { return client_->pjrt_c_api(); }

 private:
  PjRtCApiClient* client_;
  std::unique_ptr<PJRT_Executable, pjrt::PJRT_ExecutableDeleter> executable_;
  std::vector<PjRtDevice*> addressable_devices_;

  void InitDevices();
};

StatusOr<std::unique_ptr<PjRtClient>> GetCApiClient();

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_PJRT_C_API_CLIENT_H_
