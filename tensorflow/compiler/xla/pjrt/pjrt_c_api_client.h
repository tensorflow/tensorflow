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
#include "tensorflow/compiler/xla/pjrt/pjrt_compiler.h"

namespace xla {
// If false, PjRtCApiClient will raise an error on methods unimplemented in the
// PJRT C API. If true, PjRtCApiClient and related classes will assume the
// wrapper impl is being used and call directly into the wrapped C++ PJRT
// client. This can be useful for testing, but is generally not safe to use
// across library boundaries.
extern bool kPjRtCApiBypass;

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
    if (kPjRtCApiBypass) {
      VLOG(1) << "PJRT C API BYPASS: TransferToInfeed";
      return wrapped_->TransferToInfeed(literal);
    }
    return Unimplemented("PJRT C API does not support TransferToInfeed");
  }

  Status TransferFromOutfeed(MutableBorrowingLiteral literal) override {
    if (kPjRtCApiBypass) {
      VLOG(1) << "PJRT C API BYPASS: TransferFromOutfeed";
      return wrapped_->TransferFromOutfeed(std::move(literal));
    }
    return Unimplemented("PJRT C API does not support TransferFromOutfeed");
  }

  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override {
    if (kPjRtCApiBypass) {
      VLOG(1) << "PJRT C API BYPASS: CreateAsyncTrackingEvent";
      return wrapped_->CreateAsyncTrackingEvent(description);
    }
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
      int local_hardware_id) const override;

  PjRtPlatformId platform_id() const override {
    if (kPjRtCApiBypass) {
      VLOG(1) << "PJRT C API BYPASS: platform_id";
      return wrapped_->platform_id();
    }
    CHECK(false) << "PJRT C API does not support platform_id.";
  }

  absl::string_view platform_name() const override;

  absl::string_view platform_version() const override;

  // TODO(b/244756954): Rethink this function altogether
  PjRtRuntimeType runtime_type() const override {
    if (kPjRtCApiBypass) {
      VLOG(1) << "PJRT C API BYPASS: runtime_type";
      return wrapped_->runtime_type();
    }
    return PjRtRuntimeType::kTfrt;
  }

  StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis()
      const override {
    return Unimplemented("PJRT C API does not support GetHloCostAnalysis");
  }

  StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) override;

  StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      mlir::ModuleOp module, CompileOptions options) override;

  StatusOr<std::optional<std::string>> ExecutableFingerprint(
      const PjRtLoadedExecutable& executable) const override;

  // `PjRtCApiClient::DeserializeExecutable()` ignores `CompileOptions` arg
  StatusOr<std::unique_ptr<PjRtLoadedExecutable>> DeserializeExecutable(
      absl::string_view serialized,
      std::optional<CompileOptions> options) override;

  StatusOr<std::unique_ptr<PjRtBuffer>> CreateUninitializedBuffer(
      const Shape& shape, PjRtDevice* device) override {
    return Unimplemented(
        "PJRT C API does not support CreateUninitializedBuffer");
  }

  StatusOr<std::unique_ptr<AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(absl::Span<const Shape> shapes,
                                    PjRtDevice* device) override {
    return Unimplemented(
        "PJRT C API does not support CreateBuffersForAsyncHostToDevice");
  }

  StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      std::function<void()> on_done_with_host_buffer,
      PjRtDevice* device) override;

  StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtDevice* device) override {
    if (kPjRtCApiBypass) {
      VLOG(1) << "PJRT C API BYPASS: BufferFromHostLiteral";
      return WrapBuffer(wrapped_->BufferFromHostLiteral(
          literal, PjRtCApiDevice::GetWrapped(device)));
    }
    return Unimplemented("PJRT C API does not support BufferFromHostLiteral");
  }

  StatusOr<std::unique_ptr<PjRtBuffer>> CreateViewOfDeviceBuffer(
      void* device_ptr, const Shape& shape, PjRtDevice* device,
      std::function<void()> on_delete_callback) override {
    if (kPjRtCApiBypass) {
      VLOG(1) << "PJRT C API BYPASS: CreateViewOfDeviceBuffer";
      return WrapBuffer(wrapped_->CreateViewOfDeviceBuffer(
          device_ptr, shape, PjRtCApiDevice::GetWrapped(device),
          on_delete_callback));
    }
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

  Status Defragment() override {
    return Unimplemented("PJRT C API does not support Defragment");
  }

  bool SupportsSendRecvCallbacks() const override { return true; }

  StatusOr<std::unique_ptr<PjRtBuffer>> WrapBuffer(
      StatusOr<std::unique_ptr<PjRtBuffer>> to_wrap);

  const PJRT_Api* pjrt_c_api() const;

  PJRT_Client* pjrt_c_client() { return c_client_.get(); }

  PjRtCApiDevice* GetCppDevice(PJRT_Device* c_device) const {
    auto it = c_to_cpp_device_map_.find(c_device);
    CHECK(it != c_to_cpp_device_map_.end());
    return it->second;
  }

  // Returns nullptr if `kPjRtCApiBypass` is not set, until the C API
  // device manager is implemented.
  // TODO(b/267063498) return the PjRtHostMemoryForDeviceManager for the wrapped
  // client.
  PjRtHostMemoryForDeviceManager* GetPjRtHostMemoryForDeviceManager()
      const override {
    if (kPjRtCApiBypass) {
      VLOG(1) << "PJRT C API BYPASS: GetPjRtHostMemoryForDeviceManager";
      return wrapped_->GetPjRtHostMemoryForDeviceManager();
    }
    return nullptr;
  }

 private:
  void InitDevices();

  const PJRT_Api* c_api_;
  std::unique_ptr<PJRT_Client, ::pjrt::PJRT_ClientDeleter> c_client_;

  std::vector<std::unique_ptr<PjRtCApiDevice>> owned_devices_;
  std::vector<PjRtDevice*> devices_;
  std::vector<PjRtDevice*> addressable_devices_;
  absl::flat_hash_map<PJRT_Device*, PjRtCApiDevice*> c_to_cpp_device_map_;

  const std::string platform_version_;

  // TODO(skyewm): this is a shim so we can run PjRtCApiClient code without the
  // C API being fully implemented. All methods using wrapped_ should either be
  // marked unimplemented or implemented in terms of the C API, at which point
  // wrapped_ and related functionality should be removed.
  PjRtClient* wrapped_;
};

class PjRtCApiBuffer : public PjRtBuffer {
 public:
  PjRtCApiBuffer(PjRtCApiClient* client, PJRT_Buffer* buffer);

  const Shape& on_device_shape() const override;

  StatusOr<Shape> logical_on_device_shape() override;

  PjRtDevice* device() const override;

  PjRtClient* client() const override { return client_; }

  StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference()
      override {
    return Unimplemented(
        "PJRT C API does not support AcquireExternalReference");
  }

  PjRtFuture<Status> ToLiteral(MutableLiteralBase* literal) override;

  StatusOr<size_t> GetOnDeviceSizeInBytes() const override;

  PjRtFuture<Status> CopyRawToHost(void* dst, int64_t offset,
                                   int64_t transfer_size) override {
    return PjRtFuture<Status>(
        Unimplemented("PJRT C API does not support CopyRawToHost"));
  }

  void Delete() override;

  StatusOr<std::unique_ptr<ExternalReference>> ReleaseDeviceMemoryOwnership(
      bool wait_for_operations_to_complete) override {
    return Unimplemented(
        "PJRT C API does not support ReleaseDeviceMemoryOwnership");
  }

  bool IsDeleted() override;

  StatusOr<std::unique_ptr<PjRtBuffer>> CopyToDevice(
      PjRtDevice* dst_device) override;

  void CopyToRemoteDevice(
      PjRtFuture<StatusOr<std::string>> serialized_descriptor,
      RemoteSendCallback on_done) override {
    LOG(ERROR) << "PJRT C API does not support CopyToRemoteDevice";
  }

  void CopyToRemoteDeviceScattered(
      PjRtFuture<StatusOr<std::vector<std::string>>> serialized_descriptors,
      std::vector<RemoteSendCallback> callbacks,
      const ScatterDetails& scatter_details) override {
    LOG(ERROR) << "PJRT C API does not support CopyToRemoteDeviceScattered";
  }

  PjRtFuture<Status> GetReadyFuture() override;

  bool IsOnCpu() const override;

  PJRT_Buffer* c_buffer() const { return buffer_.get(); }

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
};

class PjRtCApiExecutable : public PjRtExecutable {
 public:
  PjRtCApiExecutable(const PJRT_Api* c_api, PJRT_Executable* executable);

  absl::string_view name() const override;
  int num_replicas() const override;
  int num_partitions() const override;

  int64_t SizeOfGeneratedCodeInBytes() const override;

  StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override;

  const PJRT_Api* pjrt_c_api() const { return c_api_; }
  PJRT_Executable* c_executable() const { return executable_.get(); }

  StatusOr<std::string> SerializeExecutable() const override;

 private:
  const PJRT_Api* c_api_;
  std::unique_ptr<PJRT_Executable, pjrt::PJRT_ExecutableDeleter> executable_;
};

class PjRtCApiLoadedExecutable : public PjRtLoadedExecutable {
 public:
  PjRtCApiLoadedExecutable(PjRtCApiClient* client,
                           PJRT_LoadedExecutable* executable);

  PjRtClient* client() const override { return client_; }
  absl::string_view name() const override { return executable_->name(); }
  int num_replicas() const override { return executable_->num_replicas(); }
  int num_partitions() const override { return executable_->num_partitions(); }

  int64_t SizeOfGeneratedCodeInBytes() const override {
    return executable_->SizeOfGeneratedCodeInBytes();
  }

  StatusOr<absl::flat_hash_map<std::string, PjRtValueType>> GetCostAnalysis()
      const override;

  const DeviceAssignment& device_assignment() const override {
    CHECK(false) << "PJRT C API does not support device_assignment";
  }

  absl::Span<const LogicalDeviceIds> addressable_device_logical_ids()
      const override {
    CHECK(false)
        << "PJRT C API does not support addressable_device_logical_ids";
  }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return addressable_devices_;
  }

  StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override {
    return executable_->GetHloModules();
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

  StatusOr<std::string> SerializeExecutable() const override {
    return executable_->SerializeExecutable();
  }

  const PJRT_Api* pjrt_c_api() const { return client_->pjrt_c_api(); }
  PJRT_Executable* c_executable() const { return executable_->c_executable(); }

  PJRT_LoadedExecutable* c_loaded_executable() const {
    return loaded_executable_.get();
  }

  // True if the `returned_futures` output parameter is supported in the
  // Execute*() methods.
  bool IsReturnedFutureSupported() const override { return true; }

  // std::function version of PJRT_SendCallback
  using SendCallbackFunction = std::function<bool(PJRT_Chunk*, size_t, bool)>;
  // std::function version of PJRT_RecvCallback
  using RecvCallbackFunction = std::function<void(PJRT_CopyToDeviceStream*)>;

 private:
  // Groups data needed to support send/recv execution callbacks.
  struct SendRecvCallbackData {
    std::vector<std::vector<PJRT_SendCallbackInfo>> c_send_callbacks;
    std::vector<PJRT_SendCallbackInfo*> c_send_callback_lists;
    std::vector<std::vector<PJRT_RecvCallbackInfo>> c_recv_callbacks;
    std::vector<PJRT_RecvCallbackInfo*> c_recv_callback_lists;
    std::vector<SendCallbackFunction> send_callback_functions;
    std::vector<RecvCallbackFunction> recv_callback_functions;
  };

  // Gets common Execute_Args between Execute, ExecuteSharded and
  // ExecutePortable. device_complete_events in the return is set if the input
  // device_complete_events has value.
  xla::StatusOr<PJRT_LoadedExecutable_Execute_Args> GetCommonExecuteArgs(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options, PJRT_ExecuteOptions& c_options,
      std::vector<std::vector<PJRT_Buffer*>>& c_argument_lists_storage,
      std::vector<PJRT_Buffer**>& c_arguments,
      std::vector<std::vector<PJRT_Buffer*>>& c_output_lists_storage,
      std::vector<PJRT_Buffer**>& c_output_lists,
      std::optional<std::vector<PJRT_Event*>>& device_complete_events,
      SendRecvCallbackData& send_recv_callback_data);

  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteWithSingleDevice(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<Status>>& returned_future, bool fill_future);

  PjRtCApiClient* client_;
  std::unique_ptr<PJRT_LoadedExecutable, pjrt::PJRT_LoadedExecutableDeleter>
      loaded_executable_;
  std::unique_ptr<PjRtCApiExecutable> executable_;
  std::vector<PjRtDevice*> addressable_devices_;

  void InitDevices();
};

class PjRtCApiCompiler : public PjRtCompiler {
 public:
  explicit PjRtCApiCompiler(const PJRT_Api* c_api) : c_api_(c_api) {}

  StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, const XlaComputation& computation,
      const PjRtDeviceTopology& topology, PjRtClient* client) override;

  StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, mlir::ModuleOp module,
      const PjRtDeviceTopology& topology, PjRtClient* client) override;

 private:
  const PJRT_Api* c_api_;
};

class PjRtCApiDeviceTopology : public PjRtDeviceTopology {
 public:
  PjRtCApiDeviceTopology(const PJRT_Api* c_api,
                         PJRT_DeviceTopology* c_topology);

  PjRtPlatformId platform_id() const override {
    CHECK(false) << "PJRT C API does not support platform_id.";
  }

  absl::string_view platform_name() const override;

  absl::string_view platform_version() const override;

  std::optional<PjRtCompiler*> compiler() const override {
    return compiler_.get();
  }

  const PJRT_DeviceTopology* c_topology() const { return c_topology_.get(); }

 private:
  std::unique_ptr<PjRtCApiCompiler> compiler_;
  const PJRT_Api* c_api_;
  std::unique_ptr<PJRT_DeviceTopology, ::pjrt::PJRT_DeviceTopologyDeleter>
      c_topology_;
};

class CApiCopyToDeviceStream : public CopyToDeviceStream {
 public:
  CApiCopyToDeviceStream(PJRT_CopyToDeviceStream* c_stream,
                         const PJRT_Api* c_api);

  PjRtFuture<Status> AddChunk(PjRtChunk chunk) override;

 private:
  PJRT_CopyToDeviceStream* c_stream_;
  const PJRT_Api* c_api_;
};

StatusOr<std::unique_ptr<PjRtClient>> GetCApiClient(
    absl::string_view device_type,
    const absl::flat_hash_map<std::string, PjRtValueType>& create_options = {});

StatusOr<std::unique_ptr<PjRtDeviceTopology>> GetCApiTopology(
    absl::string_view device_type);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_PJRT_C_API_CLIENT_H_
