/* Copyright 2023 The OpenXLA Authors.

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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

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
  absl::StatusOr<Shape> logical_on_device_shape() override {
    return wrapped_->logical_on_device_shape();
  }
  PjRtMemorySpace* memory_space() const override {
    return wrapped_->memory_space();
  }
  PjRtDevice* device() const override { return wrapped_->device(); }
  PjRtClient* client() const override;
  absl::StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference()
      override {
    return wrapped_->AcquireExternalReference();
  }
  PjRtFuture<> ToLiteral(MutableLiteralBase* literal) override {
    return wrapped_->ToLiteral(literal);
  }
  PjRtFuture<> LazyToLiteral(
      absl::AnyInvocable<PjRtFuture<MutableLiteralBase*>() &&> generator)
      override {
    return wrapped_->LazyToLiteral(std::move(generator));
  }
  absl::StatusOr<size_t> GetOnDeviceSizeInBytes() const override {
    return wrapped_->GetOnDeviceSizeInBytes();
  }
  PjRtFuture<> CopyRawToHost(void* dst, int64_t offset,
                             int64_t transfer_size) override {
    return wrapped_->CopyRawToHost(dst, offset, transfer_size);
  }
  void Delete() override { wrapped_->Delete(); }
  absl::StatusOr<std::unique_ptr<ExternalReference>>
  ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete) override {
    return wrapped_->ReleaseDeviceMemoryOwnership(
        wait_for_operations_to_complete);
  }
  bool IsDeleted() const override { return wrapped_->IsDeleted(); }
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(
      PjRtMemorySpace* dst_memory_space) override;
  void CopyToRemoteDevice(PjRtFuture<std::string> serialized_descriptor,
                          RemoteSendCallback on_done) override {
    wrapped_->CopyToRemoteDevice(std::move(serialized_descriptor),
                                 std::move(on_done));
  }
  PjRtFuture<> GetReadyFuture() override { return wrapped_->GetReadyFuture(); }
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
  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override {
    return wrapped_->GetHloModules();
  }
  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override {
    return wrapped_->GetOutputMemoryKinds();
  }
  using PjRtLoadedExecutable::Execute;
  absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options,
      std::optional<std::vector<PjRtFuture<>>>& returned_futures)
      const override;
  using PjRtLoadedExecutable::ExecuteSharded;
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<>>& returned_future,
      bool fill_future) const override;
  using PjRtLoadedExecutable::ExecutePortable;
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<>>& returned_future,
      bool fill_future) const override;

  void Delete() override { return wrapped_->Delete(); }
  bool IsDeleted() const override { return wrapped_->IsDeleted(); }

  absl::StatusOr<std::string> SerializeExecutable() const override {
    return wrapped_->SerializeExecutable();
  }

  absl::StatusOr<struct CompileOptions> GetCompileOptions() const override {
    return wrapped_->GetCompileOptions();
  }

  absl::StatusOr<std::string> FingerprintExecutable() const override {
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
  absl::StatusOr<PjRtDevice*> LookupDevice(
      PjRtGlobalDeviceId global_device_id) const override {
    return wrapped_->LookupDevice(global_device_id);
  }
  absl::StatusOr<PjRtDevice*> LookupAddressableDevice(
      PjRtLocalDeviceId local_device_id) const override {
    if (wrapped_ == nullptr) {
      return absl::InternalError(
          "Wrapped PJRT client in TfPjRtClient is already destroyed.");
    }
    return wrapped_->LookupAddressableDevice(local_device_id);
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
  std::optional<PjRtPluginAttributes> plugin_attributes() const override;
  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override {
    return wrapped_->GetDefaultDeviceAssignment(num_replicas, num_partitions);
  }
  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type, absl::Span<const int64_t> dims) override {
    return wrapped_->GetDefaultLayout(element_type, dims);
  }
  absl::StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis()
      const override {
    return wrapped_->GetHloCostAnalysis();
  }
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      const XlaComputation& computation, CompileOptions options) override {
    return WrapExecutable(wrapped_->CompileAndLoad(computation, options));
  }
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      mlir::ModuleOp module, CompileOptions options) override {
    return WrapExecutable(wrapped_->CompileAndLoad(std::move(module), options));
  }

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
  LoadSerializedExecutable(absl::string_view serialized,
                           std::optional<CompileOptions> options,
                           const LoadOptions& load_options) override {
    return WrapExecutable(
        wrapped_->LoadSerializedExecutable(serialized, options, load_options));
  }

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateUninitializedBuffer(
      const Shape& shape, PjRtMemorySpace* memory_space) override {
    return Unimplemented(
        "CreateUninitializedBuffer not supported for TfPjRtClient.");
  }
  absl::StatusOr<std::unique_ptr<AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(absl::Span<const Shape> shapes,
                                    PjRtMemorySpace* memory_space) override {
    return Unimplemented(
        "AsyncHostToDeviceTransferManager not supported for Tf.");
  }
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtMemorySpace* memory_space, const Layout* device_layout) override {
    return WrapBuffer(wrapped_->BufferFromHostBuffer(
        data, type, dims, byte_strides, host_buffer_semantics,
        std::move(on_done_with_host_buffer), memory_space, device_layout));
  }
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtMemorySpace* memory_space,
      const Layout* device_layout) override {
    return WrapBuffer(
        wrapped_->BufferFromHostLiteral(literal, memory_space, device_layout));
  }
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateViewOfDeviceBuffer(
      void* device_ptr, const Shape& shape, PjRtMemorySpace* memory_space,
      std::function<void()> on_delete_callback,
      std::optional<std::intptr_t> stream) override {
    return WrapBuffer(wrapped_->CreateViewOfDeviceBuffer(
        device_ptr, shape, memory_space, on_delete_callback, stream));
  }
  absl::StatusOr<std::uintptr_t> UnsafeBufferPointer(
      PjRtBuffer* buffer) override {
    return wrapped_->UnsafeBufferPointer(UnwrapBuffer(buffer));
  }
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeCrossHostReceiveBuffers(absl::Span<const Shape> shapes,
                              PjRtDevice* device,
                              PjRtCrossHostRecvNotifier notifier) override {
    return wrapped_->MakeCrossHostReceiveBuffers(shapes, device,
                                                 std::move(notifier));
  }
  absl::StatusOr<const PjRtTopologyDescription*> GetTopologyDescription()
      const override {
    return wrapped_->GetTopologyDescription();
  }

  PjRtClient* wrapped() const { return wrapped_.get(); }

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> WrapBuffer(
      absl::StatusOr<std::unique_ptr<PjRtBuffer>> to_wrap);

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

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> WrapExecutable(
      absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> to_wrap);

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
