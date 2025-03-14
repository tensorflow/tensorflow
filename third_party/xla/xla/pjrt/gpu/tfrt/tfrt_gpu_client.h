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

#ifndef XLA_PJRT_GPU_TFRT_TFRT_GPU_CLIENT_H_
#define XLA_PJRT_GPU_TFRT_TFRT_GPU_CLIENT_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "mlir/IR/BuiltinOps.h"
#include "xla/client/local_client.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/pjrt/gpu/gpu_topology.h"
#include "xla/pjrt/gpu/tfrt/gpu_event.h"
#include "xla/pjrt/gpu/tfrt/stream_pool.h"
#include "xla/pjrt/gpu/tfrt/tracked_tfrt_gpu_device_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_stream_executor_device_description.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

class TfrtGpuMemorySpace : public PjRtMemorySpace {
 public:
  TfrtGpuMemorySpace(int id, PjRtDevice* device, absl::string_view kind,
                     int kind_id);

  PjRtClient* client() const override { return device_->client(); }

  absl::Span<PjRtDevice* const> devices() const override {
    return absl::Span<PjRtDevice* const>(&device_, device_ != nullptr ? 1 : 0);
  }

  int id() const override { return id_; }

  absl::string_view kind() const override { return kind_; }

  int kind_id() const override { return kind_id_; }

  absl::string_view DebugString() const override { return debug_string_; }

  absl::string_view ToString() const override { return to_string_; }

 private:
  int id_;
  PjRtDevice* device_ = nullptr;
  absl::string_view kind_;
  int kind_id_;
  std::string debug_string_;
  std::string to_string_;
};

class TfrtGpuDeviceMemorySpace : public TfrtGpuMemorySpace {
 public:
  static constexpr absl::string_view kKind = "device";
  static const int kKindId;

  TfrtGpuDeviceMemorySpace(int id, PjRtDevice* device);
};

class TfrtGpuDevice final : public PjRtDevice {
 public:
  struct Options {
    int id;
    PjRtLocalDeviceId local_device_id;
    PjRtLocalHardwareId local_hardware_id;
    se::StreamExecutor* executor;
    std::unique_ptr<tsl::Allocator> allocator;
    int stream_capacity;
    int max_inflight_computations;
    std::string platform_version;
  };

  explicit TfrtGpuDevice(Options&& options);

  void SetClient(PjRtClient* client) {
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

  const PjRtStreamExecutorDeviceDescription& description() const override {
    return description_;
  }

  PjRtClient* client() const override { return client_; }

  bool IsAddressable() const override {
    return process_index() == client()->process_index();
  }

  int id() const override { return id_; }

  PjRtLocalDeviceId local_device_id() const override {
    return local_device_id_;
  }

  // Used as `device_ordinal`.
  PjRtLocalHardwareId local_hardware_id() const override {
    return local_hardware_id_;
  }

  absl::Status TransferToInfeed(const LiteralSlice& literal) override;

  absl::Status TransferFromOutfeed(MutableBorrowingLiteral literal) override;

  void AttachMemorySpace(PjRtMemorySpace* memory_space,
                         bool is_default = false);

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

  absl::StatusOr<PjRtMemorySpace*> memory_space_by_kind_id(int id) const;

  absl::StatusOr<PjRtMemorySpace*> memory_space_by_kind(
      absl::string_view kind) const override;

  absl::StatusOr<PjRtMemorySpace*> default_memory_space() const override;

  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override {
    return nullptr;
  }

  tsl::Allocator* allocator() { return allocator_.get(); }

  BoundedStreamPool& stream_pool() { return stream_pool_; }

  BoundedStreamPool& compute_stream_pool() { return compute_stream_pool_; }

 private:
  friend class TfrtGpuClient;

  int id_;
  PjRtClient* client_ = nullptr;
  PjRtLocalDeviceId local_device_id_;
  PjRtLocalHardwareId local_hardware_id_;
  BoundedStreamPool stream_pool_;
  // TODO(b/400541410): Support H2D transfers on compute streams.
  //   Have a dedicated compute stream pool to avoid blocking the stream pool
  //   for H2D transfers.
  BoundedStreamPool compute_stream_pool_;
  std::unique_ptr<tsl::Allocator> allocator_;
  absl::InlinedVector<PjRtMemorySpace*, 1> memory_spaces_;
  absl::flat_hash_map<int, PjRtMemorySpace*> memory_spaces_by_kind_id_;

  PjRtStreamExecutorDeviceDescription description_;
  PjRtMemorySpace* default_memory_space_ = nullptr;
};

class TfrtGpuClient final : public PjRtClient {
 public:
  TfrtGpuClient(int process_index, xla::LocalClient* xla_client,
                std::vector<std::unique_ptr<TfrtGpuDevice>> devices,
                std::unique_ptr<tsl::Allocator> host_memory_allocator,
                std::shared_ptr<const GpuTopology> gpu_topology);

  int process_index() const override { return process_index_; }

  int device_count() const override { return devices_.size(); }

  int addressable_device_count() const override {
    return addressable_devices_.size();
  }

  absl::Span<PjRtDevice* const> devices() const override { return devices_; }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return addressable_devices_;
  }

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

  PjRtPlatformId platform_id() const override {
    // TODO(b/382117736): Add support for ROCM and SYCL.
    return tsl::Fingerprint64(xla::CudaName());
  }

  absl::string_view platform_name() const override { return xla::CudaName(); }

  absl::string_view platform_version() const override {
    return platform_version_;
  }

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      const XlaComputation& computation, CompileOptions options) override;
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      mlir::ModuleOp mlir_module, CompileOptions options) override;

 private:
  // Helper function for creating PjRtStreamExecutorExecutables. Modifies
  // `options` in-place.
  struct ExecutableExtras {
    std::shared_ptr<DeviceAssignment> device_assignment;
    std::vector<PjRtLoadedExecutable::LogicalDeviceIds>
        addressable_device_logical_ids;
    std::vector<PjRtDevice*> addressable_devices;
  };
  absl::StatusOr<ExecutableExtras> GetExecutableExtras(CompileOptions* options);

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileInternal(
      const XlaComputation& computation,
      const std::vector<const Shape*>& argument_layout_pointers,
      LayoutCanonicalizationCallback layout_canonicalization_callback,
      CompileOptions options);

  int process_index_;

  xla::LocalClient* xla_client_;

  const std::string platform_version_;

  // Device memory allocator. If owned, the allocator must outlive the devices,
  // because it is the device destructor that waits for any outstanding work to
  // complete.
  se::DeviceMemoryAllocator* allocator_;
  std::unique_ptr<se::DeviceMemoryAllocator> owned_allocator_;

  // Includes all devices, including non-local devices on multi-host platforms.
  std::vector<std::unique_ptr<TfrtGpuDevice>> owned_devices_;
  // Pointers to `owned_devices_`.
  std::vector<PjRtDevice*> devices_;
  // Maps Device::id() to the corresponding Device. Includes all devices.
  absl::flat_hash_map<PjRtGlobalDeviceId, TfrtGpuDevice*> id_to_device_;
  // Local devices indexed by local device ordinal.
  std::vector<PjRtDevice*> addressable_devices_;

  // Addressable memory spaces.
  std::vector<std::unique_ptr<PjRtMemorySpace>> owned_memory_spaces_;
  // Pointers to `owned_memory_spaces_`.
  std::vector<PjRtMemorySpace*> memory_spaces_;

  std::unique_ptr<tsl::thread::ThreadPool> compile_thread_pool_;
};

absl::StatusOr<std::unique_ptr<PjRtClient>> GetTfrtGpuClient(
    const GpuClientOptions& options);

class TfrtGpuBuffer final : public PjRtBuffer {
 public:
  TfrtGpuBuffer(
      Shape on_device_shape,
      std::unique_ptr<TrackedTfrtGpuDeviceBuffer> tracked_device_buffer,
      TfrtGpuClient* client, TfrtGpuDevice* device,
      PjRtMemorySpace* memory_space);
  ~TfrtGpuBuffer() override;

  TfrtGpuBuffer(const TfrtGpuBuffer&) = delete;
  TfrtGpuBuffer(TfrtGpuBuffer&&) = delete;
  TfrtGpuBuffer& operator=(const TfrtGpuBuffer&) = delete;
  TfrtGpuBuffer& operator=(TfrtGpuBuffer&&) = delete;

  PjRtMemorySpace* memory_space() const override { return memory_space_; }
  const Shape& on_device_shape() const override { return on_device_shape_; }
  TfrtGpuDevice* device() const override { return device_; }
  TfrtGpuClient* client() const override { return client_; }

  absl::StatusOr<Shape> logical_on_device_shape() override {
    return Unimplemented("logical_on_device_shape not implemented.");
  }

  absl::StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference()
      override;

  absl::StatusOr<std::unique_ptr<ExternalReference>>
  ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete) override;

  using PjRtBuffer::ToLiteralSync;
  PjRtFuture<> ToLiteral(MutableLiteralBase* literal) override {
    // TODO(b/382117736): Implement ToLiteral.
    return PjRtFuture<>(Unimplemented("ToLiteral not implemented."));
  }
  PjRtFuture<> LazyToLiteral(
      absl::AnyInvocable<absl::StatusOr<MutableLiteralBase*>() &&> generator)
      override {
    // TODO(b/382117736): Implement LazyToLiteral.
    return PjRtFuture<>(Unimplemented("LazyToLiteral not implemented."));
  }

  absl::StatusOr<size_t> GetOnDeviceSizeInBytes() const override;

  PjRtFuture<> CopyRawToHost(void* dst, int64_t offset,
                             int64_t transfer_size) override {
    return CopyRawToHostFuture(PjRtFuture<void*>(dst), offset, transfer_size);
  }

  PjRtFuture<> CopyRawToHostFuture(PjRtFuture<void*> dst, int64_t offset,
                                   int64_t transfer_size) override {
    // TODO(b/382117736): Implement CopyRawToHostFuture.
    return PjRtFuture<>(Unimplemented("CopyRawToHostFuture not implemented."));
  }

  void Delete() override;

  bool IsDeleted() override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(
      PjRtMemorySpace* dst_memory_space) override {
    // TODO(b/382117736): Implement CopyToMemorySpace.
    return Unimplemented("CopyToMemorySpace not implemented.");
  }

  void CopyToRemoteDevice(PjRtFuture<std::string> serialized_descriptor,
                          RemoteSendCallback on_done) override {
    on_done(Unimplemented("CopyToRemoteDevice not implemented."),
            /*sends_were_enqueued=*/false);
  }

  PjRtFuture<> GetReadyFuture() override {
    return PjRtFuture<>(Unimplemented("GetReadyFuture not implemented."));
  }

  bool IsOnCpu() const override { return false; }

 private:
  tsl::AsyncValueRef<bool> GetDonationEvent() {
    absl::MutexLock lock(&mu_);
    return donation_event_;
  }

  void DropExternalReference();

  // Similar to Delete, drops the buffer's reference to its associated device
  // memory, leaving the buffer in an invalid state, but returns the
  // TrackedTfrtGpuDeviceBuffer rather than freeing the device memory, so that
  // another framework can take ownership of it. The buffer returned from
  // Release may be safely dropped at any time even if it still has pending
  // async operations. The client should call Await before calling Release with
  // wait_for_operations_to_complete=false, to ensure that the host has
  // synchronized past any outstanding write operations to the buffer. If
  // wait_for_operations_to_complete=true the host will block until any
  // potentially outstanding asynchronous operations have completed before
  // returning, in which case it is safe to read or mutate the returned buffer.
  // If the buffer was shared via an external reference it is the client's
  // responsibility that accesses via that reference do not interfere with
  // accesses via the buffer returned from Release.
  absl::StatusOr<std::unique_ptr<TrackedTfrtGpuDeviceBuffer>> Release(
      bool wait_for_operations_to_complete);

  // Releases the device buffer by returning a unique_ptr of it.
  std::unique_ptr<TrackedTfrtGpuDeviceBuffer> ReleaseBufferLocked()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  TfrtGpuClient* client_;
  const Shape on_device_shape_;
  TfrtGpuDevice* const device_;
  PjRtMemorySpace* const memory_space_;

  mutable absl::Mutex mu_;
  std::unique_ptr<TrackedTfrtGpuDeviceBuffer> tracked_device_buffer_
      ABSL_GUARDED_BY(mu_);
  // Count of external references on the buffer.
  int external_reference_counter_ ABSL_GUARDED_BY(mu_) = 0;

  // `pending_donation_` indicates whether a donation is pending. The destructor
  // of the TfrtGpuBuffer will wait for a pending donation, as the donation
  // might fail. Note that concurrent calls to AcquireUsage() and
  // AcquireDonation() might fail even if the pending donation is aborted later.
  tsl::AsyncValueRef<bool> donation_event_ ABSL_GUARDED_BY(mu_);
  PjRtFuture<>::Promise definition_promise_ ABSL_GUARDED_BY(mu_);

  // This event is triggered when the last external reference is released.
  // It is used to make sure that the buffer is not deleted before all external
  // references are dropped.
  // Notice that this event won't be triggered if there is never an external
  // reference.
  tsl::AsyncValueRef<GpuEvent> external_references_dropped_event_
      ABSL_GUARDED_BY(mu_);

  friend class TfrtGpuClient;
  friend class TfrtGpuExecutable;
};

class TfrtGpuExecutable final : public PjRtLoadedExecutable {
 public:
  TfrtGpuExecutable(
      std::vector<std::unique_ptr<LocalExecutable>> executables,
      bool parameter_is_tupled_arguments,
      std::shared_ptr<DeviceAssignment> device_assignment,
      CompileOptions compile_options,
      std::vector<LogicalDeviceIds> addressable_device_logical_ids,
      std::vector<PjRtDevice*> addressable_devices, TfrtGpuClient* client);

  TfrtGpuClient* client() const override { return client_; }

  absl::string_view name() const override;

  int num_replicas() const override {
    return executables_[0]->build_options().num_replicas();
  }

  int num_partitions() const override {
    return executables_[0]->build_options().num_partitions();
  }

  int64_t SizeOfGeneratedCodeInBytes() const override {
    int64_t size = 0;
    for (auto& executable : executables_) {
      size += executable->executable()->SizeOfGeneratedCodeInBytes();
    }
    return size;
  }

  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override;

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override;

  const DeviceAssignment& device_assignment() const override {
    return *device_assignment_;
  }

  absl::Span<const LogicalDeviceIds> addressable_device_logical_ids()
      const override {
    return addressable_device_logical_ids_;
  }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return addressable_devices_;
  }

  using PjRtLoadedExecutable::Execute;
  absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options,
      std::optional<std::vector<PjRtFuture<>>>& returned_futures) override {
    return Unimplemented("Not implemented");
  }

  using PjRtLoadedExecutable::ExecuteSharded;
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<>>& returned_future, bool fill_future) override {
    return Unimplemented("Not implemented");
  }

  using PjRtLoadedExecutable::ExecutePortable;
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<>>& returned_future, bool fill_future) override {
    return Unimplemented("Not implemented");
  }

  void Delete() override { executables_.clear(); }

  bool IsDeleted() override { return executables_.empty(); }

  absl::StatusOr<CompileOptions> GetCompileOptions() const override {
    return compile_options_;
  }

  absl::StatusOr<std::string> FingerprintExecutable() const override {
    return fingerprint_;
  };

  void SetInputHloSnapshotBits(HloModuleProto hlo_module,
                               DebugOptions debug_options) {
    input_hlo_snapshot_bits_ =
        std::make_optional<InputHloSnapshotBits>(InputHloSnapshotBits{
            HloModuleProto(std::move(hlo_module)), std::move(debug_options)});
  }

 private:
  friend class TfrtGpuClient;

  // Initializes information about which arguments to which executables must be
  // donated due to aliases that were specified by the computation.
  absl::Status SetUpDonation(bool tuple_inputs);

  // Create shared pointers so we can free them after the execution: with
  // asynchronous execution, the process being executed can outlive the
  // executable itself.
  TfrtGpuClient* const client_;
  // One executable per partition.
  std::vector<std::shared_ptr<LocalExecutable>> executables_;
  // Per-executable sorted vector of parameters that have any aliased buffers
  // and thus must be donated when executing the computation.
  std::vector<std::vector<int>> parameters_that_must_be_donated_;
  std::shared_ptr<DeviceAssignment> device_assignment_;
  CompileOptions compile_options_;

  // True if the executables were compiled expecting arguments in a single
  // tuple.
  const bool parameter_is_tupled_arguments_;

  // The replica and partition indices of device_assignment_ to be run by this
  // client. On single-host platforms without partitioning, this is all replicas
  // (i.e. addressable_device_logical_ids_[i] = (i, 0)), but this may not be the
  // case on multi-host platforms. If there are 4 replicas and 2 partitions on a
  // single host platform, size of addressable_device_logical_ids_ is 4*2 = 8.
  std::vector<LogicalDeviceIds> addressable_device_logical_ids_;

  // addressable_devices_[i] is the Device to which
  // addressable_device_logical_ids_[i] is assigned. shared_ptrs instead of
  // unique_ptrs to play well with the Python bindings (see xla.cc).
  std::vector<PjRtDevice*> addressable_devices_;
  std::string fingerprint_;

  struct InputHloSnapshotBits {
    HloModuleProto hlo_module;
    DebugOptions debug_options;
  };

  // The unoptimized (unsharded) HloModule. Primarily used for debugging.
  std::optional<InputHloSnapshotBits> input_hlo_snapshot_bits_;
};

}  // namespace xla

#endif  // XLA_PJRT_GPU_TFRT_TFRT_GPU_CLIENT_H_
