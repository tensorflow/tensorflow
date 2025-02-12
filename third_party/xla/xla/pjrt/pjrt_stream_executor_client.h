/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_PJRT_PJRT_STREAM_EXECUTOR_CLIENT_H_
#define XLA_PJRT_PJRT_STREAM_EXECUTOR_CLIENT_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/tracked_device_buffer.h"
#include "xla/pjrt/transpose.h"
#include "xla/pjrt/utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

namespace xla {

class PjRtStreamExecutorDeviceDescription : public PjRtDeviceDescription {
 public:
  explicit PjRtStreamExecutorDeviceDescription(int id, std::string device_kind,
                                               int process_index = 0)
      : id_(id),
        process_index_(process_index),
        device_kind_(std::move(device_kind)) {}

  int id() const override { return id_; }

  int process_index() const override { return process_index_; }

  absl::string_view device_kind() const override { return device_kind_; }

  absl::string_view ToString() const override { return to_string_; }

  absl::string_view DebugString() const override { return debug_string_; }

  absl::Span<int const> coords() const { return absl::MakeSpan(coords_); }

  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }

  void SetAttributes(
      absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes) {
    attributes_ = std::move(attributes);
  }

  void SetDebugString(std::string debug_string) {
    debug_string_ = std::move(debug_string);
  }

  void SetToString(std::string to_string) { to_string_ = std::move(to_string); }

  void SetCoords(std::array<int, 1> coords) { coords_ = coords; }

 private:
  const int id_;
  const int process_index_;
  const std::string device_kind_;
  std::string debug_string_ = "<unknown SE device>";
  std::string to_string_ = "<unknown SE device>";
  absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes_;
  std::array<int, 1> coords_;
};

class PjRtStreamExecutorDevice : public PjRtDevice {
 public:
  explicit PjRtStreamExecutorDevice(
      int id, std::unique_ptr<LocalDeviceState> local_device_state,
      std::string device_kind, int process_index = 0)
      : description_(id, std::move(device_kind), process_index),
        local_device_id_(local_device_state
                             ? local_device_state->local_device_id()
                             : PjRtLocalDeviceId(-1)),
        local_hardware_id_(local_device_state
                               ? local_device_state->local_hardware_id()
                               : PjRtLocalHardwareId(-1)),
        local_device_state_(std::move(local_device_state)) {}
  ~PjRtStreamExecutorDevice() override = default;

  // Must set client exactly once.
  void SetClient(PjRtClient* client) {
    CHECK(client_ == nullptr);
    client_ = client;
    // We have to define debug_string_ and to_string_ here, because
    // platform_name() requires client_ to be set.
    std::string device_name =
        absl::StrCat(MakeAsciiTitlecase(platform_name()), "Device");

    description().SetDebugString(absl::StrCat(platform_name(), ":", id()));
    description().SetToString(absl::StrCat(device_name, "(id=", id(), ")"));
  }

  PjRtStreamExecutorDeviceDescription& description() { return description_; }
  const PjRtStreamExecutorDeviceDescription& description() const override {
    return description_;
  }

  // Return `platform_id` from client.
  PjRtPlatformId platform_id() const;

  // Return `platform_name` from client.
  absl::string_view platform_name() const;

  PjRtClient* client() const override { return client_; }

  bool IsAddressable() const override { return local_device_id_ != -1; }

  PjRtLocalDeviceId local_device_id() const override {
    return local_device_id_;
  }

  PjRtLocalHardwareId local_hardware_id() const override {
    return local_hardware_id_;
  }

  // If this is a device local to this host, returns a LocalDeviceState object
  // that can be used to manipulate the device. Returns nullptr if the device is
  // not local to this host.
  LocalDeviceState* local_device_state() const {
    return local_device_state_.get();
  }

  // If this is a device local to this host, returns a LocalDeviceState object
  // that can be used to manipulate the device. Returns an error if the device
  // is not local to this host.
  absl::StatusOr<LocalDeviceState*> GetLocalDeviceState() const;

  absl::Status TransferToInfeed(const LiteralSlice& literal) override;

  absl::Status TransferFromOutfeed(MutableBorrowingLiteral literal) override;

  void AttachMemorySpace(PjRtMemorySpace* memory_space,
                         bool is_default = false);

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

  absl::StatusOr<PjRtMemorySpace*> default_memory_space() const override;

  absl::StatusOr<PjRtMemorySpace*> memory_space_by_kind(
      absl::string_view memory_space_kind) const override;

  absl::StatusOr<PjRtMemorySpace*> memory_space_by_kind_id(int id) const;

  absl::StatusOr<std::intptr_t> GetStreamForExternalReadyEvents()
      const override;

  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override {
    return nullptr;
  }

 private:
  PjRtStreamExecutorDeviceDescription description_;
  const PjRtLocalDeviceId local_device_id_;
  const PjRtLocalHardwareId local_hardware_id_;
  const std::unique_ptr<LocalDeviceState> local_device_state_;
  PjRtClient* client_ = nullptr;
  absl::InlinedVector<PjRtMemorySpace*, 1> memory_spaces_;
  absl::flat_hash_map<int, PjRtMemorySpace*> memory_spaces_by_id_;
  PjRtMemorySpace* default_memory_space_ = nullptr;
};

class PjRtStreamExecutorMemorySpace : public PjRtMemorySpace {
 public:
  PjRtStreamExecutorMemorySpace(int id, PjRtDevice* device,
                                absl::string_view kind, int kind_id);

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

class PjRtStreamExecutorClient : public PjRtClient {
 public:
  // `allocator` may null, in which case the platform default allocator is used.
  explicit PjRtStreamExecutorClient(
      std::string platform_name, LocalClient* client,
      std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices,
      int process_index,
      std::vector<std::unique_ptr<PjRtMemorySpace>> memory_spaces,
      std::unique_ptr<se::DeviceMemoryAllocator> allocator,
      std::unique_ptr<tsl::Allocator> host_memory_allocator,
      bool should_stage_host_to_device_transfers,
      std::unique_ptr<gpu::GpuExecutableRunOptions> gpu_run_options);
  ~PjRtStreamExecutorClient() override = default;

  int process_index() const override { return process_index_; }

  int device_count() const override { return devices_.size(); }
  int addressable_device_count() const override {
    return addressable_devices_.size();
  }
  absl::Span<PjRtDevice* const> devices() const override { return devices_; }
  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return addressable_devices_;
  }

  absl::StatusOr<PjRtDevice*> LookupDevice(
      PjRtGlobalDeviceId global_device_id) const override {
    auto it = id_to_device_.find(global_device_id.value());
    if (it != id_to_device_.end()) {
      return it->second;
    }
    return InvalidArgument("No matching device found for device_id %d",
                           global_device_id.value());
  }

  absl::StatusOr<PjRtDevice*> LookupAddressableDevice(
      PjRtLocalDeviceId local_device_id) const override;

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

  PjRtPlatformId platform_id() const override { return platform_id_; }
  absl::string_view platform_name() const override { return platform_name_; }
  absl::string_view platform_version() const override { return "<unknown>"; }

  // Most platforms expect device-to-device transfers to be enqueued on the
  // source d2d stream, but some platforms use the destination d2d stream. This
  // function specifies which one the platform expects.
  virtual bool EnqueueD2DTransfersOnSrcStream() const { return true; }

  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type, absl::Span<const int64_t> dims) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) override;
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      mlir::ModuleOp mlir_module, CompileOptions options) override;

  virtual absl::StatusOr<std::string> SerializeExecutable(
      const PjRtLoadedExecutable& executable) const;

  // For PjRtStreamExecutorClient, `options` is mandatory.
  // This function returns an InvalidArgument error if `std::nullopt` is passed.
  // TODO(b/237720161): make it actually optional
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> DeserializeExecutable(
      absl::string_view serialized,
      std::optional<CompileOptions> options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
  LoadSerializedExecutable(absl::string_view serialized,
                           std::optional<CompileOptions> options,
                           const LoadOptions& load_options) override;

  absl::StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis()
      const override;

  // Creates a buffer on the device without initializing or copying any data.
  // An optional `definition_event` may be speficied that can be used to
  // ensure the buffer isn't referenced until some external mechanism has
  // initialized the data.
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateUninitializedBuffer(
      const Shape& shape, PjRtMemorySpace* memory_space) override;
  using PjRtClient::CreateUninitializedBuffer;
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateUninitializedBuffer(
      const Shape& shape, PjRtMemorySpace* memory_space,
      std::shared_ptr<BufferSequencingEvent> definition_event);

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateErrorBuffer(
      absl::Status error, const Shape& shape, PjRtMemorySpace* memory) override;

  absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(absl::Span<const Shape> shapes,
                                    PjRtDevice* device) override {
    return Unimplemented("Async transfer to buffers not implemented");
  };

  absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(absl::Span<const Shape> shapes,
                                    PjRtMemorySpace* memory_space) override {
    return Unimplemented("Async transfer to buffers not implemented");
  };

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtMemorySpace* memory_space, const Layout* device_layout) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtMemorySpace* memory_space) override;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeCrossHostReceiveBuffers(absl::Span<const Shape> shapes,
                              PjRtDevice* device,
                              PjRtCrossHostRecvNotifier notifier) override;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeCrossHostReceiveBuffersForGather(
      absl::Span<const Shape> shapes, std::vector<GatherDetails> gather_details,
      PjRtDevice* device, PjRtCrossHostRecvNotifier notifier) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateViewOfDeviceBuffer(
      void* device_ptr, const Shape& shape, PjRtMemorySpace* memory_space,
      std::function<void()> on_delete_callback,
      std::optional<std::intptr_t> stream) override;

  // TODO(zhangqiaorjc): Experimental. Will be removed.
  absl::Status Defragment() override {
    return Unimplemented("Defragment not implemented");
  }

  LocalDeviceState& device_state(int device_ordinal) const {
    return *tensorflow::down_cast<PjRtStreamExecutorDevice*>(
                LookupAddressableDevice(xla::PjRtLocalDeviceId(device_ordinal))
                    .value())
                ->local_device_state();
  }
  LocalClient* client() const { return client_; }
  se::DeviceMemoryAllocator* allocator() const { return allocator_; }
  tsl::Allocator* host_memory_allocator() const {
    return host_memory_allocator_.get();
  }
  bool should_stage_host_to_device_transfers() const {
    return should_stage_host_to_device_transfers_;
  }

  gpu::GpuExecutableRunOptions* gpu_run_options() const {
    return gpu_run_options_.get();
  }

  tsl::thread::ThreadPool* thread_pool() { return &thread_pool_; }

 protected:
  friend class PjRtStreamExecutorBuffer;

  virtual absl::Status EnqueueCrossHostReceive(
      absl::Span<const std::unique_ptr<PjRtBuffer>> buffers,
      std::shared_ptr<BufferSequencingEvent> definition_event,
      PjRtCrossHostRecvNotifier notifier,
      std::optional<std::vector<GatherDetails>> gather_details) const {
    return Unimplemented("Cross host receives not implemented.");
  }

  virtual void CopyToRemoteDevice(
      PjRtBuffer* buffer, absl::string_view serialized_descriptor,
      PjRtBuffer::RemoteSendCallback on_done) const {
    on_done(Unimplemented("Cross host sends not implemented."),
            /*sends_were_enqueued=*/false);
  }

  virtual void CopyToRemoteDeviceScattered(
      PjRtBuffer* buffer, std::vector<std::string> serialized_descriptors,
      std::vector<PjRtBuffer::RemoteSendCallback> callbacks,
      const PjRtBuffer::ScatterDetails& scatter_details) const {
    for (const auto& cb : callbacks) {
      cb(Unimplemented("Scattered cross host sends not implemented."),
         /*sends_were_enqueued=*/false);
    }
  }

  virtual PjRtFuture<> CopyRawSubBufferToHost(PjRtBuffer* buffer,
                                              PjRtFuture<void*> dst,
                                              int64_t offset,
                                              int64_t transfer_size) {
    return PjRtFuture<>(Unimplemented("Raw copies to host not implemented."));
  }

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

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBufferInternal(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtDevice* device, const Layout* device_layout,
      PjRtMemorySpace* memory_space);

  const PjRtPlatformId platform_id_;
  const std::string platform_name_;
  LocalClient* client_;

  // Allocator to be used for staging memory transfers to devices.
  std::unique_ptr<tsl::Allocator> host_memory_allocator_;

  // Device memory allocator. If owned, the allocator must outlive the devices,
  // because it is the device destructor that waits for any outstanding work to
  // complete.
  se::DeviceMemoryAllocator* allocator_;
  std::unique_ptr<se::DeviceMemoryAllocator> owned_allocator_;

  // Includes all devices, including non-local devices on multi-host platforms.
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> owned_devices_;
  // Pointers to `owned_devices_`.
  std::vector<PjRtDevice*> devices_;
  // Maps Device::id() to the corresponding Device. Includes all devices.
  std::map<int, PjRtDevice*> id_to_device_;
  // Local devices indexed by local device ordinal.
  std::vector<PjRtDevice*> addressable_devices_;
  int process_index_;

  std::vector<std::unique_ptr<PjRtMemorySpace>> owned_memory_spaces_;
  // Pointers to `owned_memory_spaces_`.
  std::vector<PjRtMemorySpace*> memory_spaces_;

  // Should we always prefer to stage host-to-device transfers via memory
  // allocated on host_memory_allocator_? True only on GPU, where we prefer to
  // transfer via pinned memory.
  bool should_stage_host_to_device_transfers_;

  std::unique_ptr<gpu::GpuExecutableRunOptions> gpu_run_options_;

  tsl::thread::ThreadPool thread_pool_;

  absl::Mutex transpose_mu_;
  TransposePlanCache transpose_cache_ ABSL_GUARDED_BY(transpose_mu_);
};

// Converts a 2D set of Device objects indexed by [replica][partition] into an
// xla::DeviceAssignment.
absl::StatusOr<DeviceAssignment> DevicesToDeviceAssignment(
    absl::Span<const std::vector<PjRtDevice*>> devices);

class PjRtStreamExecutorBuffer : public PjRtBuffer {
 public:
  // Helper class to retain a "hold" on a PjRtStreamExecutorBuffer. A ScopedHold
  // may not outlive its parent PjRtStreamExecutorBuffer.
  //
  // There are three types of hold, as follows:
  //
  // 1) Usage hold: a transient hold while an operation using the buffer is
  //    being enqueued onto a stream.
  // A client acquires a usage hold by calling
  // PjRtStreamExecutorBuffer::GetBufferWithHold(kUsage) or the convenience
  // wrapper GetBufferWithUsageHold(). If the enqueue completes successfully the
  // hold should be released using a call to ConvertUsageHold. If the ScopedHold
  // is deleted without ConvertUsageHold being called, e.g., on error, the hold
  // is dropped. It is legal to drop a usage hold instead of calling
  // ConvertUsageHold, even if the buffer was successfully enqueued, as long as
  // the client ensures that all necessary synchronization has been done.
  //
  // 2) External hold: a potentially long-lived hold while the buffer is being
  //    shared by an external framework, e.g., NumPy.
  // A client acquires an external hold by calling
  // PjRtStreamExecutorBuffer::GetBufferWithHold(kExternal) or the convenience
  // wrapper GetBufferWithExternalReference and releases it by deleting the
  // ScopedHold. The external framework should not modify the underlying buffer
  // unless it is confident via its own synchronization that modifications do
  // not race with reads from the PjRtStreamExecutorBuffer.
  //
  // 3) Donation hold: a transient hold while an execution that donates the
  //    buffer is being enqueued onto the compute stream.
  // A client acquires a donation hold by calling
  // PjRtStreamExecutorBuffer::GetBufferWithHold(kDonation). If the enqueue
  // completes successfully the hold should be released using a call to
  // ConfirmDonation after which the buffer is invalid. If the ScopedHold is
  // deleted without ConfirmDonation being called, e.g., on error, the hold is
  // dropped and the buffer remains valid. If the buffer is successfully
  // enqueued the client *must* call ConfirmDonation.
  //
  // Donation holds behave like exclusive write locks: when a donation hold
  // has been acquired, any attempt to acquire another hold of any type will
  // block until the donation hold is dropped or confirmed. Acquiring a donation
  // hold will fail with an error if there is any outstanding external hold, and
  // will block if there are any outstanding usage holds until those holds are
  // dropped or converted.
  //
  // Calls to PjRtStreamExecutorBuffer::Release (and transitively to
  // PjRtStreamExecutorBuffer::Delete() and ~PjRtStreamExecutorBuffer()) will
  // block until all usage and donation holds are either deleted or
  // converted/confirmed.
  class ScopedHold {
   public:
    enum Type { kUsage = 0, kExternalReference, kDonation, kMaxValue };
    // Use a State enum instead of encoding the state in an error absl::Status
    // to avoid creating absl::Status values in non-error cases. Creating a
    // absl::Status entails several allocations and can add O(us) to every use
    // of a hold.
    enum State {
      kUninitialized = 0,
      kValid,
      kMoved,
      kConverted,
      kReleased,
      kDonated,
      kError
    };

    ~ScopedHold();
    ScopedHold(ScopedHold&& other);
    ScopedHold(const ScopedHold&) = delete;
    ScopedHold& operator=(const ScopedHold&) = delete;

    Type type() const { return type_; }

    absl::Status status() const {
      // Lazily create absl::Status values only when they are requested.
      switch (state_) {
        case kUninitialized:
          return InvalidArgument("Buffer has not been initialized");
        case kValid:
          return absl::OkStatus();
        case kMoved:
          return InvalidArgument("Buffer has been moved.");
        case kConverted:
          return InvalidArgument("Buffer has been converted");
        case kReleased:
          return InvalidArgument("Buffer has been released");
        case kDonated:
          return InvalidArgument("Buffer has been donated");
        case kError:
          return status_;
        default:
          CHECK(false) << "Unexpected state value " << state_;
      }
    }
    bool ok() const { return state_ == kValid; }

    // Access to the underlying device buffer storage. Requires this->ok().
    const std::shared_ptr<TrackedDeviceBuffer>& buffer() const {
      CHECK_EQ(state_, kValid);
      CHECK_NE(buffer_, nullptr);
      return buffer_;
    }
    TrackedDeviceBuffer* operator->() const { return buffer().get(); }
    const TrackedDeviceBuffer& operator*() const { return *buffer(); }

    // Converts the hold into a usage event. Only valid for holds of type
    // kUsage.
    //
    //   usage_stream:   the stream that the buffer was used on.
    //   event:          an event that has been recorded on usage_stream after
    //                   the buffer was used.
    //   reference_held: true if and only if the caller has caused a
    //                   reference to this->buffer() to stay live until after
    //                   the host is sure that the usage (transfer or execution)
    //                   has completed.
    void ConvertUsageHold(se::Stream* usage_stream,
                          std::shared_ptr<BufferSequencingEvent> event,
                          bool reference_held);

    // Confirms that the buffer was successfully donated to an execution.
    // Only valid for holds of type kDonation. Causes the buffer to become
    // invalid.
    void ConfirmDonation();

    // Adds the held device buffers in order to 'iterator'. Used to add the
    // buffers to an ExecutionInput. We require but do not verify that
    // 'iterator' when passed in is pointing to a sub-tuple of the
    // ExecutionInput whose on_device_shape matches that of the
    // TrackedDeviceBuffer. 'end' is used to check that 'iterator' doesn't run
    // out of bounds. Donates the device buffers if the hold type is kDonation,
    // otherwise retains ownership of the device buffers.
    void AddToInput(ShapeTree<MaybeOwningDeviceMemory>::iterator* iterator,
                    const ShapeTree<MaybeOwningDeviceMemory>::iterator& end,
                    ExecutionInput* execution_input,
                    se::DeviceMemoryAllocator* allocator) const;

   private:
    friend class PjRtStreamExecutorBuffer;
    friend class PjRtStreamExecutorClient;

    // Helper struct that makes it possible to move a ScopedHold through a
    // closure.
    using ForClosure =
        std::tuple<PjRtStreamExecutorBuffer*, Type, State, absl::Status,
                   std::shared_ptr<TrackedDeviceBuffer>>;

    ScopedHold(PjRtStreamExecutorBuffer* parent, Type type)
        : parent_(parent), type_(type), state_(kUninitialized) {}
    explicit ScopedHold(const ForClosure& closure_helper)
        : parent_(std::get<0>(closure_helper)),
          type_(std::get<1>(closure_helper)),
          state_(std::get<2>(closure_helper)),
          status_(std::get<3>(closure_helper)),
          buffer_(std::get<4>(closure_helper)) {
      // Check the buffer is not in an error state.
      CHECK(status_.ok() && buffer_ != nullptr);
    }

    // Sets buffer state.
    void SetState(State state) { state_ = state; }

    // Sets buffer_ and status_. Called by parent_ to initialize the hold.
    void Acquire(
        absl::StatusOr<std::shared_ptr<TrackedDeviceBuffer>>&& buffer_or);
    // Releases the contents of *this, so *this can subsequently be
    // deleted without releasing the parent's hold. Should be passed to the
    // appropriate constructor of another ScopedHold, e.g., when a hold must be
    // passed through a closure that is incompatible with std::move.
    ForClosure ToClosure();

    PjRtStreamExecutorBuffer* const parent_;
    const Type type_;

    // There is an invariant that if ok() then
    // buffer_.value() != nullptr.
    State state_;
    absl::Status status_;
    std::shared_ptr<TrackedDeviceBuffer> buffer_;
  };

  PjRtStreamExecutorBuffer(Shape on_device_shape,
                           std::shared_ptr<TrackedDeviceBuffer> device_buffer,
                           PjRtClient* client, PjRtDevice* device,
                           PjRtMemorySpace* memory_space);
  ~PjRtStreamExecutorBuffer() override;

  PjRtStreamExecutorBuffer(const PjRtStreamExecutorBuffer&) = delete;
  PjRtStreamExecutorBuffer(PjRtStreamExecutorBuffer&&) = delete;
  PjRtStreamExecutorBuffer& operator=(const PjRtStreamExecutorBuffer&) = delete;
  PjRtStreamExecutorBuffer& operator=(PjRtStreamExecutorBuffer&&) = delete;

  const Shape& on_device_shape() const override { return on_device_shape_; }
  absl::StatusOr<Shape> logical_on_device_shape() override;
  PjRtMemorySpace* memory_space() const override { return memory_space_; }
  PjRtStreamExecutorDevice* device() const override { return device_; }
  PjRtPlatformId platform_id() const { return client_->platform_id(); }
  absl::string_view platform_name() const { return client_->platform_name(); }
  PjRtStreamExecutorClient* client() const override { return client_; }
  bool IsEmptyTuple() const {
    return on_device_shape_.IsTuple() &&
           on_device_shape_.tuple_shapes_size() == 0;
  }

  absl::StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference()
      override;

  absl::StatusOr<std::unique_ptr<ExternalReference>>
  ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete) override;

  using PjRtBuffer::ToLiteralSync;
  PjRtFuture<> ToLiteral(MutableLiteralBase* literal) override;
  PjRtFuture<> LazyToLiteral(
      absl::AnyInvocable<absl::StatusOr<MutableLiteralBase*>() &&> generator)
      override;

  absl::StatusOr<size_t> GetOnDeviceSizeInBytes() const override;

  PjRtFuture<> CopyRawToHost(void* dst, int64_t offset,
                             int64_t transfer_size) override;

  PjRtFuture<> CopyRawToHostFuture(PjRtFuture<void*> dst, int64_t offset,
                                   int64_t transfer_size) override;

  // Drops the buffer's reference to its associated device memory, leaving the
  // buffer in an invalid state. The memory will be freed lazily when all async
  // operations using the buffer have completed, according to the allocation
  // semantics of the underlying platform. Delete may briefly block if another
  // thread is in the process of enqueuing an operation on this buffer, but it
  // will never block for a stream operation to complete. If an external
  // framework holds a reference to the TrackedDeviceBuffer via
  // GetBufferWithExternalReference, the memory will not be freed until the
  // external framework drops the reference.
  void Delete() override;

  bool IsDeleted() override;

  // Returns a view of the PjRtBuffer device memory as a ShapedBuffer. The
  // PjRtBuffer retains ownership of the device buffers.
  absl::StatusOr<ShapedBuffer> AsShapedBuffer() const;

  // Returns a hold on the TrackedDeviceBuffer holding the device
  // buffers. See comment on ScopedHold.
  ScopedHold GetBufferWithHold(ScopedHold::Type type);
  ScopedHold GetBufferWithUsageHold() {
    return GetBufferWithHold(ScopedHold::kUsage);
  }
  ScopedHold GetBufferWithExternalReference() {
    return GetBufferWithHold(ScopedHold::kExternalReference);
  }

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(
      PjRtMemorySpace* dst_memory_space) override;

  void CopyToRemoteDevice(PjRtFuture<std::string> serialized_descriptor,
                          RemoteSendCallback on_done) override;

  void CopyToRemoteDeviceScattered(
      PjRtFuture<std::vector<std::string>> serialized_descriptors,
      std::vector<RemoteSendCallback> callbacks,
      const ScatterDetails& scatter_details) override;

  PjRtFuture<> GetReadyFuture() override;

  bool IsOnCpu() const override;

  // Similar to Delete, drops the buffer's reference to its associated device
  // memory, leaving the buffer in an invalid state, but returns the
  // TrackedDeviceBuffer rather than freeing the device memory, so that another
  // framework can take ownership of it.
  //
  // When called with wait_for_operations_to_complete=false, the buffer returned
  // from Release should be dropped on the compute stream, since the only events
  // that Release doesn't wait for are events defined on the compute stream.
  //
  // If wait_for_operations_to_complete=true, the host will block until any
  // potentially outstanding asynchronous operations have completed before
  // returning, in which case it is safe to read or mutate the returned buffer.
  // If the buffer was shared via an external reference it is the client's
  // responsibility that accesses via that reference do not interfere with
  // accesses via the buffer returned from Release.
  absl::StatusOr<std::shared_ptr<TrackedDeviceBuffer>> Release(
      bool wait_for_operations_to_complete);

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> DonateWithControlDependency(
      PjRtFuture<> dependency) override;

 private:
  friend class PjRtClient;

  // Blocks in mu_.Await until there are no more usage holds.
  void WaitForOutstandingUsageHolds() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Blocks in mu_.Await until there is no donation hold.
  void WaitForOutstandingDonationHold() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Adds a hold of 'type' and returns device_buffer_. Returns an error if
  // device_buffer_ is null, or if a donation hold was requested when there is
  // an outstanding external hold.
  // Requires holds_[kDonation] == 0 (i.e., WaitForOutstandingDonationHolds()
  // must be called first.)
  absl::StatusOr<std::shared_ptr<TrackedDeviceBuffer>> GetBufferForHoldLocked(
      ScopedHold::Type type) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Adds a hold of hold->type() and initializes `hold` with device_buffer_.
  // Initializes hold with an error if device_buffer_ is null, or if a donation
  // hold was requested when there is an outstanding external hold.
  // Requires holds_[kDonation] == 0 (i.e., WaitForOutstandingDonationHolds()
  // must be called first.)
  void AcquireHoldLocked(ScopedHold* hold) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Drops a usage hold and calls device_buffer_->AddUsageEvent. Does a sanity
  // check that buffer==device_buffer_ or device_buffer_==nullptr. Called after
  // device_buffer_ was successfully enqueued on a stream.
  void ConvertUsageHold(TrackedDeviceBuffer* buffer, se::Stream* usage_stream,
                        std::shared_ptr<BufferSequencingEvent> event,
                        bool reference_held);

  // Drops a donation hold and makes *this invalid for further use. Does a
  // sanity check that buffer==device_buffer_. Called after device_buffer_ was
  // successfully donated to an execution.
  void ConfirmDonation(TrackedDeviceBuffer* device_buffer);

  // Drops a hold without taking any other action. Does a sanity check that
  // buffer==device_buffer_ or device_buffer_==nullptr.
  void DropHold(ScopedHold::Type type, TrackedDeviceBuffer* buffer);

  absl::StatusOr<std::pair<std::unique_ptr<PjRtBuffer>,
                           std::shared_ptr<BufferSequencingEvent>>>
  CopyToDeviceHelper(PjRtDevice* dst_device, LocalDeviceState* dst_local_device,
                     PjRtMemorySpace* dst_memory_space,
                     LocalDeviceState* transfer_local_device,
                     LocalDeviceState* src_local_device,
                     se::Stream* transfer_stream,
                     std::shared_ptr<TrackedDeviceBuffer> src_device_buffer);
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToDeviceMemorySpace(
      PjRtDevice* dst_device, PjRtMemorySpace* dst_memory_space = nullptr);

  PjRtStreamExecutorClient* const client_;
  const Shape on_device_shape_;
  PjRtStreamExecutorDevice* const device_;
  PjRtMemorySpace* const memory_space_;

  mutable absl::Mutex mu_;
  std::shared_ptr<TrackedDeviceBuffer> device_buffer_ ABSL_GUARDED_BY(mu_);
  // Count of holds on the buffer.
  std::array<int, ScopedHold::Type::kMaxValue> holds_ ABSL_GUARDED_BY(mu_);
  PjRtFuture<>::Promise definition_promise_ ABSL_GUARDED_BY(mu_);
};

// Allocates the device buffers for a buffer that will be used as the
// destination of a copy, either from the host or another device. copy_stream
// may be nullptr, e.g., when allocating a buffer for a cross-host copy. If the
// buffer is a tuple then the tuple tables are allocated, and all necessary
// synchronization for them is dealt with, before the buffer is returned.
//
// It is safe to delete the returned PjRtBuffer without further
// synchronization if an error occurs before the buffer is used.
//
// The caller may optionally provide a definition event to be recorded in
// the buffer.
// TODO(phawkins): replace on_host_shape here with on_device_shape.
absl::StatusOr<std::unique_ptr<PjRtStreamExecutorBuffer>>
AllocateDestinationBuffer(
    const Shape& on_host_shape, PjRtDevice* device,
    LocalDeviceState* local_device, se::Stream* copy_stream,
    bool is_uninitialized_create, PjRtStreamExecutorClient* client,
    std::shared_ptr<BufferSequencingEvent> definition_event = nullptr,
    PjRtMemorySpace* memory_space = nullptr);

// Wraps one or more XLA LocalExecutables (one per partition, as specified by
// the build options).
class PjRtStreamExecutorLoadedExecutable : public PjRtLoadedExecutable {
 public:
  PjRtStreamExecutorLoadedExecutable(
      std::vector<std::unique_ptr<LocalExecutable>> executables,
      bool parameter_is_tupled_arguments,
      std::shared_ptr<DeviceAssignment> device_assignment,
      CompileOptions compile_options,
      std::vector<LogicalDeviceIds> addressable_device_logical_ids,
      std::vector<PjRtDevice*> addressable_devices,
      PjRtStreamExecutorClient* client);

  ~PjRtStreamExecutorLoadedExecutable() override = default;

  PjRtStreamExecutorClient* client() const override { return client_; }

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

  absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override {
    if (executables_.size() != 1) {
      return Unimplemented(
          "Retrieving CompiledMemoryStats is not supported for multiple "
          "executables.");
    }
    CompiledMemoryStats memory_stats = CompiledMemoryStats();
    memory_stats.generated_code_size_in_bytes = SizeOfGeneratedCodeInBytes();
    const HloProto* proto = executables_[0]->executable()->hlo_proto();
    if (proto != nullptr) {
      memory_stats.serialized_hlo_proto = proto->SerializeAsString();
    }
    memory_stats.PopulateBufferStatsFromAllocations(
        executables_[0]->executable()->GetAllocations());
    return memory_stats;
  }

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

  // Return an HloModule per partition.
  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override;

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override;

  using PjRtLoadedExecutable::Execute;
  absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options,
      std::optional<std::vector<PjRtFuture<>>>& returned_futures) override;

  using PjRtLoadedExecutable::ExecuteSharded;
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<>>& returned_future, bool fill_future) override;

  using PjRtLoadedExecutable::ExecutePortable;
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<>>& returned_future, bool fill_future) override;

  void Delete() override { executables_.clear(); }

  bool IsDeleted() override { return executables_.empty(); }

  absl::StatusOr<std::string> SerializeExecutable() const override {
    return client_->SerializeExecutable(*this);
  }

  bool IsReturnedFutureSupported() const override { return true; }

  absl::Span<const std::shared_ptr<LocalExecutable>> executables() const {
    return executables_;
  }

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

 protected:
  bool parameter_is_tupled_arguments() const {
    return parameter_is_tupled_arguments_;
  }

 private:
  friend class PjRtStreamExecutorClient;
  friend class PjRtTpuClient;
  friend class InternalPjRtTpuClient;
  friend class StreamExecutorGpuClient;
  // Initializes information about which arguments to which executables must be
  // donated due to aliases that were specified by the computation.
  absl::Status SetUpDonation(bool tuple_inputs);

  // Returns a sorted list of the parameters that must be donated. Derived
  // classes may use custom logic.
  virtual absl::Span<int const> ParametersThatMustBeDonated(
      int executable_idx) const;

  virtual absl::StatusOr<std::vector<ExecutionInput>>
  MakeExecutionInputsAndWaitForEvents(
      int device_ordinal, const ExecuteOptions& options,
      absl::Span<const Shape> executable_parameter_shapes,
      absl::Span<PjRtBuffer* const> argument_handles,
      absl::Span<const PjRtStreamExecutorBuffer::ScopedHold> device_buffers,
      absl::flat_hash_set<BufferSequencingEvent*>& events) const;

  absl::StatusOr<ScopedShapedBuffer> EnqueueExecution(
      absl::Span<PjRtBuffer* const> argument_handles, int replica,
      int partition, int executable_idx, const RunId& run_id,
      const ExecuteOptions& options, PjRtDevice* device,
      std::vector<PjRtStreamExecutorBuffer::ScopedHold>* device_buffers,
      std::shared_ptr<DeviceAssignment> device_assignment,
      std::vector<std::function<void()>>& compute_callbacks) const;

  virtual absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeOutputBuffers(int device_ordinal, const ExecuteOptions& options,
                    ScopedShapedBuffer result_buffer,
                    std::shared_ptr<BufferSequencingEvent> definition_event,
                    PjRtDevice* device,
                    std::vector<std::function<void()>>& compute_callbacks,
                    std::vector<std::shared_ptr<TrackedDeviceBuffer>>&
                        buffers_to_release) const;

  absl::StatusOr<Result> ExecuteHelper(
      absl::Span<PjRtBuffer* const> argument_handles, int replica,
      int partition, const RunId& run_id, const ExecuteOptions& options,
      bool fill_future, PjRtDevice* device = nullptr) const;

  absl::Status VerifyCompatibleDevices() const;

  // Create shared pointers so we can free them after the execution: with
  // asynchronous execution, the process being executed can outlive the
  // executable itself.
  PjRtStreamExecutorClient* const client_;
  // One executable per partition.
  std::vector<std::shared_ptr<LocalExecutable>> executables_;
  // On device shapes of the executable parameters.
  std::vector<std::vector<Shape>> on_device_executable_parameter_shapes_;
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

#endif  // XLA_PJRT_PJRT_STREAM_EXECUTOR_CLIENT_H_
