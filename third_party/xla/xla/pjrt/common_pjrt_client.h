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

#ifndef XLA_PJRT_COMMON_PJRT_CLIENT_H_
#define XLA_PJRT_COMMON_PJRT_CLIENT_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/xla_data.pb.h"

namespace xla {

// A common base class for Pjrt clients based on raw buffers.
class CommonPjRtClient : public PjRtClient {
 public:
  using PjRtClient::PjRtClient;

  // A thread pool for dispatching background work.
  // TODO(parkers): make pure virtual and update all clients.
  virtual AsyncWorkRunner* async_work_runner() const { return nullptr; }

  // Some clients do not support recursion eg: calling to_literal in host
  // callbacks. Those clients should return false here.
  virtual bool allows_recursion() const { return true; }

  // Backend specific handlers for when an oom is detected during execute.
  virtual void CallOomHandlers() const {}

  // Computes the memory requirements for storing shape on memory_space.
  // TODO(parkers): make pure virtual and update all clients.
  virtual absl::StatusOr<int64_t> GetOnDeviceBytesCount(
      PjRtMemorySpace* memory_space, const xla::Shape& shape) const {
    return absl::UnimplementedError("GetOnDeviceBytesCount is not supported.");
  }

  // Allocates a raw buffer of a particular size after an optional
  // allocate_after. Backends may support retrying allocation on oom which
  // can be controlled via retry_on_oom.
  virtual absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>>
  AllocateRawBuffer(PjRtMemorySpace* memory_space, size_t on_device_bytes_count,
                    bool retry_on_oom,
                    tsl::AsyncValueRef<bool> allocate_after) {
    return absl::UnimplementedError("AllocateRawBuffer is not supported");
  }

  // Allocates a raw buffer of a particular size. Backends may support retrying
  // allocation on oom which can be controlled via retry_on_oom.
  // This is separate from AllocateRawBuffer so that backends can specialize
  // allocating buffers used in the execute path.
  virtual absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>>
  AllocateRawBufferForExecute(PjRtMemorySpace* memory_space,
                              size_t on_device_bytes_count, bool retry_on_oom) {
    return AllocateRawBuffer(memory_space, on_device_bytes_count, retry_on_oom,
                             {});
  }

  // Imports foreign memory as a raw buffer.
  virtual absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>>
  ImportForeignMemory(void* device_ptr,
                      absl::AnyInvocable<void() &&> on_delete_callback,
                      size_t on_device_bytes_count,
                      PjRtMemorySpace* memory_space, bool is_mutable) {
    return absl::UnimplementedError("ImportForeignMemory is not supported");
  }

  // Linearizes a literal into a raw buffer and returns a DeviceEvent
  // for when the linearization is complete.
  virtual absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> LinearizeInto(
      const LiteralSlice& literal, const xla::Shape& device_shape,
      HostBufferSemantics host_buffer_semantics,
      tsl::RCReference<CommonPjRtRawBuffer> raw_buffer) {
    return absl::UnimplementedError("LinearizeInto is not supported");
  }

  // Defines a pjrt buffer from a shape, raw_buffer and definition events.
  virtual absl::StatusOr<std::unique_ptr<PjRtBuffer>> DefineBuffer(
      const Shape& on_device_shape, PjRtMemorySpace* memory_space,
      tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
      absl::InlinedVector<tsl::RCReference<PjRtDeviceEvent>, 4>
          definition_device_events) {
    return absl::UnimplementedError("DefineBuffer is not supported");
  }

  // When calling APIs that take extra debug information, we may want
  // to omit this debug information if it is not going to be used.
  virtual bool event_tracking_enabled() { return false; }

  // Create a linked device-event and device-event-promise such that
  // setting an event into the event promise populates the device-event.
  virtual absl::StatusOr<std::pair<tsl::RCReference<PjRtDeviceEventPromise>,
                                   tsl::RCReference<PjRtDeviceEvent>>>
  CreateLinkedEventPromise(PjRtMemorySpace* memory_space,
                           absl::string_view debug_info) {
    return absl::UnimplementedError(
        "CreateLinkedEventPromise is not supported");
  }

  // Track a user-provided future with attached debug_info (if
  // event_tracking_enabled()).
  virtual void TrackFuture(PjRtMemorySpace* memory_space,
                           absl::string_view debug_info,
                           const Future<>& future);

  // Creates a future from a user-provided future with profiling and
  // traceme scopes.
  virtual Future<> CreateProfiledFuture(PjRtMemorySpace* memory_space,
                                        const char* callee_type,
                                        const char* callee_method,
                                        Future<> future);

  // Create a linked Future<> and Promise<> pair for operations on
  // buffers in memory_space which populates debug information like linked
  // tracmes.
  std::pair<Promise<>, Future<>> CreateLinkedUserPromise(
      PjRtMemorySpace* memory_space, const char* callee_type,
      const char* callee_method, absl::string_view debug_info);

  template <typename T, std::enable_if_t<std::is_invocable_v<T>, bool> = true>
  absl::StatusOr<std::pair<tsl::RCReference<PjRtDeviceEventPromise>,
                           tsl::RCReference<PjRtDeviceEvent>>>
  CreateLinkedEventPromise(PjRtMemorySpace* memory_space, T&& debug_info_cb) {
    if (event_tracking_enabled()) {
      return CreateLinkedEventPromise(memory_space,
                                      std::forward<T>(debug_info_cb)());
    }
    return CreateLinkedEventPromise(memory_space, "CreateLinkedEventPromise");
  }

  virtual std::unique_ptr<PjRtDeviceEventSet> CreateDeviceEventSet(
      size_t preallocated_size) const {
    LOG(FATAL) << "Implement";
  }

  // Registers the necessary debug information for an allocation event.
  // TODO(parkers): Once everything is unified this should be controlled
  // by a non-device-specific config instead of delegating this control
  // to a device-specific config.
  virtual tsl::AsyncValueRef<bool> CreateAllocationEventForTransfers(
      PjRtMemorySpace* memory_space,
      const std::optional<std::string>& debug_info);

  // Returns the shape+layout that would result from copying a buffer of
  // shape+layout shape from src_memory_space to dst_memory_space.
  virtual absl::StatusOr<xla::Shape> GetCopyDestinationShape(
      const xla::Shape& shape, PjRtMemorySpace* src_memory_space,
      PjRtMemorySpace* dst_memory_space);

  virtual bool IsOnCpu(PjRtMemorySpace* memory_space) { return false; }

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtMemorySpace* memory_space, const Layout* device_layout) override;
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtBuffer* donated_dst, const Layout* device_layout) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtMemorySpace* memory_space,
      const Layout* device_layout) override;

  absl::StatusOr<
      std::pair<std::unique_ptr<PjRtBuffer>, PjRtFulfillAliasBufferCallback>>
  CreateAliasBuffer(const Shape& shape, PjRtMemorySpace* memory_space) override;

  // Creates a raw buffer channel. Returns a tuple containing:
  // 1.  A tsl::RCReference<CommonPjRtRawBuffer> which is an alias for a future
  //     raw buffer.
  // 3.  A PjRtFulfillAliasRawBufferCallback to fulfill the alias.
  using PjRtFulfillAliasRawBufferCallback = absl::AnyInvocable<absl::Status(
      absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>>) &&>;
  virtual absl::StatusOr<std::pair<tsl::RCReference<CommonPjRtRawBuffer>,
                                   PjRtFulfillAliasRawBufferCallback>>
  CreateRawBufferChannel(PjRtMemorySpace* memory_space,
                         size_t on_device_bytes_count) {
    return absl::UnimplementedError("CreateRawBufferChannel is not supported");
  }

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateUninitializedBuffer(
      const Shape& shape, PjRtMemorySpace* memory_space) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateViewOfDeviceBuffer(
      void* device_ptr, const Shape& shape, PjRtMemorySpace* memory_space,
      std::function<void()> on_delete_callback,
      std::optional<std::intptr_t> stream) override;

  // Applies memory-space normalization logic on top of
  // GetTopologyDescription()->GetDefaultLayout() to select the default
  // device layout (if not provided).
  virtual absl::StatusOr<xla::Shape> MakeDefaultShapeForMemorySpace(
      PjRtMemorySpace* memory_space, xla::Shape shape,
      const xla::Layout* layout) const;

  virtual bool BufferFromHostBufferSupportsZeroCopy(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides, const Shape& shape,
      PjRtMemorySpace* memory_space, const Layout* device_layout) const {
    return false;
  }

  virtual absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
  LinearizeHostBufferInto(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      const xla::Shape& device_shape,
      tsl::RCReference<CommonPjRtRawBuffer> raw_buffer) {
    return absl::UnimplementedError("LinearizeHostBufferInto is not supported");
  }

  virtual void ScheduleRemoteSend(
      PjRtMemorySpace* memory_space,
      tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
      std::vector<tsl::RCReference<tsl::AsyncValue>> definition_events,
      tsl::RCReference<PjRtDeviceEventPromise> usage_event_promise,
      Future<std::string> serialized_descriptor,
      PjRtBuffer::RemoteSendCallback on_done);

  static absl::Status PrepareArguments(
      const ExecuteOptions& options,
      absl::Span<PjRtBuffer* const> argument_handles,
      absl::Span<int const> donated_params, PjRtDeviceEventSet& extra_deps,
      PjRtDeviceEventSet& control_deps,
      absl::InlinedVector<tsl::RCReference<CommonPjRtRawBuffer>, 4>&
          input_buffers,
      absl::InlinedVector<CommonPjRtBuffer::ScopedHold, 4>& device_buffers,
      PjRtDevice* device, int replica, int partition,
      absl::Span<const Shape> parameter_device_shapes, bool& is_error,
      bool allow_fallback_for_donation = false);

  absl::StatusOr<absl::InlinedVector<tsl::RCReference<CommonPjRtRawBuffer>, 4>>
  AllocateOutputBuffersWithInputReuse(
      const Shape& output_device_shape,
      absl::Span<const CommonPjRtBuffer::ScopedHold> input_device_buffer_holds,
      const HloInputOutputAliasConfig& alias_config, PjRtDevice* device,
      absl::Span<const int> output_memory_space_kind_ids);

  std::vector<std::unique_ptr<PjRtBuffer>> CreateOutputs(
      const Shape& output_device_shape,
      tsl::RCReference<PjRtDeviceEvent> definition_event, PjRtDevice* device,
      absl::Span<const int> output_memory_space_kind_ids,
      absl::InlinedVector<tsl::RCReference<CommonPjRtRawBuffer>, 4>
          output_leaf_buffers,
      bool is_predetermined_error);

  absl::Mutex& gang_scheduler() const { return gang_scheduler_mu_; }

 private:
  mutable absl::Mutex gang_scheduler_mu_;
};

// Represents the launch state for a loaded executable. This state must be
// reconstructed each time we want to launch the executable.
class PjRtRawLoadedExecutable {
 public:
  virtual ~PjRtRawLoadedExecutable() = default;

  virtual PjRtDevice* device() = 0;

  virtual absl::Status Load(const ExecuteOptions& options,
                            size_t host_callback_idx) = 0;

  struct RawExecuteResult {
    std::optional<tsl::Future<>> future;
    tsl::RCReference<PjRtDeviceEvent> primary_execute_event;
  };
  virtual RawExecuteResult Execute(
      const ExecuteOptions& options,
      absl::Span<const tsl::RCReference<CommonPjRtRawBuffer>> inputs,
      absl::Span<const tsl::RCReference<CommonPjRtRawBuffer>> results,
      PjRtDeviceEventSet& extra_deps, PjRtDeviceEventSet& control_deps,
      bool is_predetermined_error, bool fill_future) && = 0;
};

class CommonPjRtLoadedExecutable : public PjRtLoadedExecutable {
 public:
  CommonPjRtLoadedExecutable(
      std::vector<Shape> parameter_device_shapes, Shape output_device_shape,
      std::vector<int> output_memory_space_kind_ids,
      std::vector<PjRtDevice*> addressable_devices,
      std::vector<LogicalDeviceIds> addressable_device_logical_ids)
      : parameter_device_shapes_(std::move(parameter_device_shapes)),
        output_device_shape_(std::move(output_device_shape)),
        output_memory_space_kind_ids_(std::move(output_memory_space_kind_ids)),
        addressable_devices_(std::move(addressable_devices)),
        addressable_device_logical_ids_(
            std::move(addressable_device_logical_ids)) {}

  CommonPjRtClient* client() const override = 0;

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return addressable_devices_;
  }

  using PjRtLoadedExecutable::Execute;
  absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options,
      std::optional<std::vector<tsl::Future<void>>>& returned_futures)
      const override;

  using PjRtLoadedExecutable::ExecuteSharded;
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<tsl::Future<void>>& returned_future,
      bool fill_future) const override;

  using PjRtLoadedExecutable::ExecutePortable;
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<tsl::Future<void>>& returned_future,
      bool fill_future) const override;

 protected:
  // Execute is split into Prepare and Launch.
  // Prepare can fail and be retried, while Launch is guaranteed to succeed.
  struct ExecuteLaunchArgs {
    PjRtDevice* device;
    std::unique_ptr<PjRtRawLoadedExecutable> executable;
    absl::InlinedVector<tsl::RCReference<CommonPjRtRawBuffer>, 4> input_buffers;
    absl::InlinedVector<CommonPjRtBuffer::ScopedHold, 4> device_buffers;
    std::unique_ptr<PjRtDeviceEventSet> extra_deps;
    std::unique_ptr<PjRtDeviceEventSet> control_deps;
    absl::InlinedVector<tsl::RCReference<CommonPjRtRawBuffer>, 4>
        output_leaf_buffers;
    bool is_predetermined_error;
    const ExecuteOptions* options;
  };

  virtual absl::StatusOr<std::unique_ptr<PjRtRawLoadedExecutable>>
  StartRawExecutable(const ExecuteOptions& options, int replica, int partition,
                     PjRtDevice* device) const = 0;

  // Returns a sorted list of the parameters that must be donated as a
  // side-effect of the execution. Derived classes may use custom logic.
  absl::Span<int const> ParametersThatMustBeDonated() const;

  virtual const HloInputOutputAliasConfig& input_output_alias_config()
      const = 0;

  // Checks that the input buffers passed in by the user have the correct size
  // on device for the compiled program.
  absl::Status CheckBufferCompatibilities(
      const ExecuteOptions& options,
      absl::Span<const tsl::RCReference<CommonPjRtRawBuffer>> input_buffers,
      absl::Span<PjRtBuffer* const> argument_handles) const;

  absl::Status ExecutePrepare(ExecuteLaunchArgs& launch_args,
                              absl::Span<PjRtBuffer* const> argument_handles,
                              int replica, int partition,
                              const ExecuteOptions& options,
                              size_t host_callback_idx,
                              PjRtDevice* device) const;

  // Run Prepare and Launch phases on a single device.
  absl::StatusOr<Result> ExecuteHelperOnSingleDevice(
      absl::Span<PjRtBuffer* const> argument_handles, int replica,
      int partition, const ExecuteOptions& options, bool fill_future,
      PjRtDevice* device = nullptr) const;

  absl::Status ExecutePrepareWithOomRetries(
      std::optional<ExecuteLaunchArgs>& launch_args,
      absl::Span<PjRtBuffer* const> argument_handles, int replica,
      int partition, const ExecuteOptions& options, size_t host_callback_idx,
      PjRtDevice* device = nullptr) const;

  virtual void LaunchOnDevice(PjRtDevice* device,
                              absl::AnyInvocable<void()> execute_fn) const = 0;

  virtual bool ShouldRetryOnOom(int attempts, PjRtDevice* device,
                                absl::Status perpare_status) const {
    return false;
  }

  Result ExecuteLaunch(ExecuteLaunchArgs& launch_args, bool fill_future) const;

  // Parameter shapes.
  std::vector<Shape> parameter_device_shapes_;
  // A sorted vector of parameters that have any aliased buffers and thus must
  // be donated when executing the computation.
  std::vector<int> parameters_that_must_be_donated_;
  // Result layouts (device shapes).
  Shape output_device_shape_;
  // memory_space()->kind_id() for each output buffer.
  std::vector<int> output_memory_space_kind_ids_;
  // Size on device of each leaf buffer of the compiled program, cached here
  // for performance reasons.
  std::vector<int64_t> input_buffer_sizes_in_bytes_;
  // addressable_devices_[i] is the Device to which
  // addressable_device_logical_ids_[i] is assigned. shared_ptrs instead of
  // unique_ptrs to play well with the Python bindings (see xla.cc).
  std::vector<PjRtDevice*> addressable_devices_;
  // The replica and partition indices of device_assignment_ to be run by this
  // client. On single-host platforms without partitioning, this is all
  // replicas (i.e. addressable_device_logical_ids_[i] = (i, 0)), but this may
  // not be the case on multi-host platforms. If there are 4 replicas and 2
  // partitions on a single host platform, size of
  // addressable_device_logical_ids_ is 4*2 = 8.
  std::vector<LogicalDeviceIds> addressable_device_logical_ids_;
};

// TODO(parkers): Merge everything here into CommonPjRtBuffer.
class CommonPjRtBufferImpl : public CommonPjRtBuffer {
 public:
  CommonPjRtBufferImpl(
      const Shape& on_device_shape,
      std::unique_ptr<AbstractTrackedDeviceBuffer> tracked_device_buffer,
      PjRtMemorySpace* memory_space);

  ~CommonPjRtBufferImpl() override;

  CommonPjRtBufferImpl(const CommonPjRtBufferImpl&) = delete;
  CommonPjRtBufferImpl(CommonPjRtBufferImpl&&) = delete;
  CommonPjRtBufferImpl& operator=(const CommonPjRtBufferImpl&) = delete;
  CommonPjRtBufferImpl& operator=(CommonPjRtBufferImpl&&) = delete;

  const Shape& on_device_shape() const override { return on_device_shape_; }
  ABSL_DEPRECATED(
      "Buffers are associated with memories. Use memory_space() instead when "
      "possible.")
  PjRtDevice* device() const override;
  CommonPjRtClient* client() const override;
  PjRtMemorySpace* memory_space() const override { return memory_space_; }

  absl::StatusOr<size_t> GetOnDeviceSizeInBytes() const override;

  absl::StatusOr<std::unique_ptr<ExternalReference>>
  ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> DonateWithControlDependency(
      Future<> dependency) override;

  Future<> GetReadyFuture() override;

  // The implementation of logical_on_device_shape may involve a blocking
  // device to host transfer to read the metadata of dynamic shape.
  absl::StatusOr<Shape> logical_on_device_shape() override;

  void CopyToRemoteDevice(Future<std::string> serialized_descriptor,
                          RemoteSendCallback on_done) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(
      PjRtMemorySpace* dst_memory_space) override;
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(
      PjRtBuffer* donated_dst) override;

  // This behaves like CopyToMemorySpace for memory space pairs which
  // require no layout changes.
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> DirectCopyToMemorySpace(
      PjRtMemorySpace* dst_memory_space);
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> DirectCopyToMemorySpace(
      PjRtBuffer* donated_dst);

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToCpuMemorySpace(
      const xla::Shape& shape, PjRtMemorySpace* dst_memory_space);

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyFromCpuToMemorySpace(
      const xla::Shape& shape, PjRtMemorySpace* dst_memory_space);

  absl::StatusOr<std::unique_ptr<PjRtBuffer>>
  CopyToMemorySpaceFallbackThroughLiteral(PjRtMemorySpace* dst_memory_space);

  absl::StatusOr<std::unique_ptr<PjRtBuffer>>
  CopyToMemorySpaceSyncThroughLiteral(PjRtMemorySpace* dst_memory_space);

  using PjRtBuffer::ToLiteralSync;
  Future<> ToLiteral(MutableLiteralBase* literal) override;
  Future<> LazyToLiteral(
      absl::AnyInvocable<Future<MutableLiteralBase*>() &&> generator) override;

  absl::StatusOr<tsl::RCReference<PjRtRawBuffer>> CreateRawAliasOfBuffer();

  absl::StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference()
      override;

  Future<> CopyRawToHost(void* dst, int64_t offset,
                         int64_t transfer_size) override;

  Future<> CopyRawToHostFuture(Future<void*> dst, int64_t offset,
                               int64_t transfer_size) override;

  void Delete() override;

  bool IsOnCpu() const override;

 protected:
  // Shared implementation for ToLiteral and LazyToLiteral. If `literal` is
  // null, will call the function in the generator.
  Future<> ToLiteralImpl(
      MutableLiteralBase* literal,
      absl::AnyInvocable<Future<MutableLiteralBase*>() &&> generator);

 private:
  const Shape on_device_shape_;
};

}  // namespace xla

#endif  // XLA_PJRT_COMMON_PJRT_CLIENT_H_
