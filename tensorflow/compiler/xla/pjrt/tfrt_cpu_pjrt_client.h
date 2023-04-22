/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_TFRT_CPU_PJRT_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_TFRT_CPU_PJRT_CLIENT_H_

#include <memory>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/semaphore.h"
#include "tensorflow/compiler/xla/pjrt/tracked_tfrt_cpu_device_buffer.h"
#include "tensorflow/compiler/xla/pjrt/worker_thread.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime

namespace xla {

class TfrtCpuDevice final : public PjRtDevice {
 public:
  TfrtCpuDevice(int id, bool asynchronous);

  void SetClient(PjRtClient* client) {
    CHECK(client_ == nullptr);
    client_ = client;
  }

  PjRtClient* client() const override { return client_; }

  bool IsAddressable() const override {
    return process_index() == client()->process_index();
  }

  int id() const override { return id_; }

  int process_index() const override { return 0; }

  // Used as `device_ordinal`.
  int local_hardware_id() const override { return id_; }

  absl::string_view device_kind() const override;

  std::string DebugString() const override;

  Status TransferToInfeed(const LiteralSlice& literal) override;

  Status TransferFromOutfeed(MutableBorrowingLiteral literal) override;

  // Returns a semaphore for admission control on inflight computations.
  Semaphore& max_inflight_computations_semaphore() {
    return max_inflight_computations_semaphore_;
  }

 private:
  int id_;
  PjRtClient* client_ = nullptr;

  // TODO(zhangqiaorjc): Optimize semaphore related overhead.
  // Semaphore used to limit how many programs can be enqueued by the host
  // ahead of the device.
  Semaphore max_inflight_computations_semaphore_;
};

class TfrtCpuExecutable;

class TfrtCpuClient final : public PjRtClient {
 public:
  TfrtCpuClient(int process_index,
                std::vector<std::unique_ptr<TfrtCpuDevice>> devices,
                std::unique_ptr<tfrt::HostContext> host_ctx);

  int process_index() const override { return process_index_; }

  int device_count() const override { return devices_.size(); }

  int addressable_device_count() const override {
    return addressable_devices_.size();
  }

  absl::Span<PjRtDevice* const> devices() const override { return devices_; }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return addressable_devices_;
  }

  StatusOr<PjRtDevice*> LookupDevice(int device_id) const override;

  StatusOr<PjRtDevice*> LookupAddressableDevice(
      int local_hardware_id) const override;

  PjRtPlatformId platform_id() const override {
    return tensorflow::Fingerprint64(kCpuName);
  }

  absl::string_view platform_name() const override { return kCpuName; }

  absl::string_view platform_version() const override { return "<unknown>"; }

  PjRtRuntimeType runtime_type() const override { return kTfrt; }

  StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis() override;

  StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) override;

  StatusOr<absl::optional<std::string>> ExecutableFingerprint(
      const PjRtExecutable& executable) const override;

  StatusOr<std::string> SerializeExecutable(
      const PjRtExecutable& executable) const override {
    return Unimplemented("SerializeExecutable not implemented on %s",
                         platform_name());
  }

  StatusOr<std::unique_ptr<PjRtExecutable>> DeserializeExecutable(
      absl::string_view serialized, std::unique_ptr<HloModule> hlo_module,
      CompileOptions options) override {
    return Unimplemented("DeserializeExecutable not implemented on %s",
                         platform_name());
  }

  StatusOr<std::unique_ptr<PjRtBuffer>> CreateUninitializedBuffer(
      const Shape& shape, PjRtDevice* device) override;

  StatusOr<std::unique_ptr<PjRtClient::AsyncBufferTransferManager>>
  CreateBuffersForAsyncTransfer(absl::Span<const Shape> shapes,
                                PjRtDevice* device) override {
    return Unimplemented("Async transfer to buffers not implemented");
  };

  StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, const Shape& shape,
      HostBufferSemantics host_buffer_semantics,
      std::function<void()> on_done_with_host_buffer,
      PjRtDevice* device) override;

  StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtDevice* device) override;

  void MakeCrossHostReceiveBuffers(
      absl::Span<const Shape> shapes, PjRtDevice* device,
      PjRtCrossHostRecvNotifier&& notifier) override {
    LOG(FATAL) << "MakeCrossHostReceiveBuffers not implemented.";
  }

  void MakeCrossHostReceiveBuffersForGather(
      absl::Span<const Shape> shapes, std::vector<GatherDetails> gather_details,
      PjRtDevice* device, PjRtCrossHostRecvNotifier&& notifier) override {
    LOG(FATAL) << "MakeCrossHostReceiveBuffersForGather not implemented.";
  }

  StatusOr<std::unique_ptr<PjRtBuffer>> CreateViewOfDeviceBuffer(
      void* device_ptr, const Shape& shape, PjRtDevice* device,
      std::function<void()> on_delete_callback) override;

  StatusOr<ChannelHandle> CreateChannelHandle() override {
    return Unimplemented("CreateChannelHandle not implemented.");
  }
  StatusOr<ChannelHandle> CreateDeviceToHostChannelHandle() override {
    return Unimplemented("CreateDeviceToHostChannelHandle not implemented.");
  }
  StatusOr<ChannelHandle> CreateHostToDeviceChannelHandle() override {
    return Unimplemented("CreateHostToDeviceChannelHandle not implemented.");
  }

  Status Defragment() override {
    return Unimplemented("Defragment not implemented.");
  }

  tfrt::HostContext* GetHostContext() const { return host_ctx_.get(); }

  Eigen::ThreadPoolDevice* eigen_intraop_device() const {
    return eigen_intraop_device_.get();
  }

  tfrt::AsyncValueRef<CpuEvent> GetLastCollectiveLaunchEvent() {
    absl::MutexLock lock(&mu_);
    return last_collective_launch_event_.CopyRef();
  }

  void SetLastCollectiveLaunchEvent(tfrt::AsyncValueRef<CpuEvent> event) {
    absl::MutexLock lock(&mu_);
    last_collective_launch_event_ = std::move(event);
  }

 private:
  int process_index_;
  // Includes all devices, including non-addressable devices.
  std::vector<std::unique_ptr<TfrtCpuDevice>> owned_devices_;
  // Pointers to `owned_devices_`.
  std::vector<PjRtDevice*> devices_;
  // Maps Device::id() to the corresponding Device. Includes all devices.
  absl::flat_hash_map<int, TfrtCpuDevice*> id_to_device_;
  // Addressable devices indexed by core_id.
  std::vector<PjRtDevice*> addressable_devices_;
  std::unique_ptr<tfrt::HostContext> host_ctx_;
  std::unique_ptr<ComputationPlacer> computation_placer_;

  // TODO(zhangqiaorjc): Use tfrt::compat::EigenHostContextThreadPool.
  std::unique_ptr<tensorflow::thread::ThreadPool> eigen_intraop_pool_;
  std::unique_ptr<Eigen::ThreadPoolDevice> eigen_intraop_device_;

  // Launching collectives are prone to deadlock when we use fixed-sized
  // threadpools since ExecuteHelper will block until all replicas reach the
  // barrier. We ensure that
  // 1. Threadpool size is at least as large as device_count so one collective
  //    launch over all devices can succeed.
  // 2. Gang-schedule each collective by conservatively ensuring a total order
  //    of collectives and launching only one collective at a time to avoid
  //    having no active threads to make progress
  // TODO(zhangqiaorjc): Explore alternatives that allow multiple concurrent
  // collectives.
  mutable absl::Mutex mu_;
  tfrt::AsyncValueRef<CpuEvent> last_collective_launch_event_
      ABSL_GUARDED_BY(mu_);
};

class TfrtCpuBuffer final : public PjRtBuffer {
 public:
  // Helper class to retain a "hold" on a TfrtCpuBuffer. A ScopedHold may not
  // outlive its parent TfrtCpuBuffer.
  //
  // There are three types of hold, as follows:
  //
  // 1) Usage hold: a transient hold while an operation using the buffer is
  //    being enqueued to the runtime.
  // A client acquires a usage hold by calling
  // TfrtCpuBuffer::GetBufferWithHold(kUsage) or the convenience
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
  // TfrtCpuBuffer::GetBufferWithHold(kExternal) or the convenience
  // wrapper GetBufferWithExternalReference and releases it by deleting the
  // ScopedHold. The external framework should not modify the underlying buffer
  // unless it is confident via its own synchronization that modifications do
  // not race with reads from the TfrtCpuBuffer.
  //
  // 3) Donation hold: a transient hold while an execution that donates the
  //    buffer is being enqueued to the runtime.
  // A client acquires a donation hold by calling
  // TfrtCpuBuffer::GetBufferWithHold(kDonation). If the enqueue
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
  // Calls to TfrtCpuBuffer::ReleaseDeviceMemoryOwnership (and transitively to
  // TfrtCpuBuffer::Delete() and ~TfrtCpuBuffer()) will block until all usage
  // and donation holds are either deleted or converted/confirmed.
  class ScopedHold {
   public:
    enum Type { kUsage = 0, kExternalReference, kDonation, kMaxValue };
    // Use a State enum instead of encoding the state in an error Status to
    // avoid creating Status values in non-error cases. Creating a Status
    // entails several allocations and can add O(us) to every use of a hold.
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

    Status status() const {
      // Lazily create Status values only when they are requested.
      switch (state_) {
        case kUninitialized:
          return InvalidArgument("Buffer has not been initialized");
        case kValid:
          return Status::OK();
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
    const std::shared_ptr<TrackedTfrtCpuDeviceBuffer>& buffer() const {
      CHECK_EQ(state_, kValid);
      CHECK_NE(buffer_, nullptr);
      return buffer_;
    }
    TrackedTfrtCpuDeviceBuffer* operator->() const { return buffer().get(); }
    const TrackedTfrtCpuDeviceBuffer& operator*() const { return *buffer(); }

    // Converts the hold into a usage event. Only valid for holds of type
    // kUsage.
    void ConvertUsageHold(absl::Span<tfrt::AsyncValueRef<CpuEvent>> events);

    // Confirms that the buffer was successfully donated to an execution.
    // Only valid for holds of type kDonation. Causes the buffer to become
    // invalid.
    void ConfirmDonation();

   private:
    friend class TfrtCpuClient;
    friend class TfrtCpuBuffer;

    // Helper struct that makes it possible to move a ScopedHold through a
    // closure.
    using ForClosure = std::tuple<TfrtCpuBuffer*, Type, State, Status,
                                  std::shared_ptr<TrackedTfrtCpuDeviceBuffer>>;

    ScopedHold(TfrtCpuBuffer* parent, Type type)
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
        StatusOr<std::shared_ptr<TrackedTfrtCpuDeviceBuffer>>&& buffer_or);
    // Releases the contents of *this, so *this can subsequently be
    // deleted without releasing the parent's hold. Should be passed to the
    // appropriate constructor of another ScopedHold, e.g., when a hold must be
    // passed through a closure that is incompatible with std::move.
    ForClosure ToClosure();

    TfrtCpuBuffer* const parent_;
    const Type type_;

    // There is an invariant that if ok() then buffer_ != nullptr.
    State state_;
    Status status_;
    std::shared_ptr<TrackedTfrtCpuDeviceBuffer> buffer_;
  };

  TfrtCpuBuffer(
      Shape on_device_shape,
      std::shared_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer,
      TfrtCpuClient* client, TfrtCpuDevice* device);
  ~TfrtCpuBuffer() override;

  TfrtCpuBuffer(const TfrtCpuBuffer&) = delete;
  TfrtCpuBuffer(TfrtCpuBuffer&&) = delete;
  TfrtCpuBuffer& operator=(const TfrtCpuBuffer&) = delete;
  TfrtCpuBuffer& operator=(TfrtCpuBuffer&&) = delete;

  const Shape& on_device_shape() const override { return on_device_shape_; }
  TfrtCpuDevice* device() const override { return device_; }
  TfrtCpuClient* client() const override { return client_; }

  StatusOr<Shape> logical_on_device_shape() override;

  StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference()
      override;

  StatusOr<std::unique_ptr<ExternalReference>> ReleaseDeviceMemoryOwnership(
      bool wait_for_operations_to_complete) override;

  using PjRtBuffer::ToLiteral;
  void ToLiteral(MutableLiteralBase* literal,
                 std::function<void(Status)> on_ready) override;

  StatusOr<size_t> GetOnDeviceSizeInBytes() const override;

  Status CopyRawToHost(void* dst, int64 offset, int64 transfer_size,
                       std::function<void(Status)> on_ready) override {
    return Unimplemented("CopyRawToHost not implemented");
  }

  void Delete() override;

  bool IsDeleted() override;

  StatusOr<std::unique_ptr<PjRtBuffer>> CopyToDevice(
      PjRtDevice* dst_device) override;

  Status CopyToRemoteDevice(absl::string_view serialized_descriptor) override {
    return Unimplemented("CopyToRemoteDevice not implemented.");
  }

  Status CopyToRemoteDeviceScattered(
      absl::Span<const std::string> serialized_descriptors,
      const ScatterDetails& scatter_details) override {
    return Unimplemented("CopyToRemoteDeviceScattered not implemented.");
  }

  Status BlockHostUntilReady() override;

  bool IsOnCpu() const override { return true; }

  // Returns a hold on the TrackedTfrtCpuDeviceBuffer holding the device
  // buffers. See comment on ScopedHold.
  ScopedHold GetBufferWithHold(ScopedHold::Type type);
  ScopedHold GetBufferWithUsageHold() {
    return GetBufferWithHold(ScopedHold::kUsage);
  }
  ScopedHold GetBufferWithExternalReference() {
    return GetBufferWithHold(ScopedHold::kExternalReference);
  }

 private:
  bool IsEmptyTuple() const {
    return on_device_shape_.IsTuple() &&
           on_device_shape_.tuple_shapes_size() == 0;
  }

  StatusOr<tfrt::AsyncValueRef<Literal>> CopyToHostAsyncInternal(
      bool discard_cached_copy, absl::optional<xla::Layout> layout);

  // Requires holds_[kDonation] == 0 (i.e., WaitForOutstandingDonationHolds()
  // must be called first.)
  StatusOr<std::shared_ptr<TrackedTfrtCpuDeviceBuffer>> GetBufferForHoldLocked(
      ScopedHold::Type type) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Requires holds_[kDonation] == 0 (i.e., WaitForOutstandingDonationHolds()
  // must be called first.)
  void AcquireHoldLocked(ScopedHold* hold) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void ConvertUsageHold(TrackedTfrtCpuDeviceBuffer* buffer,
                        absl::Span<tfrt::AsyncValueRef<CpuEvent>> events);

  void ConfirmDonation(TrackedTfrtCpuDeviceBuffer* device_buffer);

  void DropHold(ScopedHold::Type type, TrackedTfrtCpuDeviceBuffer* buffer);

  void WaitForOutstandingUsageHolds() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void WaitForOutstandingDonationHold() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Similar to Delete, drops the buffer's reference to its associated device
  // memory, leaving the buffer in an invalid state, but returns the
  // TrackedTfrtCpuDeviceBuffer rather than freeing the device memory, so that
  // another framework can take ownership of it. The buffer returned from
  // Release may be safely dropped at any time even if it still has pending
  // async operations. The client should call BlockHostUntilReady before calling
  // Release with wait_for_operations_to_complete=false, to ensure that the host
  // has synchronized past any outstanding write operations to the buffer. If
  // wait_for_operations_to_complete=true the host will block until any
  // potentially outstanding asynchronous operations have completed before
  // returning, in which case it is safe to read or mutate the returned buffer.
  // If the buffer was shared via an external reference it is the client's
  // responsibility that accesses via that reference do not interfere with
  // accesses via the buffer returned from Release.
  StatusOr<std::shared_ptr<TrackedTfrtCpuDeviceBuffer>> Release(
      bool wait_for_operations_to_complete);

  TfrtCpuClient* client_;
  const Shape on_device_shape_;
  TfrtCpuDevice* const device_;

  mutable absl::Mutex mu_;
  std::shared_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer_
      ABSL_GUARDED_BY(mu_);
  // Count of holds on the buffer.
  std::array<int, ScopedHold::Type::kMaxValue> holds_ ABSL_GUARDED_BY(mu_);
};

class TfrtCpuExecutable final : public PjRtExecutable {
 public:
  TfrtCpuExecutable(
      int num_replicas, int num_partitions,
      std::shared_ptr<DeviceAssignment> device_assignment,
      bool parameter_is_tupled_arguments,
      std::unique_ptr<Executable> cpu_executable,
      BufferAllocation::Index result_buffer_index,
      absl::InlinedVector<BufferAllocation::Index, 4> result_buffer_indices,
      std::vector<LogicalDeviceIds> addressable_device_logical_ids,
      std::vector<PjRtDevice*> addressable_devices, TfrtCpuClient* client);

  ~TfrtCpuExecutable() override = default;

  TfrtCpuClient* client() const override { return client_; }

  absl::string_view name() const override {
    return cpu_executable_->shared_module()->name();
  }

  int num_replicas() const override { return num_replicas_; }

  int num_partitions() const override { return num_partitions_; }

  int64 SizeOfGeneratedCodeInBytes() const override {
    return cpu_executable_->SizeOfGeneratedCodeInBytes();
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

  StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override {
    return std::vector<std::shared_ptr<HloModule>>{
        cpu_executable_->shared_module()};
  }

  StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options) override;

  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options) override;

  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options) override;

  void Delete() override;

  bool IsDeleted() override;

  StatusOr<absl::optional<std::string>> Fingerprint() const;

 private:
  friend class TfrtCpuClient;

  Status SetUpDonation(bool tuple_inputs);

  // Checks that the input buffers passed in by the user have the correct size
  // on device for the compiled program.
  Status CheckBufferCompatibilities(
      absl::Span<const std::shared_ptr<TrackedTfrtCpuDeviceBuffer>>
          input_buffers) const;

  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteHelper(
      absl::Span<PjRtBuffer* const> argument_handles, int replica,
      int partition, const RunId& run_id, const ExecuteOptions& options,
      tfrt::AsyncValueRef<CpuEvent> last_collective_launch_event,
      TfrtCpuDevice* device = nullptr);

  TfrtCpuClient* client_;

  int num_replicas_;
  int num_partitions_;
  std::shared_ptr<DeviceAssignment> device_assignment_;
  bool parameter_is_tupled_arguments_;

  std::shared_ptr<Executable> cpu_executable_;

  // Caching `result_buffer_index_` and `result_buffer_indices_` to avoid lookup
  // HLO dataflow analysis data structures in program execution critical path.

  // Buffer allocation index corresponding to root buffer buffer.
  BufferAllocation::Index result_buffer_index_;
  // Buffer allocation indices corresponding to each result buffer leaf buffer.
  absl::InlinedVector<BufferAllocation::Index, 4> result_buffer_indices_;

  // Size on device of each leaf buffer of the compiled program, cached here
  // for performance reasons.
  std::vector<int64_t> input_buffer_sizes_in_bytes_;

  // A sorted vector of parameters that have any aliased buffers and thus must
  // be donated when executing the computation.
  std::vector<int> parameters_that_must_be_donated_;

  // The replica and partition indices of device_assignment_ to be run by this
  // client. On single-host platforms without partitioning, this is all
  // replicas (i.e. addressable_device_logical_ids_[i] = (i, 0)), but this may
  // not be the case on multi-host platforms. If there are 4 replicas and 2
  // partitions on a single host platform, size of
  // addressable_device_logical_ids_ is 4*2 = 8.
  std::vector<LogicalDeviceIds> addressable_device_logical_ids_;

  // addressable_devices_[i] is the Device to which
  // addressable_device_logical_ids_[i] is assigned. shared_ptrs instead of
  // unique_ptrs to play well with the Python bindings (see xla.cc).
  std::vector<PjRtDevice*> addressable_devices_;

  // Cached result of comparing HloCostAnalysis FLOP estimate for execute
  // critical path.
  bool cheap_computation_;
};

StatusOr<std::unique_ptr<PjRtClient>> GetTfrtCpuClient(bool asynchronous);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_TFRT_CPU_PJRT_CLIENT_H_
