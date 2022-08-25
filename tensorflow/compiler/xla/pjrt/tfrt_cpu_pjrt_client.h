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

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_future.h"
#include "tensorflow/compiler/xla/pjrt/semaphore.h"
#include "tensorflow/compiler/xla/pjrt/tracked_tfrt_cpu_device_buffer.h"
#include "tensorflow/compiler/xla/pjrt/transpose.h"
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

  absl::string_view DebugString() const override;

  absl::string_view ToString() const override;

  Status TransferToInfeed(const LiteralSlice& literal) override;

  Status TransferFromOutfeed(MutableBorrowingLiteral literal) override;

  // Returns a semaphore for admission control on inflight computations.
  Semaphore& max_inflight_computations_semaphore() {
    return max_inflight_computations_semaphore_;
  }

  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override {
    return nullptr;
  }

  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }

 private:
  int id_;
  PjRtClient* client_ = nullptr;
  std::string debug_string_;
  std::string to_string_;

  // TODO(zhangqiaorjc): Optimize semaphore related overhead.
  // Semaphore used to limit how many programs can be enqueued by the host
  // ahead of the device.
  Semaphore max_inflight_computations_semaphore_;
  absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes_ = {};
};

class TfrtCpuExecutable;

class TfrtCpuClient final : public PjRtClient {
 public:
  TfrtCpuClient(int process_index,
                std::vector<std::unique_ptr<TfrtCpuDevice>> devices,
                std::unique_ptr<tfrt::HostContext> host_ctx);
  ~TfrtCpuClient();

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
    return tensorflow::Fingerprint64(CpuName());
  }

  absl::string_view platform_name() const override { return CpuName(); }

  absl::string_view platform_version() const override { return "<unknown>"; }

  PjRtRuntimeType runtime_type() const override { return kTfrt; }

  StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis() override;

  StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) override;
  StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      mlir::ModuleOp module, CompileOptions options) override;

  StatusOr<std::optional<std::string>> ExecutableFingerprint(
      const PjRtLoadedExecutable& executable) const override;

  StatusOr<std::string> SerializeExecutable(
      const PjRtLoadedExecutable& executable) const override {
    return Unimplemented("SerializeExecutable not implemented on %s",
                         platform_name());
  }

  StatusOr<std::unique_ptr<PjRtLoadedExecutable>> DeserializeExecutable(
      absl::string_view serialized, CompileOptions options) override {
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
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      std::function<void()> on_done_with_host_buffer,
      PjRtDevice* device) override;

  StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtDevice* device) override;

  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeCrossHostReceiveBuffers(absl::Span<const Shape> shapes,
                              PjRtDevice* device,
                              PjRtCrossHostRecvNotifier notifier) override {
    return Unimplemented("MakeCrossHostReceiveBuffers not implemented.");
  }

  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeCrossHostReceiveBuffersForGather(
      absl::Span<const Shape> shapes, std::vector<GatherDetails> gather_details,
      PjRtDevice* device, PjRtCrossHostRecvNotifier notifier) override {
    return Unimplemented(
        "MakeCrossHostReceiveBuffersForGather not implemented.");
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

  // A cache for transpose plans. We use transposes to convert
  // (possibly strided) buffers provided to BufferFromHostBuffer into dense
  // major-to-minor layout.
  absl::Mutex transpose_mu_;
  TransposePlanCache transpose_cache_ ABSL_GUARDED_BY(transpose_mu_);
};

class TfrtCpuBuffer final : public PjRtBuffer {
 public:
  TfrtCpuBuffer(
      Shape on_device_shape,
      std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer,
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

  using PjRtBuffer::ToLiteralSync;
  PjRtFuture<Status> ToLiteral(MutableLiteralBase* literal) override;

  StatusOr<size_t> GetOnDeviceSizeInBytes() const override;

  PjRtFuture<Status> CopyRawToHost(void* dst, int64_t offset,
                                   int64_t transfer_size) override {
    return PjRtFuture<Status>(Unimplemented("CopyRawToHost not implemented"));
  }

  void Delete() override;

  bool IsDeleted() override;

  StatusOr<std::unique_ptr<PjRtBuffer>> CopyToDevice(
      PjRtDevice* dst_device) override;

  void CopyToRemoteDevice(absl::string_view serialized_descriptor,
                          RemoteSendCallback on_done) override {
    on_done(Unimplemented("CopyToRemoteDevice not implemented."),
            /*sends_were_enqueued=*/false);
  }

  void CopyToRemoteDeviceScattered(
      absl::Span<const std::pair<std::string, RemoteSendCallback>>
          serialized_descriptors_and_callbacks,
      const ScatterDetails& scatter_details) override {
    for (const auto& d_and_cb : serialized_descriptors_and_callbacks) {
      d_and_cb.second(
          Unimplemented("CopyToRemoteDeviceScattered not implemented."),
          /*sends_were_enqueued=*/false);
    }
  }

  PjRtFuture<Status> GetReadyFuture() override;

  bool IsOnCpu() const override { return true; }

 private:
  bool IsEmptyTuple() const {
    return on_device_shape_.IsTuple() &&
           on_device_shape_.tuple_shapes_size() == 0;
  }

  StatusOr<tfrt::AsyncValueRef<Literal>> CopyToHostAsyncInternal(
      bool discard_cached_copy, std::optional<xla::Layout> layout);

  // Acquires the device buffer for shared read-only usages, and it also adds
  // the `usage_event` to it. Any donation event in the future is expected to be
  // serialized after all the usage events added through this method. Returns
  // nullptr if the buffer is already donated or there is outstanding external
  // references.
  TrackedTfrtCpuDeviceBuffer* AcquireUsage(
      tfrt::AsyncValueRef<CpuEvent> usage_event);

  // A helper class for managing a pending donation. It should be committed upon
  // success. Otherwise, the donated buffer is returned to the TfrtCpuBuffer.
  class DonationTransaction {
   public:
    explicit DonationTransaction(
        TfrtCpuBuffer* buffer,
        std::unique_ptr<TrackedTfrtCpuDeviceBuffer> device_buffer)
        : buffer_(buffer), device_buffer_(std::move(device_buffer)) {
      CHECK(buffer_);
    }
    DonationTransaction(const DonationTransaction&) = delete;
    DonationTransaction& operator=(const DonationTransaction&) = delete;
    DonationTransaction(DonationTransaction&&) = default;
    DonationTransaction& operator=(DonationTransaction&& other) {
      Abort();

      buffer_ = other.buffer_;
      device_buffer_ = std::move(other.device_buffer_);
      return *this;
    }

    ~DonationTransaction() { Abort(); }

    // Commit the donation. The rvalue ref qualifier is used to ensure the
    // semantic that it can be committed at most once.
    void Commit() && {
      buffer_->CommitDonation();
      device_buffer_.reset();
    }

    TrackedTfrtCpuDeviceBuffer* device_buffer() const {
      return device_buffer_.get();
    }

   private:
    void Abort() {
      if (device_buffer_) buffer_->AbortDonation(std::move(device_buffer_));
    }

    TfrtCpuBuffer* buffer_ = nullptr;
    std::unique_ptr<TrackedTfrtCpuDeviceBuffer> device_buffer_;
  };

  // Acquires the device buffer for exclusive donation. The caller of this
  // method is expected to use the usage events and definition events to
  // serialize this donation with previous usages. After this method is called,
  // calls to AcquireUsage() will fail. Returns error status if the buffer is
  // already donated or there is outstanding external references.
  StatusOr<DonationTransaction> AcquireDonation();

  void DropExternalReference() {
    absl::MutexLock lock(&mu_);
    CHECK_GT(external_reference_counter_, 0);
    --external_reference_counter_;
  }

  // Commits the pending donation by setting `pending_donation_` to false.
  // `pending_donation_` must be true before calling this method.
  void CommitDonation();

  // Aborts the pending donation by returning the donated buffer, and setting
  // `pending_donation_` to false. `pending_donation_` must be true before
  // calling this method.
  void AbortDonation(std::unique_ptr<TrackedTfrtCpuDeviceBuffer> device_buffer);

  // Similar to Delete, drops the buffer's reference to its associated device
  // memory, leaving the buffer in an invalid state, but returns the
  // TrackedTfrtCpuDeviceBuffer rather than freeing the device memory, so that
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
  StatusOr<std::unique_ptr<TrackedTfrtCpuDeviceBuffer>> Release(
      bool wait_for_operations_to_complete);

  // Releases the device buffer by returning a unique_ptr of it. If there is
  // outstanding donation or usage holds, this method blocks until those holds
  // are commited or dropped.
  std::unique_ptr<TrackedTfrtCpuDeviceBuffer> ReleaseBufferLocked()
      ABSL_LOCKS_EXCLUDED(mu_);

  TfrtCpuClient* client_;
  const Shape on_device_shape_;
  TfrtCpuDevice* const device_;

  mutable absl::Mutex mu_;
  std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer_
      ABSL_GUARDED_BY(mu_);
  // Count of external references on the buffer.
  int external_reference_counter_ ABSL_GUARDED_BY(mu_) = 0;

  // `pending_donation_` indicates whether a donation is pending. The destructor
  // of the TfrtCpuBuffer will wait for a pending donation, as the donation
  // might fail. Note that concurrent calls to AcquireUsage() and
  // AcquireDonation() might fail even if the pending donation is aborted later.
  bool pending_donation_ ABSL_GUARDED_BY(mu_) = false;

  friend class TfrtCpuClient;
  friend class TfrtCpuExecutable;
};

class TfrtCpuExecutable final : public PjRtLoadedExecutable {
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

  int64_t SizeOfGeneratedCodeInBytes() const override {
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

  void Delete() override;

  bool IsDeleted() override;

  bool IsReturnedFutureSupported() const override { return true; }

  StatusOr<std::optional<std::string>> Fingerprint() const;

 private:
  friend class TfrtCpuClient;

  Status SetUpDonation(bool tuple_inputs);

  // Checks that the input buffers passed in by the user have the correct size
  // on device for the compiled program.
  Status CheckBufferCompatibilities(
      absl::Span<TrackedTfrtCpuDeviceBuffer* const> input_buffers) const;

  StatusOr<Result> ExecuteHelper(
      absl::Span<PjRtBuffer* const> argument_handles, int replica,
      int partition, const RunId& run_id, const ExecuteOptions& options,
      tfrt::AsyncValueRef<CpuEvent> last_collective_launch_event,
      bool fill_future, TfrtCpuDevice* device = nullptr);

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

// Creates a CPU client with one Device. For testing purposes, you can set the
// number of devices passing the --xla_force_host_platform_device_count flag to
// the XLA_FLAGS environment variable.
StatusOr<std::unique_ptr<PjRtClient>> GetTfrtCpuClient(bool asynchronous);

// Similar to the function above, but you can set the number of devices
// explicitly.
StatusOr<std::unique_ptr<PjRtClient>> GetTfrtCpuClient(bool asynchronous,
                                                       int cpu_device_count);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_TFRT_CPU_PJRT_CLIENT_H_
