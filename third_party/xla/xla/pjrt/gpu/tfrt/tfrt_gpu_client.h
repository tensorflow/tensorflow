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
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "mlir/IR/BuiltinOps.h"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/maybe_owning.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/gpu/gpu_topology.h"
#include "xla/pjrt/gpu/se_gpu_topology_description.h"
#include "xla/pjrt/gpu/tfrt/gpu_event.h"
#include "xla/pjrt/gpu/tfrt/host_memory_allocator.h"
#include "xla/pjrt/gpu/tfrt/tracked_tfrt_gpu_device_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_stream_executor_device_description.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/semaphore.h"
#include "xla/pjrt/transpose.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
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
    int32_t process_index;
    int slice_index;
    PjRtLocalDeviceId local_device_id;
    PjRtLocalHardwareId local_hardware_id;
    se::StreamExecutor* executor;
    int max_inflight_computations;
    std::string platform_version;
    std::string compute_capability;
    std::string device_vendor;
    int core_count;
  };

  explicit TfrtGpuDevice(Options&& options);

  ~TfrtGpuDevice() override;

  void SetClient(PjRtClient* client);

  const PjRtStreamExecutorDeviceDescription& description() const override {
    return description_;
  }

  PjRtClient* client() const override { return client_; }

  bool IsAddressable() const override { return local_device_id_ != -1; }

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

  // Returns the semaphore to control the max inflight computations.
  Semaphore& max_inflight_computations_semaphore() {
    return max_inflight_computations_semaphore_;
  }

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

  absl::StatusOr<tsl::AllocatorStats> GetAllocatorStats() const override;

  se::DeviceMemoryAllocator* allocator() const;

  // Returns a fresh, PRNG-generated random seed for an XLA computation.
  int GetNewPrngSeed();

  se::Stream* stream() const { return stream_.get(); }

  se::StreamExecutor* executor() const { return executor_; }

  tsl::AsyncValueRef<GpuEvent> SetLastCollectiveLaunchEvent(
      tsl::AsyncValueRef<GpuEvent> event);

 private:
  friend class TfrtGpuClient;
  friend class TfrtGpuExecutable;
  friend class TfrtGpuBuffer;

  absl::StatusOr<TransferManager*> GetTransferManager();

  int id_;
  PjRtClient* client_ = nullptr;
  const PjRtLocalDeviceId local_device_id_;
  const PjRtLocalHardwareId local_hardware_id_;
  se::StreamExecutor* executor_;
  std::unique_ptr<se::Stream> stream_;
  absl::InlinedVector<PjRtMemorySpace*, 1> memory_spaces_;
  absl::flat_hash_map<int, PjRtMemorySpace*> memory_spaces_by_kind_id_;

  absl::Mutex mu_;
  std::random_device prng_seed_device_ ABSL_GUARDED_BY(mu_);
  std::mt19937 prng_seed_generator_ ABSL_GUARDED_BY(mu_);
  std::uniform_int_distribution<> prng_seed_distribution_ ABSL_GUARDED_BY(mu_);
  // Launching collectives are prone to deadlock when we use fixed-sized
  // thread pools and stream pools, since ExecuteHelper will block until all
  // replicas reach the barrier. We ensure that
  // 1. Thread pool size is at least as large as device_count so one collective
  //    launch over all devices can succeed.
  // 2. Gang-schedule each collective by conservatively ensuring a total order
  //    of collectives and launching only one collective at a time to avoid
  //    having no active threads to make progress
  tsl::AsyncValueRef<GpuEvent> last_collective_launch_event_
      ABSL_GUARDED_BY(mu_);

  PjRtStreamExecutorDeviceDescription description_;
  PjRtMemorySpace* default_memory_space_ = nullptr;

  // Semaphore used to limit how many programs can be enqueued by the host
  // ahead of the device.
  xla::Semaphore max_inflight_computations_semaphore_;
};

class TfrtGpuClient final : public PjRtClient {
 public:
  TfrtGpuClient(std::string platform_name, int process_index,
                xla::LocalClient* xla_client,
                std::vector<std::unique_ptr<TfrtGpuDevice>> devices,
                bool should_stage_host_to_device_transfers,
                MaybeOwning<se::DeviceMemoryAllocator> allocator,
                std::unique_ptr<tsl::Allocator> host_memory_allocator,
                std::unique_ptr<gpu::GpuExecutableRunOptions> gpu_run_options,
                std::shared_ptr<KeyValueStoreInterface> kv_store,
                std::shared_ptr<const GpuTopology> gpu_topology);

  ~TfrtGpuClient() override;

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
      PjRtGlobalDeviceId global_device_id) const override;

  absl::StatusOr<PjRtDevice*> LookupAddressableDevice(
      PjRtLocalDeviceId local_device_id) const override;

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

  xla::LocalClient* xla_client() const { return xla_client_; }

  se::DeviceMemoryAllocator* allocator() { return allocator_.get_mutable(); }

  bool should_stage_host_to_device_transfers() const {
    return should_stage_host_to_device_transfers_;
  }

  HostMemoryAllocator* host_memory_allocator() const {
    return host_memory_allocator_.get();
  }

  PjRtPlatformId platform_id() const override {
    // TODO(b/382117736): Add support for ROCM and SYCL.
    return tsl::Fingerprint64(xla::CudaName());
  }

  absl::string_view platform_name() const override { return platform_name_; }

  absl::string_view platform_version() const override;

  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type, absl::Span<const int64_t> dims) override;

  absl::StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis()
      const override;

  tsl::thread::ThreadPool* blocking_thread_pool() const {
    return blocking_thread_pool_.get();
  }

  tsl::thread::ThreadPool* non_blocking_thread_pool() const {
    return non_blocking_thread_pool_.get();
  }

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) override;
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      const XlaComputation& computation, CompileOptions options) override;
  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      mlir::ModuleOp mlir_module, CompileOptions options) override;
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      mlir::ModuleOp mlir_module, CompileOptions options) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateUninitializedBuffer(
      const Shape& shape, PjRtMemorySpace* memory_space) override;

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> DeserializeExecutable(
      absl::string_view serialized,
      std::optional<CompileOptions> options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
  LoadSerializedExecutable(absl::string_view serialized,
                           std::optional<CompileOptions> options,
                           const LoadOptions& load_options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Load(
      std::unique_ptr<PjRtExecutable> executable,
      const LoadOptions& load_options) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateErrorBuffer(
      absl::Status error, const Shape& shape, PjRtMemorySpace* memory) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateViewOfDeviceBuffer(
      void* device_ptr, const Shape& shape, PjRtMemorySpace* memory_space,
      std::function<void()> on_delete_callback,
      std::optional<std::intptr_t> stream) override;

  gpu::GpuExecutableRunOptions* gpu_run_options() const {
    return gpu_run_options_.get();
  }

  absl::StatusOr<const xla::PjRtTopologyDescription*> GetTopologyDescription()
      const override {
    return &topology_;
  }

  std::optional<std::shared_ptr<KeyValueStoreInterface>> key_value_store()
      const override {
    return kv_store_;
  }

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtMemorySpace* memory_space, const Layout* device_layout) override;

  using PjRtClient::BufferFromHostLiteral;
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtMemorySpace* memory_space,
      const Layout* device_layout) override;

  absl::StatusOr<std::unique_ptr<AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(
      absl::Span<const ShapeSpec> shape_specs,
      std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
      PjRtMemorySpace* memory_space) override;

  // Caller is responsible to ensure that `data` has allocated enough memory
  // for `buffer_size` to do DMA mapping.
  absl::Status DmaMap(void* data, size_t buffer_size) override;

  absl::Status DmaUnmap(void* data) override;

  bool IsDmaMapped(const void* data_start, int64_t transfer_size);

 private:
  friend class TfrtGpuBuffer;

  // Helper function for creating PjRtStreamExecutorExecutables. Modifies
  // `options` in-place.
  struct ExecutableExtras {
    std::shared_ptr<DeviceAssignment> device_assignment;
    std::vector<PjRtLoadedExecutable::LogicalDeviceIds>
        addressable_device_logical_ids;
    std::vector<PjRtDevice*> addressable_devices;
  };
  absl::StatusOr<ExecutableExtras> GetExecutableExtras(CompileOptions* options);

  // Updates `options` for compilation.
  absl::Status UpdateCompileOptions(CompileOptions* options);

  // Same as above, but also returns the executable extras.
  absl::StatusOr<ExecutableExtras> UpdateCompileOptionsAndGetExecutableExtras(
      CompileOptions* options);

  // Updates `options` for compilation, and gets the executable extras if
  // `returned_extras` is not null.
  absl::Status UpdateCompileOptionsInternal(CompileOptions* options,
                                            ExecutableExtras* returned_extras);

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> CompileInternal(
      const XlaComputation& computation,
      const std::vector<const Shape*>& argument_layout_pointers,
      LayoutCanonicalizationCallback layout_canonicalization_callback,
      CompileOptions options);

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> BuildPjRtExecutable(
      std::vector<std::unique_ptr<LocalExecutable>> local_executables,
      CompileOptions compile_options);

  absl::StatusOr<
      std::pair<std::vector<std::unique_ptr<LocalExecutable>>, CompileOptions>>
  DeserializeToLocalExecutable(absl::string_view serialized,
                               std::optional<CompileOptions> options);

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> LoadInternal(
      std::vector<std::unique_ptr<LocalExecutable>> local_executables,
      CompileOptions compile_options);

  int process_index_;

  // Platform name must be initialized before SetClient is called on devices.
  const std::string platform_name_;

  xla::LocalClient* xla_client_;

  bool should_stage_host_to_device_transfers_;

  // Device memory allocator. If owned, the allocator must outlive the devices,
  // because it is the device destructor that waits for any outstanding work to
  // complete.
  MaybeOwning<se::DeviceMemoryAllocator> allocator_;
  // Allocator to be used for staging memory transfers to devices.
  std::unique_ptr<HostMemoryAllocator> host_memory_allocator_;

  // Pointers to `owned_devices_`.
  std::vector<PjRtDevice*> devices_;
  // Maps Device::id() to the corresponding Device. Includes all devices.
  absl::flat_hash_map<PjRtGlobalDeviceId, TfrtGpuDevice*> id_to_device_;
  // Local devices indexed by local device ordinal.
  std::vector<PjRtDevice*> addressable_devices_;
  std::unique_ptr<ComputationPlacer> computation_placer_;

  // Addressable memory spaces.
  std::vector<std::unique_ptr<PjRtMemorySpace>> owned_memory_spaces_;
  // Pointers to `owned_memory_spaces_`.
  std::vector<PjRtMemorySpace*> memory_spaces_;

  const std::unique_ptr<gpu::GpuExecutableRunOptions> gpu_run_options_;

  // A cache for transpose plans. We use transposes to convert
  // (possibly strided) buffers provided to BufferFromHostBuffer into dense
  // major-to-minor layout.
  absl::Mutex transpose_mu_;
  TransposePlanCache transpose_cache_ ABSL_GUARDED_BY(transpose_mu_);

  StreamExecutorGpuTopologyDescription topology_;
  std::shared_ptr<KeyValueStoreInterface> kv_store_;

  absl::Mutex dma_maps_mutex_;
  // Maps dma mapped start pointers to their sizes.
  absl::btree_map<const void*, size_t, std::greater<const void*>> dma_maps_
      ABSL_GUARDED_BY(dma_maps_mutex_);

  // Includes all devices, including non-local devices on multi-host platforms.
  // Destructed after the thread pools, to ensure that all kernels in the
  // streams are finished.
  std::vector<std::unique_ptr<TfrtGpuDevice>> owned_devices_;

  // Thread pools must be destructed first, to make all the pending tasks are
  // completed before the client is destructed.
  std::unique_ptr<tsl::thread::ThreadPool> compile_thread_pool_;
  std::unique_ptr<tsl::thread::ThreadPool> blocking_thread_pool_;
  std::unique_ptr<tsl::thread::ThreadPool> non_blocking_thread_pool_;
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

  absl::StatusOr<Shape> logical_on_device_shape() override;

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
                             int64_t transfer_size) override {
    return CopyRawToHostFuture(PjRtFuture<void*>(dst), offset, transfer_size);
  }

  PjRtFuture<> CopyRawToHostFuture(PjRtFuture<void*> dst, int64_t offset,
                                   int64_t transfer_size) override;

  void Delete() override;

  bool IsDeleted() override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(
      PjRtMemorySpace* dst_memory_space) override;

  void CopyToRemoteDevice(PjRtFuture<std::string> serialized_descriptor,
                          RemoteSendCallback on_done) override {
    on_done(Unimplemented("CopyToRemoteDevice not implemented."),
            /*sends_were_enqueued=*/false);
  }

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> DonateWithControlDependency(
      PjRtFuture<> dependency) override;

  PjRtFuture<> GetReadyFuture() override;

  bool IsOnCpu() const override;

  const tsl::AsyncValueRef<MaybeOwningGpuMemory>& GetBufferPtr() const;

 private:
  // Acquires the device buffer for shared read-only usages, and it also adds
  // the `usage_event` to it. Any donation event in the future is expected to be
  // serialized after all the usage events added through this method. Returns
  // nullptr if the buffer is already donated or there is outstanding external
  // references.
  TrackedTfrtGpuDeviceBuffer* AcquireUsage(
      tsl::AsyncValueRef<GpuEvent> usage_event);

  // A helper class for managing a pending donation. It should be committed upon
  // success. Otherwise, the donated buffer is returned to the TfrtGpuBuffer.
  class DonationTransaction {
   public:
    explicit DonationTransaction(
        tsl::AsyncValueRef<bool> donation_event,
        std::unique_ptr<TrackedTfrtGpuDeviceBuffer> device_buffer)
        : donation_event_(donation_event),
          device_buffer_(std::move(device_buffer)) {
      VLOG(3) << "DonationTransaction::DonationTransaction";
    }
    DonationTransaction(const DonationTransaction&) = delete;
    DonationTransaction& operator=(const DonationTransaction&) = delete;
    DonationTransaction(DonationTransaction&&) = default;
    DonationTransaction& operator=(DonationTransaction&& other) {
      Abort();

      donation_event_ = other.donation_event_;
      device_buffer_ = std::move(other.device_buffer_);
      return *this;
    }

    ~DonationTransaction() { Abort(); }

    // Commit the donation. The rvalue ref qualifier is used to ensure the
    // semantic that it can be committed at most once.
    void Commit() && {
      donation_event_.emplace(true);
      device_buffer_->SetUnOwned();
      device_buffer_.reset();
    }

    TrackedTfrtGpuDeviceBuffer* device_buffer() const {
      return device_buffer_.get();
    }

   private:
    void Abort() {
      if (device_buffer_) {
        VLOG(0) << "DonationTransaction::Abort is going to "
                   "abort donation: "
                << device_buffer_.get();
        donation_event_.emplace(false);
        device_buffer_.reset();  // TODO(b/382117736): We should put this back
                                 // into the TfrtGpuBuffer instead.
      }
    }

    tsl::AsyncValueRef<bool> donation_event_;
    std::unique_ptr<TrackedTfrtGpuDeviceBuffer> device_buffer_;
  };

  // Acquires the device buffer for exclusive donation. The caller of this
  // method is expected to use the usage events and definition events to
  // serialize this donation with previous usages. After this method is called,
  // calls to AcquireUsage() will fail. Returns error status if the buffer is
  // already donated or there is outstanding external references.
  absl::StatusOr<DonationTransaction> AcquireDonation()
      ABSL_LOCKS_EXCLUDED(mu_);

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
  friend class DonationTransactionPeer;
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

  absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override;

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

  absl::Span<const std::shared_ptr<LocalExecutable>> executables() const {
    return executables_;
  }

  absl::StatusOr<std::string> SerializeExecutable() const override;

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

  absl::StatusOr<Result> ExecuteHelper(
      absl::Span<PjRtBuffer* const> argument_handles, int replica,
      int partition, const RunId& run_id, const ExecuteOptions& options,
      bool fill_future, TfrtGpuDevice* device = nullptr);

  // Create shared pointers so we can free them after the execution: with
  // asynchronous execution, the process being executed can outlive the
  // executable itself.
  TfrtGpuClient* const client_;
  // One executable per partition.
  std::vector<std::shared_ptr<LocalExecutable>> executables_;
  // On device shapes of the executable parameters.
  std::vector<std::shared_ptr<std::vector<Shape>>>
      on_device_executable_parameter_shapes_;

  // Size on device of each leaf buffer of the compiled program, cached here
  // for performance reasons.
  std::vector<std::shared_ptr<std::vector<int64_t>>>
      input_buffer_sizes_in_bytes_;

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
