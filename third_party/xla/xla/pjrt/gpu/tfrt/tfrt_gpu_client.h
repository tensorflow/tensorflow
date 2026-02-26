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
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
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
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/gpu/se_gpu_topology_description.h"
#include "xla/pjrt/gpu/tfrt/tfrt_gpu_buffer.h"
#include "xla/pjrt/gpu/tfrt/tfrt_gpu_device.h"
#include "xla/pjrt/gpu/tfrt/tracked_gpu_device_buffer.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/transpose.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

inline constexpr absl::string_view kPjRtClientName = "TfrtGpuClient";

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

class TfrtGpuClient final : public PjRtClient {
 public:
  TfrtGpuClient(std::string platform_name, int process_index,
                xla::LocalClient* xla_client,
                std::vector<std::unique_ptr<TfrtGpuDevice>> devices,
                bool should_stage_host_to_device_transfers,
                bool abort_collectives_on_failure,
                MaybeOwning<se::DeviceAddressAllocator> allocator,
                HostMemoryAllocator::Factory host_memory_allocator_factory,
                std::unique_ptr<gpu::GpuExecutableRunOptions> gpu_run_options,
                std::shared_ptr<KeyValueStoreInterface> kv_store,
                std::shared_ptr<const GpuTopology> gpu_topology);

  ~TfrtGpuClient() override;

  int process_index() const override { return process_index_; }

  int device_count() const override { return devices_.size(); }

  int addressable_device_count() const override {
    return addressable_devices_.size();
  }

  std::optional<PjRtPluginAttributes> plugin_attributes() const override;

  absl::Span<PjRtDevice* const> devices() const override { return devices_; }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return addressable_devices_;
  }

  absl::StatusOr<PjRtDevice*> LookupDevice(
      GlobalDeviceId global_device_id) const override;

  absl::StatusOr<PjRtDevice*> LookupAddressableDevice(
      LocalDeviceId local_device_id) const override;

  void UpdateGlobalProcessInfo(
      absl::Span<tensorflow::CoordinatedTaskStateInfo> infos) override;

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

  xla::LocalClient* xla_client() const { return xla_client_; }

  se::DeviceAddressAllocator* allocator() { return allocator_.get_mutable(); }

  bool ShouldStageHostToDeviceTransfers(const void* data, int64_t size) {
    // Disable staging buffers for large transfers because allocation and extra
    // memcpy overheads for multi-gigabyte buffers will likely offset the
    // benefit of using a staging buffer. The current threshold is arbitrarily
    // chosen and may need to be adjusted in the future.
    return should_stage_host_to_device_transfers_ &&
           size < (int64_t{1} << 30) && !IsDmaMapped(data, size);
  }

  HostMemoryAllocator* GetHostMemoryAllocator() const override {
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

  AsyncWorkRunner* blocking_thread_pool() const {
    return blocking_thread_pool_.get();
  }

  AsyncWorkRunner* non_blocking_thread_pool() const {
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

  absl::StatusOr<
      std::pair<std::unique_ptr<PjRtBuffer>, PjRtFulfillAliasBufferCallback>>
  CreateAliasBuffer(const Shape& shape, PjRtMemorySpace* memory_space) override;

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> DeserializeExecutable(
      absl::string_view serialized,
      std::optional<CompileOptions> options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
  LoadSerializedExecutable(absl::string_view serialized,
                           std::optional<CompileOptions> options,
                           const LoadOptions& load_options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Load(
      std::shared_ptr<PjRtExecutable> executable,
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
  absl::Status UpdateCompileOptions(CompileOptions* options,
                                    bool lookup_addressable_devices);

  // Same as above, but also returns the executable extras.
  absl::StatusOr<ExecutableExtras> UpdateCompileOptionsAndGetExecutableExtras(
      CompileOptions* options);

  // Updates `options` for compilation, and gets the executable extras if
  // `returned_extras` is not null. It skips addressable device lookup if
  // `lookup_addressable_devices` is false.
  absl::Status UpdateCompileOptionsInternal(CompileOptions* options,
                                            ExecutableExtras* returned_extras,
                                            bool lookup_addressable_devices);

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options,
      bool lookup_addressable_devices);
  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      mlir::ModuleOp mlir_module, CompileOptions options,
      bool lookup_addressable_devices);

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> CompileInternal(
      const XlaComputation& computation,
      const std::vector<const Shape*>& argument_layout_pointers,
      LayoutCanonicalizationCallback layout_canonicalization_callback,
      CompileOptions options, bool lookup_addressable_devices);

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> BuildPjRtExecutable(
      std::optional<HloModuleProto> unoptimized_hlo_module_proto,
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
  const bool abort_collectives_on_failure_ = false;

  // Device memory allocator. If owned, the allocator must outlive the devices,
  // because it is the device destructor that waits for any outstanding work to
  // complete.
  MaybeOwning<se::DeviceAddressAllocator> allocator_;
  // Allocator to be used for staging memory transfers to devices.
  std::shared_ptr<HostMemoryAllocator> host_memory_allocator_;

  // Pointers to `owned_devices_`.
  std::vector<PjRtDevice*> devices_;
  // Maps Device::id() to the corresponding Device. Includes all devices.
  absl::flat_hash_map<GlobalDeviceId, TfrtGpuDevice*> id_to_device_;
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
  std::unique_ptr<AsyncWorkRunner> blocking_thread_pool_;
  std::unique_ptr<AsyncWorkRunner> non_blocking_thread_pool_;
};

absl::StatusOr<std::unique_ptr<PjRtClient>> GetTfrtGpuClient(
    const GpuClientOptions& options);

}  // namespace xla

#endif  // XLA_PJRT_GPU_TFRT_TFRT_GPU_CLIENT_H_
