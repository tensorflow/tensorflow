/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_NANORT_IFRT_CLIENT_H_
#define XLA_BACKENDS_CPU_NANORT_IFRT_CLIENT_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/backends/cpu/nanort/nanort_client.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/topology.h"
#include "xla/python/ifrt/tuple.h"
#include "xla/python/ifrt/value.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace Eigen {
class ThreadPoolInterface;
struct ThreadPoolDevice;
}  // namespace Eigen

namespace xla::cpu {

// Options for creating a NanoIfrtClient client.
struct NanoIfrtOptions {
  // If set, this thread pool will be used as an intra-op thread pool for
  // the underlying NanoRtClient. It is a user's responsibility to ensure that
  // the thread pool outlives all pending executions.
  Eigen::ThreadPoolInterface* intra_op_threadpool = nullptr;
};

// NanoIfrtClient is a thin wrapper around NanoRtClient that implements the
// ifrt::Client interface.
//
// Unlike NanoRtClient, this class will honor sharding annotations in XLA
// programs, mostly to satisfy IFRT callers. The sharding will be undone as soon
// as possible and reused (either when the sharded arrays is assembled or when
// it is first accessed by an executable). Even so, this client will have much
// better performance with unsharded inputs.
//
// Note: Array remapping is currently unimplemented.
//
// Note: We may add support for callers to access the underlying executables and
// buffers directly in the future, this would allow the "load path" that
// initializes programs and variables to be reused while still getting the
// performance wins of NanoRt at execution time.
class NanoIfrtClient : public llvm::RTTIExtends<NanoIfrtClient, ifrt::Client> {
 public:
  ~NanoIfrtClient() override;

  // Creates a client with a single device. Typically this is how this client
  // should be used.
  static std::shared_ptr<NanoIfrtClient> Create(
      const NanoIfrtOptions& options = {});

  // Creates a client with the given number of devices, this is provided for
  // testing and to allow the client to be used in applications that expect
  // programs to be sharded.
  static std::shared_ptr<NanoIfrtClient> CreateWithDevices(
      int32_t num_devices, const NanoIfrtOptions& options = {});

  // Returns a single device sharding. Generally callers should prefer to use
  // this when possible for optimal performance.
  ifrt::ShardingRef default_sharding() const;

  // Returns the underlying NanoRtClient.
  NanoRtClient* nano_client() { return &client_; }

  // Returns the optional intra-op device constructed from the Eigen thread
  // pool passed to the constructor via NanoIfrtOptions.
  const Eigen::ThreadPoolDevice* intra_op_device() const {
    return intra_op_device_.get();
  }

  using HostBufferSemantics = ifrt::Client::HostBufferSemantics;

  // Creates an array from a host buffer. The buffer will be used directly
  // without a copy if the copy semantics allow it and the layout is row major
  // and dense.
  absl::StatusOr<ifrt::ArrayRef> MakeArrayFromHostBuffer(
      const void* data, ifrt::DType dtype, ifrt::Shape shape,
      std::optional<absl::Span<const int64_t>> byte_strides,
      ifrt::ShardingRef sharding, ifrt::LayoutRef layout,
      HostBufferSemantics semantics,
      std::function<void()> on_done_with_host_buffer) override;
  // Expose the base class's `MakeArrayFromHostBuffer` overloads.
  using xla::ifrt::Client::MakeArrayFromHostBuffer;

  absl::StatusOr<std::vector<ifrt::ArrayRef>> MakeArraysFromHostBufferShards(
      absl::Span<MakeArraysFromHostBufferShardsSpec> specs,
      HostBufferSemantics semantics) override;

  absl::StatusOr<std::vector<ifrt::ArrayRef>> MakeErrorArrays(
      const absl::Status& error,
      absl::Span<const ifrt::ArraySpec> array_specs) override;

  // Assembles a sharded array from a list of single device arrays. If the
  // provided sharding is specific enough to assemble a dense array, this method
  // will actually return an assembled array that pretends it is sharded.
  //
  // Otherwise we will produce an assembled array on demand when it is first
  // accessed by an XLA program.
  absl::StatusOr<ifrt::ArrayRef> AssembleArrayFromSingleDeviceArrays(
      ifrt::DType dtype, ifrt::Shape shape, ifrt::ShardingRef sharding,
      absl::Span<ifrt::ArrayRef> arrays,
      ifrt::ArrayCopySemantics array_copy_semantics,
      ifrt::SingleDeviceShardSemantics single_device_shard_semantics) override;

  absl::StatusOr<std::vector<ifrt::ArrayRef>> CopyArrays(
      absl::Span<ifrt::ArrayRef> arrays,
      std::optional<ifrt::DeviceListRef> devices,
      std::optional<ifrt::MemoryKind> memory_kind,
      ifrt::ArrayCopySemantics semantics) override;

  absl::StatusOr<std::vector<ifrt::ArrayRef>> RemapArrays(
      const ifrt::RemapPlan& plan, absl::Span<ifrt::ArrayRef> arrays,
      ifrt::ArrayCopySemantics semantics) override;

  absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> ReshardArrays(
      absl::Span<xla::ifrt::ArrayRef> arrays,
      absl::Span<const xla::ifrt::ArraySpec> specs,
      xla::ifrt::ArrayCopySemantics semantics) override;

  tsl::Future<> GetReadyFuture(
      absl::Span<const ifrt::ValueRef> values) override;

  absl::StatusOr<tsl::RCReference<ifrt::Tuple>> MakeTuple(
      absl::Span<ifrt::ValueRef> values) override;

  void CancelExecution(
      xla::ifrt::LoadedExecutable::CancellationHandle cancellation_handle,
      absl::Status error) override {}

  absl::string_view runtime_type() const override;

  absl::string_view platform_name() const override;
  absl::string_view platform_version() const override;
  ifrt::PlatformId platform_id() const override;

  const ifrt::AttributeMap& Attributes() const override;

  int device_count() const override;
  int addressable_device_count() const override;
  absl::Span<ifrt::Device* const> devices() const override;
  absl::Span<ifrt::Device* const> addressable_devices() const override;
  int process_index() const override;

  absl::Span<ifrt::Device* const> GetAllDevices() const override;

  absl::StatusOr<ifrt::DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;
  absl::StatusOr<ifrt::Device*> LookupDevice(
      ifrt::DeviceId device_id) const override;
  absl::StatusOr<ifrt::Device*> LookupAddressableDevice(
      int local_hardware_id) const override;

  absl::StatusOr<ifrt::DeviceListRef> MakeDeviceList(
      absl::Span<ifrt::Device* const> devices) const override;

  ifrt::Compiler* GetDefaultCompiler() override;

  absl::StatusOr<std::shared_ptr<ifrt::Topology>> GetTopologyForDevices(
      const ifrt::DeviceListRef& devices) const override;

  absl::StatusOr<std::shared_ptr<const PjRtLayout>> GetDefaultPjRtLayout(
      ifrt::DType dtype, absl::Span<const int64_t> dims, ifrt::Device* device,
      ifrt::MemoryKind memory_kind) const override;
  absl::StatusOr<ifrt::CustomLayoutRef> GetDefaultLayout(
      ifrt::DType dtype, const ifrt::Shape& shape,
      const ifrt::ShardingRef& sharding) const override;

  absl::StatusOr<std::unique_ptr<xla::ifrt::DeviceAttributeSubscription>>
  SubscribeToAttributeChanges(
      absl::Span<xla::ifrt::Device* const> devices,
      std::optional<absl::Span<const std::string>> attribute_names,
      xla::ifrt::OnDeviceAttributeChangeCallback callback) override {
    return absl::UnimplementedError(
        "SubscribeToAttributeChanges is not implemented in NanoIfrtClient.");
  }

  static char ID;  // NOLINT

 private:
  NanoIfrtClient(int32_t num_devices, const NanoIfrtOptions& options);

  // The underlying NanoRtClient.
  NanoRtClient client_;

  // The compiler, memory, and device objects. See cc file for implementation
  // details.
  std::unique_ptr<ifrt::Compiler> compiler_;
  std::unique_ptr<ifrt::Memory> memory_;
  std::vector<std::unique_ptr<ifrt::Device>> owned_devices_;

  // Some of the ifrt::Client methods return a span of devices, so we need to
  // keep storage for them here.
  std::vector<ifrt::Device*> devices_;

  // Single-device device lists pre-constructed for all `devices_`. We cache it
  // in the client to avoid constructing them all the time on a hot path.
  std::vector<ifrt::DeviceListRef> single_device_lists_;

  // Optional intra-op device constructed from the Eigen thread pool passed to
  // the constructor via NanoIfrtOptions.
  std::unique_ptr<Eigen::ThreadPoolDevice> intra_op_device_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_NANORT_IFRT_CLIENT_H_
