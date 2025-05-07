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
#include <vector>

#include "absl/base/nullability.h"
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
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/topology.h"
#include "xla/python/ifrt/tuple.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt/value.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla::cpu {

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
  static std::shared_ptr<NanoIfrtClient> Create();

  // Creates a client with the given number of devices, this is provided for
  // testing and to allow the client to be used in applications that expect
  // programs to be sharded.
  static std::shared_ptr<NanoIfrtClient> CreateWithDevices(int32_t num_devices);

  // Returns a single device sharding. Generally callers should prefer to use
  // this when possible for optimal performance.
  ifrt::ShardingRef default_sharding() const;

  // Returns the underlying NanoRtClient.
  NanoRtClient* nano_client() { return &client_; }

  using HostBufferSemantics = xla::ifrt::Client::HostBufferSemantics;

  // Creates an array from a host buffer. The buffer will be used directly
  // without a copy if the copy semantics allow it and the layout is row major
  // and dense.
  absl::StatusOr<ifrt::ArrayRef> MakeArrayFromHostBuffer(
      const void* data, ifrt::DType dtype, ifrt::Shape shape,
      std::optional<absl::Span<const int64_t>> byte_strides,
      ifrt::ShardingRef sharding, HostBufferSemantics semantics,
      std::function<void()> on_done_with_host_buffer,
      tsl::RCReference<xla::ifrt::UserContext> user_context) override;

  absl::StatusOr<std::vector<ifrt::ArrayRef>> MakeArraysFromHostBufferShards(
      absl::Span<MakeArraysFromHostBufferShardsSpec> specs,
      HostBufferSemantics semantics,
      tsl::RCReference<xla::ifrt::UserContext> user_context) override;

  absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> MakeErrorArrays(
      const absl::Status& error,
      absl::Span<const xla::ifrt::ArraySpec> array_specs,
      tsl::RCReference<xla::ifrt::UserContext> user_context) override;

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

  absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> RemapArrays(
      const ifrt::RemapPlan& plan, absl::Span<xla::ifrt::ArrayRef> arrays,
      ifrt::ArrayCopySemantics semantics) override;

  ifrt::Future<> GetReadyFuture(
      absl::Span<const tsl::RCReference<ifrt::Value>> values) override;

  absl::StatusOr<tsl::RCReference<ifrt::Tuple>> MakeTuple(
      absl::Span<tsl::RCReference<ifrt::Value>> values) override;

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

  absl::Span<xla::ifrt::Device* const> GetAllDevices() const override;

  absl::StatusOr<ifrt::DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;
  absl::StatusOr<ifrt::Device*> LookupDevice(
      ifrt::DeviceId device_id) const override;
  absl::StatusOr<ifrt::Device*> LookupAddressableDevice(
      int local_hardware_id) const override;

  ifrt::DeviceListRef MakeDeviceList(
      absl::Span<ifrt::Device* const> devices) const override;

  ifrt::Compiler* GetDefaultCompiler() override;

  absl::StatusOr<std::shared_ptr<ifrt::Topology>> GetTopologyForDevices(
      const ifrt::DeviceListRef& devices) const override;

  absl::StatusOr<std::shared_ptr<const PjRtLayout>> GetDefaultLayout(
      ifrt::DType dtype, absl::Span<const int64_t> dims, ifrt::Device* device,
      xla::ifrt::MemoryKind memory_kind) const override;

  tsl::RCReference<xla::ifrt::UserContext> CreateUserContext() override {
    return tsl::RCReference<xla::ifrt::UserContext>();
  }

  static char ID;  // NOLINT

 private:
  explicit NanoIfrtClient(int32_t num_devices);

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
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_NANORT_IFRT_CLIENT_H_
