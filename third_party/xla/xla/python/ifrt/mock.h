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

#ifndef XLA_PYTHON_IFRT_MOCK_H_
#define XLA_PYTHON_IFRT_MOCK_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/test.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/executable_serdes.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/topology.h"
#include "xla/python/ifrt/tuple.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt/value.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

// array.h

class MockArray : public llvm::RTTIExtends<MockArray, Array> {
 public:
  MockArray() = default;
  explicit MockArray(tsl::RCReference<xla::ifrt::Array> delegated);

  // LINT.IfChange
  MOCK_METHOD(Client*, client, (), (const, final));
  MOCK_METHOD(Future<>, GetReadyFuture, (), (const, final));
  MOCK_METHOD(Future<>, Delete, (), (final));
  MOCK_METHOD(bool, IsDeleted, (), (const, final));

  MOCK_METHOD(DType, dtype, (), (const, final));
  MOCK_METHOD(const Shape&, shape, (), (const, final));
  MOCK_METHOD(const Sharding&, sharding, (), (const, final));
  MOCK_METHOD(absl::Nonnull<std::shared_ptr<const Sharding>>,
              shared_ptr_sharding, (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::shared_ptr<const PjRtLayout>>, layout, (),
              (const, final));
  MOCK_METHOD(absl::StatusOr<std::vector<tsl::RCReference<Array>>>,
              DisassembleIntoSingleDeviceArrays,
              (ArrayCopySemantics array_copy_semantics,
               SingleDeviceShardSemantics single_device_shard_semantics),
              (final));
  MOCK_METHOD(absl::StatusOr<tsl::RCReference<Array>>, FullyReplicatedShard,
              (ArrayCopySemantics semantics), (final));
  MOCK_METHOD(Future<>, CopyToHostBuffer,
              (void* data,
               std::optional<absl::Span<const int64_t>> byte_strides,
               ArrayCopySemantics semantics),
              (final));
  // LINT.ThenChange(mock.cc:MockArrayDelegation)

  tsl::RCReference<xla::ifrt::Array> delegated() const { return delegated_; }

  std::string DebugString() const final { return "MockArray"; }

  static char ID;  // NOLINT

 private:
  const tsl::RCReference<xla::ifrt::Array> delegated_;
};

// client.h

class MockClient : public llvm::RTTIExtends<MockClient, Client> {
 public:
  MockClient() = default;
  explicit MockClient(std::unique_ptr<xla::ifrt::Client> delegated);

  // LINT.IfChange
  MOCK_METHOD(absl::StatusOr<tsl::RCReference<Array>>, MakeArrayFromHostBuffer,
              (const void* data, DType dtype, Shape shape,
               std::optional<absl::Span<const int64_t>> byte_strides,
               absl::Nonnull<std::shared_ptr<const Sharding>> sharding,
               HostBufferSemantics semantics,
               std::function<void()> on_done_with_host_buffer,
               tsl::RCReference<UserContext> user_context),
              (final));
  MOCK_METHOD(absl::StatusOr<std::vector<tsl::RCReference<Array>>>,
              MakeArraysFromHostBufferShards,
              (absl::Span<MakeArraysFromHostBufferShardsSpec> specs,
               HostBufferSemantics semantics,
               tsl::RCReference<UserContext> user_context),
              (final));
  MOCK_METHOD(absl::StatusOr<std::vector<tsl::RCReference<Array>>>,
              MakeErrorArrays,
              (const absl::Status& error,
               absl::Span<const ArraySpec> array_specs,
               tsl::RCReference<UserContext> user_context),
              (final));
  MOCK_METHOD(absl::StatusOr<tsl::RCReference<Array>>,
              AssembleArrayFromSingleDeviceArrays,
              (DType dtype, Shape shape,
               absl::Nonnull<std::shared_ptr<const Sharding>> sharding,
               absl::Span<tsl::RCReference<Array>> arrays,
               ArrayCopySemantics array_copy_semantics,
               SingleDeviceShardSemantics single_device_shard_semantics),
              (final));
  MOCK_METHOD(absl::StatusOr<std::vector<tsl::RCReference<Array>>>, CopyArrays,
              (absl::Span<tsl::RCReference<Array>> arrays,
               std::optional<DeviceListRef> devices,
               std::optional<MemoryKind> memory_kind,
               ArrayCopySemantics semantics),
              (final));
  MOCK_METHOD(absl::StatusOr<std::vector<tsl::RCReference<Array>>>, RemapArrays,
              (const RemapPlan& plan,
               absl::Span<tsl::RCReference<Array>> arrays,
               ArrayCopySemantics semantics),
              (final));
  MOCK_METHOD(Future<>, GetReadyFuture,
              (absl::Span<const tsl::RCReference<Value>> values), (final));
  MOCK_METHOD(absl::StatusOr<tsl::RCReference<Tuple>>, MakeTuple,
              (absl::Span<tsl::RCReference<Value>> values), (final));
  MOCK_METHOD(absl::string_view, runtime_type, (), (const, final));
  MOCK_METHOD(absl::string_view, platform_name, (), (const, final));
  MOCK_METHOD(absl::string_view, platform_version, (), (const, final));
  MOCK_METHOD((const AttributeMap&), Attributes, (), (const, final));
  MOCK_METHOD(int, device_count, (), (const, final));
  MOCK_METHOD(PlatformId, platform_id, (), (const, final));
  MOCK_METHOD(int, addressable_device_count, (), (const, final));
  MOCK_METHOD(absl::Span<Device* const>, devices, (), (const, final));
  MOCK_METHOD(absl::Span<Device* const>, addressable_devices, (),
              (const, final));
  MOCK_METHOD(int, process_index, (), (const, final));
  MOCK_METHOD(absl::Span<xla::ifrt::Device* const>, GetAllDevices, (),
              (const, final));
  MOCK_METHOD(absl::StatusOr<DeviceAssignment>, GetDefaultDeviceAssignment,
              (int num_replicas, int num_partitions), (const, final));
  MOCK_METHOD(absl::StatusOr<Device*>, LookupDevice, (DeviceId device_id),
              (const, final));
  MOCK_METHOD(absl::StatusOr<Device*>, LookupAddressableDevice,
              (int local_hardware_id), (const, final));
  MOCK_METHOD(DeviceListRef, MakeDeviceList,
              (absl::Span<Device* const> devices), (const));
  MOCK_METHOD(Compiler*, GetDefaultCompiler, (), (final));
  MOCK_METHOD(absl::StatusOr<std::shared_ptr<Topology>>, GetTopologyForDevices,
              (const xla::ifrt::DeviceListRef& devices), (const, final));
  MOCK_METHOD(absl::StatusOr<std::shared_ptr<const PjRtLayout>>,
              GetDefaultLayout,
              (xla::ifrt::DType dtype, absl::Span<const int64_t> dims,
               xla::ifrt::Device* device, xla::ifrt::MemoryKind memory_kind),
              (const, final));
  MOCK_METHOD(tsl::RCReference<xla::ifrt::UserContext>, CreateUserContext, (),
              (final));
  // LINT.ThenChange(mock.cc:MockClientDelegation)

  xla::ifrt::Client* delegated() const { return delegated_.get(); }

  static char ID;  // NOLINT

 private:
  const std::unique_ptr<xla::ifrt::Client> delegated_;
};

// compiler.h

class MockCompiler : public llvm::RTTIExtends<MockCompiler, Compiler> {
 public:
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<LoadedExecutable>>, Compile,
              (std::unique_ptr<Program> program,
               std::unique_ptr<CompileOptions> options),
              (final));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Executable>>, Compile,
              (std::unique_ptr<Program> program, const Topology& topology,
               std::unique_ptr<CompileOptions> options),
              (final));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<LoadedExecutable>>,
              DeserializeLoadedExecutable,
              (absl::string_view serialized,
               std::unique_ptr<DeserializeExecutableOptions> options),
              (final));

  static char ID;  // NOLINT
};

// device.h

class MockDevice : public Device {
 public:
  MockDevice() = default;
  explicit MockDevice(Device* delegated);

  // LINT.IfChange
  MOCK_METHOD(Client*, client, (), (const, final));
  MOCK_METHOD(bool, IsAddressable, (), (const, final));
  MOCK_METHOD(int, ProcessIndex, (), (const, final));
  MOCK_METHOD(DeviceId, Id, (), (const, final));
  MOCK_METHOD(absl::string_view, Kind, (), (const, final));
  MOCK_METHOD((const AttributeMap&), Attributes, (), (const, final));
  MOCK_METHOD(absl::StatusOr<Memory*>, DefaultMemory, (), (const, final));
  MOCK_METHOD(absl::Span<Memory* const>, Memories, (), (const, final));
  // LINT.ThenChange(mock.cc:MockDeviceDelegation)

  Device* delegated() const { return delegated_; }

  absl::string_view DebugString() const final { return "MockDevice"; }
  absl::string_view ToString() const final { return "MockDevice"; }

 private:
  Device* const delegated_ = nullptr;
};

// memory.h

class MockMemory : public Memory {
 public:
  MOCK_METHOD(MemoryId, Id, (), (const, final));
  MOCK_METHOD(absl::Span<Device* const>, Devices, (), (const, final));
  MOCK_METHOD(const MemoryKind&, Kind, (), (const, final));
  MOCK_METHOD(absl::string_view, ToString, (), (const, final));

  absl::string_view DebugString() const final { return "MockMemory"; }
};

// executable.h

class MockExecutable : public llvm::RTTIExtends<MockExecutable, Executable> {
 public:
  MOCK_METHOD(absl::string_view, name, (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::optional<std::string>>, Fingerprint, (),
              (const, final));
  MOCK_METHOD(absl::StatusOr<std::string>, Serialize, (), (const, final));
  MOCK_METHOD(int, num_devices, (), (const, final));
  MOCK_METHOD(int64_t, SizeOfGeneratedCodeInBytes, (), (const, final));
  MOCK_METHOD(absl::StatusOr<CompiledMemoryStats>, GetCompiledMemoryStats, (),
              (const, final));
  MOCK_METHOD(std::optional<std::vector<OpSharding>>, GetParameterShardings, (),
              (const, final));
  MOCK_METHOD(std::optional<std::vector<OpSharding>>, GetOutputShardings, (),
              (const, final));
  MOCK_METHOD(absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>,
              GetParameterLayouts, (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>,
              GetOutputLayouts, (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::vector<std::shared_ptr<HloModule>>>,
              GetHloModules, (), (const, final));
  MOCK_METHOD(absl::StatusOr<xla::ifrt::AttributeMap>, GetCostAnalysis, (),
              (const, final));

  static char ID;  // NOLINT
};

class MockLoadedExecutable
    : public llvm::RTTIExtends<MockLoadedExecutable, LoadedExecutable> {
 public:
  MOCK_METHOD(Client*, client, (), (const, final));
  MOCK_METHOD(absl::string_view, name, (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::optional<std::string>>, Fingerprint, (),
              (const, final));
  MOCK_METHOD(absl::StatusOr<std::string>, Serialize, (), (const, final));
  MOCK_METHOD(Future<>, GetReadyFuture, (), (const, override));
  MOCK_METHOD(int, num_devices, (), (const, final));
  MOCK_METHOD(int64_t, SizeOfGeneratedCodeInBytes, (), (const, final));
  MOCK_METHOD(absl::StatusOr<CompiledMemoryStats>, GetCompiledMemoryStats, (),
              (const, final));
  MOCK_METHOD(std::optional<std::vector<OpSharding>>, GetParameterShardings, (),
              (const, final));
  MOCK_METHOD(std::optional<std::vector<OpSharding>>, GetOutputShardings, (),
              (const, final));
  MOCK_METHOD(absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>,
              GetParameterLayouts, (), (const, final));
  MOCK_METHOD(absl::StatusOr<absl::Span<const int>>, GetDonatableInputIndices,
              (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>,
              GetOutputLayouts, (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::vector<std::vector<absl::string_view>>>,
              GetOutputMemoryKinds, (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::vector<std::shared_ptr<HloModule>>>,
              GetHloModules, (), (const, final));
  MOCK_METHOD(absl::StatusOr<xla::ifrt::AttributeMap>, GetCostAnalysis, (),
              (const, final));
  MOCK_METHOD(absl::StatusOr<ExecuteResult>, Execute,
              (absl::Span<tsl::RCReference<Array>> args,
               const ExecuteOptions& options,
               std::optional<DeviceListRef> devices),
              (final));
  MOCK_METHOD(Future<>, Delete, (), (final));
  MOCK_METHOD(bool, IsDeleted, (), (const, final));
  MOCK_METHOD(absl::Span<Device* const>, addressable_devices, (),
              (const, final));

  static char ID;  // NOLINT
};

// host_callback.h

class MockHostCallback final
    : public llvm::RTTIExtends<MockHostCallback, HostCallback> {
 public:
  MOCK_METHOD(std::string, Serialize, (), (const, final));

  static char ID;  // NOLINT
};

class MockLoadedHostCallback final
    : public llvm::RTTIExtends<MockLoadedHostCallback, LoadedHostCallback> {
 public:
  MOCK_METHOD(Client*, client, (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::string>, Serialize, (), (const, final));

  static char ID;  // NOLINT
};

// sharding.h

class MockSharding : public llvm::RTTIExtends<MockSharding, Sharding> {
 public:
  MockSharding()
      : llvm::RTTIExtends<MockSharding, Sharding>(
            BasicDeviceList::Create({}), MemoryKind(),
            /*is_fully_replicated=*/false) {}

  MockSharding(DeviceListRef devices, MemoryKind memory_kind,
               bool is_fully_replicated)
      : llvm::RTTIExtends<MockSharding, Sharding>(devices, memory_kind,
                                                  is_fully_replicated) {}

  MOCK_METHOD(
      (absl::StatusOr<std::vector<
           std::pair<Shape, absl::Nonnull<std::shared_ptr<const Sharding>>>>>),
      Disassemble, (const Shape& shape), (const, final));
  MOCK_METHOD(
      (absl::StatusOr<std::vector<
           std::pair<Shape, absl::Nonnull<std::shared_ptr<const Sharding>>>>>),
      Disassemble,
      (const Shape& shape,
       SingleDeviceShardSemantics single_device_shard_semantics),
      (const, final));
  MOCK_METHOD(
      (absl::StatusOr<std::vector<std::pair<
           DynamicShape, absl::Nonnull<std::shared_ptr<const Sharding>>>>>),
      Disassemble, (const DynamicShape& dynamic_shape), (const final));
  MOCK_METHOD(
      (absl::StatusOr<std::vector<std::pair<
           DynamicShape, absl::Nonnull<std::shared_ptr<const Sharding>>>>>),
      Disassemble,
      (const DynamicShape& dynamic_shape,
       SingleDeviceShardSemantics single_device_shard_semantics),
      (const final));
  MOCK_METHOD(absl::StatusOr<std::vector<IndexDomain>>, IndexDomains,
              (const Shape& shape), (const, final));
  MOCK_METHOD(absl::StatusOr<std::vector<IndexDomain>>, IndexDomains,
              (const Shape& shape,
               SingleDeviceShardSemantics single_device_shard_semantics),
              (const, final));
  MOCK_METHOD(absl::StatusOr<Shape>, GetShardShape, (const Shape& shape),
              (const, final));
  MOCK_METHOD(bool, HasSamePartitioning, (const Sharding& other),
              (const final));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Sharding>>, WithDeviceAssignment,
              (std::optional<DeviceListRef> devices,
               std::optional<MemoryKind> memory_kind),
              (const final));
  MOCK_METHOD(void, Hash, (absl::HashState), (const final));

  std::string DebugString() const final { return "MockSharding"; }

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_MOCK_H_
