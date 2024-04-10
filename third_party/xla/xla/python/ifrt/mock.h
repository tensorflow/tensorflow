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

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/test.h"
#include "tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

// array.h

class MockArray : public llvm::RTTIExtends<MockArray, Array> {
 public:
  MockArray() = default;
  explicit MockArray(tsl::RCReference<xla::ifrt::Array> delegated);

  // LINT.IfChange
  MOCK_METHOD(Client*, client, (), (const, final));
  MOCK_METHOD(Future<Status>, GetReadyFuture, (), (const, final));
  MOCK_METHOD(Future<Status>, Delete, (), (final));
  MOCK_METHOD(bool, IsDeleted, (), (const, final));
  MOCK_METHOD(std::string, DebugString, (), (const, final));

  MOCK_METHOD(DType, dtype, (), (const, final));
  MOCK_METHOD(const Shape&, shape, (), (const, final));
  MOCK_METHOD(const Sharding&, sharding, (), (const, final));
  MOCK_METHOD(std::shared_ptr<const Sharding>, shared_ptr_sharding, (),
              (const, final));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<PjRtLayout>>, layout, (),
              (const, final));
  MOCK_METHOD(absl::StatusOr<std::vector<tsl::RCReference<Array>>>,
              DisassembleIntoSingleDeviceArrays, (ArrayCopySemantics semantics),
              (final));
  MOCK_METHOD(absl::StatusOr<tsl::RCReference<Array>>, FullyReplicatedShard,
              (ArrayCopySemantics semantics), (final));
  MOCK_METHOD(Future<Status>, CopyToHostBuffer,
              (void* data,
               std::optional<absl::Span<const int64_t>> byte_strides,
               ArrayCopySemantics semantics),
              (final));
  MOCK_METHOD(absl::StatusOr<tsl::RCReference<Array>>, Reshard,
              (std::shared_ptr<const Sharding> new_sharding,
               ArrayCopySemantics semantics),
              (final));
  // LINT.ThenChange(mock.cc:MockArrayDelegation)

  tsl::RCReference<xla::ifrt::Array> delegated() const { return delegated_; }

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
               std::shared_ptr<const Sharding> sharding,
               HostBufferSemantics semantics,
               std::function<void()> on_done_with_host_buffer),
              (final));
  MOCK_METHOD(absl::StatusOr<tsl::RCReference<Array>>,
              AssembleArrayFromSingleDeviceArrays,
              (Shape shape, std::shared_ptr<const Sharding> sharding,
               absl::Span<tsl::RCReference<Array>> arrays,
               ArrayCopySemantics semantics),
              (final));
  MOCK_METHOD(absl::StatusOr<tsl::RCReference<Tuple>>, MakeTuple,
              (absl::Span<tsl::RCReference<Value>> values), (final));
  MOCK_METHOD(absl::string_view, runtime_type, (), (const, final));
  MOCK_METHOD(absl::string_view, platform_name, (), (const, final));
  MOCK_METHOD(absl::string_view, platform_version, (), (const, final));
  MOCK_METHOD((absl::flat_hash_map<std::string, Client::ClientAttribute>),
              attributes, (), (const, final));
  MOCK_METHOD(int, device_count, (), (const, final));
  MOCK_METHOD(PlatformId, platform_id, (), (const, final));
  MOCK_METHOD(int, addressable_device_count, (), (const, final));
  MOCK_METHOD(absl::Span<Device* const>, devices, (), (const, final));
  MOCK_METHOD(absl::Span<Device* const>, addressable_devices, (),
              (const, final));
  MOCK_METHOD(int, process_index, (), (const, final));
  MOCK_METHOD(absl::StatusOr<DeviceAssignment>, GetDefaultDeviceAssignment,
              (int num_replicas, int num_partitions), (const, final));
  MOCK_METHOD(absl::StatusOr<Device*>, LookupDevice, (int device_id),
              (const, final));
  MOCK_METHOD(absl::StatusOr<Device*>, LookupAddressableDevice,
              (int local_hardware_id), (const, final));
  MOCK_METHOD(Compiler*, GetDefaultCompiler, (), (final));
  MOCK_METHOD(
      absl::StatusOr<std::shared_ptr<const xla::PjRtTopologyDescription>>,
      GetTopologyForDevices, (const xla::ifrt::DeviceList& devices),
      (const, final));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<xla::PjRtLayout>>,
              GetDefaultLayoutForDevice,
              (xla::ifrt::DType dtype, absl::Span<const int64_t> dims,
               xla::ifrt::Device* device),
              (const, final));
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
  MOCK_METHOD(xla::PjRtClient*, client, (), (const, final));
  MOCK_METHOD(bool, IsAddressable, (), (const, final));
  MOCK_METHOD(const xla::PjRtDeviceDescription&, description, (),
              (const, final));
  MOCK_METHOD(xla::PjRtGlobalDeviceId, global_device_id, (), (const, final));
  MOCK_METHOD(int, process_index, (), (const, final));
  MOCK_METHOD(int, local_hardware_id, (), (const, final));
  MOCK_METHOD(xla::PjRtLocalDeviceId, local_device_id, (), (const, final));
  MOCK_METHOD(xla::PjRtLocalHardwareId, local_hardware_id_typed, (),
              (const, final));
  MOCK_METHOD(absl::string_view, device_kind, (), (const, final));
  MOCK_METHOD(absl::string_view, DebugString, (), (const, final));
  MOCK_METHOD(absl::string_view, ToString, (), (const, final));
  MOCK_METHOD(
      (const absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute>&),
      Attributes, (), (const, final));
  MOCK_METHOD(std::unique_ptr<ScopedAsyncTrackingEvent>,
              CreateAsyncTrackingEvent, (absl::string_view description),
              (const, final));
  MOCK_METHOD(Status, TransferToInfeed, (const LiteralSlice& literal), (final));
  MOCK_METHOD(Status, TransferFromOutfeed, (MutableBorrowingLiteral literal),
              (final));
  MOCK_METHOD(absl::StatusOr<xla::PjRtMemorySpace*>, default_memory_space, (),
              (const, final));
  MOCK_METHOD(absl::StatusOr<tsl::AllocatorStats>, GetAllocatorStats, (),
              (const, final));
  MOCK_METHOD(absl::Span<xla::PjRtMemorySpace* const>, memory_spaces, (),
              (const, final));
  // LINT.ThenChange(mock.cc:MockDeviceDelegation)

  Device* delegated() const { return delegated_; }

 private:
  Device* const delegated_ = nullptr;
};

// memory.h

class MockMemory : public Memory {
 public:
  MOCK_METHOD(xla::PjRtClient*, client, (), (const, final));
  MOCK_METHOD(absl::Span<Device* const>, devices, (), (const, final));
  MOCK_METHOD(int, id, (), (const, final));
  MOCK_METHOD(absl::string_view, kind, (), (const, final));
  MOCK_METHOD(int, kind_id, (), (const, final));
  MOCK_METHOD(absl::string_view, DebugString, (), (const, final));
  MOCK_METHOD(absl::string_view, ToString, (), (const, final));
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
  MOCK_METHOD(absl::StatusOr<std::vector<std::unique_ptr<Layout>>>,
              GetParameterLayouts, (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::vector<std::unique_ptr<Layout>>>,
              GetOutputLayouts, (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::vector<std::shared_ptr<HloModule>>>,
              GetHloModules, (), (const, final));
  MOCK_METHOD(
      (absl::StatusOr<absl::flat_hash_map<std::string, CostAnalysisValue>>),
      GetCostAnalysis, (), (const, final));

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
  MOCK_METHOD(Future<absl::Status>, GetReadyFuture, (), (const, override));
  MOCK_METHOD(int, num_devices, (), (const, final));
  MOCK_METHOD(int64_t, SizeOfGeneratedCodeInBytes, (), (const, final));
  MOCK_METHOD(absl::StatusOr<CompiledMemoryStats>, GetCompiledMemoryStats, (),
              (const, final));
  MOCK_METHOD(std::optional<std::vector<OpSharding>>, GetParameterShardings, (),
              (const, final));
  MOCK_METHOD(std::optional<std::vector<OpSharding>>, GetOutputShardings, (),
              (const, final));
  MOCK_METHOD(absl::StatusOr<std::vector<std::unique_ptr<Layout>>>,
              GetParameterLayouts, (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::vector<std::unique_ptr<Layout>>>,
              GetOutputLayouts, (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::vector<std::vector<absl::string_view>>>,
              GetOutputMemoryKinds, (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::vector<std::shared_ptr<HloModule>>>,
              GetHloModules, (), (const, final));
  MOCK_METHOD(
      (absl::StatusOr<
          absl::flat_hash_map<std::string, Executable::CostAnalysisValue>>),
      GetCostAnalysis, (), (const, final));
  MOCK_METHOD(absl::StatusOr<ExecuteResult>, Execute,
              (absl::Span<tsl::RCReference<Array>> args,
               const ExecuteOptions& options,
               std::optional<DeviceList> devices),
              (final));
  MOCK_METHOD(Future<Status>, Delete, (), (final));
  MOCK_METHOD(bool, IsDeleted, (), (const, final));
  MOCK_METHOD(absl::Span<const LogicalDeviceIds>,
              addressable_device_logical_ids, (), (const, final));
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
  MOCK_METHOD(
      (absl::StatusOr<
          std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>),
      Disassemble, (const Shape& shape), (const, final));
  MOCK_METHOD(absl::StatusOr<std::vector<IndexDomain>>, IndexDomains,
              (const Shape& shape), (const, final));
  MOCK_METHOD(std::string, DebugString, (), (const, final));

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_MOCK_H_
