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

#include "xla/python/ifrt/mock.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/value.h"

namespace xla {
namespace ifrt {

char MockArray::ID = 0;
char MockClient::ID = 0;
char MockCompiler::ID = 0;
char MockExecutable::ID = 0;
char MockLoadedExecutable::ID = 0;
char MockMpmdLoadedExecutable::ID = 0;
char MockHostCallback::ID = 0;
char MockLoadedHostCallback::ID = 0;
char MockSharding::ID = 0;

namespace {
using ::testing::_;
}

// LINT.IfChange(MockArrayDelegation)
MockArray::MockArray(xla::ifrt::ArrayRef delegated)
    : delegated_(std::move(delegated)) {
  ON_CALL(*this, user_context).WillByDefault([this]() {
    return delegated_->user_context();
  });
  ON_CALL(*this, GetReadyFuture).WillByDefault([this]() {
    return delegated_->GetReadyFuture();
  });
  ON_CALL(*this, Delete).WillByDefault([this]() {
    return delegated_->Delete();
  });
  ON_CALL(*this, IsDeleted).WillByDefault([this]() {
    return delegated_->IsDeleted();
  });
  ON_CALL(*this, dtype).WillByDefault([this]() { return delegated_->dtype(); });
  ON_CALL(*this, shape).WillByDefault([this]() -> const Shape& {
    return delegated_->shape();
  });
  ON_CALL(*this, sharding).WillByDefault([this]() -> const Sharding& {
    return delegated_->sharding();
  });
  ON_CALL(*this, shared_ptr_sharding).WillByDefault([this]() {
    return delegated_->shared_ptr_sharding();
  });
  ON_CALL(*this, pjrt_layout)
      .WillByDefault(
          [this]() -> absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> {
            return delegated_->pjrt_layout();
          });
  ON_CALL(*this, layout).WillByDefault([this]() -> CustomLayoutRef {
    return delegated_->layout();
  });
  ON_CALL(*this, DisassembleIntoSingleDeviceArrays(_, _))
      .WillByDefault(
          [this](ArrayCopySemantics array_copy_semantics,
                 SingleDeviceShardSemantics single_device_shard_semantics) {
            return delegated_->DisassembleIntoSingleDeviceArrays(
                array_copy_semantics, single_device_shard_semantics);
          });
  ON_CALL(*this, FullyReplicatedShard)
      .WillByDefault([this](ArrayCopySemantics semantics) {
        return delegated_->FullyReplicatedShard(semantics);
      });
  ON_CALL(*this, CopyToHostBuffer)
      .WillByDefault(
          [this](void* data,
                 std::optional<absl::Span<const int64_t>> byte_strides,
                 ArrayCopySemantics semantics) {
            return delegated_->CopyToHostBuffer(data, byte_strides, semantics);
          });
}
// LINT.ThenChange()

// LINT.IfChange(MockClientDelegation)
MockClient::MockClient(std::unique_ptr<xla::ifrt::Client> delegated)
    : delegated_(std::move(delegated)) {
  ON_CALL(*this, MakeArrayFromHostBuffer)
      .WillByDefault([this](
                         const void* data, DType dtype, Shape shape,
                         std::optional<absl::Span<const int64_t>> byte_strides,
                         ShardingRef sharding, LayoutRef layout,
                         HostBufferSemantics semantics,
                         std::function<void()> on_done_with_host_buffer) {
        return delegated_->MakeArrayFromHostBuffer(
            data, dtype, std::move(shape), byte_strides, std::move(sharding),
            std::move(layout), semantics, std::move(on_done_with_host_buffer));
      });
  ON_CALL(*this, MakeArraysFromHostBufferShards)
      .WillByDefault(
          [this](absl::Span<MakeArraysFromHostBufferShardsSpec> specs,
                 HostBufferSemantics semantics) {
            return delegated_->MakeArraysFromHostBufferShards(specs, semantics);
          });
  ON_CALL(*this, MakeErrorArrays)
      .WillByDefault([this](const absl::Status& error,
                            absl::Span<const ArraySpec> array_specs) {
        return delegated_->MakeErrorArrays(error, array_specs);
      });
  ON_CALL(*this, AssembleArrayFromSingleDeviceArrays(_, _, _, _, _, _))
      .WillByDefault(
          [this](DType dtype, Shape shape, ShardingRef sharding,
                 absl::Span<ArrayRef> arrays,
                 ArrayCopySemantics array_copy_semantics,
                 SingleDeviceShardSemantics single_device_shard_semantics) {
            return delegated_->AssembleArrayFromSingleDeviceArrays(
                std::move(dtype), std::move(shape), std::move(sharding), arrays,
                array_copy_semantics, single_device_shard_semantics);
          });
  ON_CALL(*this, CopyArrays)
      .WillByDefault([this](absl::Span<ArrayRef> arrays,
                            std::optional<DeviceListRef> devices,
                            std::optional<MemoryKind> memory_kind,
                            ArrayCopySemantics semantics) {
        return delegated_->CopyArrays(arrays, std::move(devices), memory_kind,
                                      semantics);
      });
  ON_CALL(*this, RemapArrays)
      .WillByDefault([this](const RemapPlan& plan, absl::Span<ArrayRef> arrays,
                            ArrayCopySemantics semantics) {
        return delegated_->RemapArrays(plan, arrays, semantics);
      });
  ON_CALL(*this, ReshardArrays)
      .WillByDefault([this](absl::Span<ArrayRef> arrays,
                            absl::Span<const ArraySpec> specs,
                            ArrayCopySemantics semantics) {
        return delegated_->ReshardArrays(arrays, specs, semantics);
      });
  ON_CALL(*this, GetReadyFuture)
      .WillByDefault([this](absl::Span<const ValueRef> values) {
        return delegated_->GetReadyFuture(values);
      });
  ON_CALL(*this, MakeTuple).WillByDefault([this](absl::Span<ValueRef> values) {
    return delegated_->MakeTuple(values);
  });
  ON_CALL(*this, CancelExecution)
      .WillByDefault([this](xla::ifrt::LoadedExecutable::CancellationHandle
                                cancellation_handle,
                            absl::Status error) {
        delegated_->CancelExecution(cancellation_handle, std::move(error));
      });

  ON_CALL(*this, runtime_type).WillByDefault([this]() {
    return delegated_->runtime_type();
  });
  ON_CALL(*this, platform_name).WillByDefault([this]() {
    return delegated_->platform_name();
  });
  ON_CALL(*this, platform_version).WillByDefault([this]() {
    return delegated_->platform_version();
  });
  ON_CALL(*this, platform_id).WillByDefault([this]() {
    return delegated_->platform_id();
  });
  ON_CALL(*this, Attributes).WillByDefault([this]() {
    return delegated_->Attributes();
  });
  ON_CALL(*this, device_count).WillByDefault([this]() {
    return delegated_->device_count();
  });
  ON_CALL(*this, addressable_device_count).WillByDefault([this]() {
    return delegated_->addressable_device_count();
  });
  ON_CALL(*this, devices).WillByDefault([this]() {
    return delegated_->devices();
  });
  ON_CALL(*this, addressable_devices).WillByDefault([this]() {
    return delegated_->addressable_devices();
  });
  ON_CALL(*this, process_index).WillByDefault([this]() {
    return delegated_->process_index();
  });
  ON_CALL(*this, GetAllDevices).WillByDefault([this]() {
    return delegated_->GetAllDevices();
  });
  ON_CALL(*this, GetDefaultDeviceAssignment)
      .WillByDefault([this](int num_replicas, int num_partitions) {
        return delegated_->GetDefaultDeviceAssignment(num_replicas,
                                                      num_partitions);
      });
  ON_CALL(*this, LookupDevice).WillByDefault([this](DeviceId device_id) {
    return delegated_->LookupDevice(device_id);
  });
  ON_CALL(*this, LookupAddressableDevice)
      .WillByDefault([this](int local_hardware_id) {
        return delegated_->LookupAddressableDevice(local_hardware_id);
      });
  ON_CALL(*this, MakeDeviceList)
      .WillByDefault([this](absl::Span<xla::ifrt::Device* const> devices) {
        return delegated_->MakeDeviceList(devices);
      });
  ON_CALL(*this, GetDefaultCompiler).WillByDefault([this]() {
    return delegated_->GetDefaultCompiler();
  });
  ON_CALL(*this, GetTopologyForDevices)
      .WillByDefault([this](const DeviceListRef& devices) {
        return delegated_->GetTopologyForDevices(devices);
      });
  ON_CALL(*this, GetDefaultPjRtLayout)
      .WillByDefault(
          [this](xla::ifrt::DType dtype, absl::Span<const int64_t> dims,
                 xla::ifrt::Device* device, xla::ifrt::MemoryKind memory_kind)
              -> absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> {
            return delegated_->GetDefaultPjRtLayout(dtype, dims, device,
                                                    memory_kind);
          });
  ON_CALL(*this, GetDefaultLayout)
      .WillByDefault(
          [this](
              DType dtype, const Shape& shape,
              const ShardingRef& sharding) -> absl::StatusOr<CustomLayoutRef> {
            return delegated_->GetDefaultLayout(dtype, shape, sharding);
          });
  ON_CALL(*this, Attributes).WillByDefault([this]() -> const AttributeMap& {
    return delegated_->Attributes();
  });
  ON_CALL(*this, SubscribeToAttributeChanges)
      .WillByDefault(
          [this](absl::Span<xla::ifrt::Device* const> devices,
                 std::optional<absl::Span<const std::string>> attribute_names,
                 xla::ifrt::OnDeviceAttributeChangeCallback callback) {
            return delegated_->SubscribeToAttributeChanges(
                devices, attribute_names, std::move(callback));
          });
}
// LINT.ThenChange()

// LINT.IfChange(MockDeviceDelegation)
MockDevice::MockDevice(Device* delegated) : delegated_(delegated) {
  ON_CALL(*this, client).WillByDefault([this]() {
    return delegated_->client();
  });
  ON_CALL(*this, IsAddressable).WillByDefault([this]() {
    return delegated_->IsAddressable();
  });
  ON_CALL(*this, Id).WillByDefault([this]() { return delegated_->Id(); });
  ON_CALL(*this, ProcessIndex).WillByDefault([this]() {
    return delegated_->ProcessIndex();
  });
  ON_CALL(*this, PlatformName).WillByDefault([this]() {
    return delegated_->PlatformName();
  });
  ON_CALL(*this, Kind).WillByDefault([this]() { return delegated_->Kind(); });
  ON_CALL(*this, Attributes).WillByDefault([this]() -> const AttributeMap& {
    return delegated_->Attributes();
  });
  ON_CALL(*this, DefaultMemory).WillByDefault([this]() {
    return delegated_->DefaultMemory();
  });
  ON_CALL(*this, Memories).WillByDefault([this]() {
    return delegated_->Memories();
  });
}
// LINT.ThenChange()

}  // namespace ifrt
}  // namespace xla
