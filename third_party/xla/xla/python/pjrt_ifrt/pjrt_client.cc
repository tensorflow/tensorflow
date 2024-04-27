/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/python/pjrt_ifrt/pjrt_client.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/tuple.h"
#include "xla/python/ifrt/value.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/pjrt_device.h"
#include "xla/python/pjrt_ifrt/pjrt_memory.h"
#include "xla/python/pjrt_ifrt/pjrt_tuple.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

// A nullptr std::function implicitly converts to a non-nullptr
// absl::AnyInvocable, which later crashes when being invoked. absl team
// explicitly said this is WAI. See b/258212655#comment10.
absl::AnyInvocable<void() &&> FromStdFunction(std::function<void()>&& f) {
  return f ? std::move(f) : absl::AnyInvocable<void() &&>();
}

}  // namespace

char PjRtCompatibleClient::ID = 0;
char PjRtClient::ID = 0;

std::unique_ptr<PjRtClient> PjRtClient::Create(
    std::shared_ptr<xla::PjRtClient> pjrt_client) {
  return absl::WrapUnique(new PjRtClient(std::move(pjrt_client)));
}

PjRtClient::PjRtClient(std::shared_ptr<xla::PjRtClient> pjrt_client)
    : pjrt_client_(std::move(pjrt_client)), default_compiler_(this) {
  devices_.reserve(pjrt_client_->devices().size());
  device_map_.reserve(pjrt_client_->devices().size());
  for (xla::PjRtDevice* device : pjrt_client_->devices()) {
    auto ifrt_device = std::make_unique<PjRtDevice>(this, device);
    devices_.push_back(ifrt_device.get());
    CHECK(device_map_.insert({device, std::move(ifrt_device)}).second);
  }
  addressable_devices_.reserve(pjrt_client_->addressable_devices().size());
  for (xla::PjRtDevice* device : pjrt_client_->addressable_devices()) {
    auto it = device_map_.find(device);
    CHECK(it != device_map_.end());
    addressable_devices_.push_back(it->second.get());
  }

  memory_map_.reserve(pjrt_client_->memory_spaces().size());
  for (xla::PjRtMemorySpace* memory_space : pjrt_client_->memory_spaces()) {
    auto ifrt_memory_space = std::make_unique<PjRtMemory>(this, memory_space);
    memory_map_[memory_space] = std::move(ifrt_memory_space);
  }

  for (Device* device : devices_) {
    auto* pjrt_device = tensorflow::down_cast<PjRtDevice*>(device);
    pjrt_device->memories_.reserve(
        pjrt_device->pjrt_device()->memory_spaces().size());
    for (xla::PjRtMemorySpace* pjrt_memory_space :
         pjrt_device->pjrt_device()->memory_spaces()) {
      pjrt_device->memories_.push_back(*LookupPjRtMemory(pjrt_memory_space));
    }
  }
}

PjRtClient::~PjRtClient() = default;

absl::StatusOr<PjRtCompatibleDevice*> PjRtClient::LookupPjRtDevice(
    xla::PjRtDevice* pjrt_device) const {
  auto it = device_map_.find(pjrt_device);
  if (it == device_map_.end()) {
    return InvalidArgument("PjRtDevice not found: %s",
                           pjrt_device->DebugString());
  }
  return it->second.get();
}

absl::StatusOr<PjRtCompatibleMemory*> PjRtClient::LookupPjRtMemory(
    xla::PjRtMemorySpace* pjrt_memory) const {
  auto it = memory_map_.find(pjrt_memory);
  if (it == memory_map_.end()) {
    return InvalidArgument("PjRtMemorySpace not found: %s",
                           pjrt_memory->DebugString());
  }
  return it->second.get();
}

absl::StatusOr<Device*> PjRtClient::LookupDevice(DeviceId device_id) const {
  DCHECK(this);
  TF_ASSIGN_OR_RETURN(xla::PjRtDevice * pjrt_device,
                      pjrt_client_->LookupDevice(device_id.value()));
  return LookupPjRtDevice(pjrt_device);
}

absl::StatusOr<Device*> PjRtClient::LookupAddressableDevice(
    int local_hardware_id) const {
  DCHECK(this);
  TF_ASSIGN_OR_RETURN(xla::PjRtDevice * pjrt_device,
                      pjrt_client_->LookupAddressableDevice(local_hardware_id));
  return LookupPjRtDevice(pjrt_device);
}

absl::flat_hash_map<std::string, Client::ClientAttribute>
PjRtClient::attributes() const {
  absl::flat_hash_map<std::string, ClientAttribute> attributes;
  attributes.insert({"supports_executable_serialization", true});

  if (std::optional<PjRtPluginAttributes> plugin_attributes =
          pjrt_client_->plugin_attributes();
      plugin_attributes.has_value()) {
    attributes.insert(
        {"pjrt_c_api_major_version",
         ClientAttribute(plugin_attributes->pjrt_c_api_major_version)});
    attributes.insert(
        {"pjrt_c_api_minor_version",
         ClientAttribute(plugin_attributes->pjrt_c_api_minor_version)});
    for (const auto& [key, value] : plugin_attributes->attributes) {
      attributes.insert({key, value});
    }
  }

  return attributes;
}

absl::StatusOr<tsl::RCReference<PjRtCompatibleArray>>
PjRtClient::CreatePjRtArray(std::shared_ptr<PjRtBuffer> pjrt_buffer) {
  TF_ASSIGN_OR_RETURN(auto array,
                      PjRtArray::Create(this, std::move(pjrt_buffer)));
  return tsl::RCReference<PjRtCompatibleArray>(std::move(array));
}

absl::StatusOr<tsl::RCReference<PjRtCompatibleArray>>
PjRtClient::CreatePjRtArray(Shape shape, PjRtBuffers pjrt_buffers) {
  TF_ASSIGN_OR_RETURN(auto array, PjRtArray::Create(this, std::move(shape),
                                                    std::move(pjrt_buffers)));
  return tsl::RCReference<PjRtCompatibleArray>(std::move(array));
}

absl::StatusOr<tsl::RCReference<Array>> PjRtClient::MakeArrayFromHostBuffer(
    const void* data, DType dtype, Shape shape,
    std::optional<absl::Span<const int64_t>> byte_strides,
    std::shared_ptr<const Sharding> sharding,
    Client::HostBufferSemantics semantics,
    std::function<void()> on_done_with_host_buffer) {
  DCHECK(this);
  if (!llvm::isa<const SingleDeviceSharding>(sharding.get())) {
    return InvalidArgument(
        "Only SingleDeviceSharding is supported: sharding=%s",
        sharding->DebugString());
  }
  TF_ASSIGN_OR_RETURN(auto primitive_type, ToPrimitiveType(dtype));

  std::unique_ptr<PjRtBuffer> buffer;
  // If the sharding has memory_kind specified, use a version of
  // `PjRtClient::BufferFromHostBuffer` that accepts `PjRtMemorySpace`.
  // Otherwise, use a non-`PjRtMemorySpace` version that is compatible with PjRt
  // implementations without memories support.
  if (sharding->memory_kind().memory_kind().has_value()) {
    // Find `PjRtMemorySpace` that is associated with the sharding's device and
    // matches the sharding's memory_kind.
    Memory* memory_space = nullptr;
    for (Memory* ms : sharding->devices().front()->Memories()) {
      if (ms->Kind() == sharding->memory_kind()) {
        memory_space = ms;
        break;
      }
    }
    if (memory_space == nullptr) {
      return InvalidArgument(
          "Invalid memory kind: %s; available memory kinds: %s",
          *sharding->memory_kind().memory_kind(),
          absl::StrJoin(sharding->devices().front()->Memories(), ", ",
                        [](std::string* out, Memory* ms) {
                          absl::StrAppend(out, *ms->Kind().memory_kind());
                        }));
    }
    TF_ASSIGN_OR_RETURN(
        buffer,
        pjrt_client_->BufferFromHostBuffer(
            data, primitive_type, shape.dims(), byte_strides, semantics,
            FromStdFunction(std::move(on_done_with_host_buffer)),
            tensorflow::down_cast<PjRtMemory*>(memory_space)->pjrt_memory(),
            /*device_layout=*/nullptr));
  } else {
    TF_ASSIGN_OR_RETURN(
        buffer,
        pjrt_client_->BufferFromHostBuffer(
            data, primitive_type, shape.dims(), byte_strides, semantics,
            FromStdFunction(std::move(on_done_with_host_buffer)),
            tensorflow::down_cast<PjRtDevice*>(sharding->devices().front())
                ->pjrt_device()));
  }
  return PjRtArray::Create(
      this, dtype, std::move(shape), std::move(sharding),
      PjRtArray::PjRtBuffers({std::shared_ptr<PjRtBuffer>(buffer.release())}));
}

absl::StatusOr<tsl::RCReference<Array>>
PjRtClient::AssembleArrayFromSingleDeviceArrays(
    Shape shape, std::shared_ptr<const Sharding> sharding,
    absl::Span<tsl::RCReference<Array>> arrays, ArrayCopySemantics semantics) {
  DCHECK(this);
  if (llvm::isa<const SingleDeviceSharding>(sharding.get())) {
    // Assemble with SingleDeviceSharding is No-op.
    if (arrays.size() != 1) {
      return InvalidArgument(
          "When the sharding is SingleDeviceSharding, the input arrays size "
          "must be one, but the actual size is %d",
          arrays.size());
    }
    return arrays[0];
  } else if (!llvm::isa<const OpaqueSharding, const ConcreteSharding,
                        const ConcreteEvenSharding, const ShardingParamSharding,
                        const HloSharding>(sharding.get())) {
    return InvalidArgument(
        "Only SingleDeviceSharding, OpaqueSharding, ConcreteSharding, "
        "ConcreteEvenSharding, ShardingParamSharding, HloSharding are "
        "supported: sharding=%s",
        sharding->DebugString());
  }
  if (sharding->devices().size() != arrays.size()) {
    return InvalidArgument(
        "Number of output shards must match the number of single-shard arrays: "
        "%d vs. %d",
        sharding->devices().size(), arrays.size());
  }
  PjRtArray::PjRtBuffers buffers;
  buffers.reserve(arrays.size());
  DType dtype = arrays[0]->dtype();
  for (int i = 0; i < arrays.size(); ++i) {
    if (!llvm::isa<PjRtCompatibleArray>(arrays[i].get())) {
      return InvalidArgument(
          "Only PjRtCompatibleArray is supported: arrays[%d]=%s", i,
          arrays[i]->DebugString());
    }
    auto* array = static_cast<PjRtCompatibleArray*>(arrays[i].get());
    if (array->dtype() != dtype) {
      return InvalidArgument(
          "Every input must have the same dtype: %s (shard 0) vs. %s (shard "
          "%d)",
          dtype.DebugString(), array->dtype().DebugString(), i);
    }
    if (array->sharding().devices().size() != 1) {
      return InvalidArgument(
          "Every input must use a single device sharding, but input %d has "
          "sharding=%s",
          i, array->sharding().DebugString());
    }
    switch (semantics) {
      case ArrayCopySemantics::kAlwaysCopy:
        // TODO(hyeontaek): kAlwaysCopy should clone the buffer, but the PjRt
        // API does not have efficient buffer cloning on the same device.
        buffers.push_back(array->pjrt_buffers().front());
        break;
      case ArrayCopySemantics::kReuseInput:
        buffers.push_back(array->pjrt_buffers().front());
        break;
      case ArrayCopySemantics::kDonateInput:
        buffers.push_back(std::move(array->pjrt_buffers().front()));
        break;
    }
  }
  return PjRtArray::Create(this, dtype, std::move(shape), std::move(sharding),
                           std::move(buffers));
}

absl::StatusOr<tsl::RCReference<Tuple>> PjRtClient::MakeTuple(
    absl::Span<tsl::RCReference<Value>> values) {
  return PjRtTuple::Create(this, values);
}

absl::StatusOr<std::shared_ptr<const xla::PjRtTopologyDescription>>
PjRtClient::GetTopologyForDevices(const xla::ifrt::DeviceList& devices) const {
  // TODO(parkers): Consider constructing a sub-slice topology based on the
  // provided devices.
  TF_ASSIGN_OR_RETURN(auto topology, pjrt_client_->GetTopologyDescription());
  return std::shared_ptr<const xla::PjRtTopologyDescription>(pjrt_client_,
                                                             topology);
}

absl::StatusOr<std::unique_ptr<PjRtLayout>>
PjRtClient::GetDefaultLayoutForDevice(DType dtype,
                                      absl::Span<const int64_t> dims,
                                      Device* device) const {
  TF_ASSIGN_OR_RETURN(PrimitiveType element_type, ToPrimitiveType(dtype));
  TF_ASSIGN_OR_RETURN(xla::Layout layout,
                      pjrt_client_->GetDefaultLayout(element_type, dims));
  return std::make_unique<PjRtXlaLayout>(std::move(layout));
}

absl::Status PjRtClient::TransferToInfeed(PjRtDevice* device,
                                          const LiteralSlice& literal) {
  return device->pjrt_device()->TransferToInfeed(literal);
}

absl::Status PjRtClient::TransferFromOutfeed(PjRtDevice* device,
                                             MutableBorrowingLiteral literal) {
  return device->pjrt_device()->TransferFromOutfeed(literal);
}

}  // namespace ifrt
}  // namespace xla
