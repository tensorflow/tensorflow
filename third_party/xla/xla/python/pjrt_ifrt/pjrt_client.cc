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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
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
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/tuple.h"
#include "xla/python/ifrt/value.h"
#include "xla/python/pjrt_ifrt/basic_string_array.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/pjrt_device.h"
#include "xla/python/pjrt_ifrt/pjrt_memory.h"
#include "xla/python/pjrt_ifrt/pjrt_remap.h"
#include "xla/python/pjrt_ifrt/pjrt_tuple.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/errors.h"
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

absl::StatusOr<tsl::RCReference<Array>> MakeStringArrayFromHostBuffer(
    Client* client, const void* data, DType dtype, Shape shape,
    std::optional<absl::Span<const int64_t>> byte_strides,
    std::shared_ptr<const Sharding> sharding,
    Client::HostBufferSemantics semantics,
    std::function<void()> on_done_with_host_buffer) {
  auto param_validation = [&]() -> absl::Status {
    if (byte_strides.has_value()) {
      return absl::InvalidArgumentError(
          "byte_strides is not currently supported for making "
          "BasicStringArrays.");
    }
    if (semantics != Client::HostBufferSemantics::kImmutableOnlyDuringCall) {
      return absl::InvalidArgumentError(
          "HostBufferSemantics other than kImmutableOnlyDuringCall are not "
          "currently supported for making BasicStringArrays.");
    }
    if (!llvm::isa<const SingleDeviceSharding>(sharding.get())) {
      return absl::InvalidArgumentError(
          absl::StrCat("Only SingleDeviceSharding is supported for making "
                       "BasicStringArrays: got: ",
                       sharding->DebugString()));
    }
    return absl::OkStatus();
  }();

  TF_RETURN_IF_ERROR(param_validation);

  auto num_elements = shape.num_elements();
  auto strings = std::make_shared<std::vector<std::string>>();
  strings->reserve(num_elements);
  auto string_views = std::make_shared<std::vector<absl::string_view>>();
  string_views->reserve(num_elements);
  auto element = static_cast<const absl::string_view*>(data);
  for (int i = 0; i < num_elements; ++i, ++element) {
    strings->push_back(std::string(*element));
    string_views->push_back(absl::string_view(strings->back()));
  }
  std::move(on_done_with_host_buffer)();

  BasicStringArray::Buffers buffers;
  buffers.push_back(*string_views);
  auto buffer_releaser = [strings = std::move(strings),
                          string_views = std::move(string_views)]() {};

  return BasicStringArray::Create(
      client, std::move(shape), std::move(sharding),
      Future<BasicStringArray::Buffers>(std::move(buffers)),
      std::move(buffer_releaser));
}

absl::StatusOr<tsl::RCReference<Array>>
AssembleStringArrayFromSingleDeviceStringArrays(
    Shape shape, std::shared_ptr<const Sharding> sharding,
    absl::Span<tsl::RCReference<Array>> arrays, ArrayCopySemantics semantics) {
  // BufferBackingState contains the per-shard vectors of the strings and
  // string_views underlying a BasicString::Buffer.  Not thread safe.
  struct BufferBackingStore {
    explicit BufferBackingStore(int num_shards)
        : per_shard_strings(num_shards), per_shard_string_views(num_shards) {}
    void clear() {
      per_shard_strings.clear();
      per_shard_string_views.clear();
    }
    void CopyBuffer(absl::Span<const absl::string_view> strbuf, int shard_index,
                    BasicStringArray::Buffers* buffers) {
      auto& strings = per_shard_strings[shard_index];
      strings.reserve(strbuf.size());
      auto& views = per_shard_string_views[shard_index];
      views.reserve(strbuf.size());

      for (int i = 0; i < strbuf.size(); ++i) {
        strings.push_back(std::string(strbuf[i].data(), strbuf[i].size()));
      }
      for (const auto& str : strings) {
        views.push_back(str);
      }
      (*buffers)[shard_index] = absl::MakeConstSpan(views);
    }
    std::vector<std::vector<std::string>> per_shard_strings;
    std::vector<std::vector<absl::string_view>> per_shard_string_views;
  };
  auto buffer_backing_store =
      std::make_shared<BufferBackingStore>(sharding->devices().size());
  auto on_done_with_buffer = [buffer_holder = buffer_backing_store]() {};

  struct BufferCopyingState {
    BufferCopyingState(int num_buffers_to_copy,
                       std::shared_ptr<BufferBackingStore> buffer_backing_store)
        : num_buffers_to_copy(num_buffers_to_copy),
          buffer_backing_store(std::move(buffer_backing_store)),
          buffers(num_buffers_to_copy) {}
    absl::Mutex mu;
    int num_buffers_to_copy ABSL_GUARDED_BY(mu);
    std::shared_ptr<BufferBackingStore> buffer_backing_store
        ABSL_GUARDED_BY(mu);
    BasicStringArray::Buffers buffers ABSL_GUARDED_BY(mu);
  };
  auto buffer_copying_state = std::make_shared<BufferCopyingState>(
      arrays.size(), std::move(buffer_backing_store));

  auto buffers_promise = Future<BasicStringArray::Buffers>::CreatePromise();
  auto buffers_future = Future<BasicStringArray::Buffers>(buffers_promise);

  auto buffer_copier = [state = buffer_copying_state,
                        promise = buffers_promise](
                           absl::StatusOr<BasicStringArray::Buffers> strbuf,
                           int shard_index) mutable {
    absl::MutexLock lock(&state->mu);
    if (state->num_buffers_to_copy == 0) {
      // Nothing to be done. We get here if the buffers of a single
      // device array became ready with a an error previously.
      return;
    }
    if (!strbuf.ok()) {
      promise.Set(strbuf.status());
      state->num_buffers_to_copy = 0;  // Don't copy any more buffers.

      // Release the partially copied buffers and reclaim the memory.
      // These are no longer needed. The empty buffer_holder itself lives
      // on until the on_done_with_buffer is called.
      state->buffer_backing_store->clear();
      state->buffer_backing_store = nullptr;
      return;
    }

    state->buffer_backing_store->CopyBuffer(strbuf->front(), shard_index,
                                            &state->buffers);

    if (--state->num_buffers_to_copy > 0) {
      return;  // We have more single device arrays we need to wait for.
    }
    // We have all the buffers. Set the promise.
    promise.Set(std::move(state->buffers));
  };

  for (int i = 0; i < arrays.size(); ++i) {
    auto basic_string_array = llvm::dyn_cast<BasicStringArray>(arrays[i].get());
    if (!basic_string_array) {
      return absl::InvalidArgumentError(
          "All single device arrays must be BasicStringArrays");
    }
    if (!llvm::isa<SingleDeviceSharding>(basic_string_array->sharding())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "All single device arrays must have single device sharding. got: %s "
          "for shard index: %d",
          basic_string_array->sharding().DebugString(), i));
    }

    basic_string_array->buffers().OnReady(
        [shard_index = i, buffer_copier = buffer_copier](
            absl::StatusOr<BasicStringArray::Buffers> strbuf) mutable {
          buffer_copier(std::move(strbuf), shard_index);
        });
  }

  return BasicStringArray::Create(arrays[0]->client(), std::move(shape),
                                  std::move(sharding), buffers_future,
                                  std::move(on_done_with_buffer));
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
    auto ifrt_device = std::make_unique<PjRtDevice>(
        this, DeviceId(device->global_device_id().value()),
        std::string(device->device_kind()), std::string(device->ToString()),
        std::string(device->DebugString()), device->process_index(),
        device->Attributes(), device->IsAddressable() ? device : nullptr);
    devices_.push_back(ifrt_device.get());
    CHECK(device_id_map_.emplace(ifrt_device->Id(), ifrt_device.get()).second);
    CHECK(device_map_.emplace(device, std::move(ifrt_device)).second);
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

  for (size_t i = 0; i < devices_.size(); ++i) {
    auto* device = tensorflow::down_cast<PjRtDevice*>(devices_[i]);
    auto* pjrt_device = pjrt_client_->devices()[i];
    device->memories_.reserve(pjrt_device->memory_spaces().size());
    for (xla::PjRtMemorySpace* pjrt_memory_space :
         pjrt_device->memory_spaces()) {
      device->memories_.push_back(*LookupPjRtMemory(pjrt_memory_space));
    }
    absl::StatusOr<PjRtMemorySpace*> memory =
        pjrt_device->default_memory_space();
    if (memory.ok()) {
      device->default_memory_ = *LookupPjRtMemory(*memory);
    } else {
      device->default_memory_ = memory.status();
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
  auto it = device_id_map_.find(device_id);
  if (it != device_id_map_.end()) {
    return it->second;
  }
  return InvalidArgument("No matching device found for device_id %d",
                         device_id.value());
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
  if (dtype.kind() == DType::kString) {
    return MakeStringArrayFromHostBuffer(this, data, dtype, shape, byte_strides,
                                         sharding, semantics,
                                         on_done_with_host_buffer);
  }
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
    Memory* memory = nullptr;
    for (Memory* ms : sharding->devices().front()->Memories()) {
      if (ms->Kind() == sharding->memory_kind()) {
        memory = ms;
        break;
      }
    }
    if (memory == nullptr) {
      return InvalidArgument(
          "Invalid memory kind: %s; available memory kinds: %s",
          *sharding->memory_kind().memory_kind(),
          absl::StrJoin(sharding->devices().front()->Memories(), ", ",
                        [](std::string* out, Memory* ms) {
                          absl::StrAppend(out, *ms->Kind().memory_kind());
                        }));
    }
    TF_ASSIGN_OR_RETURN(
        buffer, pjrt_client_->BufferFromHostBuffer(
                    data, primitive_type, shape.dims(), byte_strides, semantics,
                    FromStdFunction(std::move(on_done_with_host_buffer)),
                    tensorflow::down_cast<PjRtMemory*>(memory)->pjrt_memory(),
                    /*device_layout=*/nullptr));
  } else {
    Device* device = sharding->devices().front();
    if (!device->IsAddressable()) {
      return InvalidArgument("Cannot copy array to non-addressable device %s",
                             device->DebugString());
    }
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
  if (arrays[0]->dtype().kind() == DType::kString) {
    return AssembleStringArrayFromSingleDeviceStringArrays(shape, sharding,
                                                           arrays, semantics);
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

absl::StatusOr<std::vector<tsl::RCReference<xla::ifrt::Array>>>
PjRtClient::RemapArrays(const RemapPlan& plan,
                        absl::Span<tsl::RCReference<xla::ifrt::Array>> arrays,
                        ArrayCopySemantics semantics) {
  return PjRtCompatibleClientRemapArrays(this, plan, arrays, semantics);
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
  if (!device->IsAddressable()) {
    return InvalidArgument(
        "Infeed is only supported on addressable devices "
        "but device %s is not addressable",
        device->DebugString());
  }
  return device->pjrt_device()->TransferToInfeed(literal);
}

absl::Status PjRtClient::TransferFromOutfeed(PjRtDevice* device,
                                             MutableBorrowingLiteral literal) {
  if (!device->IsAddressable()) {
    return InvalidArgument(
        "Outfeed is only supported on addressable devices "
        "but device %s is not addressable",
        device->DebugString());
  }
  return device->pjrt_device()->TransferFromOutfeed(literal);
}

}  // namespace ifrt
}  // namespace xla
