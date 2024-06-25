// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/python/ifrt_proxy/client/client.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/value.h"
#include "xla/python/ifrt_proxy/client/array.h"
#include "xla/python/ifrt_proxy/client/device.h"
#include "xla/python/ifrt_proxy/client/memory.h"
#include "xla/python/ifrt_proxy/client/rpc_helper.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/types.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace proxy {

char Client::ID = 0;

absl::StatusOr<std::unique_ptr<Client>> Client::Create(
    std::shared_ptr<RpcHelper> rpc_helper, InitResponse init_response) {
  absl::flat_hash_set<int> addressable_device_ids(
      init_response.addressable_device_ids().begin(),
      init_response.addressable_device_ids().end());

  absl::flat_hash_map<int, std::unique_ptr<Memory>> memories;
  for (const auto& m : init_response.memories()) {
    auto memory =
        std::make_unique<Memory>(m.id(), m.memory_space_kind(), m.kind_id(),
                                 m.debug_string(), m.to_string());
    memories.insert({m.id(), std::move(memory)});
  }

  absl::flat_hash_map<int, std::unique_ptr<Device>> devices;
  std::vector<xla::ifrt::Device*> device_ptrs;
  std::vector<xla::ifrt::Device*> addressable_device_ptrs;

  for (const auto& d : init_response.devices()) {
    absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute> attributes;
    for (const auto& [key, attr] : d.attributes()) {
      TF_ASSIGN_OR_RETURN(xla::PjRtDeviceAttribute value,
                          FromVariantProto(attr));
      attributes.insert({key, std::move(value)});
    }

    DeviceDescription desc(d.id(), init_response.process_index(),
                           d.device_kind(), d.debug_string(), d.to_string(),
                           std::move(attributes));
    bool is_addressable = addressable_device_ids.contains(d.id());

    auto device =
        std::make_unique<Device>(std::move(desc), d.local_device_id(),
                                 d.local_hardware_id(), is_addressable);
    device_ptrs.push_back(device.get());
    if (is_addressable) {
      addressable_device_ptrs.push_back(device.get());
    }

    if (d.has_default_memory_id()) {
      const auto it = memories.find(d.default_memory_id());
      if (it == memories.end()) {
        return absl::NotFoundError(
            absl::StrCat("Memory ", d.default_memory_id(), " not found"));
      }
      device->default_memory_ = it->second.get();
    }
    for (const int memory_id : d.memory_ids()) {
      const auto it = memories.find(memory_id);
      if (it == memories.end()) {
        return absl::NotFoundError(
            absl::StrCat("Memory ", memory_id, " not found"));
      }
      device->memories_.push_back(it->second.get());
    }

    devices.insert({d.id(), std::move(device)});
  }

  for (const auto& m : init_response.memories()) {
    Memory* memory = memories.at(m.id()).get();
    for (const int device_id : m.device_ids()) {
      const auto device = devices.find(device_id);
      if (device == devices.end()) {
        return absl::NotFoundError(
            absl::StrCat("Device ", device_id, " not found"));
      }
      memory->devices_.push_back(device->second.get());
    }
  }

  // Prefix the runtime_type string received from the server with "proxy/" so
  // that the users (of this proxy client, such as JAX) do not erroneously
  // conclude that they are talking with the backend runtime directly.
  std::string runtime_type =
      absl::StrCat("proxy/", init_response.runtime_type());

  auto client = absl::WrapUnique(new Client(
      std::move(rpc_helper), init_response.session_id(),
      init_response.platform_name(), init_response.platform_version(),
      init_response.platform_id(), init_response.process_index(), runtime_type,
      std::move(devices), device_ptrs, std::move(addressable_device_ptrs),
      std::move(memories)));
  for (ifrt::Device* device : device_ptrs) {
    tensorflow::down_cast<Device*>(device)->client_ = client.get();
  }
  return client;
}

Client::Client(std::shared_ptr<RpcHelper> rpc_helper, uint64_t session_id,
               std::string platform_name, std::string platform_version,
               uint64_t platform_id, uint64_t process_index,
               std::string runtime_type,
               absl::flat_hash_map<int, std::unique_ptr<Device>> devices,
               std::vector<xla::ifrt::Device*> device_ptrs,
               std::vector<xla::ifrt::Device*> addressable_device_ptrs,
               absl::flat_hash_map<int, std::unique_ptr<Memory>> memories)
    : rpc_helper_(rpc_helper),
      platform_name_(std::move(platform_name)),
      platform_version_(std::move(platform_version)),
      platform_id_(platform_id),
      process_index_(process_index),
      runtime_type_(std::move(runtime_type)),
      devices_(std::move(devices)),
      device_ptrs_(device_ptrs),
      addressable_device_ptrs_(std::move(addressable_device_ptrs)),
      memories_(std::move(memories)),
      default_compiler_(this, rpc_helper) {}

Client::~Client() { rpc_helper_->Disconnect(); }

absl::StatusOr<xla::ifrt::Device*> Client::LookupDevice(
    DeviceId device_id) const {
  auto it = devices_.find(device_id.value());
  if (it == devices_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Device ", device_id.value(), " not found."));
  }
  return it->second.get();
}

absl::StatusOr<tsl::RCReference<xla::ifrt::Array>>
Client::MakeArrayFromHostBuffer(
    const void* data, DType dtype, Shape shape,
    std::optional<absl::Span<const int64_t>> byte_strides,
    std::shared_ptr<const Sharding> sharding,
    xla::ifrt::Client::HostBufferSemantics semantics,
    std::function<void()> on_done_with_host_buffer) {
  return Array::MakeArrayFromHostBuffer(
      this, rpc_helper_, data, dtype, std::move(shape), std::move(byte_strides),
      std::move(sharding), semantics, std::move(on_done_with_host_buffer));
}

absl::StatusOr<tsl::RCReference<xla::ifrt::Array>>
Client::AssembleArrayFromSingleDeviceArrays(
    Shape shape, std::shared_ptr<const Sharding> sharding,
    absl::Span<tsl::RCReference<xla::ifrt::Array>> arrays,
    ArrayCopySemantics semantics) {
  return Array::AssembleArrayFromSingleDeviceArrays(
      this, rpc_helper_, std::move(shape), sharding, arrays, semantics);
}

absl::StatusOr<std::vector<tsl::RCReference<xla::ifrt::Array>>>
Client::CopyArrays(absl::Span<tsl::RCReference<xla::ifrt::Array>> arrays,
                   std::optional<DeviceList> devices,
                   std::optional<MemoryKind> memory_kind,
                   ArrayCopySemantics semantics) {
  if (arrays.empty()) {
    return std::vector<tsl::RCReference<xla::ifrt::Array>>();
  }

  for (int i = 1; i < arrays.size(); ++i) {
    const auto& sharding = arrays[i]->sharding();
    if (sharding.devices() != arrays[0]->sharding().devices() ||
        sharding.memory_kind() != arrays[0]->sharding().memory_kind()) {
      return absl::InvalidArgumentError(
          "CopyArrays only supports arrays with the same device list and "
          "memory kind");
    }
  }

  if (rpc_helper_->version().protocol_version() <= 2) {
    std::vector<tsl::RCReference<xla::ifrt::Array>> new_arrays;
    new_arrays.reserve(arrays.size());
    for (const auto& array : arrays) {
      TF_ASSIGN_OR_RETURN(
          auto new_sharding,
          array->sharding().WithDeviceAssignment(devices, memory_kind));
      TF_ASSIGN_OR_RETURN(new_arrays.emplace_back(),
                          array->Reshard(std::move(new_sharding), semantics));
    }
    return new_arrays;
  }

  auto req = std::make_unique<CopyArraysRequest>();
  for (const auto& array : arrays) {
    if (auto* proxy_array =
            llvm::dyn_cast<xla::ifrt::proxy::Array>(array.get())) {
      req->add_array_handles(proxy_array->handle().handle);
    } else {
      return absl::InvalidArgumentError(
          "CopyArrays only supports arrays created via IFRT Proxy client");
    }
  }
  if (devices.has_value()) {
    for (auto* const device : devices->devices()) {
      req->add_device_ids(device->Id().value());
    }
  }
  if (memory_kind.has_value()) {
    // Use an empty string to indicate the default memory kind.
    req->set_memory_kind(std::string(memory_kind->memory_kind().value_or("")));
  }
  req->set_copy_semantics(ToArrayCopySemanticsProto(semantics));

  auto future = rpc_helper_->CopyArrays(std::move(req));
  TF_ASSIGN_OR_RETURN(auto response, future.Await());

  std::vector<tsl::RCReference<xla::ifrt::Array>> new_arrays;
  new_arrays.reserve(arrays.size());
  for (int i = 0; i < response->array_handles_size(); ++i) {
    TF_ASSIGN_OR_RETURN(
        auto new_sharding,
        arrays[i]->sharding().WithDeviceAssignment(devices, memory_kind));
    new_arrays.push_back(tsl::MakeRef<Array>(
        this, rpc_helper_, arrays[i]->dtype(), arrays[i]->shape(),
        std::move(new_sharding), ArrayHandle{response->array_handles(i)}));
  }
  return new_arrays;
}

absl::StatusOr<std::vector<tsl::RCReference<xla::ifrt::Array>>>
Client::RemapArrays(const RemapPlan& plan,
                    absl::Span<tsl::RCReference<xla::ifrt::Array>> arrays,
                    ArrayCopySemantics semantics) {
  return Array::RemapArrays(this, rpc_helper_, plan, arrays, semantics);
}

xla::ifrt::Future<> Client::GetReadyFuture(
    absl::Span<const tsl::RCReference<xla::ifrt::Value>> values) {
  if (rpc_helper_->version().protocol_version() <= 1) {
    // Legacy implementation for servers that do not support
    // `Client::GetReadyFuture`.
    std::vector<xla::ifrt::Future<>> futures;
    futures.reserve(values.size());
    for (const auto& value : values) {
      futures.push_back(value->GetReadyFuture());
    }
    return xla::ifrt::JoinFutures(futures);
  }

  absl::InlinedVector<Future<>, 1> futures;

  auto req = std::make_unique<CheckValueReadyRequest>();
  for (const auto& value : values) {
    // TODO(b/261991179): IFRT Proxy currently supports Arrays as the only value
    // type, but this may be extended later to other types such as Tuples.
    if (auto proxy_array =
            llvm::dyn_cast<xla::ifrt::proxy::Array>(value.get())) {
      req->add_value_handles(proxy_array->handle().handle);
    } else {
      futures.push_back(value->GetReadyFuture());
    }
  }

  auto promise = Future<>::CreatePromise();
  rpc_helper_->CheckValueReady(std::move(req))
      .OnReady(
          [promise](absl::StatusOr<std::shared_ptr<CheckValueReadyResponse>>
                        resp) mutable { promise.Set(resp.status()); });
  futures.push_back(Future<>(std::move(promise)));

  return JoinFutures(futures);
}

absl::StatusOr<DeviceAssignment> Client::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  auto req = std::make_unique<GetDefaultDeviceAssignmentRequest>();
  req->set_num_replicas(num_replicas);
  req->set_num_partitions(num_partitions);

  auto future = rpc_helper_->GetDefaultDeviceAssignment(std::move(req));
  TF_ASSIGN_OR_RETURN(auto response, future.Await());

  TF_ASSIGN_OR_RETURN(
      auto assignment_to_return,
      DeviceAssignment::Deserialize(response->device_assignment()));

  return *std::move(assignment_to_return);
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
