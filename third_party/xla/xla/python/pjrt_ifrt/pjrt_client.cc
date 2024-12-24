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

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
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
#include "xla/pjrt/distributed/protocol.pb.h"
#include "xla/pjrt/distributed/topology_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"
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
#include "xla/python/ifrt/value.h"
#include "xla/python/pjrt_ifrt/basic_string_array.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/pjrt_attribute_map_util.h"
#include "xla/python/pjrt_ifrt/pjrt_device.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/python/pjrt_ifrt/pjrt_memory.h"
#include "xla/python/pjrt_ifrt/pjrt_remap.h"
#include "xla/python/pjrt_ifrt/pjrt_topology.h"
#include "xla/python/pjrt_ifrt/pjrt_tuple.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

// Returns an `AttributeMap` with the attributes of the given `PjRtClient`.
AttributeMap MakeAttributeMap(xla::PjRtClient* pjrt_client) {
  absl::flat_hash_map<std::string, PjRtValueType> attributes;
  attributes.insert({"supports_executable_serialization", true});
  if (std::optional<PjRtPluginAttributes> plugin_attributes =
          pjrt_client->plugin_attributes();
      plugin_attributes.has_value()) {
    attributes.insert(
        {"pjrt_c_api_major_version",
         PjRtValueType(plugin_attributes->pjrt_c_api_major_version)});
    attributes.insert(
        {"pjrt_c_api_minor_version",
         PjRtValueType(plugin_attributes->pjrt_c_api_minor_version)});
    for (const auto& [key, value] : plugin_attributes->attributes) {
      attributes.insert({key, value});
    }
  }
  return FromPjRtAttributeMap(std::move(attributes));
}

void SerializePjRtDeviceAttributes(
    const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& attributes,
    DeviceProto& device_proto) {
  for (const auto& [key, value] : attributes) {
    DeviceAttributeProto& attribute = (*device_proto.mutable_attributes())[key];
    if (std::holds_alternative<std::string>(value)) {
      attribute.set_string_value(std::get<std::string>(value));
    } else if (std::holds_alternative<int64_t>(value)) {
      attribute.set_int_value(std::get<int64_t>(value));
    } else if (std::holds_alternative<std::vector<int64_t>>(value)) {
      auto values = std::get<std::vector<int64_t>>(value);
      attribute.mutable_int_values()->mutable_values()->Assign(values.begin(),
                                                               values.end());
    } else if (std::holds_alternative<bool>(value)) {
      attribute.set_bool_value(std::get<bool>(value));
    } else if (std::holds_alternative<float>(value)) {
      attribute.set_float_value(std::get<float>(value));
    }
  }
}

absl::Status DeserializePjRtDeviceAttributes(
    const DeviceProto& device_proto,
    absl::flat_hash_map<std::string, PjRtDeviceAttribute>& attributes) {
  for (const auto& [key, value] : device_proto.attributes()) {
    if (value.has_string_value()) {
      attributes[key] = value.string_value();
    } else if (value.has_int_value()) {
      attributes[key] = value.int_value();
    } else if (value.has_int_values()) {
      attributes[key] =
          std::vector<int64_t>(value.int_values().values().begin(),
                               value.int_values().values().end());
    } else if (value.has_bool_value()) {
      attributes[key] = value.bool_value();
    } else if (value.has_float_value()) {
      attributes[key] = value.float_value();
    }
  }
  return absl::OkStatus();
}

// All information to construct a global view.
struct GlobalTopology {
  GlobalTopologyProto global_topology_proto;
  // My process (node) index in `global_topology_proto`. This is not to confused
  // with a process ID or node ID, which is not necessarily an index.
  int my_process_index;
  // Mapping from IFRT device ID to PjRt global device ID. Made for the devices
  // that are accessible via `pjrt_client_->devices()`.
  absl::flat_hash_map<DeviceId, xla::PjRtGlobalDeviceId>
      ifrt_device_id_to_pjrt_global_device_id;
};

// Constructs a `GlobalTopologyProto` and my node ID from the `PjRtClient` and
// the `CreateOptions` without external topology exchange. The result is
// directly used for materializing IFRT devices.
absl::StatusOr<GlobalTopology> MakeGlobalTopologyFromPjRtClient(
    xla::PjRtClient* pjrt_client, const PjRtClient::CreateOptions& options) {
  // Discovered process ID and number of processes.
  std::optional<int> my_process_index;
  int num_processes;
  // Mapping from IFRT device ID to PjRt global device ID. Made for the
  // devices that are accessible via `pjrt_client->devices()`.
  absl::flat_hash_map<DeviceId, xla::PjRtGlobalDeviceId>
      ifrt_device_id_to_pjrt_global_device_id;
  // Process index to IFRT device IDs. Made for all IFRT devices.
  std::vector<std::vector<DeviceId>> process_index_to_ifrt_device_ids;

  if (options.global_device_mapping.has_value()) {
    const auto& addressable_device_ids =
        options.global_device_mapping->addressable_device_ids;
    const auto& device_id_to_process_index =
        options.global_device_mapping->device_id_to_process_index;

    if (addressable_device_ids.size() !=
        pjrt_client->addressable_device_count()) {
      return InvalidArgument(
          "global device mapping contains different number of addressable "
          "devices from PjRtClient's: %d vs. %d",
          addressable_device_ids.size(),
          pjrt_client->addressable_device_count());
    }
    if (pjrt_client->device_count() !=
        pjrt_client->addressable_device_count()) {
      // If any non-addressable devices are present in `pjrt_client`, we expect
      // that the total count of devices matches.
      if (device_id_to_process_index.size() != pjrt_client->device_count()) {
        return InvalidArgument(
            "global device mapping contains different number of global devices "
            "from PjRtClient's: %d vs. %d",
            device_id_to_process_index.size(), pjrt_client->device_count());
      }
    }

    num_processes = 0;
    for (const auto& [_, process_index] : device_id_to_process_index) {
      num_processes = std::max(num_processes, process_index + 1);
    }
    if (num_processes == 0) {
      return InvalidArgument("global device mapping contains no processes");
    }

    for (const DeviceId addressable_device_id : addressable_device_ids) {
      const auto it = device_id_to_process_index.find(addressable_device_id);
      if (it == device_id_to_process_index.end()) {
        return InvalidArgument(
            "global device mapping contains an addressable device ID that is "
            "not in the device_id_to_process_index mapping: %d",
            addressable_device_id.value());
      }
      if (my_process_index.has_value()) {
        if (*my_process_index != it->second) {
          return InvalidArgument(
              "addressable device IDs are mapped to multiple processes: %d vs. "
              "%d",
              *my_process_index, it->second);
        }
      } else {
        my_process_index = it->second;
      }
    }

    // Match IFRT device IDs and PjRt device IDs in sorted order by their own
    // device IDs. We currently do not support reordering device IDs because
    // `GlobalDeviceMapping` does not provide a way to specify the order.
    std::vector<DeviceId> sorted_addressable_device_ids(
        addressable_device_ids.begin(), addressable_device_ids.end());
    absl::c_sort(sorted_addressable_device_ids);
    int next_addressable_device_index = 0;

    std::vector<DeviceId> sorted_non_addressable_device_ids;
    sorted_non_addressable_device_ids.reserve(
        device_id_to_process_index.size());
    for (const auto& [device_id, _] : device_id_to_process_index) {
      if (!addressable_device_ids.contains(device_id)) {
        sorted_non_addressable_device_ids.push_back(device_id);
      }
    }
    absl::c_sort(sorted_non_addressable_device_ids);
    int next_non_addressable_device_index = 0;

    std::vector<xla::PjRtDevice*> pjrt_devices(pjrt_client->devices().begin(),
                                               pjrt_client->devices().end());
    absl::c_sort(pjrt_devices, [](xla::PjRtDevice* a, xla::PjRtDevice* b) {
      return a->global_device_id() < b->global_device_id();
    });

    for (xla::PjRtDevice* pjrt_device : pjrt_devices) {
      const xla::PjRtGlobalDeviceId pjrt_global_device_id =
          pjrt_device->global_device_id();
      DeviceId ifrt_device_id;
      if (pjrt_device->IsAddressable()) {
        ifrt_device_id =
            sorted_addressable_device_ids[next_addressable_device_index++];
      } else {
        ifrt_device_id = sorted_non_addressable_device_ids
            [next_non_addressable_device_index++];
      }
      ifrt_device_id_to_pjrt_global_device_id.insert(
          {ifrt_device_id, pjrt_global_device_id});
    }
    process_index_to_ifrt_device_ids.resize(num_processes);
    for (const auto& [device_id, process_index] : device_id_to_process_index) {
      process_index_to_ifrt_device_ids[process_index].push_back(device_id);
    }
  } else {
    num_processes = 1;
    for (xla::PjRtDevice* pjrt_device : pjrt_client->devices()) {
      if (pjrt_device->IsAddressable()) {
        if (my_process_index.has_value()) {
          if (*my_process_index != pjrt_device->process_index()) {
            return InvalidArgument(
                "addressable devices are mapped to multiple processes: %d vs. "
                "%d",
                *my_process_index, pjrt_device->process_index());
          }
        } else {
          my_process_index = pjrt_device->process_index();
        }
      }
      num_processes = std::max(num_processes, pjrt_device->process_index() + 1);
    }

    process_index_to_ifrt_device_ids.resize(num_processes);
    for (xla::PjRtDevice* pjrt_device : pjrt_client->devices()) {
      const xla::PjRtGlobalDeviceId pjrt_global_device_id =
          pjrt_device->global_device_id();
      // Use PjRt device ID as IFRT device ID.
      const DeviceId ifrt_device_id = DeviceId(pjrt_global_device_id.value());
      ifrt_device_id_to_pjrt_global_device_id.insert(
          {ifrt_device_id, pjrt_global_device_id});
      process_index_to_ifrt_device_ids[pjrt_device->process_index()].push_back(
          ifrt_device_id);
    }
  }

  // Generate `GlobalTopologyProto` based on collected device mapping
  // information.
  GlobalTopologyProto global_topology_proto;
  for (int process_index = 0; process_index < num_processes; ++process_index) {
    LocalTopologyProto& node = *global_topology_proto.add_nodes();
    node.set_node_id(process_index);

    const std::vector<DeviceId>& process_device_ids =
        process_index_to_ifrt_device_ids[process_index];
    for (int local_device_ordinal = 0;
         local_device_ordinal < process_device_ids.size();
         ++local_device_ordinal) {
      const DeviceId ifrt_device_id = process_device_ids[local_device_ordinal];
      DeviceProto& device = *node.add_devices();

      xla::PjRtDevice* pjrt_device;
      if (auto it =
              ifrt_device_id_to_pjrt_global_device_id.find(ifrt_device_id);
          it == ifrt_device_id_to_pjrt_global_device_id.end()) {
        pjrt_device = nullptr;
      } else {
        TF_ASSIGN_OR_RETURN(pjrt_device, pjrt_client->LookupDevice(it->second));
      }

      if (pjrt_device == nullptr) {
        device.set_local_device_ordinal(local_device_ordinal);
      } else {
        // Respect the local device ordinal of `PjRtDevice` if it is present
        // (i.e., addressable). During IFRT device materialization, the local
        // device ordinal is used as the key to look up the corresponding
        // `PjRtDevice` from `pjrt_client`.
        device.set_local_device_ordinal(pjrt_device->local_device_id().value());
      }
      // Put IFRT device ID (instead of PjRt global device ID) here to skip any
      // further device ID remapping before IFRT devices are materialized.
      device.set_global_device_id(ifrt_device_id.value());
      device.set_device_kind(
          std::string(pjrt_client->addressable_devices()[0]->device_kind()));

      // TODO(hyeontaek): Take optional device->slice_index mapping in
      // GlobalDeviceMapping and generate the `slice_index` attribute for both
      // addressable and non-addressable devices.
      if (pjrt_device == nullptr) {
        device.set_to_string("NonAddressable");
        device.set_debug_string("NonAddressable");
      } else {
        device.set_to_string(std::string(pjrt_device->ToString()));
        device.set_debug_string(std::string(pjrt_device->DebugString()));
        SerializePjRtDeviceAttributes(pjrt_device->Attributes(), device);
      }
    }
  }

  if (!my_process_index.has_value()) {
    return InvalidArgument(
        "Could not determine process_index for this process");
  }
  return GlobalTopology{std::move(global_topology_proto), *my_process_index,
                        std::move(ifrt_device_id_to_pjrt_global_device_id)};
}

// Constructs a `LocalTopologyProto` from the `PjRtClient` and the
// `CreateOptions` that can be used for topology exchange to obtain
// `GlobalTopologyProto`.
LocalTopologyProto MakeLocalTopologyFromPjRtClient(
    xla::PjRtClient* pjrt_client, const PjRtClient::CreateOptions& options) {
  LocalTopologyProto local_topology_proto;
  local_topology_proto.set_node_id(options.process_id);
  std::string boot_id_str;
  auto boot_id_str_or_status = GetBootIdString();
  if (!boot_id_str_or_status.ok()) {
    LOG(INFO) << boot_id_str_or_status.status();
  } else {
    boot_id_str = boot_id_str_or_status.value();
  }
  local_topology_proto.set_boot_id(boot_id_str);
  // We ignore any non-addressable devices. We're going to do our own topology
  // exchange, so we don't care what devices any given client things that some
  // other process has.
  for (xla::PjRtDevice* device : pjrt_client->addressable_devices()) {
    DeviceProto& device_proto = *local_topology_proto.add_devices();
    device_proto.set_global_device_id(device->global_device_id().value());
    device_proto.set_local_device_ordinal(device->local_device_id().value());
    device_proto.set_device_kind(
        std::string(device->description().device_kind()));
    device_proto.set_to_string(std::string(device->ToString()));
    device_proto.set_debug_string(std::string(device->DebugString()));
    SerializePjRtDeviceAttributes(device->Attributes(), device_proto);
  }

  return local_topology_proto;
}

// Constructs a `GlobalTopology` from topology exchange.
absl::StatusOr<GlobalTopology> MakeGlobalTopologyWithLocalTopology(
    xla::PjRtClient* pjrt_client, const PjRtClient::CreateOptions& options,
    const LocalTopologyProto& local_topology_proto) {
  GlobalTopologyProto global_topology_proto;
  TF_RETURN_IF_ERROR(ExchangeTopologies(
      pjrt_client->platform_name(), options.process_id, options.num_processes,
      options.get_local_topology_timeout, options.get_global_topology_timeout,
      options.kv_store.get(), local_topology_proto, &global_topology_proto,
      /*assign_global_device_ids=*/false));

  std::optional<int> my_process_index;
  absl::flat_hash_map<DeviceId, xla::PjRtGlobalDeviceId>
      ifrt_device_id_to_pjrt_global_device_id;
  for (int process_index = 0;
       process_index < global_topology_proto.nodes_size(); ++process_index) {
    const LocalTopologyProto& node = global_topology_proto.nodes(process_index);
    if (node.node_id() == options.process_id) {
      if (my_process_index.has_value()) {
        if (*my_process_index != process_index) {
          return InvalidArgument(
              "GlobalTopologyProto contains multiple nodes with the same "
              "process ID");
        }
      } else {
        my_process_index = process_index;
      }
    }
    for (const DeviceProto& device_proto : node.devices()) {
      // Use the same PjRt global device ID as IFRT device ID when topology
      // exchange is used.
      ifrt_device_id_to_pjrt_global_device_id.insert(
          {DeviceId(device_proto.global_device_id()),
           xla::PjRtGlobalDeviceId(device_proto.global_device_id())});
    }
  }

  if (!my_process_index.has_value()) {
    return InvalidArgument(
        "Could not determine process_index for this process");
  }
  return GlobalTopology{std::move(global_topology_proto), *my_process_index,
                        std::move(ifrt_device_id_to_pjrt_global_device_id)};
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtDevice>>>
MakePjRtDevicesFromGlobalTopology(PjRtClient* client,
                                  xla::PjRtClient* pjrt_client,
                                  const GlobalTopology& global_topology) {
  std::vector<std::unique_ptr<PjRtDevice>> devices;

  // Some PJRT implementations (e.g., TPU) assign their own "slice_index"
  // values. If these are present, leave them alone. Otherwise, we assign
  // the same slice_index to all devices of the same host, as determined by
  // the boot_id.
  int next_slice_index = 0;
  absl::flat_hash_map<std::string, int> boot_id_to_slice_index;
  for (int process_index = 0;
       process_index < global_topology.global_topology_proto.nodes_size();
       ++process_index) {
    const LocalTopologyProto& node =
        global_topology.global_topology_proto.nodes(process_index);
    int64_t slice_index = -1;
    if (!node.boot_id().empty()) {
      // Every new boot_id seen is treated as a new host/slice.
      absl::string_view boot_id = node.boot_id();
      auto [it, inserted] =
          boot_id_to_slice_index.try_emplace(boot_id, next_slice_index);
      slice_index = it->second;
      if (inserted) {
        ++next_slice_index;
      }
    }

    const bool node_is_me = process_index == global_topology.my_process_index;
    for (const DeviceProto& device_proto : node.devices()) {
      absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes;
      TF_RETURN_IF_ERROR(
          DeserializePjRtDeviceAttributes(device_proto, attributes));
      if (slice_index != -1) {
        // Sets a generated `slice_index` attribute if not already present.
        attributes.insert(
            {"slice_index",
             xla::PjRtDeviceAttribute(static_cast<int64_t>(slice_index))});
      }
      const DeviceId ifrt_device_id(device_proto.global_device_id());
      xla::PjRtDevice* pjrt_device = nullptr;
      std::string to_string(device_proto.to_string());
      std::string debug_string(device_proto.debug_string());
      if (node_is_me) {
        xla::PjRtGlobalDeviceId pjrt_global_device_id =
            global_topology.ifrt_device_id_to_pjrt_global_device_id.at(
                ifrt_device_id);
        TF_ASSIGN_OR_RETURN(pjrt_device,
                            pjrt_client->LookupDevice(pjrt_global_device_id));
        // Only append any device ID remapping to the device debug string. The
        // user code often uses a pattern matching on the debug string (which is
        // discouraged but does exist), and changing the debug string format
        // significantly would break the backward compatibility.
        if (pjrt_global_device_id.value() != device_proto.global_device_id()) {
          absl::StrAppend(&to_string,
                          "[PjRtIFRTDeviceId=", ifrt_device_id.value(), "]");
          absl::StrAppend(&debug_string,
                          "[PjRtIFRTDeviceId=", ifrt_device_id.value(), "]");
        }
      }
      auto ifrt_device = std::make_unique<PjRtDevice>(
          client, ifrt_device_id, device_proto.device_kind(),
          std::move(to_string), std::move(debug_string), process_index,
          std::move(attributes), pjrt_device);
      devices.push_back(std::move(ifrt_device));
    }
  }
  return devices;
}

// Logs a summary of the devices in the client.
void LogDeviceSummary(PjRtClient* client) {
  LOG(INFO) << "PjRt-IFRT device count: total=" << client->devices().size()
            << ", addressable=" << client->addressable_devices().size();
  for (int i = 0; i < client->addressable_devices().size(); ++i) {
    if (i < 10) {
      LOG(INFO) << "Addressable PjRt-IFRT device: "
                << client->addressable_devices()[i]->ToString();
    } else {
      LOG(INFO) << "... (omitted) ...";
      break;
    }
  }
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
  auto strings = std::make_shared<std::vector<absl::Cord>>();
  strings->reserve(num_elements);
  auto element = static_cast<const absl::Cord*>(data);
  for (int i = 0; i < num_elements; ++i, ++element) {
    strings->push_back(*element);
  }
  std::move(on_done_with_host_buffer)();

  BasicStringArray::Buffers buffers;
  buffers.push_back(*strings);
  auto buffer_releaser = [strings = std::move(strings)]() {};

  return BasicStringArray::Create(
      client, std::move(shape), std::move(sharding),
      Future<BasicStringArray::Buffers>(std::move(buffers)),
      std::move(buffer_releaser));
}

absl::StatusOr<tsl::RCReference<Array>>
AssembleStringArrayFromSingleDeviceStringArrays(
    Shape shape, std::shared_ptr<const Sharding> sharding,
    absl::Span<tsl::RCReference<Array>> arrays,
    ArrayCopySemantics array_copy_semantics,
    SingleDeviceShardSemantics single_device_shard_semantics) {
  if (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards &&
      !sharding->devices()->IsFullyAddressable()) {
    return InvalidArgument(
        "All shards are requested but the sharding has non-addressable "
        "devices: %v",
        *sharding->devices());
  }
  // BufferBackingState contains the per-shard vectors of the strings and
  // string_views underlying a BasicString::Buffer.  Not thread safe.
  struct BufferBackingStore {
    explicit BufferBackingStore(int num_shards)
        : per_shard_strings(num_shards) {}
    void clear() { per_shard_strings.clear(); }

    void CopyBuffer(absl::Span<const absl::Cord> strbuf, int shard_index,
                    BasicStringArray::Buffers* buffers) {
      auto& strings = per_shard_strings[shard_index];
      strings.reserve(strbuf.size());
      for (int i = 0; i < strbuf.size(); ++i) {
        strings.push_back(strbuf[i]);
      }
      (*buffers)[shard_index] = absl::MakeConstSpan(strings);
    }
    std::vector<std::vector<absl::Cord>> per_shard_strings;
  };
  auto buffer_backing_store =
      std::make_shared<BufferBackingStore>(sharding->devices()->size());
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

absl::StatusOr<std::unique_ptr<PjRtClient>> PjRtClient::Create(
    PjRtClient::CreateOptions options) {
  auto client =
      absl::WrapUnique(new PjRtClient(std::move(options.pjrt_client)));
  xla::PjRtClient* pjrt_client = client->pjrt_client();

  GlobalTopology global_topology;
  if (!options.kv_store) {
    TF_ASSIGN_OR_RETURN(global_topology,
                        MakeGlobalTopologyFromPjRtClient(pjrt_client, options));
  } else {
    if (options.global_device_mapping.has_value()) {
      return InvalidArgument(
          "global_device_mapping and kv_store cannot be set at the same time");
    }
    // If a KV-store was provided, we perform a topology exchange to aggregate
    // topology information from all processes.
    const LocalTopologyProto local_topology_proto =
        MakeLocalTopologyFromPjRtClient(pjrt_client, options);
    TF_ASSIGN_OR_RETURN(global_topology,
                        MakeGlobalTopologyWithLocalTopology(
                            pjrt_client, options, local_topology_proto));
  }
  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<PjRtDevice>> devices,
                      MakePjRtDevicesFromGlobalTopology(
                          client.get(), pjrt_client, global_topology));

  if (options.sort_devices_by_process_index) {
    absl::c_sort(devices, [](const std::unique_ptr<PjRtDevice>& a,
                             const std::unique_ptr<PjRtDevice>& b) {
      return a->ProcessIndex() < b->ProcessIndex() ||
             (a->ProcessIndex() == b->ProcessIndex() && a->Id() < b->Id());
    });
  } else {
    absl::c_sort(devices, [](const std::unique_ptr<PjRtDevice>& a,
                             const std::unique_ptr<PjRtDevice>& b) {
      return a->Id() < b->Id();
    });
  }

  client->my_process_index_ = global_topology.my_process_index;
  client->ifrt_device_id_to_pjrt_global_device_id_ =
      std::move(global_topology.ifrt_device_id_to_pjrt_global_device_id);

  client->devices_.reserve(devices.size());
  client->device_map_.reserve(pjrt_client->addressable_device_count());
  for (auto& ifrt_device : devices) {
    client->devices_.push_back(ifrt_device.get());
    TF_RET_CHECK(
        client->device_id_map_.emplace(ifrt_device->Id(), ifrt_device.get())
            .second);
    xla::PjRtDevice* pjrt_device = ifrt_device->pjrt_device();
    if (pjrt_device) {
      TF_RET_CHECK(
          client->device_map_.emplace(pjrt_device, ifrt_device.get()).second);
    }
    client->owned_devices_.push_back(std::move(ifrt_device));
  }

  client->addressable_devices_.reserve(
      pjrt_client->addressable_devices().size());
  for (xla::PjRtDevice* device : pjrt_client->addressable_devices()) {
    auto it = client->device_map_.find(device);
    CHECK(it != client->device_map_.end());
    client->addressable_devices_.push_back(it->second);
  }

  client->memory_map_.reserve(pjrt_client->memory_spaces().size());
  for (xla::PjRtMemorySpace* memory_space : pjrt_client->memory_spaces()) {
    auto ifrt_memory = std::make_unique<PjRtMemory>(client.get(), memory_space);
    client->memory_map_[memory_space] = ifrt_memory.get();
    client->owned_memories_.push_back(std::move(ifrt_memory));
  }

  for (Device* ifrt_device : client->addressable_devices_) {
    auto* device = tensorflow::down_cast<PjRtDevice*>(ifrt_device);
    auto* pjrt_device = device->pjrt_device();
    device->memories_.reserve(pjrt_device->memory_spaces().size());
    for (xla::PjRtMemorySpace* pjrt_memory_space :
         pjrt_device->memory_spaces()) {
      device->memories_.push_back(*client->LookupPjRtMemory(pjrt_memory_space));
    }

    absl::StatusOr<PjRtMemorySpace*> memory =
        pjrt_device->default_memory_space();
    if (memory.ok()) {
      device->default_memory_ = *client->LookupPjRtMemory(*memory);
    } else {
      device->default_memory_ = memory.status();
    }
  }

  LogDeviceSummary(client.get());
  return client;
}

std::unique_ptr<PjRtClient> PjRtClient::Create(
    std::shared_ptr<xla::PjRtClient> pjrt_client) {
  PjRtClient::CreateOptions options;
  options.pjrt_client = std::move(pjrt_client);
  return *Create(std::move(options));
}

PjRtClient::PjRtClient(std::shared_ptr<xla::PjRtClient> pjrt_client)
    : pjrt_client_(std::move(pjrt_client)),
      default_compiler_(this),
      attributes_(MakeAttributeMap(pjrt_client_.get())) {}

PjRtClient::~PjRtClient() = default;

absl::StatusOr<PjRtCompatibleDevice*> PjRtClient::LookupPjRtDevice(
    xla::PjRtDevice* pjrt_device) const {
  auto it = device_map_.find(pjrt_device);
  if (it == device_map_.end()) {
    return InvalidArgument("PjRtDevice not found: %s",
                           pjrt_device->DebugString());
  }
  return it->second;
}

absl::StatusOr<PjRtCompatibleMemory*> PjRtClient::LookupPjRtMemory(
    xla::PjRtMemorySpace* pjrt_memory) const {
  auto it = memory_map_.find(pjrt_memory);
  if (it == memory_map_.end()) {
    return InvalidArgument("PjRtMemorySpace not found: %s",
                           pjrt_memory->DebugString());
  }
  return it->second;
}

absl::StatusOr<xla::PjRtGlobalDeviceId> PjRtClient::GetPjRtGlobalDeviceId(
    DeviceId device_id) const {
  auto it = ifrt_device_id_to_pjrt_global_device_id_.find(device_id);
  if (it == ifrt_device_id_to_pjrt_global_device_id_.end()) {
    return InvalidArgument(
        "Unknown PjRt global device ID for IFRT device ID %d",
        device_id.value());
  }
  return it->second;
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
                      pjrt_client_->LookupAddressableDevice(
                          xla::PjRtLocalDeviceId(local_hardware_id)));
  return LookupPjRtDevice(pjrt_device);
}

const AttributeMap& PjRtClient::Attributes() const { return attributes_; }

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
  if (!llvm::isa<const SingleDeviceSharding>(sharding.get()) &&
      !sharding->IsFullyReplicated()) {
    return InvalidArgument(
        "Only SingleDeviceSharding or fully-replicated sharding is supported: "
        "sharding=%s",
        sharding->DebugString());
  }
  TF_ASSIGN_OR_RETURN(auto primitive_type, ToPrimitiveType(dtype));

  auto count = std::make_shared<std::atomic<int>>(sharding->devices()->size());
  std::function<void()> on_done_with_host_buffer_per_device;
  if (on_done_with_host_buffer) {
    on_done_with_host_buffer_per_device =
        [on_done_with_host_buffer = std::move(on_done_with_host_buffer),
         count]() {
          if (count->fetch_sub(1, std::memory_order_relaxed) == 1) {
            on_done_with_host_buffer();
          }
        };
  } else {
    on_done_with_host_buffer_per_device = []() {};
  }

  PjRtArray::PjRtBuffers buffers;
  buffers.reserve(sharding->devices()->size());
  for (xla::ifrt::Device* const device : sharding->devices()->devices()) {
    std::unique_ptr<PjRtBuffer> buffer;
    // If the sharding has memory_kind specified, use a version of
    // `PjRtClient::BufferFromHostBuffer` that accepts `PjRtMemorySpace`.
    // Otherwise, use a non-`PjRtMemorySpace` version that is compatible with
    // PjRt implementations without memories support.
    if (sharding->memory_kind().memory_kind().has_value()) {
      // Find `PjRtMemorySpace` that is associated with the sharding's device
      // and matches the sharding's memory_kind.
      Memory* memory = nullptr;
      for (Memory* ms : device->Memories()) {
        if (ms->Kind() == sharding->memory_kind()) {
          memory = ms;
          break;
        }
      }
      if (memory == nullptr) {
        return InvalidArgument(
            "Invalid memory kind: %s; available memory kinds: %s",
            *sharding->memory_kind().memory_kind(),
            absl::StrJoin(sharding->devices()->devices().front()->Memories(),
                          ", ", [](std::string* out, Memory* ms) {
                            absl::StrAppend(out, *ms->Kind().memory_kind());
                          }));
      }
      TF_ASSIGN_OR_RETURN(
          buffer, pjrt_client_->BufferFromHostBuffer(
                      data, primitive_type, shape.dims(), byte_strides,
                      semantics, on_done_with_host_buffer_per_device,
                      tensorflow::down_cast<PjRtMemory*>(memory)->pjrt_memory(),
                      /*device_layout=*/nullptr));
    } else {
      if (!device->IsAddressable()) {
        return InvalidArgument("Cannot copy array to non-addressable device %s",
                               device->DebugString());
      }
      TF_ASSIGN_OR_RETURN(
          buffer,
          pjrt_client_->BufferFromHostBuffer(
              data, primitive_type, shape.dims(), byte_strides, semantics,
              on_done_with_host_buffer_per_device,
              tensorflow::down_cast<PjRtDevice*>(device)->pjrt_device()));
    }
    buffers.push_back(std::move(buffer));
  }
  return PjRtArray::Create(this, dtype, std::move(shape), std::move(sharding),
                           std::move(buffers));
}

absl::StatusOr<tsl::RCReference<Array>>
PjRtClient::AssembleArrayFromSingleDeviceArrays(
    Shape shape, std::shared_ptr<const Sharding> sharding,
    absl::Span<tsl::RCReference<Array>> arrays, ArrayCopySemantics semantics) {
  DCHECK(this);
  return AssembleArrayFromSingleDeviceArrays(
      std::move(shape), std::move(sharding), arrays, semantics,
      SingleDeviceShardSemantics::kAllShards);
}

absl::StatusOr<tsl::RCReference<Array>>
PjRtClient::AssembleArrayFromSingleDeviceArrays(
    Shape shape, std::shared_ptr<const Sharding> sharding,
    absl::Span<tsl::RCReference<Array>> arrays,
    ArrayCopySemantics array_copy_semantics,
    SingleDeviceShardSemantics single_device_shard_semantics) {
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
  if (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards &&
      !sharding->devices()->IsFullyAddressable()) {
    return InvalidArgument(
        "All shards are requested but the sharding has non-addressable "
        "devices: %v",
        *sharding->devices());
  }
  if (sharding->devices()->AddressableDeviceList()->size() != arrays.size()) {
    return InvalidArgument(
        "Number of addressable output shards must match the number of "
        "single-shard arrays: %d vs. %d",
        sharding->devices()->AddressableDeviceList()->size(), arrays.size());
  }
  if (arrays[0]->dtype().kind() == DType::kString) {
    return AssembleStringArrayFromSingleDeviceStringArrays(
        shape, sharding, arrays, array_copy_semantics,
        single_device_shard_semantics);
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
    if (array->sharding().devices()->size() != 1) {
      return InvalidArgument(
          "Every input must use a single device sharding, but input %d has "
          "sharding=%s",
          i, array->sharding().DebugString());
    }
    switch (array_copy_semantics) {
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

absl::StatusOr<std::vector<tsl::RCReference<Array>>> PjRtClient::CopyArrays(
    absl::Span<tsl::RCReference<Array>> arrays,
    std::optional<tsl::RCReference<DeviceList>> devices,
    std::optional<MemoryKind> memory_kind, ArrayCopySemantics semantics) {
  if (arrays.empty()) {
    return std::vector<tsl::RCReference<Array>>();
  }

  for (int i = 1; i < arrays.size(); ++i) {
    const auto& sharding = arrays[i]->sharding();
    if (*sharding.devices() != *arrays[0]->sharding().devices() ||
        sharding.memory_kind() != arrays[0]->sharding().memory_kind()) {
      return absl::InvalidArgumentError(
          "CopyArrays only supports arrays with the same device list and "
          "memory kind");
    }
  }

  std::vector<tsl::RCReference<Array>> new_arrays;
  new_arrays.reserve(arrays.size());
  for (const auto& array : arrays) {
    if (auto* const pjrt_array = llvm::dyn_cast<PjRtArray>(array.get())) {
      TF_ASSIGN_OR_RETURN(new_arrays.emplace_back(),
                          pjrt_array->Copy(devices, memory_kind, semantics));
    } else if (auto* const string_array =
                   llvm::dyn_cast<BasicStringArray>(array.get())) {
      TF_ASSIGN_OR_RETURN(new_arrays.emplace_back(),
                          string_array->Copy(devices, memory_kind, semantics));
    } else {
      return absl::InvalidArgumentError(
          "Unsupported array type for PjRtClient::CopyArrays");
    }
  }
  return new_arrays;
}

absl::StatusOr<std::vector<tsl::RCReference<xla::ifrt::Array>>>
PjRtClient::RemapArrays(const RemapPlan& plan,
                        absl::Span<tsl::RCReference<xla::ifrt::Array>> arrays,
                        ArrayCopySemantics semantics) {
  return PjRtCompatibleClientRemapArrays(this, plan, arrays, semantics);
}

Future<> PjRtClient::GetReadyFuture(
    absl::Span<const tsl::RCReference<Value>> values) {
  absl::InlinedVector<Future<>, 1> futures;
  futures.reserve(values.size());
  for (const auto& value : values) {
    futures.push_back(value->GetReadyFuture());
  }
  return JoinFutures(futures);
}

absl::StatusOr<tsl::RCReference<Tuple>> PjRtClient::MakeTuple(
    absl::Span<tsl::RCReference<Value>> values) {
  return PjRtTuple::Create(this, values);
}

absl::StatusOr<std::shared_ptr<Topology>> PjRtClient::GetTopologyForDevices(
    const tsl::RCReference<xla::ifrt::DeviceList>& devices) const {
  // TODO(parkers): Consider constructing a sub-slice topology based on the
  // provided devices.
  TF_ASSIGN_OR_RETURN(auto topology, pjrt_client_->GetTopologyDescription());
  return std::make_shared<PjRtTopology>(
      std::shared_ptr<const xla::PjRtTopologyDescription>(pjrt_client_,
                                                          topology));
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
