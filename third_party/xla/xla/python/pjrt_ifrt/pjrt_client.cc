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
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/distributed/coordination/coordination_service_agent.h"
#include "xla/pjrt/distributed/protocol.pb.h"
#include "xla/pjrt/distributed/topology_util.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/client_impl_util.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/layout.h"
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
#include "xla/python/pjrt_ifrt/pjrt_layout.h"
#include "xla/python/pjrt_ifrt/pjrt_memory.h"
#include "xla/python/pjrt_ifrt/pjrt_remap.h"
#include "xla/python/pjrt_ifrt/pjrt_topology.h"
#include "xla/python/pjrt_ifrt/pjrt_tuple.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "xla/runtime/device_id.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/distributed_runtime/call_options.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

namespace xla {
namespace ifrt {
namespace {

// Returns an `AttributeMap` with the attributes of the given `PjRtClient`.
AttributeMap MakeAttributeMap(xla::PjRtClient* pjrt_client) {
  absl::flat_hash_map<std::string, PjRtValueType> attributes;
  attributes.insert({"supports_executable_serialization", true});
  attributes.insert({"serialize_with_sdy", PjRtValueType(true)});
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
          // OSS requires explicit string conversion
          // NOLINTNEXTLINE(*-redundant-string-conversions)
          std::string(pjrt_client->addressable_devices()[0]->device_kind()));

      // TODO(hyeontaek): Take optional device->partition_index mapping in
      // GlobalDeviceMapping and generate the `partition_index` attribute for
      // both addressable and non-addressable devices.
      if (pjrt_device == nullptr) {
        device.set_to_string("NonAddressable");
        device.set_debug_string("NonAddressable");
      } else {
        // OSS requires explicit string conversion
        // NOLINTBEGIN(*-redundant-string-conversions)
        device.set_to_string(std::string(pjrt_device->ToString()));
        device.set_debug_string(std::string(pjrt_device->DebugString()));
        // NOLINTEND(*-redundant-string-conversions)
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
    // OSS requires explicit string conversion
    // NOLINTBEGIN(*-redundant-string-conversions)
    device_proto.set_device_kind(
        std::string(device->description().device_kind()));
    device_proto.set_to_string(std::string(device->ToString()));
    device_proto.set_debug_string(std::string(device->DebugString()));
    // NOLINTEND(*-redundant-string-conversions)
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

  // Some PJRT implementations (e.g., GPU) assign their own "partition_index"
  // values. If these are present, leave them alone. Otherwise, we assign
  // the same partition_index to all devices of the same host, as determined by
  // the boot_id.
  int next_partition_index = 0;
  absl::flat_hash_map<std::string, int> boot_id_to_partition_index;
  for (int process_index = 0;
       process_index < global_topology.global_topology_proto.nodes_size();
       ++process_index) {
    const LocalTopologyProto& node =
        global_topology.global_topology_proto.nodes(process_index);
    int64_t partition_index = -1;
    if (!node.boot_id().empty()) {
      // Every new boot_id seen is treated as a new host/partition.
      absl::string_view boot_id = node.boot_id();
      auto [it, inserted] =
          boot_id_to_partition_index.try_emplace(boot_id, next_partition_index);
      partition_index = it->second;
      if (inserted) {
        ++next_partition_index;
      }
    }

    std::string platform_name(pjrt_client->platform_name());
    const bool node_is_me = process_index == global_topology.my_process_index;
    for (const DeviceProto& device_proto : node.devices()) {
      absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes;
      TF_RETURN_IF_ERROR(
          DeserializePjRtDeviceAttributes(device_proto, attributes));
      if (partition_index != -1) {
        // Sets a generated `partition_index` attribute if not already present.
        attributes.insert(
            {"partition_index",
             xla::PjRtDeviceAttribute(static_cast<int64_t>(partition_index))});
        // TODO - b/435521225: `slice_index` is deprecated. Use
        // `partition_index`, which better aligns with NVIDIA's terminology.
        attributes.insert(
            {"slice_index",
             xla::PjRtDeviceAttribute(static_cast<int64_t>(partition_index))});
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
          client, ifrt_device_id, platform_name, device_proto.device_kind(),
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

absl::StatusOr<ArrayRef> MakeStringArrayFromHostBuffer(
    Client* client, const void* data, DType dtype, Shape shape,
    std::optional<absl::Span<const int64_t>> byte_strides, ShardingRef sharding,
    Client::HostBufferSemantics semantics,
    std::function<void()> on_done_with_host_buffer) {
  auto param_validation = [&]() -> absl::Status {
    if (byte_strides.has_value()) {
      return absl::InvalidArgumentError(
          "byte_strides is not currently supported for making "
          "BasicStringArrays.");
    }
    if (!(semantics == Client::HostBufferSemantics::kImmutableOnlyDuringCall ||
          semantics ==
              Client::HostBufferSemantics::kImmutableUntilTransferCompletes)) {
      return absl::InvalidArgumentError(
          "HostBufferSemantics other than kImmutableOnlyDuringCall and "
          "kImmutableUntilTransferCompletes are not "
          "currently supported for making BasicStringArrays.");
    }
    if (!sharding->IsFullyReplicated()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Only fully replicated shardings are supported for making "
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
      tsl::Future<BasicStringArray::Buffers>(std::move(buffers)),
      std::move(buffer_releaser));
}

absl::StatusOr<ArrayRef> AssembleStringArrayFromSingleDeviceStringArrays(
    PjRtClient* client, Shape shape, ShardingRef sharding,
    absl::Span<ArrayRef> arrays, ArrayCopySemantics array_copy_semantics,
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

  auto [buffers_promise, buffers_future] =
      tsl::MakePromise<BasicStringArray::Buffers>();

  auto buffer_copier = [state = buffer_copying_state,
                        promise = std::move(buffers_promise).ToShared()](
                           absl::StatusOr<BasicStringArray::Buffers> strbuf,
                           int shard_index) mutable {
    absl::MutexLock lock(state->mu);
    if (state->num_buffers_to_copy == 0) {
      // Nothing to be done. We get here if the buffers of a single
      // device array became ready with a an error previously.
      return;
    }
    if (!strbuf.ok()) {
      promise->Set(strbuf.status());
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
    promise->Set(std::move(state->buffers));
  };

  for (int i = 0; i < arrays.size(); ++i) {
    auto basic_string_array = llvm::dyn_cast<BasicStringArray>(arrays[i].get());
    if (!basic_string_array) {
      return absl::InvalidArgumentError(
          "All single device arrays must be BasicStringArrays");
    }

    if (basic_string_array->sharding().devices()->size() != 1) {
      return absl::InvalidArgumentError(
          absl::StrFormat("All single device arrays must have single device "
                          "sharding. got: %s for shard index: %d",
                          basic_string_array->sharding().DebugString(), i));
    }

    basic_string_array->buffers().OnReady(
        [shard_index = i, buffer_copier](
            absl::StatusOr<BasicStringArray::Buffers> strbuf) mutable {
          buffer_copier(std::move(strbuf), shard_index);
        });
  }

  return BasicStringArray::Create(client, std::move(shape), std::move(sharding),
                                  buffers_future,
                                  std::move(on_done_with_buffer));
}

// Copies the buffer at the given index in each array to a buffer on the given
// device.
absl::StatusOr<std::vector<std::shared_ptr<PjRtBuffer>>>
CopyPjRtBuffersToLocalDevice(int index, absl::Span<ArrayRef> arrays,
                             Device* dst_device,
                             std::optional<MemoryKind> memory_kind,
                             ArrayCopySemantics semantics) {
  std::vector<std::shared_ptr<PjRtBuffer>> buffers;
  buffers.reserve(arrays.size());
  for (ArrayRef& array : arrays) {
    if (auto* const pjrt_array = llvm::dyn_cast<PjRtArray>(array.get());
        pjrt_array != nullptr) {
      TF_ASSIGN_OR_RETURN(std::shared_ptr<PjRtBuffer> buffer,
                          pjrt_array->CopySinglePjRtBuffer(
                              index, dst_device, memory_kind, semantics));
      buffers.push_back(std::move(buffer));
    } else {
      return absl::InvalidArgumentError(
          "Unsupported array type for CopyPjRtBuffersToLocalDevice");
    }
  }
  return buffers;
}

const char kKeyPrefix[] = "ifrt_cross_host_transfer_";

}  // namespace

char PjRtCompatibleClient::ID = 0;
char PjRtClient::ID = 0;

absl::StatusOr<tsl::RCReference<PjRtCompatibleArray>>
PjRtCompatibleClient::CreatePjRtArray(std::shared_ptr<PjRtBuffer> pjrt_buffer) {
  return CreatePjRtArray(std::move(pjrt_buffer), /*has_custom_layout=*/true);
}

absl::StatusOr<tsl::RCReference<PjRtCompatibleArray>>
PjRtCompatibleClient::CreatePjRtArray(Shape shape, PjRtBuffers pjrt_buffers) {
  return CreatePjRtArray(std::move(shape), std::move(pjrt_buffers),
                         /*has_custom_layout=*/true);
}

absl::StatusOr<std::unique_ptr<PjRtClient>> PjRtClient::Create(
    PjRtClient::CreateOptions options) {
  auto client =
      absl::WrapUnique(new PjRtClient(std::move(options.pjrt_client)));
  xla::PjRtClient* pjrt_client = client->pjrt_client();

  GlobalTopology global_topology;
  if (!options.kv_store || !options.use_kv_store_for_topology_exchange) {
    TF_ASSIGN_OR_RETURN(global_topology,
                        MakeGlobalTopologyFromPjRtClient(pjrt_client, options));
  } else {
    if (options.global_device_mapping.has_value()) {
      return InvalidArgument(
          "global_device_mapping and kv_store cannot be set at the same time "
          "if use_kv_store_for_topology_exchange is true.");
    }
    // If a KV-store was provided and `use_kv_store_for_topology_exchange` is
    // true, we perform a topology exchange to aggregate topology information
    // from all processes.
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

  // For non-addressable devices, pjrt_device is null, so the default memory is
  // set to that of an addressable device.
  auto default_memory = client->addressable_devices_.front()->DefaultMemory();
  for (const auto& device : client->owned_devices_) {
    if (!device->pjrt_device()) {
      for (const Memory* memory :
           client->addressable_devices_.front()->Memories()) {
        auto ifrt_memory = std::make_unique<PjRtMemory>(
            client.get(), memory->Kind(), device.get());
        device->memories_.push_back(ifrt_memory.get());
        if (absl::IsUnknown(device->default_memory_.status())) {
          if (default_memory.ok()) {
            if (memory == *default_memory) {
              device->default_memory_ = ifrt_memory.get();
            }
          } else {
            device->default_memory_ = default_memory.status();
          }
        }
        client->owned_memories_.push_back(std::move(ifrt_memory));
      }
    }
  }

  client->distributed_client_ = std::move(options.distributed_client);
  client->kv_store_ = std::move(options.kv_store);
  client->cross_host_transfer_timeout_ = options.cross_host_transfer_timeout;
  client->transfer_server_factory_ = std::move(options.transfer_server_factory);
  client->force_dcn_cross_host_transfers_ =
      options.force_dcn_cross_host_transfers;

  if (client->pjrt_client()->plugin_attributes().has_value()) {
    auto attrs = client->pjrt_client()->plugin_attributes()->attributes;
    if (attrs.contains("supports_cross_host_transfers")) {
      client->pjrt_supports_cross_host_transfers_ =
          std::get<bool>(attrs.at("supports_cross_host_transfers"));
    }
  }

  // Start a background thread to monitor the status of all processes.
  if (client->distributed_client_) {
    absl::StatusOr<xla::CoordinationServiceAgent*> agent =
        client->distributed_client_->GetCoordinationServiceAgent();
    if (agent.ok()) {
      client->global_process_info_thread_.reset(
          tsl::Env::Default()->StartThread(
              tsl::ThreadOptions(), "global_process_info",
              [client = client.get(), agent = *agent]() {
                absl::Status s = client->WatchGlobalProcessInfo(*agent);
                if (!s.ok()) {
                  LOG(ERROR) << s;
                }
              }));
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

static int NumCompilationThreads(xla::PjRtPlatformId platform_id) {
  if (platform_id == xla::CudaId()) {
    // Disable asynchronous compilation on GPUs since sharded autotuning may
    // require in-order compilation.
    return 0;
  }
  return 8;
}

PjRtClient::PjRtClient(std::shared_ptr<xla::PjRtClient> pjrt_client)
    : pjrt_client_(std::move(pjrt_client)),
      default_compiler_(this,
                        NumCompilationThreads(pjrt_client_->platform_id())),
      attributes_(MakeAttributeMap(pjrt_client_.get())) {}

PjRtClient::~PjRtClient() {
  absl::MutexLock lock(shutting_down_mu_);
  shutting_down_ = true;
}

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

absl::StatusOr<DeviceListRef> PjRtClient::MakeDeviceList(
    absl::Span<Device* const> devices) const {
  return xla::ifrt::BasicDeviceList::Create(devices);
}

const AttributeMap& PjRtClient::Attributes() const { return attributes_; }

absl::StatusOr<tsl::RCReference<PjRtCompatibleArray>>
PjRtClient::CreatePjRtArray(std::shared_ptr<PjRtBuffer> pjrt_buffer,
                            bool has_custom_layout) {
  TF_ASSIGN_OR_RETURN(
      auto array,
      PjRtArray::Create(this, std::move(pjrt_buffer), has_custom_layout));
  return tsl::RCReference<PjRtCompatibleArray>(std::move(array));
}

absl::StatusOr<tsl::RCReference<PjRtCompatibleArray>>
PjRtClient::CreatePjRtArray(Shape shape, PjRtBuffers pjrt_buffers,
                            bool has_custom_layout) {
  std::shared_ptr<const xla::PjRtLayout> layout;
  TF_ASSIGN_OR_RETURN(auto array, PjRtArray::Create(this, std::move(shape),
                                                    std::move(pjrt_buffers),
                                                    has_custom_layout));
  return tsl::RCReference<PjRtCompatibleArray>(std::move(array));
}

absl::StatusOr<ArrayRef> PjRtClient::MakeArrayFromHostBuffer(
    const void* data, DType dtype, Shape shape,
    std::optional<absl::Span<const int64_t>> byte_strides, ShardingRef sharding,
    LayoutRef layout, Client::HostBufferSemantics semantics,
    std::function<void()> on_done_with_host_buffer) {
  DCHECK(this);
  if (dtype.kind() == DType::kString) {
    if (layout != nullptr) {
      return InvalidArgument(
          "String arrays do not support custom layouts: layout=%v", *layout);
    }
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

  absl::Span<xla::ifrt::Device* const> ifrt_addressable_devices =
      sharding->devices()->AddressableDeviceList()->devices();
  auto count =
      std::make_shared<std::atomic<int>>(ifrt_addressable_devices.size());
  if (ifrt_addressable_devices.empty()) {
    return InvalidArgument("Cannot copy array to non-addressable device: %s",
                           sharding->devices()->DebugString());
  }
  std::shared_ptr<const xla::PjRtLayout> pjrt_layout;
  const xla::Layout* xla_layout;
  if (layout == nullptr) {
    xla_layout = nullptr;
  } else {
    TF_ASSIGN_OR_RETURN(Shape shard_shape, sharding->GetShardShape(shape));
    TF_ASSIGN_OR_RETURN(pjrt_layout, ToPjRtLayout(dtype, shard_shape, layout));
    xla_layout = &pjrt_layout->xla_layout();
  }
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
  buffers.reserve(ifrt_addressable_devices.size());
  for (xla::ifrt::Device* const device : ifrt_addressable_devices) {
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
            absl::StrJoin(ifrt_addressable_devices.front()->Memories(), ", ",
                          [](std::string* out, Memory* ms) {
                            absl::StrAppend(out, *ms->Kind().memory_kind());
                          }));
      }
      TF_ASSIGN_OR_RETURN(
          buffer, pjrt_client_->BufferFromHostBuffer(
                      data, primitive_type, shape.dims(), byte_strides,
                      semantics, on_done_with_host_buffer_per_device,
                      tensorflow::down_cast<PjRtMemory*>(memory)->pjrt_memory(),
                      xla_layout));
    } else {
      TF_ASSIGN_OR_RETURN(xla::PjRtMemorySpace * memory_space,
                          tensorflow::down_cast<PjRtDevice*>(device)
                              ->pjrt_device()
                              ->default_memory_space());
      TF_ASSIGN_OR_RETURN(
          buffer,
          pjrt_client_->BufferFromHostBuffer(
              data, primitive_type, shape.dims(), byte_strides, semantics,
              on_done_with_host_buffer_per_device, memory_space, xla_layout));
    }
    buffers.push_back(std::move(buffer));
  }
  return PjRtArray::Create(this, dtype, std::move(shape), std::move(sharding),
                           std::move(buffers), std::move(pjrt_layout));
}

absl::StatusOr<std::vector<ArrayRef>>
PjRtClient::MakeArraysFromHostBufferShards(
    absl::Span<MakeArraysFromHostBufferShardsSpec> specs,
    HostBufferSemantics semantics) {
  return ClientMakeArraysFromHostBufferShards(this, specs, semantics);
}

absl::StatusOr<std::vector<ArrayRef>> PjRtClient::MakeErrorArrays(
    const absl::Status& error, absl::Span<const ArraySpec> array_specs) {
  if (error.ok()) {
    return absl::InvalidArgumentError("Error status must not be OK");
  }
  DCHECK(this);
  std::vector<ArrayRef> arrays;
  arrays.reserve(array_specs.size());
  for (const auto& array_spec : array_specs) {
    if (array_spec.dtype.kind() == DType::kString) {
      TF_ASSIGN_OR_RETURN(arrays.emplace_back(),
                          BasicStringArray::Create(
                              this, array_spec.shape, array_spec.sharding,
                              tsl::Future<BasicStringArray::Buffers>(error),
                              /*on_done_with_buffer=*/[]() {}));
      continue;
    }

    TF_ASSIGN_OR_RETURN(auto primitive_type, ToPrimitiveType(array_spec.dtype));
    absl::Span<xla::ifrt::Device* const> ifrt_addressable_devices =
        array_spec.sharding->devices()->AddressableDeviceList()->devices();
    TF_ASSIGN_OR_RETURN(Shape shard_shape,
                        array_spec.sharding->GetShardShape(array_spec.shape));
    xla::Shape xla_shape =
        xla::ShapeUtil::MakeShape(primitive_type, shard_shape.dims());

    PjRtArray::PjRtBuffers buffers;
    buffers.reserve(ifrt_addressable_devices.size());
    for (xla::ifrt::Device* const device : ifrt_addressable_devices) {
      std::unique_ptr<PjRtBuffer> buffer;
      // Find `PjRtMemorySpace` that is associated with the sharding's device
      // and matches the sharding's memory_kind.
      Memory* memory = nullptr;
      for (Memory* ms : device->Memories()) {
        if (ms->Kind() == array_spec.sharding->memory_kind()) {
          memory = ms;
          break;
        }
      }
      if (memory == nullptr) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Invalid memory kind: %s; available memory kinds: %s",
            *array_spec.sharding->memory_kind().memory_kind(),
            absl::StrJoin(ifrt_addressable_devices.front()->Memories(), ", ",
                          [](std::string* out, Memory* ms) {
                            absl::StrAppend(out, *ms->Kind().memory_kind());
                          })));
      }
      TF_ASSIGN_OR_RETURN(
          buffers.emplace_back(),
          pjrt_client_->CreateErrorBuffer(
              error, xla_shape,
              tensorflow::down_cast<PjRtMemory*>(memory)->pjrt_memory()));
    }
    TF_ASSIGN_OR_RETURN(
        arrays.emplace_back(),
        PjRtArray::Create(this, array_spec.dtype, std::move(shard_shape),
                          array_spec.sharding, std::move(buffers),
                          array_spec.layout));
  }
  return arrays;
}

absl::StatusOr<ArrayRef> PjRtClient::AssembleArrayFromSingleDeviceArrays(
    DType dtype, Shape shape, ShardingRef sharding, absl::Span<ArrayRef> arrays,
    ArrayCopySemantics array_copy_semantics,
    SingleDeviceShardSemantics single_device_shard_semantics) {
  DCHECK(this);
  if (!arrays.empty() &&
      llvm::isa<const SingleDeviceSharding>(sharding.get())) {
    // Assemble with SingleDeviceSharding is No-op.
    if (arrays.size() != 1) {
      return InvalidArgument(
          "When the sharding is SingleDeviceSharding, the input arrays size "
          "must be one, but the actual size is %d",
          arrays.size());
    }
    return arrays[0];
  } else if (!llvm::isa<const SingleDeviceSharding, const OpaqueSharding,
                        const ConcreteSharding, const ConcreteEvenSharding,
                        const ShardingParamSharding, const HloSharding>(
                 sharding.get())) {
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
  if (dtype.kind() == DType::kString) {
    return AssembleStringArrayFromSingleDeviceStringArrays(
        this, shape, sharding, arrays, array_copy_semantics,
        single_device_shard_semantics);
  }
  PjRtArray::PjRtBuffers buffers;
  buffers.reserve(arrays.size());
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
  // TODO(emilyaf): Remove the following logic once layout is plumbed through.
  std::shared_ptr<const xla::PjRtLayout> layout;
  if (!arrays.empty()) {
    TF_ASSIGN_OR_RETURN(layout, arrays.front()->pjrt_layout());
  }
  return PjRtArray::Create(this, dtype, std::move(shape), std::move(sharding),
                           std::move(buffers), std::move(layout));
}

absl::StatusOr<std::vector<ArrayRef>> PjRtClient::CopyArrays(
    absl::Span<ArrayRef> arrays, std::optional<DeviceListRef> devices,
    std::optional<MemoryKind> memory_kind, ArrayCopySemantics semantics) {
  if (arrays.empty()) {
    return std::vector<ArrayRef>();
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

  DeviceListRef src_devices = arrays[0]->sharding().devices();
  DeviceListRef dst_devices;
  bool all_host_local_transfers = true;
  if (devices.has_value()) {
    dst_devices = *devices;
    if (src_devices->size() != dst_devices->size()) {
      return absl::InvalidArgumentError(
          "CopyArrays only supports destination device list of the same size "
          "as the array device lists.");
    };
    if (src_devices->size() > 0 && (src_devices->devices()[0]->client() ==
                                    dst_devices->devices()[0]->client())) {
      for (int i = 0; i < dst_devices->size(); ++i) {
        if (dst_devices->devices()[i]->ProcessIndex() !=
            src_devices->devices()[i]->ProcessIndex()) {
          all_host_local_transfers = false;
          break;
        }
      }
    }
  }

  if (all_host_local_transfers) {
    std::vector<ArrayRef> new_arrays;
    new_arrays.reserve(arrays.size());
    for (const ArrayRef& array : arrays) {
      if (auto* const pjrt_array = llvm::dyn_cast<PjRtArray>(array.get())) {
        TF_ASSIGN_OR_RETURN(new_arrays.emplace_back(),
                            pjrt_array->Copy(devices, memory_kind, semantics));
      } else if (auto* const string_array =
                     llvm::dyn_cast<BasicStringArray>(array.get())) {
        TF_ASSIGN_OR_RETURN(
            new_arrays.emplace_back(),
            string_array->Copy(devices, memory_kind, semantics));
      } else {
        return absl::InvalidArgumentError(
            "Unsupported array type for PjRtClient::CopyArrays");
      }
    }
    return new_arrays;
  }
  if (pjrt_supports_cross_host_transfers_ && !force_dcn_cross_host_transfers_) {
    return CopyArraysForCrossHost(arrays, src_devices, dst_devices, memory_kind,
                                  semantics);
  }
  if (transfer_server_factory_ != nullptr) {
    return CopyArraysForCrossHostFallback(arrays, src_devices, dst_devices,
                                          memory_kind);
  }
  return absl::UnimplementedError(
      "Cross-host transfers are not supported by this backend. Set the "
      "`--jax_cross_host_transfer_socket_address` flag to enable DCN transfers "
      "on Linux for any backend.");
}

absl::StatusOr<std::vector<xla::ifrt::ArrayRef>>
PjRtClient::CopyArraysForCrossHost(absl::Span<ArrayRef> arrays,
                                   DeviceListRef src_devices,
                                   DeviceListRef dst_devices,
                                   std::optional<MemoryKind> memory_kind,
                                   ArrayCopySemantics semantics) {
  std::vector<ArrayRef> new_arrays;
  new_arrays.reserve(arrays.size());
  std::vector<std::vector<std::shared_ptr<PjRtBuffer>>> recv_buffers;
  recv_buffers.reserve(dst_devices->AddressableDeviceList()->size());
  auto on_send_done = [](absl::Status status) {
    if (!status.ok()) {
      LOG(ERROR) << "xla::PjRtClient::CrossHostSendBuffers failed: " << status;
    }
  };
  auto on_recv_done = [](absl::Status status) {
    if (!status.ok()) {
      LOG(ERROR) << "Cross-host receive buffer failed: " << status;
    }
  };
  int j = 0;  // Counter for the addressable buffers.
  for (int i = 0; i < dst_devices->size(); ++i) {
    // TODO(emilyaf): Extend CreateNewTransferKey to take N and return N keys
    // as a performance optimization.
    std::vector<CrossHostTransferKey> transfer_keys;
    transfer_keys.reserve(arrays.size());
    for (int k = 0; k < arrays.size(); ++k) {
      transfer_keys.push_back(CreateNewTransferKey());
    }

    if (src_devices->devices()[i]->IsAddressable()) {
      // Create send buffers.
      std::vector<PjRtBuffer*> send_buffers;
      send_buffers.reserve(arrays.size());
      for (ArrayRef& array : arrays) {
        if (auto* const pjrt_array = llvm::dyn_cast<PjRtArray>(array.get())) {
          auto buffers = pjrt_array->pjrt_buffers();
          send_buffers.push_back(buffers[j].get());
        } else {
          // TODO(emilyaf): Support string arrays.
          return absl::InvalidArgumentError(
              "Unsupported array type for cross-host "
              "PjRtClient::CopyArraysForCrossHost");
        }
      }

      if (dst_devices->devices()[i]->IsAddressable()) {
        // This transfer is between two addressable devices.
        TF_ASSIGN_OR_RETURN(
            recv_buffers.emplace_back(),
            CopyPjRtBuffersToLocalDevice(j, arrays, dst_devices->devices()[i],
                                         memory_kind, semantics));
      } else {
        // Create vector of (remote) dst devices; we send each array to
        // dst_devices->devices()[i].
        TF_ASSIGN_OR_RETURN(
            xla::PjRtGlobalDeviceId dst_global_device_id,
            GetPjRtGlobalDeviceId(dst_devices->devices()[i]->Id()));
        std::vector<PjRtGlobalDeviceId> dst_global_device_ids(
            arrays.size(), dst_global_device_id);

        // If the PJRT plugin implements the `CrossHostSendBuffers` API, use it.
        // Otherwise, call this class's `CrossHostSendBuffers` method to use the
        // plugin's `CopyToRemoteDevice` API, getting the buffer descriptors
        // from the KV store.
        absl::StatusOr<std::vector<Future<>>> send_futures =
            pjrt_client_->CrossHostSendBuffers(
                send_buffers, std::move(dst_global_device_ids), transfer_keys);
        if (send_futures.ok()) {
          for (Future<>& send_future : *send_futures) {
            send_future.OnReady(on_send_done);
          }
        } else if (absl::IsUnimplemented(send_futures.status())) {
          TF_RETURN_IF_ERROR(
              CrossHostSendBuffers(send_buffers, std::move(transfer_keys)));
        } else {
          return send_futures.status();
        }
      }
      ++j;
    } else if (dst_devices->devices()[i]->IsAddressable()) {
      // Create vector of shapes to receive.
      std::vector<xla::Shape> recv_shapes;
      recv_shapes.reserve(arrays.size());
      for (const ArrayRef& array : arrays) {
        if (auto* const pjrt_array = llvm::dyn_cast<PjRtArray>(array.get())) {
          TF_ASSIGN_OR_RETURN(xla::PrimitiveType dtype,
                              ToPrimitiveType(pjrt_array->dtype()));
          TF_ASSIGN_OR_RETURN(
              Shape shard_shape,
              pjrt_array->sharding().GetShardShape(pjrt_array->shape()));
          xla::Shape recv_shape =
              xla::ShapeUtil::MakeShape(dtype, shard_shape.dims());
          recv_shapes.push_back(std::move(recv_shape));
        } else {
          return absl::InvalidArgumentError(
              "Unsupported array type for cross-host "
              "PjRtClient::CopyArraysForCrossHost");
        }
      }

      // Get the dst device we receive into.
      TF_ASSIGN_OR_RETURN(
          xla::PjRtGlobalDeviceId pjrt_global_device_id,
          GetPjRtGlobalDeviceId(dst_devices->devices()[i]->Id()));
      TF_ASSIGN_OR_RETURN(xla::PjRtDevice * pjrt_device,
                          pjrt_client_->LookupDevice(pjrt_global_device_id));

      // Create vector of src devices; we receive each array from
      // src_devices->devices()[i].
      TF_ASSIGN_OR_RETURN(
          xla::PjRtGlobalDeviceId src_global_device_id,
          GetPjRtGlobalDeviceId(src_devices->devices()[i]->Id()));
      std::vector<PjRtGlobalDeviceId> src_global_device_ids(
          arrays.size(), src_global_device_id);

      // If the PJRT plugin implements the `CrossHostReceiveBuffers` API, use
      // it. Otherwise, call this class's `CrossHostReceiveBuffers` method to
      // use the plugin's `MakeCrossHostReceiveBuffers` API, transmitting the
      // buffer descriptors via the KV store.
      absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
          received_buffers = pjrt_client_->CrossHostReceiveBuffers(
              pjrt_device, recv_shapes, std::move(src_global_device_ids),
              transfer_keys);
      if (absl::IsUnimplemented(received_buffers.status())) {
        TF_ASSIGN_OR_RETURN(received_buffers,
                            CrossHostReceiveBuffers(recv_shapes, pjrt_device,
                                                    std::move(transfer_keys)));
      }
      if (!received_buffers.ok()) {
        return received_buffers.status();
      }
      std::vector<std::shared_ptr<PjRtBuffer>> buffers;
      buffers.reserve(received_buffers->size());
      for (std::unique_ptr<PjRtBuffer>& buffer : *received_buffers) {
        buffer->GetReadyFuture().OnReady(on_recv_done);
        buffers.push_back(std::move(buffer));
      }
      recv_buffers.push_back(std::move(buffers));
    }
  }

  for (int i = 0; i < arrays.size(); ++i) {
    PjRtArray::PjRtBuffers new_buffers;
    new_buffers.reserve(dst_devices->AddressableDeviceList()->size());
    int k = 0;
    for (Device* device : dst_devices->devices()) {
      if (device->IsAddressable()) {
        new_buffers.push_back(std::move(recv_buffers[k++][i]));
      }
    }
    TF_ASSIGN_OR_RETURN(ShardingRef new_sharding,
                        arrays[i]->shared_ptr_sharding()->WithDeviceAssignment(
                            dst_devices, memory_kind));
    TF_ASSIGN_OR_RETURN(auto new_layout, arrays[i]->pjrt_layout());
    TF_ASSIGN_OR_RETURN(
        new_arrays.emplace_back(),
        PjRtArray::Create(this, arrays[i]->dtype(), arrays[i]->shape(),
                          std::move(new_sharding), std::move(new_buffers),
                          std::move(new_layout)));
  }
  return new_arrays;
}

absl::Status PjRtClient::InitializeTransferServer() {
  if (!transfer_server_.has_value()) {
    if (transfer_server_factory_ == nullptr) {
      return absl::FailedPreconditionError("Transfer server factory is null.");
    }
    TF_ASSIGN_OR_RETURN(transfer_server_,
                        transfer_server_factory_(pjrt_client_));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<xla::ifrt::ArrayRef>>
PjRtClient::CopyArraysForCrossHostFallback(
    absl::Span<ArrayRef> arrays, DeviceListRef src_devices,
    DeviceListRef dst_devices, std::optional<MemoryKind> memory_kind) {
  {
    absl::MutexLock lock(transfer_server_mu_);
    TF_RETURN_IF_ERROR(InitializeTransferServer());
  }
  return (*transfer_server_)
      ->CopyArraysForCrossHost(this, arrays, src_devices, dst_devices,
                               memory_kind);
}

CrossHostTransferKey PjRtClient::CreateNewTransferKey() {
  return CrossHostTransferKey(next_transfer_key_++);
}

absl::Status PjRtClient::WatchGlobalProcessInfo(
    xla::CoordinationServiceAgent& agent) {
  TF_ASSIGN_OR_RETURN(tensorflow::CoordinatedTask task, agent.GetOwnTask());
  VLOG(3) << "Watching global process info for task "
          << task.ShortDebugString();

  int64_t version_number = -1;  // latest job state version
  while (true) {
    // Call WatchJobStateAsync.
    VLOG(3) << "Calling WatchJobStateAsync for task " << task.ShortDebugString()
            << " with version number " << version_number;
    absl::StatusOr<tensorflow::WatchJobStateResponse> response;
    bool done = false;
    std::shared_ptr<tsl::CallOptions> call_opts = agent.WatchJobStateAsync(
        task.job_name(), version_number,
        [this, &response,
         &done](absl::StatusOr<tensorflow::WatchJobStateResponse> r) {
          response = std::move(r);
          absl::MutexLock lock(shutting_down_mu_);
          done = true;
        });

    {
      // Wait for the WatchJobStateAsync call to finish or for us to shut down,
      // whichever happens first.
      absl::MutexLock lock(shutting_down_mu_);
      auto done_or_shutting_down = [this, &done]() {
        shutting_down_mu_.AssertHeld();
        return done || shutting_down_;
      };
      shutting_down_mu_.Await(absl::Condition(&done_or_shutting_down));

      if (shutting_down_) {
        // Cancel the call the WatchJobStateAsync and wait for it to terminate.
        VLOG(3) << "WatchGlobalProcessInfo shutting down for task "
                << task.ShortDebugString();
        call_opts->StartCancel();
        shutting_down_mu_.Await(absl::Condition(&done));
        return absl::OkStatus();
      }

      if (!response.ok()) {
        // Sleep to avoid repeatedly issuing a request that fails immediately.
        //
        // TODO: mwhittaker - Perform exponential backoff.
        LOG(WARNING) << "WatchJobStateAsync failed for task "
                     << task.ShortDebugString() << ": " << response.status();
        shutting_down_mu_.AwaitWithTimeout(absl::Condition(&shutting_down_),
                                           absl::Seconds(1));
        continue;
      }
    }

    // Parse the response.
    version_number = response->version_number();
    std::vector<tensorflow::CoordinatedTaskStateInfo> state(
        response->task_state().begin(), response->task_state().end());
    absl::c_sort(state,
                 [](const tensorflow::CoordinatedTaskStateInfo& x,
                    const tensorflow::CoordinatedTaskStateInfo& y) -> bool {
                   return x.task().task_id() < y.task().task_id();
                 });

    // Pretty print the job state, if VLOG is on.
    if (VLOG_IS_ON(3)) {
      VLOG(3) << "Job state for task " << task.ShortDebugString() << ":";
      for (const auto& info : state) {
        VLOG(3) << "- " << info.DebugString();
      }
    }

    // Update the client with the job state.
    pjrt_client_->UpdateGlobalProcessInfo(absl::MakeSpan(state));
  }
}

absl::Status PjRtClient::CrossHostSendBuffers(
    std::vector<PjRtBuffer*> buffers,
    const std::vector<CrossHostTransferKey>& keys) {
  if (keys.size() != buffers.size()) {
    return absl::InternalError(
        "CrossHostSendBuffers: keys must be the same size as buffers.");
  }
  for (int i = 0; i < keys.size(); ++i) {
    auto [promise, descriptor_future] = tsl::MakePromise<std::string>();
    std::shared_ptr<tsl::CallOptions> call_opts = kv_store_->AsyncGet(
        absl::StrCat(kKeyPrefix, keys[i]),
        [promise = std::move(promise).ToShared()](
            const absl::StatusOr<std::string>& descriptor) mutable {
          promise->Set(descriptor);
        });
    if (call_opts == nullptr) {
      return absl::InternalError(
          "CrossHostSendBuffers: kv_store_->AsyncGet returned nullptr.");
    }
    xla::PjRtBuffer::RemoteSendCallback on_done =
        [call_opts = std::move(call_opts)](absl::Status status,
                                           bool sends_were_enqueued) {
          if (!status.ok()) {
            call_opts->StartCancel();
            LOG(ERROR) << "`xla::PjRtBuffer::CopyToRemoteDevice` failed: "
                       << status;
          }
          if (!sends_were_enqueued) {
            LOG(ERROR)
                << "`xla::PjRtBuffer::CopyToRemoteDevice` did not enqueue "
                   "sends.";
          }
        };
    buffers[i]->CopyToRemoteDevice(std::move(descriptor_future),
                                   std::move(on_done));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtClient::CrossHostReceiveBuffers(absl::Span<const xla::Shape> shapes,
                                    xla::PjRtDevice* device,
                                    std::vector<CrossHostTransferKey> keys) {
  auto notifier = [this, keys = std::move(keys)](
                      absl::StatusOr<xla::PjRtCrossHostRecvState> recv_state) {
    if (!recv_state.ok()) {
      LOG(FATAL) << "Invalid PjRtCrossHostRecvState passed to "
                    "xla::PjRtCrossHostRecvNotifier callback in "
                    "xla::PjRtClient::MakeCrossHostReceiveBuffers: "
                 << recv_state.status();
    }
    auto on_canceled = [](const absl::Status& status) {
      if (!status.ok()) {
        LOG(FATAL) << "Invalid status passed to `on_canceled` callback in "
                      "xla::PjRtCrossHostSendCancelNotifier in "
                      "xla::PjRtClient::MakeCrossHostReceiveBuffers: "
                   << status;
      }
    };
    if (recv_state->descriptors.size() != keys.size()) {
      absl::Status error_status = absl::InternalError(absl::StrFormat(
          "Descriptors must be the same size as keys. Descriptors: %d, "
          "keys: %d",
          recv_state->descriptors.size(), keys.size()));
      CHECK_NOTNULL(recv_state->cancel_notifier);
      for (auto& descriptor : recv_state->descriptors) {
        recv_state->cancel_notifier(descriptor.serialized_descriptors.front(),
                                    error_status, on_canceled);
      }
      return;
    }
    for (int i = 0, n = keys.size(); i < n; ++i) {
      std::string key = absl::StrCat(kKeyPrefix, keys[i].value());
      absl::Status kv_status = kv_store_->Set(
          key, recv_state->descriptors[i].serialized_descriptors.front());
      if (!kv_status.ok()) {
        CHECK_NOTNULL(recv_state->cancel_notifier);
        absl::Status error_status = absl::InternalError(absl::StrFormat(
            "Failed to set key %s: %s", key, kv_status.message()));
        recv_state->cancel_notifier(
            recv_state->descriptors[i].serialized_descriptors.front(),
            error_status, on_canceled);
        return;
      }
    }
  };
  return pjrt_client_->MakeCrossHostReceiveBuffers(shapes, device,
                                                   std::move(notifier));
}

absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> PjRtClient::RemapArrays(
    const RemapPlan& plan, absl::Span<xla::ifrt::ArrayRef> arrays,
    ArrayCopySemantics semantics) {
  return PjRtCompatibleClientRemapArrays(this, plan, arrays, semantics);
}

absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> PjRtClient::ReshardArrays(
    absl::Span<ArrayRef> arrays, absl::Span<const ArraySpec> specs,
    ArrayCopySemantics semantics) {
  return Unimplemented("ReshardArrays not available with pjrt-ifrt client.");
}

tsl::Future<> PjRtClient::GetReadyFuture(absl::Span<const ValueRef> values) {
  absl::InlinedVector<tsl::Future<>, 1> futures;
  futures.reserve(values.size());
  for (const auto& value : values) {
    futures.push_back(value->GetReadyFuture());
  }
  return JoinFutures(futures);
}

absl::StatusOr<tsl::RCReference<Tuple>> PjRtClient::MakeTuple(
    absl::Span<ValueRef> values) {
  return PjRtTuple::Create(this, values);
}

absl::StatusOr<std::shared_ptr<Topology>> PjRtClient::GetTopologyForDevices(
    const xla::ifrt::DeviceListRef& devices) const {
  // TODO(parkers): Consider constructing a sub-slice topology based on the
  // provided devices.
  TF_ASSIGN_OR_RETURN(auto topology, pjrt_client_->GetTopologyDescription());
  return std::make_shared<PjRtTopology>(
      std::shared_ptr<const xla::PjRtTopologyDescription>(pjrt_client_,
                                                          topology));
}

absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>>
PjRtClient::GetDefaultPjRtLayout(DType dtype, absl::Span<const int64_t> dims,
                                 Device* device, MemoryKind memory_kind) const {
  // PjRt-IFRT devices are currently homogeneous. The cache key omits device
  // information.
  // TODO(hyeontaek): Add device-specific information (e.g., `device->Kind()`)
  // once PjRt-IFRT supports heterogeneous devices.
  auto key = std::make_tuple(
      dtype, std::vector<int64_t>(dims.begin(), dims.end()), memory_kind);
  {
    absl::MutexLock lock(default_layout_cache_mu_);
    if (auto it = default_layout_cache_.find(key);
        it != default_layout_cache_.end()) {
      return it->second;
    }
  }

  auto layout =
      [&]() -> absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> {
    static MemoryKind kUnpinnedHostMemoryKind(UnpinnedHostMemorySpace::kKind);
    if (memory_kind == kUnpinnedHostMemoryKind) {
      return std::make_shared<xla::PjRtLayout>(
          LayoutUtil::MakeDescendingLayout(dims.size()));
    }
    TF_ASSIGN_OR_RETURN(PrimitiveType element_type, ToPrimitiveType(dtype));
    if (element_type == PrimitiveType::TOKEN) {
      return std::make_shared<xla::PjRtLayout>(
          LayoutUtil::MakeDescendingLayout(dims.size()));
    }
    TF_ASSIGN_OR_RETURN(xla::Layout layout,
                        pjrt_client_->GetDefaultLayout(element_type, dims));
    return std::make_shared<xla::PjRtLayout>(std::move(layout));
  }();
  {
    absl::MutexLock lock(default_layout_cache_mu_);
    default_layout_cache_.insert({std::move(key), layout});
  }
  return layout;
}

absl::StatusOr<CustomLayoutRef> PjRtClient::GetDefaultLayout(
    DType dtype, const Shape& shape, const ShardingRef& sharding) const {
  TF_ASSIGN_OR_RETURN(const Shape shard_shape, sharding->GetShardShape(shape));
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<const xla::PjRtLayout> layout,
      GetDefaultPjRtLayout(dtype, shard_shape.dims(),
                           sharding->devices()->devices().front(),
                           sharding->memory_kind()));
  return PjRtLayout::Create(std::move(layout));
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

absl::StatusOr<absl::flat_hash_map<int, IncarnationId>>
PjRtClient::Incarnations() const {
  if (!distributed_client_) {
    return absl::FailedPreconditionError("missing distributed client");
  }
  TF_ASSIGN_OR_RETURN(xla::CoordinationServiceAgent * agent,
                      distributed_client_->GetCoordinationServiceAgent());
  return agent->Incarnations();
}

}  // namespace ifrt
}  // namespace xla
