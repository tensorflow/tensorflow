/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/utils/string_container_utils.h"
#include "xla/array4d.h"
#include "xla/service/computation_placer.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/protobuf/tpu/topology.pb.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {

// Device coordinates are defined as (x, y, z, core), thus resulting in a rank 4
// topology.
constexpr int kTPUTopologyRank = 4;

constexpr char kDeviceTPUSystem[] = "TPU_SYSTEM";
constexpr char kDeviceTPU[] = "TPU";
constexpr char kTPUReplicatedCore[] = "TPU_REPLICATED_CORE";
constexpr char kTPUReplicatedHost[] = "TPU_REPLICATED_HOST";
constexpr char kBadIntArrayElementMsg[] =
    "bad '{0}' attribute at index {1}, not an int";

using ParsedDevice = DeviceNameUtils::ParsedName;
using ParsedDevices = llvm::ArrayRef<DeviceNameUtils::ParsedName>;

namespace {
// Find matching devices in `devices` based on pattern `spec`.
llvm::SmallVector<ParsedDevice, 8> FindMatchingDevices(
    ParsedDevices devices, const ParsedDevice& spec) {
  llvm::SmallVector<ParsedDevice, 8> matching_devices;
  for (const auto& device : devices)
    if (DeviceNameUtils::IsCompleteSpecification(spec, device))
      matching_devices.push_back(device);
  return matching_devices;
}

// Create error message for a conflicting attribute of a device.
template <typename T>
absl::Status MismatchedTPUSystemAttributeErr(absl::string_view attribute, T a,
                                             T b) {
  return absl::InvalidArgumentError(
      absl::StrCat("found ", kDeviceTPUSystem, " devices with conflicting ",
                   attribute, "s '", a, "' and '", b, "'"));
}

// Find TPU_SYSTEM:0 devices in `devices`. If multiple TPU_SYSTEM devices are
// found, the first one lexicographically is returned. If no TPU_SYSTEM device
// is found or if there are multiple TPU_SYSTEM devices with different jobs or
// replicas, a failure will be returned.
absl::StatusOr<llvm::SmallVector<ParsedDevice, 8>> GetTPUSystemDevices(
    ParsedDevices devices) {
  ParsedDevice spec;
  spec.type = kDeviceTPUSystem;
  spec.has_type = true;
  spec.id = 0;
  spec.has_id = true;

  llvm::SmallVector<ParsedDevice, 8> system_devices =
      FindMatchingDevices(devices, spec);
  if (system_devices.empty())
    return absl::InvalidArgumentError(
        absl::StrCat("no ", kDeviceTPUSystem, " devices found"));

  // Check that all system devices are part of the same job.
  const auto& job = system_devices[0].job;
  auto replica = system_devices[0].replica;
  for (const auto& device : llvm::make_range(std::next(system_devices.begin()),
                                             system_devices.end())) {
    if (device.job != job)
      return MismatchedTPUSystemAttributeErr("job", job, device.job);

    if (device.replica != replica)
      return MismatchedTPUSystemAttributeErr("replica", replica,
                                             device.replica);
  }

  // Sort by task to be deterministic.
  std::sort(system_devices.begin(), system_devices.end(),
            [](const ParsedDevice& a, const ParsedDevice& b) {
              return a.task < b.task;
            });

  return system_devices;
}

// Find TPU devices associated to system device based on spec (e.g. from
// GetTPUSystemDevices). If the number of TPU devices per host do not match for
// every host, a failure will be returned.
absl::StatusOr<llvm::SmallVector<llvm::SmallVector<ParsedDevice, 8>, 8>>
GetTPUDevices(ParsedDevices devices,
              llvm::ArrayRef<ParsedDevice> system_devices) {
  llvm::SmallVector<llvm::SmallVector<ParsedDevice, 8>, 8> tpu_devices;
  tpu_devices.reserve(system_devices.size());

  auto lookup = [&devices](ParsedDevice device_spec) {
    device_spec.has_type = true;
    device_spec.type = kDeviceTPU;
    // Enumerate all the available TPUs.
    device_spec.has_id = false;

    llvm::SmallVector<ParsedDevice, 8> host_tpu_devices =
        FindMatchingDevices(devices, device_spec);

    // Sort devices by id.
    std::sort(host_tpu_devices.begin(), host_tpu_devices.end(),
              [](const ParsedDevice& i, const ParsedDevice& j) {
                return i.id < j.id;
              });
    return host_tpu_devices;
  };

  int num_tpus_per_host = 0;
  {
    const auto& device = system_devices[0];
    auto host_tpu_devices = lookup(device);
    num_tpus_per_host = host_tpu_devices.size();
    tpu_devices.push_back(std::move(host_tpu_devices));
  }

  for (const auto& device_spec : llvm::make_range(
           std::next(system_devices.begin()), system_devices.end())) {
    auto host_tpu_devices = lookup(device_spec);
    // Check number of TPU devices per host all match.
    const int64_t host_tpu_devices_size = host_tpu_devices.size();
    if (num_tpus_per_host != host_tpu_devices_size)
      return absl::InvalidArgumentError(
          absl::StrCat("expected the number of TPU devices per host to be ",
                       num_tpus_per_host, ", got ", host_tpu_devices.size()));

    tpu_devices.push_back(std::move(host_tpu_devices));
  }

  return tpu_devices;
}

// Find the compilation device from system device with `DEVICE_CPU` as its
// type.
std::string GetTPUCompilationDevice(ParsedDevice system_device) {
  // TODO(b/110910013) GetTPUSystemDevices parses the spec and returns the
  // TPU_SYSTEM device, which we replace with the CPU device. We do this
  // replacement because we want to place the `tf._TPUCompileMlir` explicitly on
  // CPU devices of the same job as the TPU_SYSTEM device.
  system_device.type = tensorflow::DEVICE_CPU;
  return DeviceNameUtils::ParsedNameToString(system_device);
}

// Find the host CPU device for a given TPU device with `DEVICE_CPU` as its
// type. If multiple local cpu devices are disabled, always assign id 0. If
// set, use the same id as the tpu device.
absl::StatusOr<std::string> GetCPUHostDeviceForTPUDevice(
    ParsedDevice tpu_device, ParsedDevices devices) {
  tpu_device.type = DEVICE_CPU;
  bool enable_multiple_local_cpu_devices =
      tensorflow::GetMlirCommonFlags()
          ->tf_mlir_enable_multiple_local_cpu_devices;
  if (!enable_multiple_local_cpu_devices) {
    tpu_device.id = 0;
  }
  if (FindMatchingDevices(devices, tpu_device).empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Can't find device: ", DeviceNameUtils::ParsedNameToString(tpu_device),
        " in the devices list."));
  }
  return DeviceNameUtils::ParsedNameToString(tpu_device);
}

// Determine execution devices when topology and device assignment are not
// defined. This is a special case where a single core computation is replicated
// to every core in the mesh. TPU devices are simply added to
// `execution_devices` of one replica. `num_replicas` must be 1 or the total
// number of TPU devices available, and `num_cores_per_replica` must be 1.
absl::StatusOr<TPUDevicesAndHosts> GetFullMeshTPUExecutionDeviceAssignment(
    int num_replicas, int num_cores_per_replica,
    llvm::ArrayRef<llvm::SmallVector<ParsedDevice, 8>> tpu_devices,
    ParsedDevices devices) {
  const int num_tasks = tpu_devices.size();
  const int num_tpus_per_task = tpu_devices[0].size();
  const int num_tpu_devices = num_tasks * num_tpus_per_task;

  if (num_replicas != 1 && num_replicas != num_tpu_devices)
    return absl::InvalidArgumentError(
        absl::StrCat("'num_replicas' must be equal to 1 or ", num_tpu_devices,
                     ", got ", num_replicas));

  if (num_cores_per_replica != 1)
    return absl::InvalidArgumentError(
        absl::StrCat("'num_cores_per_replica' must be equal to 1, got ",
                     num_cores_per_replica));

  TPUDevicesAndHosts devices_and_hosts;
  devices_and_hosts.reserve(num_replicas);
  for (int i = 0; i < num_replicas; ++i) {
    const int task = i / num_tpus_per_task;
    const int device = i % num_tpus_per_task;
    const auto& tpu_device = tpu_devices[task][device];
    devices_and_hosts.push_back({TPUDeviceAndHost(
        /*device=*/tensorflow::DeviceNameUtils::ParsedNameToString(tpu_device),
        /*host=*/*GetCPUHostDeviceForTPUDevice(tpu_device, devices))});
  }

  return devices_and_hosts;
}

// Helper struct for keeping track of task and device for an associated TPU
// device coordinate.
struct TaskAndDevice {
  TaskAndDevice() = default;
  TaskAndDevice(int task, int device) : task(task), device(device) {}

  int task = -1;
  int device = -1;
};

// Check if device coordinate is outside of topology mesh shape bounds.
bool DeviceCoordinateOutOfBound(int x, int y, int z, int core, int bound_x,
                                int bound_y, int bound_z, int bound_core) {
  return x < 0 || x >= bound_x || y < 0 || y >= bound_y || z < 0 ||
         z >= bound_z || core < 0 || core >= bound_core;
}

// Create error message for an out of bound device coordinate.
absl::Status DeviceCoordinateErrorMsg(absl::string_view attribute, int x, int y,
                                      int z, int core, int bound_x, int bound_y,
                                      int bound_z, int bound_core) {
  return absl::InvalidArgumentError(
      absl::StrCat("device coordinate (", x, ", ", y, ", ", z, ", ", core,
                   ") in '", attribute, "' is outside of mesh shape (", bound_x,
                   ", ", bound_y, ", ", bound_z, ", ", bound_core, ")"));
}

// Create error message for a duplicate device coordinate.
absl::Status DuplicateCoordinateErrorMsg(absl::string_view attribute, int x,
                                         int y, int z, int core) {
  return absl::InvalidArgumentError(
      absl::StrCat("'", attribute, "' has duplicate device coordinate (", x,
                   ", ", y, ", ", z, ", ", core, ")"));
}

// Parse and validate topology (serialized string of TopologyProto), and maps
// device coordinate (x, y, z, core) to task and device (of available TPUs).
// Topology attribute device coordinates are ordered by task then device (major
// to minor).
//
// A valid TopologyProto must have:
//  - a valid mesh shape (rank 4 with positive dimensions)
//  - `num_tasks` and `num_tpu_devices_per_task` must match the number of
//    available TPU hosts and devices per host
//  - device coordinates within the mesh shape
//  - no duplicate device coordinates
//  - number of device coordinates (in tuple 3) match number of availabe TPUs
absl::StatusOr<xla::Array4D<TaskAndDevice>> ParseTopologyAttr(
    llvm::StringRef topology_attr, int num_tasks, int num_tpus_per_task) {
  tpu::TopologyProto topology_proto;
  if (!topology_proto.ParseFromString(topology_attr.str()))
    return absl::InvalidArgumentError(absl::StrCat(
        "failed to parse '", kTopologyAttr, "' attribute to TopologyProto"));

  if (topology_proto.mesh_shape_size() != kTPUTopologyRank)
    return absl::InvalidArgumentError(absl::StrCat(
        "'", kTopologyAttr, "' 'mesh_shape' must be rank ", kTPUTopologyRank,
        ", got rank ", topology_proto.mesh_shape_size()));

  for (auto mesh_shape_dim : llvm::enumerate(topology_proto.mesh_shape()))
    if (mesh_shape_dim.value() <= 0)
      return absl::InvalidArgumentError(
          absl::StrCat("'", kTopologyAttr, "' 'mesh_shape' dimension ",
                       mesh_shape_dim.index(), " must be positive, got ",
                       mesh_shape_dim.value()));

  if (topology_proto.num_tasks() != num_tasks)
    return absl::InvalidArgumentError(absl::StrCat(
        "number of tasks from available TPU devices must be 'num_tasks' in '",
        kTopologyAttr, "' (", topology_proto.num_tasks(), "), got ",
        num_tasks));

  if (topology_proto.num_tpu_devices_per_task() != num_tpus_per_task)
    return absl::InvalidArgumentError(absl::StrCat(
        "number of TPU devices available per task must be "
        "'num_tpu_devices_per_task' in '",
        kTopologyAttr, "' (", topology_proto.num_tpu_devices_per_task(),
        "), got ", num_tpus_per_task));

  const int expected_device_coordinates_size =
      num_tasks * num_tpus_per_task * kTPUTopologyRank;
  if (topology_proto.device_coordinates_size() !=
      expected_device_coordinates_size)
    return absl::InvalidArgumentError(absl::StrCat(
        "length of 'device_coordinates' in '", kTopologyAttr,
        "' must be 'num_tasks' * 'num_tpus_per_task' * ", kTPUTopologyRank,
        " (", num_tasks, " * ", num_tpus_per_task, " * ", kTPUTopologyRank,
        "), got ", topology_proto.device_coordinates_size()));

  const int bound_x = topology_proto.mesh_shape(0);
  const int bound_y = topology_proto.mesh_shape(1);
  const int bound_z = topology_proto.mesh_shape(2);
  const int bound_core = topology_proto.mesh_shape(3);

  xla::Array4D<TaskAndDevice> topology(bound_x, bound_y, bound_z, bound_core);
  int pos = 0;
  for (int task = 0; task < num_tasks; ++task) {
    for (int device = 0; device < num_tpus_per_task; ++device) {
      int x = topology_proto.device_coordinates(pos++);
      int y = topology_proto.device_coordinates(pos++);
      int z = topology_proto.device_coordinates(pos++);
      int core = topology_proto.device_coordinates(pos++);
      if (DeviceCoordinateOutOfBound(x, y, z, core, bound_x, bound_y, bound_z,
                                     bound_core))
        return DeviceCoordinateErrorMsg(kTopologyAttr, x, y, z, core, bound_x,
                                        bound_y, bound_z, bound_core);

      auto& task_and_device = topology(x, y, z, core);
      if (task_and_device.task != -1)
        return DuplicateCoordinateErrorMsg(kTopologyAttr, x, y, z, core);

      task_and_device = {task, device};
    }
  }

  return topology;
}

// Determine execution devices when topology and device assignment are defined.
// With a topology device coordinate to task and device mapping, device
// assignment device coordinates can then be mapped to task and device for TPU
// devices. The device assignment array is also validated.
//
// A valid device assignment array must have:
//  - device coordinates within the topology mesh shape
//  - no duplicate device coordinates
//  - number of device coordinates (in tuple 3) match number 'num_replicas' *
//    'num_cores_per_replica'
//  - a TPU device associated with each device coordinate
absl::StatusOr<std::pair<TPUDevicesAndHosts, xla::DeviceAssignmentProto>>
GetGeneralTPUExecutionDeviceAssignment(
    int num_replicas, int num_cores_per_replica,
    llvm::ArrayRef<llvm::SmallVector<ParsedDevice, 8>> tpu_devices,
    ParsedDevices devices, llvm::StringRef topology_attr,
    llvm::ArrayRef<int64_t> device_assignment_attr) {
  const int num_tasks = tpu_devices.size();
  const int num_tpus_per_task = tpu_devices[0].size();

  TF_ASSIGN_OR_RETURN(auto topology, ParseTopologyAttr(topology_attr, num_tasks,
                                                       num_tpus_per_task));

  const int expected_device_assignment_size =
      num_replicas * num_cores_per_replica * kTPUTopologyRank;
  const int device_assignment_attr_size = device_assignment_attr.size();
  if (device_assignment_attr_size != expected_device_assignment_size)
    return absl::InvalidArgumentError(absl::StrCat(
        "length of '", kDeviceAssignmentAttr,
        "' must be 'num_replicas' * 'num_cores_per_replica' * ",
        kTPUTopologyRank, " (", num_replicas, " * ", num_cores_per_replica,
        " * ", kTPUTopologyRank, "), got ", device_assignment_attr.size()));

  const int bound_x = topology.n1();
  const int bound_y = topology.n2();
  const int bound_z = topology.n3();
  const int bound_core = topology.n4();

  // TPU XLA device ID is determined by its device coordinate, from major to
  // minor coordinates (z, y, x, core).
  auto location_to_id = [&](int x, int y, int z, int core) {
    return (x + bound_x * (y + bound_y * z)) * bound_core + core;
  };

  std::vector<bool> used_device_ids(bound_x * bound_y * bound_z * bound_core,
                                    false);
  TPUDevicesAndHosts devices_and_hosts(
      num_replicas, llvm::SmallVector<TPUDeviceAndHost, 8>(
                        num_cores_per_replica, TPUDeviceAndHost()));
  xla::DeviceAssignment device_assignment(num_replicas, num_cores_per_replica);
  int pos = 0;
  for (int replica = 0; replica < num_replicas; ++replica) {
    for (int logical_core = 0; logical_core < num_cores_per_replica;
         ++logical_core) {
      int x = device_assignment_attr[pos++];
      int y = device_assignment_attr[pos++];
      int z = device_assignment_attr[pos++];
      int core = device_assignment_attr[pos++];
      if (DeviceCoordinateOutOfBound(x, y, z, core, bound_x, bound_y, bound_z,
                                     bound_core))
        return DeviceCoordinateErrorMsg(kDeviceAssignmentAttr, x, y, z, core,
                                        bound_x, bound_y, bound_z, bound_core);

      TaskAndDevice task_and_device = topology(x, y, z, core);
      const int task = task_and_device.task;
      const int device = task_and_device.device;
      if (task == -1 || device == -1)
        return absl::InvalidArgumentError(absl::StrCat(
            "no TPU device found for '", kDeviceAssignmentAttr,
            "' device coordinate (", x, ", ", y, ", ", z, ", ", core, ")"));

      const int device_id = location_to_id(x, y, z, core);
      if (used_device_ids[device_id])
        return DuplicateCoordinateErrorMsg(kDeviceAssignmentAttr, x, y, z,
                                           core);

      used_device_ids[device_id] = true;
      device_assignment(replica, logical_core) = device_id;
      auto& device_and_host = devices_and_hosts[replica][logical_core];
      const auto& tpu_device = tpu_devices[task][device];
      device_and_host.device = DeviceNameUtils::ParsedNameToString(tpu_device);
      device_and_host.host = *GetCPUHostDeviceForTPUDevice(tpu_device, devices);
    }
  }

  xla::DeviceAssignmentProto device_assignment_proto;
  device_assignment.Serialize(&device_assignment_proto);

  return std::pair<TPUDevicesAndHosts, xla::DeviceAssignmentProto>(
      std::move(devices_and_hosts), std::move(device_assignment_proto));
}

mlir::LogicalResult GetTopology(mlir::tf_device::ClusterOp cluster,
                                std::string& topology) {
  mlir::StringAttr topology_attr =
      cluster->getAttrOfType<mlir::StringAttr>(tensorflow::kTopologyAttr);
  if (topology_attr) {
    topology = topology_attr.getValue();
    return mlir::success();
  } else {
    return cluster.emitOpError(
        llvm::formatv("requires attribute '{0}'", tensorflow::kTopologyAttr)
            .str());
  }
}

mlir::LogicalResult GetDeviceAssignmentCoordinates(
    mlir::tf_device::ClusterOp cluster,
    llvm::SmallVector<int64_t, 8>& device_coordinates) {
  mlir::ArrayAttr device_assignment_attr =
      cluster->getAttrOfType<mlir::ArrayAttr>(
          tensorflow::kDeviceAssignmentAttr);
  if (!device_assignment_attr)
    return cluster.emitOpError(llvm::formatv("requires attribute '{0}'",
                                             tensorflow::kDeviceAssignmentAttr)
                                   .str());
  if (absl::StatusOr<llvm::SmallVector<int64_t, 8>> fetched_device_coordinates =
          tensorflow::GetDeviceCoordinates(device_assignment_attr);
      fetched_device_coordinates.ok()) {
    device_coordinates = *fetched_device_coordinates;
    return mlir::success();
  } else {
    return cluster.emitError() << "error in fetching tpu device coordinates: "
                               << fetched_device_coordinates.status().message();
  }
}

int GetNumCoresPerReplica(mlir::tf_device::ClusterOp cluster) {
  mlir::IntegerAttr num_cores_per_replica_attr =
      cluster->getAttrOfType<mlir::IntegerAttr>(kNumCoresPerReplicaAttr);
  if (num_cores_per_replica_attr) {
    return num_cores_per_replica_attr.getInt();
  } else {
    return 1;
  }
}

// Get the TPUDevicesAndHosts for a cluster that is not replicated.
mlir::LogicalResult GetTPUDevicesAndHostsNotReplicated(
    mlir::TF::RuntimeDevices devices, mlir::tf_device::ClusterOp cluster,
    tensorflow::TPUDevicesAndHosts& devices_and_hosts) {
  std::string topology;
  if (failed(GetTopology(cluster, topology))) {
    return mlir::failure();
  }

  llvm::SmallVector<int64_t, 8> device_coordinates;
  if (failed(GetDeviceAssignmentCoordinates(cluster, device_coordinates))) {
    return mlir::failure();
  }

  // Determine compilation and execution devices.
  if (absl::StatusOr<TPUDeviceAssignment> tpu_device_assignment =
          tensorflow::GetTPUCompilationAndExecutionDevices(
              devices.device_names(), /*num_replicas=*/1,
              GetNumCoresPerReplica(cluster), topology, device_coordinates);
      tpu_device_assignment.ok()) {
    devices_and_hosts = tpu_device_assignment->tpu_devices;
    return mlir::success();
  } else {
    return cluster.emitError()
           << "error in fetching TPU compilation/execution devices: "
           << tpu_device_assignment.status().message();
  }
}

mlir::LogicalResult GetHostDeviceOCInTPUPipeline(
    mlir::TF::RuntimeDevices devices, mlir::tf_device::ClusterOp cluster,
    std::string& host_device) {
  mlir::tf_device::ReplicateOp replicate =
      cluster->getParentOfType<mlir::tf_device::ReplicateOp>();
  if (replicate) {
    host_device = GetDeviceAliasForHostOfLogicalCore(0);
    return mlir::success();
  }

  tensorflow::TPUDevicesAndHosts devices_and_hosts;
  if (failed(GetTPUDevicesAndHostsNotReplicated(devices, cluster,
                                                devices_and_hosts))) {
    return mlir::failure();
  } else {
    host_device = devices_and_hosts[0][0].host;
    return mlir::success();
  }
}

// Get the map from `core` to `TPU_REPLICATED_HOST_{core}` for a replicated
// TPU cluster.
// TPU_REPLICATED_HOST_{core} is the host that corresponds to the TPU core.
// Different TPU_REPLICATED_HOST_*s can map to the same physical host within the
// same replica. Also, TPU_REPLICATE_HOST_{core} in different replicas can map
// to the same physical host. For example, if there are 2 hosts, num_replicas=8,
// and num_cores_per_replica=2, then all cores in the first 4 replicas will map
// to the first host and all cores in the second 4 replicas will map to the
// second host.
llvm::SmallVector<std::string, 8> GetTPUToHostMapReplicated(
    mlir::tf_device::ClusterOp cluster) {
  int num_cores_per_replica = GetNumCoresPerReplica(cluster);
  llvm::SmallVector<std::string, 8> core_to_host;
  core_to_host.reserve(num_cores_per_replica);
  for (int core = 0; core < num_cores_per_replica; ++core) {
    core_to_host.push_back(GetDeviceAliasForHostOfLogicalCore(core));
  }
  return core_to_host;
}

// Get the map from `core` to host device for a non-replicated TPU cluster.
mlir::LogicalResult GetTPUToHostMapNotReplicated(
    mlir::TF::RuntimeDevices devices, mlir::tf_device::ClusterOp cluster,
    llvm::SmallVector<std::string, 8>& core_to_host) {
  tensorflow::TPUDevicesAndHosts devices_and_hosts;
  if (failed(GetTPUDevicesAndHostsNotReplicated(devices, cluster,
                                                devices_and_hosts))) {
    return mlir::failure();
  }

  // core_to_host is the list of hosts in replica 0, which is the only replica.
  core_to_host.reserve(GetNumCoresPerReplica(cluster));
  for (const auto& device_and_host : devices_and_hosts[0]) {
    core_to_host.push_back(device_and_host.host);
  }
  return mlir::success();
}

// Get the map from `core` to host device for a TPU cluster.
mlir::LogicalResult GetTPUToHostMap(
    mlir::TF::RuntimeDevices devices, mlir::tf_device::ClusterOp cluster,
    llvm::SmallVector<std::string, 8>& core_to_host) {
  if (cluster->getParentOfType<mlir::tf_device::ReplicateOp>()) {
    core_to_host = GetTPUToHostMapReplicated(cluster);
    return mlir::success();
  }
  return GetTPUToHostMapNotReplicated(devices, cluster, core_to_host);
}

}  // anonymous namespace

absl::StatusOr<llvm::SmallVector<int64_t, 8>> GetDeviceCoordinates(
    mlir::ArrayAttr device_assignment_attr) {
  llvm::SmallVector<int64_t, 8> device_coordinates;
  device_coordinates.reserve(device_assignment_attr.size());

  for (auto device_coordinate_and_idx :
       llvm::enumerate(device_assignment_attr)) {
    auto device_coordinate =
        mlir::dyn_cast<mlir::IntegerAttr>(device_coordinate_and_idx.value());
    if (!device_coordinate)
      return absl::InvalidArgumentError(
          llvm::formatv(kBadIntArrayElementMsg, kDeviceAssignmentAttr,
                        device_coordinate_and_idx.index())
              .str());

    device_coordinates.push_back(device_coordinate.getInt());
  }

  return device_coordinates;
}

absl::StatusOr<TPUDeviceAssignment> GetTPUCompilationAndExecutionDevices(
    ParsedDevices devices, int num_replicas, int num_cores_per_replica,
    llvm::StringRef topology_attr,
    llvm::ArrayRef<int64_t> device_assignment_attr) {
  // Collect TPU_SYSTEM devices.
  TF_ASSIGN_OR_RETURN(auto system_devices, GetTPUSystemDevices(devices));

  // Collect TPU devices based on TPU_SYSTEM devices collected earlier.
  TF_ASSIGN_OR_RETURN(auto tpu_devices, GetTPUDevices(devices, system_devices));

  std::string compilation_device = GetTPUCompilationDevice(system_devices[0]);

  if (topology_attr.empty()) {
    if (!device_assignment_attr.empty())
      return absl::InvalidArgumentError(
          absl::StrCat("'", kDeviceAssignmentAttr, "' must not be set when '",
                       kTopologyAttr, "' is not set"));

    TF_ASSIGN_OR_RETURN(
        auto execution_devices,
        GetFullMeshTPUExecutionDeviceAssignment(
            num_replicas, num_cores_per_replica, tpu_devices, devices));
    return TPUDeviceAssignment(compilation_device,
                               std::move(execution_devices));
  }

  TF_ASSIGN_OR_RETURN(auto devices_and_ids,
                      GetGeneralTPUExecutionDeviceAssignment(
                          num_replicas, num_cores_per_replica, tpu_devices,
                          devices, topology_attr, device_assignment_attr));
  return TPUDeviceAssignment(compilation_device,
                             std::move(devices_and_ids.first),
                             std::move(devices_and_ids.second));
}

std::string GetDeviceAliasForLogicalCore(const int core_index) {
  return llvm::formatv("{0}_{1}", kTPUReplicatedCore, core_index).str();
}

std::string GetDeviceAliasForHostOfLogicalCore(const int core_index) {
  return llvm::formatv("{0}_{1}", kTPUReplicatedHost, core_index).str();
}

bool HasModelParallelism(mlir::tf_device::ClusterOp cluster) {
  mlir::IntegerAttr num_cores_per_replica_attr =
      cluster->getAttrOfType<mlir::IntegerAttr>(
          tensorflow::kNumCoresPerReplicaAttr);
  if (!num_cores_per_replica_attr) return false;
  return num_cores_per_replica_attr.getInt() != 1;
}

bool HasTPUDevice(const mlir::TF::RuntimeDevices& devices) {
  for (const auto& device : devices.device_names()) {
    if (device.has_type && device.type == "TPU") return true;
  }
  return false;
}

mlir::LogicalResult GetHostDeviceOutsideCompilationInGenericPipeline(
    mlir::TF::RuntimeDevices devices, std::string* host_device) {
  for (const auto& device : devices.device_names()) {
    if (device.has_type && device.type == "CPU" && device.id == 0) {
      if (!host_device->empty()) {
        // TODO(hanxiongwang): Remove this warning when TF API to bridge
        // interface is understood.
        LOG(WARNING) << "Found multiple CPU:0 host devices";
        if (device.job == "chief")
          *host_device =
              tensorflow::DeviceNameUtils::ParsedNameToString(device);
        continue;
      }
      *host_device = tensorflow::DeviceNameUtils::ParsedNameToString(device);
    }
  }
  if (host_device->empty()) {
    LOG(ERROR) << "Did not find any CPU:0 host devices";
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult GetHostDeviceOutsideComputation(
    mlir::TF::RuntimeDevices devices, mlir::tf_device::ClusterOp cluster,
    std::string* host_device) {
  if (HasTPUDevice(devices) ||
      cluster->getParentOfType<mlir::tf_device::ReplicateOp>()) {
    return GetHostDeviceOCInTPUPipeline(devices, cluster, *host_device);
  } else {
    return GetHostDeviceOutsideCompilationInGenericPipeline(devices,
                                                            host_device);
  }
}

bool IsTPUDevice(llvm::StringRef device) {
  ParsedDevice parsed_device;
  if (!DeviceNameUtils::ParseFullName(mlir::StringRefToView(device),
                                      &parsed_device))
    return false;
  return parsed_device.has_type && parsed_device.type == kDeviceTPU;
}

bool IsTPUReplicatedCore(llvm::StringRef device) {
  ParsedDevice parsed_device;
  if (!DeviceNameUtils::ParseFullName(mlir::StringRefToView(device),
                                      &parsed_device))
    return false;
  return parsed_device.has_type && parsed_device.type == kTPUReplicatedCore;
}

bool TypeValidForXLA(const mlir::Type& type) {
  const mlir::Type elem = getElementTypeOrSelf(type);
  return !mlir::isa<mlir::TF::ResourceType>(elem) &&
         !mlir::isa<mlir::TF::StringType>(elem);
}

mlir::LogicalResult GetDeviceToHostMap(
    mlir::tf_device::ClusterOp cluster,
    llvm::SmallVector<std::string, 8>& core_to_host) {
  mlir::TF::RuntimeDevices devices;
  if (failed(tensorflow::GetDevicesFromOp(
          cluster->getParentOfType<mlir::ModuleOp>(), &devices))) {
    return mlir::failure();
  }

  if (tensorflow::HasTPUDevice(devices) ||
      cluster->getParentOfType<mlir::tf_device::ReplicateOp>()) {
    return GetTPUToHostMap(devices, cluster, core_to_host);
  }

  std::string host_device;
  if (failed(tensorflow::GetHostDeviceOutsideCompilationInGenericPipeline(
          devices, &host_device))) {
    return mlir::failure();
  } else {
    core_to_host.push_back(host_device);
    return mlir::success();
  }
}

mlir::LogicalResult GetNonReplicatedTPU0(mlir::Operation* op,
                                         std::string* tpu0_device) {
  // Fetch the TPU devices.
  mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
  mlir::TF::RuntimeDevices devices;
  if (failed(tensorflow::GetDevicesFromOp(moduleOp, &devices)))
    return moduleOp.emitOpError() << "No available devices.";
  llvm::ArrayRef<tensorflow::DeviceNameUtils::ParsedName> device_names =
      devices.device_names();
  auto status_or_system_devices = GetTPUSystemDevices(device_names);
  if (!status_or_system_devices.ok())
    return moduleOp.emitOpError()
           << "error in fetching TPU_SYSTEM devices: "
           << status_or_system_devices.status().message();
  auto status_or_tpu_devices =
      GetTPUDevices(device_names, status_or_system_devices.value());
  if (!status_or_tpu_devices.ok())
    return moduleOp.emitOpError() << "error in fetching TPU devices: "
                                  << status_or_tpu_devices.status().message();

  // Select the first TPU device.
  *tpu0_device =
      DeviceNameUtils::ParsedNameToString(status_or_tpu_devices.value()[0][0]);
  return mlir::success();
}

mlir::LogicalResult GetNonReplicatedCPU0(mlir::Operation* op,
                                         std::string* cpu0_device) {
  std::string tpu0_device;
  if (failed(tensorflow::GetNonReplicatedTPU0(op, &tpu0_device)))
    return mlir::failure();
  auto status = tensorflow::DeviceNameUtils::DeviceNameToCpuDeviceName(
      tpu0_device, cpu0_device);
  if (!status.ok())
    return op->emitError()
           << "error in converting TPU0 to CPU0. The TPU device is "
           << tpu0_device;
  return mlir::success();
}

}  // namespace tensorflow
