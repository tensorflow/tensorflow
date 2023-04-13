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
#include <type_traits>
#include <utility>

#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/utils/string_container_utils.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/tpu/topology.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

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

using Device = DeviceNameUtils::ParsedName;
using Devices = llvm::ArrayRef<DeviceNameUtils::ParsedName>;

namespace {
// Finds matching devices in `devices` based on pattern `spec`.
void FindMatchingDevices(Devices devices, const Device& spec,
                         llvm::SmallVectorImpl<Device>* matched_devices) {
  for (const auto& device : devices)
    if (DeviceNameUtils::IsCompleteSpecification(spec, device))
      matched_devices->push_back(device);
}

// Creates error message for a conflicting attribute of a device.
template <typename T>
Status MismatchedTPUSystemAttributeErr(absl::string_view attribute, T a, T b) {
  return errors::InvalidArgument("found ", kDeviceTPUSystem,
                                 " devices with conflicting ", attribute, "s '",
                                 a, "' and '", b, "'");
}

// Finds TPU_SYSTEM:0 devices in `devices`. If multiple TPU_SYSTEM devices are
// found, the first one lexicographically is returned. If no TPU_SYSTEM device
// is found or if there are multiple TPU_SYSTEM devices with different jobs or
// replicas, a failure will be returned.
Status GetTPUSystemDevices(Devices devices,
                           llvm::SmallVectorImpl<Device>* matched_devices) {
  Device spec;
  spec.type = kDeviceTPUSystem;
  spec.has_type = true;
  spec.id = 0;
  spec.has_id = true;

  llvm::SmallVector<Device, 8> system_devices;
  FindMatchingDevices(devices, spec, &system_devices);
  if (system_devices.empty())
    return errors::InvalidArgument("no ", kDeviceTPUSystem, " devices found");

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
            [](const Device& a, const Device& b) { return a.task < b.task; });

  matched_devices->swap(system_devices);

  return OkStatus();
}

// Finds TPU devices associated to system device based on spec (e.g. from
// GetTPUSystemDevices). If the number of TPU devices per host do not match for
// every host, a failure will be returned.
Status GetTPUDevices(
    Devices devices, llvm::ArrayRef<Device> system_devices,
    llvm::SmallVectorImpl<llvm::SmallVector<Device, 8>>* tpu_devices) {
  tpu_devices->reserve(system_devices.size());

  auto lookup = [&devices](Device device_spec) {
    device_spec.has_type = true;
    device_spec.type = kDeviceTPU;
    // Enumerate all the available TPUs.
    device_spec.has_id = false;

    llvm::SmallVector<Device, 8> host_tpu_devices;
    FindMatchingDevices(devices, device_spec, &host_tpu_devices);

    // Sort devices by id.
    std::sort(host_tpu_devices.begin(), host_tpu_devices.end(),
              [](const Device& i, const Device& j) { return i.id < j.id; });
    return host_tpu_devices;
  };

  int num_tpus_per_host = 0;
  {
    const auto& device = system_devices[0];
    auto host_tpu_devices = lookup(device);
    num_tpus_per_host = host_tpu_devices.size();
    tpu_devices->push_back(std::move(host_tpu_devices));
  }

  for (const auto& device_spec : llvm::make_range(
           std::next(system_devices.begin()), system_devices.end())) {
    auto host_tpu_devices = lookup(device_spec);
    // Check number of TPU devices per host all match.
    const int64_t host_tpu_devices_size = host_tpu_devices.size();
    if (num_tpus_per_host != host_tpu_devices_size)
      return errors::InvalidArgument(
          "expected the number of TPU devices per host to be ",
          num_tpus_per_host, ", got ", host_tpu_devices.size());

    tpu_devices->push_back(std::move(host_tpu_devices));
  }

  return OkStatus();
}

// Finds the compilation device from system device.
std::string GetTPUCompilationDevice(Device system_device) {
  // TODO(b/110910013) GetTPUSystemDevices parses the spec and returns the
  // TPU_SYSTEM device, which we replace with the CPU device. We do this
  // replacement because we want to place the `tf._TPUCompileMlir` explicitly on
  // CPU devices of the same job as the TPU_SYSTEM device.
  system_device.type = tensorflow::DEVICE_CPU;
  return DeviceNameUtils::ParsedNameToString(system_device);
}

// Finds the host CPU device for a given TPU device.
std::string GetCPUHostDeviceForTPUDevice(Device tpu_device) {
  tpu_device.type = DEVICE_CPU;
  tpu_device.id = 0;
  return DeviceNameUtils::ParsedNameToString(tpu_device);
}

// Determines execution devices when topology and device assignment are not
// defined. This is a special case where a single core computation is replicated
// to every core in the mesh. TPU devices are simply added to
// `execution_devices` of one replica. `num_replicas` must be 1 or the total
// number of TPU devices available, and `num_cores_per_replica` must be 1.
StatusOr<TPUDevicesAndHosts> GetFullMeshTPUExecutionDeviceAssignment(
    int num_replicas, int num_cores_per_replica,
    llvm::ArrayRef<llvm::SmallVector<Device, 8>> tpu_devices) {
  const int num_tasks = tpu_devices.size();
  const int num_tpus_per_task = tpu_devices[0].size();
  const int num_tpu_devices = num_tasks * num_tpus_per_task;

  if (num_replicas != 1 && num_replicas != num_tpu_devices)
    return errors::InvalidArgument("'num_replicas' must be equal to 1 or ",
                                   num_tpu_devices, ", got ", num_replicas);

  if (num_cores_per_replica != 1)
    return errors::InvalidArgument(
        "'num_cores_per_replica' must be equal to 1, got ",
        num_cores_per_replica);

  TPUDevicesAndHosts devices_and_hosts;
  devices_and_hosts.reserve(num_replicas);
  for (int i = 0; i < num_replicas; ++i) {
    const int task = i / num_tpus_per_task;
    const int device = i % num_tpus_per_task;
    const auto& tpu_device = tpu_devices[task][device];
    devices_and_hosts.push_back({TPUDeviceAndHost(
        /*device=*/tensorflow::DeviceNameUtils::ParsedNameToString(tpu_device),
        /*host=*/GetCPUHostDeviceForTPUDevice(tpu_device))});
  }

  return devices_and_hosts;
}

// Helper struct for keeping track of task and device for an associated TPU
// device coordinate.
struct TaskAndDevice {
  TaskAndDevice() {}
  TaskAndDevice(int task, int device) : task(task), device(device) {}

  int task = -1;
  int device = -1;
};

// Checks if device coordinate is outside of topology mesh shape bounds.
bool DeviceCoordinateOutOfBound(int x, int y, int z, int core, int bound_x,
                                int bound_y, int bound_z, int bound_core) {
  return x < 0 || x >= bound_x || y < 0 || y >= bound_y || z < 0 ||
         z >= bound_z || core < 0 || core >= bound_core;
}

// Creates error message for an out of bound device coordinate.
Status DeviceCoordinateErrorMsg(absl::string_view attribute, int x, int y,
                                int z, int core, int bound_x, int bound_y,
                                int bound_z, int bound_core) {
  return errors::InvalidArgument("device coordinate (", x, ", ", y, ", ", z,
                                 ", ", core, ") in '", attribute,
                                 "' is outside of mesh shape (", bound_x, ", ",
                                 bound_y, ", ", bound_z, ", ", bound_core, ")");
}

// Creates error message for a duplicate device coordinate.
Status DuplicateCoordinateErrorMsg(absl::string_view attribute, int x, int y,
                                   int z, int core) {
  return errors::InvalidArgument("'", attribute,
                                 "' has duplicate device coordinate (", x, ", ",
                                 y, ", ", z, ", ", core, ")");
}

// Parses and validates topology (serialized string of TopologyProto), and maps
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
StatusOr<xla::Array4D<TaskAndDevice>> ParseTopologyAttr(
    llvm::StringRef topology_attr, int num_tasks, int num_tpus_per_task) {
  tpu::TopologyProto topology_proto;
  if (!topology_proto.ParseFromString(topology_attr.str()))
    return errors::InvalidArgument("failed to parse '", kTopologyAttr,
                                   "' attribute to TopologyProto");

  if (topology_proto.mesh_shape_size() != kTPUTopologyRank)
    return errors::InvalidArgument(
        "'", kTopologyAttr, "' 'mesh_shape' must be rank ", kTPUTopologyRank,
        ", got rank ", topology_proto.mesh_shape_size());

  for (auto mesh_shape_dim : llvm::enumerate(topology_proto.mesh_shape()))
    if (mesh_shape_dim.value() <= 0)
      return errors::InvalidArgument(
          "'", kTopologyAttr, "' 'mesh_shape' dimension ",
          mesh_shape_dim.index(), " must be positive, got ",
          mesh_shape_dim.value());

  if (topology_proto.num_tasks() != num_tasks)
    return errors::InvalidArgument(
        "number of tasks from available TPU devices must be 'num_tasks' in '",
        kTopologyAttr, "' (", topology_proto.num_tasks(), "), got ", num_tasks);

  if (topology_proto.num_tpu_devices_per_task() != num_tpus_per_task)
    return errors::InvalidArgument(
        "number of TPU devices available per task must be "
        "'num_tpu_devices_per_task' in '",
        kTopologyAttr, "' (", topology_proto.num_tpu_devices_per_task(),
        "), got ", num_tpus_per_task);

  const int expected_device_coordinates_size =
      num_tasks * num_tpus_per_task * kTPUTopologyRank;
  if (topology_proto.device_coordinates_size() !=
      expected_device_coordinates_size)
    return errors::InvalidArgument(
        "length of 'device_coordinates' in '", kTopologyAttr,
        "' must be 'num_tasks' * 'num_tpus_per_task' * ", kTPUTopologyRank,
        " (", num_tasks, " * ", num_tpus_per_task, " * ", kTPUTopologyRank,
        "), got ", topology_proto.device_coordinates_size());

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

// Determines execution devices when topology and device assignment are defined.
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
StatusOr<std::pair<TPUDevicesAndHosts, xla::DeviceAssignmentProto>>
GetGeneralTPUExecutionDeviceAssignment(
    int num_replicas, int num_cores_per_replica,
    llvm::ArrayRef<llvm::SmallVector<Device, 8>> tpu_devices,
    llvm::StringRef topology_attr,
    llvm::ArrayRef<int64_t> device_assignment_attr) {
  const int num_tasks = tpu_devices.size();
  const int num_tpus_per_task = tpu_devices[0].size();

  TF_ASSIGN_OR_RETURN(auto topology, ParseTopologyAttr(topology_attr, num_tasks,
                                                       num_tpus_per_task));

  const int expected_device_assignment_size =
      num_replicas * num_cores_per_replica * kTPUTopologyRank;
  const int device_assignment_attr_size = device_assignment_attr.size();
  if (device_assignment_attr_size != expected_device_assignment_size)
    return errors::InvalidArgument(
        "length of '", kDeviceAssignmentAttr,
        "' must be 'num_replicas' * 'num_cores_per_replica' * ",
        kTPUTopologyRank, " (", num_replicas, " * ", num_cores_per_replica,
        " * ", kTPUTopologyRank, "), got ", device_assignment_attr.size());

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
        return errors::InvalidArgument(
            "no TPU device found for '", kDeviceAssignmentAttr,
            "' device coordinate (", x, ", ", y, ", ", z, ", ", core, ")");

      const int device_id = location_to_id(x, y, z, core);
      if (used_device_ids[device_id])
        return DuplicateCoordinateErrorMsg(kDeviceAssignmentAttr, x, y, z,
                                           core);

      used_device_ids[device_id] = true;
      device_assignment(replica, logical_core) = device_id;
      auto& device_and_host = devices_and_hosts[replica][logical_core];
      const auto& tpu_device = tpu_devices[task][device];
      device_and_host.device = DeviceNameUtils::ParsedNameToString(tpu_device);
      device_and_host.host = GetCPUHostDeviceForTPUDevice(tpu_device);
    }
  }

  xla::DeviceAssignmentProto device_assignment_proto;
  TF_RETURN_IF_ERROR(device_assignment.Serialize(&device_assignment_proto));

  return std::pair<TPUDevicesAndHosts, xla::DeviceAssignmentProto>(
      std::move(devices_and_hosts), std::move(device_assignment_proto));
}

mlir::LogicalResult GetHostDeviceOCInGenericPipeline(
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

mlir::LogicalResult GetHostDeviceOCInTPUPipeline(
    mlir::TF::RuntimeDevices devices, mlir::tf_device::ClusterOp cluster,
    std::string* host_device) {
  auto replicate = cluster->getParentOfType<mlir::tf_device::ReplicateOp>();
  if (replicate) {
    *host_device = GetDeviceAliasForHostOfLogicalCore(0);
    return mlir::success();
  }

  auto topology_attr =
      cluster->getAttrOfType<mlir::StringAttr>(tensorflow::kTopologyAttr);
  if (!topology_attr)
    return cluster.emitOpError("cluster op missing `topology` attribute");

  auto num_cores_per_replica_attr = cluster->getAttrOfType<mlir::IntegerAttr>(
      tensorflow::kNumCoresPerReplicaAttr);
  if (!num_cores_per_replica_attr)
    return cluster.emitOpError(
        llvm::formatv("requires attribute '{0}'",
                      tensorflow::kNumCoresPerReplicaAttr)
            .str());

  auto device_assignment_attr = cluster->getAttrOfType<mlir::ArrayAttr>(
      tensorflow::kDeviceAssignmentAttr);
  if (!device_assignment_attr)
    return cluster.emitOpError(llvm::formatv("requires attribute '{0}'",
                                             tensorflow::kDeviceAssignmentAttr)
                                   .str());

  auto status_or_device_coodinates =
      tensorflow::GetDeviceCoordinates(device_assignment_attr);

  if (!status_or_device_coodinates.ok())
    return cluster.emitError()
           << "error in fetching tpu device coordinates: "
           << status_or_device_coodinates.status().error_message();

  // Determine compilation and execution devices.
  auto status_or_tpu_device_assignment =
      tensorflow::GetTPUCompilationAndExecutionDevices(
          devices.device_names(), /*num_replicas=*/1,
          num_cores_per_replica_attr.getInt(), topology_attr.getValue(),
          std::move(status_or_device_coodinates).value());
  if (!status_or_tpu_device_assignment.ok())
    return cluster.emitError()
           << "error in fetching TPU compilation/execution devices: "
           << status_or_tpu_device_assignment.status().error_message();
  auto& tpu_device_assignment = status_or_tpu_device_assignment.value();

  *host_device = tpu_device_assignment.tpu_devices[0][0].host;
  return mlir::success();
}

}  // anonymous namespace

StatusOr<llvm::SmallVector<int64_t, 8>> GetDeviceCoordinates(
    mlir::ArrayAttr device_assignment_attr) {
  llvm::SmallVector<int64_t, 8> device_coordinates;
  device_coordinates.reserve(device_assignment_attr.size());

  for (auto device_coordinate_and_idx :
       llvm::enumerate(device_assignment_attr)) {
    auto device_coordinate =
        device_coordinate_and_idx.value().dyn_cast<mlir::IntegerAttr>();
    if (!device_coordinate)
      return errors::InvalidArgument(
          llvm::formatv(kBadIntArrayElementMsg, kDeviceAssignmentAttr,
                        device_coordinate_and_idx.index())
              .str());

    device_coordinates.push_back(device_coordinate.getInt());
  }

  return device_coordinates;
}

StatusOr<TPUDeviceAssignment> GetTPUCompilationAndExecutionDevices(
    Devices devices, int num_replicas, int num_cores_per_replica,
    llvm::StringRef topology_attr,
    llvm::ArrayRef<int64_t> device_assignment_attr) {
  // Collect TPU_SYSTEM devices.
  llvm::SmallVector<Device, 8> system_devices;
  TF_RETURN_IF_ERROR(GetTPUSystemDevices(devices, &system_devices));

  // Collect TPU devices based on TPU_SYSTEM devices collected earlier.
  llvm::SmallVector<llvm::SmallVector<Device, 8>, 8> tpu_devices;
  TF_RETURN_IF_ERROR(GetTPUDevices(devices, system_devices, &tpu_devices));

  std::string compilation_device = GetTPUCompilationDevice(system_devices[0]);

  if (topology_attr.empty()) {
    if (!device_assignment_attr.empty())
      return errors::InvalidArgument("'", kDeviceAssignmentAttr,
                                     "' must not be set when '", kTopologyAttr,
                                     "' is not set");

    TF_ASSIGN_OR_RETURN(auto execution_devices,
                        GetFullMeshTPUExecutionDeviceAssignment(
                            num_replicas, num_cores_per_replica, tpu_devices));
    return TPUDeviceAssignment(compilation_device,
                               std::move(execution_devices));
  }

  TF_ASSIGN_OR_RETURN(auto devices_and_ids,
                      GetGeneralTPUExecutionDeviceAssignment(
                          num_replicas, num_cores_per_replica, tpu_devices,
                          topology_attr, device_assignment_attr));
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

mlir::LogicalResult GetHostDeviceOutsideComputation(
    mlir::TF::RuntimeDevices devices, mlir::tf_device::ClusterOp cluster,
    std::string* host_device) {
  if (HasTPUDevice(devices) ||
      cluster->getParentOfType<mlir::tf_device::ReplicateOp>())
    return GetHostDeviceOCInTPUPipeline(devices, cluster, host_device);
  return GetHostDeviceOCInGenericPipeline(devices, host_device);
}

bool IsTPUDevice(llvm::StringRef device) {
  Device parsed_device;
  if (!DeviceNameUtils::ParseFullName(mlir::StringRefToView(device),
                                      &parsed_device))
    return false;
  return parsed_device.has_type && parsed_device.type == kDeviceTPU;
}

bool IsTPUReplicatedCore(llvm::StringRef device) {
  Device parsed_device;
  if (!DeviceNameUtils::ParseFullName(mlir::StringRefToView(device),
                                      &parsed_device))
    return false;
  return parsed_device.has_type && parsed_device.type == kTPUReplicatedCore;
}

bool TypeValidForXLA(const mlir::Type& type) {
  const mlir::Type elem = getElementTypeOrSelf(type);
  return !elem.isa<mlir::TF::ResourceType>() &&
         !elem.isa<mlir::TF::StringType>();
}

}  // namespace tensorflow
