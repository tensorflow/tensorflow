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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_TPU_REWRITE_DEVICE_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_TPU_REWRITE_DEVICE_UTIL_H_

#include <optional>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
using tsl::StatusOr;

inline constexpr absl::string_view kNumCoresPerReplicaAttr =
    "num_cores_per_replica";
inline constexpr absl::string_view kTopologyAttr = "topology";
inline constexpr absl::string_view kDeviceAssignmentAttr = "device_assignment";

// A TPU device for execution alongside its associated host CPU device.
struct TPUDeviceAndHost {
  TPUDeviceAndHost() = default;
  TPUDeviceAndHost(llvm::StringRef device, llvm::StringRef host)
      : device(device), host(host) {}

  std::string device;
  std::string host;
};

// TPU devices to be used for execution (e.g. devices for TPUExecute ops) and
// their associated host CPU devices (for outside compilation). They are ordered
// by `num_replicas` followed by `num_cores_per_replica`.
using TPUDevicesAndHosts =
    llvm::SmallVector<llvm::SmallVector<TPUDeviceAndHost, 8>, 8>;

// TPU compilation device, execution and associated host devices, and optionally
// execution device IDs. Execution device IDs are populated if `topology` and
// `device_assignment` are provided.
struct TPUDeviceAssignment {
  TPUDeviceAssignment(llvm::StringRef compilation_device,
                      TPUDevicesAndHosts&& tpu_devices)
      : compilation_device(compilation_device),
        tpu_devices(std::move(tpu_devices)) {}

  TPUDeviceAssignment(llvm::StringRef compilation_device,
                      TPUDevicesAndHosts&& tpu_devices,
                      xla::DeviceAssignmentProto&& xla_device_assignment)
      : compilation_device(compilation_device),
        tpu_devices(std::move(tpu_devices)),
        xla_device_assignment(std::move(xla_device_assignment)) {}

  std::string compilation_device;
  TPUDevicesAndHosts tpu_devices;
  std::optional<xla::DeviceAssignmentProto> xla_device_assignment;
};

// Extracts device coordinates from a device assignment attribute on an op.
StatusOr<llvm::SmallVector<int64_t, 8>> GetDeviceCoordinates(
    mlir::ArrayAttr device_assignment_attr);

// Finds the TPU compilation device and execution devices from `devices` for a
// TPU computation subgraph. Compilation device is determined from looking up
// all TPU_SYSTEM:0 devices and choosing the CPU device associated to the first
// TPU_SYSTEM device sorted lexicographically by replica and task. Execution
// devices are determined by looking up all TPU devices associated with each
// TPU_SYSTEM:0 device found, alongside associated `topology_attr` and
// `device_assignment_attr`. If `topology_attr` not an empty string (parsable to
// TopologyProto), `device_assignment_attr` must not be empty also. When
// `topology_attr` and `device_assignment_attr` are not empty, a general device
// assignment based on those two attributes are used. Otherwise when
// `topology_attr` and `device_assignment_attr` are empty, a full mesh device
// assignment is used instead. A failure will be returned if it is not possible
// (e.g. invalid devices or invalid parameters).
//
//
// For example, for `devices`:
//   {
//     /job:localhost/replica:0/task:0/device:CPU:0,
//     /job:worker/replica:0/task:0/device:CPU:0,
//     /job:worker/replica:0/task:0/device:TPU_SYSTEM:0,
//     /job:worker/replica:0/task:0/device:TPU:0,
//     /job:worker/replica:0/task:0/device:TPU:1,
//     /job:worker/replica:0/task:0/device:TPU:2,
//     /job:worker/replica:0/task:0/device:TPU:3,
//     /job:worker/replica:0/task:1/device:CPU:0,
//     /job:worker/replica:0/task:1/device:TPU_SYSTEM:0,
//     /job:worker/replica:0/task:1/device:TPU:0,
//     /job:worker/replica:0/task:1/device:TPU:1,
//     /job:worker/replica:0/task:1/device:TPU:2,
//     /job:worker/replica:0/task:1/device:TPU:3
//   }
//
//
// With the following parameters (full mesh device assignment):
//   `num_replicas` = 8
//   `num_cores_per_replica` = 1
//   `topology_attr` = ""
//   `device_assignment_attr` = {}
//
// The `compilation_device` will be:
//   /job:worker/replica:0/task:0/device:CPU:0
//
// `execution_devices` will be:
//   {
//     {
//       /job:worker/replica:0/task:0/device:TPU:0
//     },
//     {
//       /job:worker/replica:0/task:0/device:TPU:1
//     },
//     {
//       /job:worker/replica:0/task:0/device:TPU:2
//     },
//     {
//       /job:worker/replica:0/task:0/device:TPU:3
//     },
//     {
//       /job:worker/replica:0/task:1/device:TPU:0
//     },
//     {
//       /job:worker/replica:0/task:1/device:TPU:1
//     },
//     {
//       /job:worker/replica:0/task:1/device:TPU:2
//     },
//     {
//       /job:worker/replica:0/task:1/device:TPU:3
//     }
//   }
//
// and `xla_device_assignment` will not be set.
//
//
// With the following parameters (general device assignment):
//   `num_replicas` = 4
//   `num_cores_per_replica` = 2
//   `topology_attr` (in proto debug string format) =
//     {
//       mesh_shape: 2
//       mesh_shape: 2
//       mesh_shape: 2
//       num_tasks: 2
//       num_tpu_devices_per_task: 4
//       device_coordinates: 0
//       device_coordinates: 0
//       device_coordinates: 0
//       device_coordinates: 0
//       device_coordinates: 1
//       device_coordinates: 0
//       device_coordinates: 1
//       device_coordinates: 1
//       device_coordinates: 0
//       device_coordinates: 1
//       device_coordinates: 0
//       device_coordinates: 0
//       device_coordinates: 1
//       device_coordinates: 0
//       device_coordinates: 1
//       device_coordinates: 1
//       device_coordinates: 1
//       device_coordinates: 1
//       device_coordinates: 0
//       device_coordinates: 1
//       device_coordinates: 1
//       device_coordinates: 0
//       device_coordinates: 0
//       device_coordinates: 1
//     }
//   `device_assignment` =
//     {0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1}
//
// The `compilation_device` will be:
//   /job:worker/replica:0/task:0/device:CPU:0
//
// `execution_devices` will be:
//   {
//     {
//       "/job:worker/replica:0/task:0/device:TPU:0",
//       "/job:worker/replica:0/task:1/device:TPU:3"
//     },
//     {
//       "/job:worker/replica:0/task:0/device:TPU:1",
//       "/job:worker/replica:0/task:1/device:TPU:2"
//     },
//     {
//       "/job:worker/replica:0/task:0/device:TPU:3",
//       "/job:worker/replica:0/task:1/device:TPU:0"
//     },
//     {
//       "/job:worker/replica:0/task:0/device:TPU:2",
//       "/job:worker/replica:0/task:1/device:TPU:1"
//     }
//   }
//
// and `xla_device_assignment` will be:
//   {
//     replica_count: 4
//     computation_count: 2
//     computation_devices {
//       replica_device_ids: 0
//       replica_device_ids: 4
//       replica_device_ids: 2
//       replica_device_ids: 6
//     }
//     computation_devices {
//       replica_device_ids: 1
//       replica_device_ids: 5
//       replica_device_ids: 3
//       replica_device_ids: 7
//     }
//   }
StatusOr<TPUDeviceAssignment> GetTPUCompilationAndExecutionDevices(
    llvm::ArrayRef<DeviceNameUtils::ParsedName> devices, int num_replicas,
    int num_cores_per_replica, llvm::StringRef topology_attr,
    llvm::ArrayRef<int64_t> device_assignment_attr);

// Virtual device name of the passed logical core. The logical core is the index
// of a core within a replica.
std::string GetDeviceAliasForLogicalCore(int core_index);

// Virtual device name of the host that is associated with the passed logical
// core. The logical core is the index of a core within a replica.
std::string GetDeviceAliasForHostOfLogicalCore(int core_index);

// Returns true if cluster contains model parallelism based on
// `num_cores_per_replica_attribute`. Otherwise returns false.
bool HasModelParallelism(mlir::tf_device::ClusterOp cluster);

// Returns true if the devices list contain any TPU devices
bool HasTPUDevice(const mlir::TF::RuntimeDevices& devices);

// Returns the host device used for outside compilation in generic pipeline.
mlir::LogicalResult GetHostDeviceOutsideCompilationInGenericPipeline(
    mlir::TF::RuntimeDevices devices, std::string* host_device);

// Parses XLA compilation and execution devices from a tf_device.cluster and
// returns the host device for the head and tail computations. For TPU device,
// if the computation is replicated, GetDeviceAliasForHostOfLogicalCore(0) is
// returned instead.
mlir::LogicalResult GetHostDeviceOutsideComputation(
    mlir::TF::RuntimeDevices devices, mlir::tf_device::ClusterOp cluster,
    std::string* host_device);

// Checks if a device string is a TPU device.
bool IsTPUDevice(llvm::StringRef device);

// Checks if a device string is a TPU replicated core device.
bool IsTPUReplicatedCore(llvm::StringRef device);

// Checks if `type` is allowed for XLA. String and resources are not XLA types.
// There are other TF types that are not XLA types which will be removed by
// successive passes in TF/XLA bridge phase 2.
bool TypeValidForXLA(const mlir::Type& type);

// Returns the map from core to the host that is associated with the
// core. If `cluster` is not replicated then the core is a physical core index
// and the host is a physical host name. If `cluster` is replicated then the
// core with index `i` is a logical core (`TPU_REPLICATED_CORE_i`), and the host
// is the associated virtual device name (`TPU_REPLICATED_HOST_i`).
mlir::LogicalResult GetDeviceToHostMap(
    mlir::tf_device::ClusterOp cluster,
    llvm::SmallVector<std::string, 8>& core_to_host);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_TPU_REWRITE_DEVICE_UTIL_H_
