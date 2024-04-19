/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/mlir/collectives_common.h"

#include <map>
#include <string>
#include <vector>

namespace tensorflow {
namespace dtensor {

// A map from a unique set of kept mesh dimension values (a partition) to
// IDs of devices in that partition.
//
// Users will typically ignore the key, but use the map values as the group
// assignment for collective operations. This is intentionally a
// std::map instead of absl::flat_hash_map to guarantee all hosts in
// a multi-host cluster will generate the same grouping, and therefore the same
// XLA program fingerprint, independently. std::map guarantees the same
// iteration order.
using AllReducePartitions = std::map<DeviceLocation, std::vector<int32>>;

// Computes AllReduce partitions using reduced mesh dimension names.
//
// Reduction groups are formed across all _non_-reduced dimensions. For example,
// in the following scenario:
//
// output_layout.dims() = [a, b]
// output_layout.mesh() = [(x, 8), (y, 4)]
// reduced_dims = `x`
//
// We first reduce over `a` locally on each device, producing 32 local
// reductions. We then AllReduce within each of the 4 partitions. Each partition
// corresponds to one unique value of `y` and has 8 devices. The end result is
// sharded over the y mesh dimension and replicated 8 times.
//
// The returned map should have four entries with key values from [0] to [3]
// (unique values of `y`). Each key maps to IDs of devices with that `y` value.
StatusOr<AllReducePartitions> GetAllReducePartitionsFromReducedDims(
    const dtensor::Layout& output_layout,
    const absl::flat_hash_set<std::string>& reduced_dims) {
  AllReducePartitions partitions;
  for (int64 device = 0; device < output_layout.num_devices(); ++device) {
    TF_ASSIGN_OR_RETURN(const DeviceLocation device_loc,
                        output_layout.mesh().device_location(device));
    DeviceLocation kept_dims;
    for (int64 dim_idx = 0; dim_idx < device_loc.size(); ++dim_idx) {
      if (!reduced_dims.contains(output_layout.mesh().dim_name(dim_idx))) {
        kept_dims.push_back(device_loc[dim_idx]);
      }
    }
    partitions[kept_dims].push_back(device);
  }
  return partitions;
}

// Use the first device in the mesh to extract the device name. For example:
//
// device_path = "/job:localhost/replica:0/task:0/device:TPU:0"
// device_type = "/job:localhost/replica:0/task:0/device:TPU"
// device_id = 0
//
// The device ID can be obtained through DeviceId as a runtime input. We may
// need it in the future to enable device ID-based branch divergence.
StatusOr<std::string> DeviceTypeFromMesh(const Mesh& mesh) {
  std::string device_path =
      mesh.is_remote() ? mesh.global_devices()[0] : mesh.local_devices()[0];
  size_t device_path_pos = device_path.find_last_of(':');
  if (device_path_pos == std::string::npos) {
    return errors::InvalidArgument("Unexpected device path: ", device_path);
  }
  return device_path.substr(0, device_path_pos);
}

}  // namespace dtensor
}  // namespace tensorflow
