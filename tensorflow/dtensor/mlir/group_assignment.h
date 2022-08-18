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

#ifndef TENSORFLOW_DTENSOR_MLIR_GROUP_ASSIGNMENT_H_
#define TENSORFLOW_DTENSOR_MLIR_GROUP_ASSIGNMENT_H_

#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/dtensor/cc/dstatus.h"

namespace tensorflow {
namespace dtensor {

// Arranges all replica IDs in a DTensor mesh in groups, used as an attribute
// on collective operations.
//
// A group assignment has two views:
//
// - The global mesh view contains replica IDs from all participant TPU slices.
//   These replica IDs are identical to global device IDs in a DTensor mesh.
// - The local slice view contains per-slice device IDs understood and used by
//   the TPU runtime on each slice. These device IDs are used to set replica
//   IDs on each slice.
//
// Some notable common cases:
//
// - In a single-slice case, `slice_size` is set to the actual slice size
//   (e.g., 32 for 4x4 DF). The global and local views are identical.
// - In a special topology case, `slice_size` is set to 8.
// - In a multi-topology case, `slice_size` is set to the size of a single
// topology.
//   All topologies must have the same size.
class GroupAssignment {
 public:
  using ReplicaId = int;

  struct DeviceId {
   public:
    int slice_id;
    int core_id;  // within `slice_id`
  };

  // Maps global replica IDs to local device IDs consisting of a slice ID and a
  // core-on-slice ID.
  class ReplicaToDeviceMap {
   public:
    // Creates a default map that orders devices according to TF task IDs
    // followed by device ordinals.
    static ReplicaToDeviceMap DefaultReplicaToDeviceMap(int num_slices,
                                                        int slice_size);

    // Constructs a map directly, checking it's valid.
    explicit ReplicaToDeviceMap(absl::flat_hash_map<ReplicaId, DeviceId> map);

    int num_slices() { return num_slices_; }
    int num_cores() { return map_.size(); }
    DeviceId device_id(ReplicaId replica_id) { return map_[replica_id]; }

   private:
    absl::flat_hash_map<ReplicaId, DeviceId> map_;
    int num_slices_;
  };

  // Creates a group assignment by converting from an MLIR attribute.
  static StatusOr<GroupAssignment> FromMLIR(
      const mlir::DenseIntElementsAttr& group_assignment_attr,
      ReplicaToDeviceMap replica_to_device_map);

  // Creates an MLIR attribute using the global view.
  mlir::DenseIntElementsAttr GlobalToMLIR(mlir::MLIRContext& context) const {
    return global_.ToMLIR(context);
  }

  // Creates an MLIR attribute for a particular slice.
  // Callers should make sure `slice_id` is >= 0 and < num_slices().
  StatusOr<mlir::DenseIntElementsAttr> SliceToMLIR(mlir::MLIRContext& context,
                                                   int slice_id) const {
    if (slice_id < 0 || slice_id >= num_slices())
      return errors::InvalidArgument("slide_id was not within bounds.");
    return slices_[slice_id].ToMLIR(context);
  }

  // Returns a string representation for debugging.
  std::string ToString() const;

  // Returns true if every group in the global view only has replica IDs from
  // the same slice.
  bool IsWithinSlices() const;

  // Returns the number of slices in the local view.
  int num_slices() const { return slices_.size(); }

  // These methods return attributes of the global view.
  int num_groups() const { return global_.num_groups(); }
  int group_size() const { return global_.group_size(); }
  int num_replica_ids() const { return global_.num_replica_ids(); }
  const std::vector<std::vector<int>>& replica_ids() const {
    return global_.replica_ids();
  }

  // These methods return attributes of a particular slice.
  // Callers should make sure `slice_id` is >= 0 and < num_slices().
  StatusOr<int> num_groups(int slice_id) const {
    if (slice_id < 0 || slice_id >= num_slices())
      return errors::InvalidArgument("slide_id was not within bounds.");
    return slices_[slice_id].num_groups();
  }
  StatusOr<int> group_size(int slice_id) const {
    if (slice_id < 0 || slice_id >= num_slices())
      return errors::InvalidArgument("slide_id was not within bounds.");
    return slices_[slice_id].group_size();
  }
  const std::vector<std::vector<int>>& replica_ids(int slice_id) const {
    return slices_[slice_id].replica_ids();
  }

  // Returns the replica groups for collectives running on a particular host.
  // Callers should make sure `slice_id` is >= 0 and < num_slices().
  const std::vector<std::vector<int>>& host_replica_ids(int slice_id) const {
    return hosts_[slice_id].replica_ids();
  }

 private:
  // Groups of consecutive replica IDs starting at 0.
  class ReplicaGroups {
   public:
    // Creates an object, enforcing the requirements on `replica_ids_`.
    explicit ReplicaGroups(std::vector<std::vector<int>> replica_ids);

    mlir::DenseIntElementsAttr ToMLIR(mlir::MLIRContext& context) const;

    std::string ToString() const;

    int num_groups() const { return replica_ids_.size(); }
    int group_size() const { return replica_ids_.front().size(); }
    int num_replica_ids() const { return num_groups() * group_size(); }
    const std::vector<std::vector<int>>& replica_ids() const {
      return replica_ids_;
    }

   private:
    // N groups of replica IDs, N > 0. All groups have the same size G, G > 0.
    // All replica IDs are distinct values >= 0;
    std::vector<std::vector<int>> replica_ids_;  // replica ID order matters
  };

  // Creates an object but leaves `slices_` empty. `GlobalToSlices` should be
  // called next to fill in `slices_`.
  explicit GroupAssignment(ReplicaGroups global,
                           ReplicaToDeviceMap replica_to_device_map)
      : global_(std::move(global)),
        replica_to_device_map_(std::move(replica_to_device_map)) {}

  // Divides the global view along slice boundaries and fill in the slice view.
  Status GlobalToSlices();

  ReplicaGroups global_;
  std::vector<ReplicaGroups> hosts_;   // sorted by increasing slice ID
  std::vector<ReplicaGroups> slices_;  // sorted by increasing slice ID
  ReplicaToDeviceMap replica_to_device_map_;
};

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_MLIR_GROUP_ASSIGNMENT_H_
