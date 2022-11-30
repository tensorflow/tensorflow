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

#include "tensorflow/dtensor/mlir/group_assignment.h"

#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/dtensor/cc/dstatus.h"

namespace tensorflow {
namespace dtensor {

GroupAssignment::ReplicaToDeviceMap
GroupAssignment::ReplicaToDeviceMap::DefaultReplicaToDeviceMap(int num_slices,
                                                               int slice_size) {
  absl::flat_hash_map<ReplicaId, DeviceId> map;
  for (int i = 0; i < num_slices; ++i) {
    for (int j = 0; j < slice_size; ++j) {
      map[ReplicaId{i * slice_size + j}] = DeviceId{i, j};
    }
  }
  return ReplicaToDeviceMap(std::move(map));
}

GroupAssignment::ReplicaToDeviceMap::ReplicaToDeviceMap(
    absl::flat_hash_map<ReplicaId, DeviceId> map)
    : map_(std::move(map)) {
  std::set<int> slice_ids;
  for (const auto& entry : map_) {
    slice_ids.insert(entry.second.slice_id);
  }
  CHECK_GT(slice_ids.size(), 0);                // Crash OK
  CHECK_EQ(map_.size() % slice_ids.size(), 0);  // Crash OK
  num_slices_ = slice_ids.size();
}

GroupAssignment::ReplicaGroups::ReplicaGroups(
    std::vector<std::vector<int>> replica_ids)
    : replica_ids_(std::move(replica_ids)) {
  int n = replica_ids_.size();
  CHECK_GT(n, 0);  // Crash OK
  int g = replica_ids_.front().size();
  CHECK_GT(g, 0);  // Crash OK
  std::set<int> seen_replica_ids;
  for (std::vector<int>& group : replica_ids_) {
    CHECK_EQ(group.size(), g);  // Crash OK
    for (int replica_id : group) {
      CHECK_GE(replica_id, 0);  // Crash OK
      bool inserted = seen_replica_ids.insert(replica_id).second;
      CHECK(inserted);  // Crash OK
    }
  }
}

mlir::DenseIntElementsAttr GroupAssignment::ReplicaGroups::ToMLIR(
    mlir::MLIRContext& context) const {
  auto shaped_type = mlir::RankedTensorType::get(
      {num_groups(), group_size()}, mlir::IntegerType::get(&context, 32));

  llvm::SmallVector<int32, 4> flat_replica_ids;
  flat_replica_ids.reserve(num_replica_ids());
  for (const std::vector<int>& group : replica_ids()) {
    flat_replica_ids.insert(flat_replica_ids.end(), group.begin(), group.end());
  }

  return mlir::DenseIntElementsAttr::get(shaped_type, flat_replica_ids);
}

std::string GroupAssignment::ReplicaGroups::ToString() const {
  return strings::StrCat(
      "[",
      str_util::Join(replica_ids(), ", ",
                     [](std::string* str, const std::vector<int>& group) {
                       strings::StrAppend(str, "[", str_util::Join(group, ", "),
                                          "]");
                     }),
      "]");
}

StatusOr<GroupAssignment> GroupAssignment::FromMLIR(
    const mlir::DenseIntElementsAttr& group_assignment_attr,
    ReplicaToDeviceMap replica_to_device_map) {
  mlir::ShapedType shaped_type = group_assignment_attr.getType();
  if (!shaped_type.hasRank()) {
    return errors::InvalidArgument("group_assignment_attr must have a rank");
  }
  if (shaped_type.getRank() != 2) {
    return errors::InvalidArgument(
        "group_assignment_attr must have a rank of 2, got ",
        shaped_type.getRank());
  }
  llvm::ArrayRef<int64_t> shape = shaped_type.getShape();
  int num_groups = shape[0];
  if (num_groups <= 0) {
    return errors::InvalidArgument(
        "group_assignment_attr must have at least 1 group, got ", num_groups);
  }
  int group_size = shape[1];
  if (group_size <= 0) {
    return errors::InvalidArgument(
        "group_assignment_attr must have non-empty groups, got ", group_size,
        " replica IDs per group");
  }
  int num_replica_ids = num_groups * group_size;
  if (num_replica_ids != replica_to_device_map.num_cores()) {
    return errors::InvalidArgument("group_assignment_attr must have ",
                                   replica_to_device_map.num_cores(),
                                   " replica IDs, got ", num_replica_ids);
  }

  // Translate the flat group assignment to a 2D array.
  std::vector<std::vector<int>> replica_ids;
  replica_ids.resize(num_groups, std::vector<int>(group_size));
  std::set<int> seen_replica_ids;
  if (group_assignment_attr.getNumElements() != num_replica_ids) {
    return errors::InvalidArgument(
        "group_assignments_attr num elements was not equal to the number of "
        "replica ids.");
  }
  for (const auto& it :
       llvm::enumerate(group_assignment_attr.getValues<llvm::APInt>())) {
    int index = it.index();
    int replica_id = it.value().getSExtValue();

    // If all replica IDs are within this range and distinct, they must be a
    // permutation of [0, ..., num_replica_ids).
    if (replica_id < 0 || replica_id >= num_replica_ids) {
      return errors::InvalidArgument("Out of range replica ID: ", replica_id);
    }
    if (!seen_replica_ids.insert(replica_id).second) {
      return errors::InvalidArgument(
          "All replica IDs in group_assigment must be distinct, seeing ",
          replica_id, " more than once");
    }

    replica_ids[index / group_size][index % group_size] = replica_id;
  }

  GroupAssignment group_assignment(
      /*global=*/ReplicaGroups(std::move(replica_ids)),
      std::move(replica_to_device_map));
  TF_RETURN_IF_ERROR(group_assignment.GlobalToSlices());
  return group_assignment;
}

std::string GroupAssignment::ToString() const {
  return strings::StrCat(
      "GroupAssignment global: ", global_.ToString(), "; hosts: ",
      hosts_.empty()
          ? "<none>"
          : str_util::Join(hosts_, ", ",
                           [](std::string* str, const ReplicaGroups& groups) {
                             strings::StrAppend(str, groups.ToString());
                           }),
      "; slices: ",
      slices_.empty()
          ? "<none>"
          : str_util::Join(slices_, ", ",
                           [](std::string* str, const ReplicaGroups& groups) {
                             strings::StrAppend(str, groups.ToString());
                           }));
}

bool GroupAssignment::IsWithinSlices() const {
  // This function returns true iff no group in the global view gets split in
  // `GlobalToSlices`, i.e., the total group count remains the same.
  int total_num_groups = 0;
  for (int i = 0; i < num_slices(); i++) {
    total_num_groups += num_groups(i).value();
  }
  if (total_num_groups != num_groups()) return false;
  return total_num_groups == num_groups();
}

Status GroupAssignment::GlobalToSlices() {
  VLOG(2) << "Original group assignment: " << ToString();

  int num_slices = replica_to_device_map_.num_slices();
  if (num_slices == 0) {
    return errors::InvalidArgument("Unexpectedly empty replica_to_device_map.");
  }

  // For each replica group in global replica groups, divide its replicas based
  // on which slices they come from. Then, for each slice, collect subgroups
  // from every such division and form a new ReplicaGroup for that slice.
  std::vector<std::vector<std::vector<int>>> replica_groups_per_host;
  std::vector<std::vector<std::vector<int>>> replica_groups_per_slice;
  replica_groups_per_host.resize(num_slices, {});
  replica_groups_per_slice.resize(num_slices, {});

  for (const std::vector<int>& replica_group : replica_ids()) {
    std::vector<std::vector<int>> replica_group_divided_by_host;
    replica_group_divided_by_host.resize(num_slices, {});
    std::vector<std::vector<int>> replica_group_divided_by_slice;
    replica_group_divided_by_slice.resize(num_slices, {});

    for (int replica_id : replica_group) {
      // TODO(b/183426911): Use DeviceId::core_id in ReplicaGroup directly for
      // now. Integrate with device assignment with proper typing.
      DeviceId device_id = replica_to_device_map_.device_id(replica_id);
      replica_group_divided_by_host[device_id.slice_id].push_back(replica_id);
      replica_group_divided_by_slice[device_id.slice_id].push_back(
          device_id.core_id);
    }

    for (int i = 0; i < num_slices; ++i) {
      if (!replica_group_divided_by_host[i].empty()) {
        // Host meshes have the same global device and replica IDs as TPU
        // meshes. Let the first replica in every group do a host collective.
        replica_groups_per_host[i].push_back(
            std::vector<int>(1, replica_group_divided_by_host[i].front()));
      }
      if (!replica_group_divided_by_slice[i].empty()) {
        replica_groups_per_slice[i].push_back(
            std::move(replica_group_divided_by_slice[i]));
      }
    }
  }

  hosts_.reserve(num_slices);
  slices_.reserve(num_slices);
  for (int i = 0; i < num_slices; ++i) {
    hosts_.push_back(ReplicaGroups(std::move(replica_groups_per_host[i])));
    slices_.push_back(ReplicaGroups(std::move(replica_groups_per_slice[i])));
  }

  VLOG(2) << "Divided group assignment: " << ToString();
  return OkStatus();
}

}  // namespace dtensor
}  // namespace tensorflow
