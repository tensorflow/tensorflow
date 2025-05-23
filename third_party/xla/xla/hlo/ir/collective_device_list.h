/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_HLO_IR_COLLECTIVE_DEVICE_LIST_H_
#define XLA_HLO_IR_COLLECTIVE_DEVICE_LIST_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/printer.h"
#include "xla/service/hlo.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {

std::string ReplicaGroupsToString(
    absl::Span<const ReplicaGroup> replica_groups);

// Represents a list of replica groups (a list of list of devices) with
// reshaping and transposing an iota array (iota tile assignment). Can be used
// to represent certain common patterns of device lists in a compact, scalable
// format.
class IotaReplicaGroupList {
 public:
  explicit IotaReplicaGroupList(int64_t num_replica_groups,
                                int64_t num_devices_per_group)
      : iota_tile_assignment_(IotaTileAssignment::Create(
            {num_replica_groups, num_devices_per_group})),
        num_replica_groups_(num_replica_groups),
        num_devices_per_group_(num_devices_per_group) {}

  explicit IotaReplicaGroupList(int64_t num_replica_groups,
                                int64_t num_devices_per_group,
                                absl::Span<const int64_t> reshape_dims,
                                absl::Span<const int> transpose_perm)
      : iota_tile_assignment_(IotaTileAssignment::Create(
            {num_replica_groups, num_devices_per_group}, reshape_dims,
            transpose_perm)),
        num_replica_groups_(num_replica_groups),
        num_devices_per_group_(num_devices_per_group) {}

  bool operator==(const IotaReplicaGroupList& other) const {
    return num_replica_groups() == other.num_replica_groups() &&
           num_devices_per_group() == other.num_devices_per_group() &&
           reshape_dims() == other.reshape_dims() &&
           transpose_perm() == other.transpose_perm();
  }

  int64_t num_replica_groups() const;
  int64_t num_devices_per_group() const;
  absl::Span<const int64_t> reshape_dims() const {
    return iota_tile_assignment_.reshape_dims();
  }
  absl::Span<const int> transpose_perm() const {
    return iota_tile_assignment_.transpose_perm();
  }
  Array<int64_t> ToArray() const { return iota_tile_assignment_.ToArray(); }

  void Print(Printer* printer) const;

  std::string ToString() const;

  IotaReplicaGroupListProto ToProto() const;

  static IotaReplicaGroupList FromProto(const IotaReplicaGroupListProto& proto);

 private:
  IotaTileAssignment iota_tile_assignment_;
  int64_t num_replica_groups_ = -1;
  int64_t num_devices_per_group_ = -1;
};

// Represents a series of devices participating in a collective operation
// (all-gather, all-reduce, etc.). While this directly translates to a list of
// replica groups, it may be used to represent these lists in compact forms.
class CollectiveDeviceList {
 public:
  explicit CollectiveDeviceList()
      : replica_groups_(std::make_shared<std::vector<ReplicaGroup>>()) {};

  explicit CollectiveDeviceList(std::vector<ReplicaGroup> replica_groups)
      : replica_groups_(std::make_shared<std::vector<ReplicaGroup>>(
            std::move(replica_groups))) {};

  explicit CollectiveDeviceList(absl::Span<const ReplicaGroup> replica_groups)
      : replica_groups_(std::make_shared<std::vector<ReplicaGroup>>(
            replica_groups.begin(), replica_groups.end())) {};

  explicit CollectiveDeviceList(
      absl::Span<const std::vector<int64_t>> replica_groups)
      : replica_groups_(ToReplicaGroupVector(replica_groups)) {};

  // Replica groups are materialized lazily upon first access.
  explicit CollectiveDeviceList(
      const IotaReplicaGroupList& iota_replica_group_list)
      : iota_replica_group_list_(iota_replica_group_list) {}

  // Lazyly explands iota if applicable.
  const std::vector<ReplicaGroup>& replica_groups() const;
  const std::optional<IotaReplicaGroupList>& iota_replica_group_list() const {
    return iota_replica_group_list_;
  }

  int64_t num_replica_groups() const {
    return iota_replica_group_list_.has_value()
               ? iota_replica_group_list_->num_replica_groups()
               : replica_groups_->size();
  }

  int64_t num_devices_per_group() const {
    return iota_replica_group_list_.has_value()
               ? iota_replica_group_list_->num_devices_per_group()
               : replica_groups_->begin()->replica_ids_size();
  }

  void Print(Printer* printer,
             bool print_full_replica_group_list = false) const;

  std::string ToString(bool print_full_replica_group_list = false) const;

  CollectiveDeviceListProto ToProto() const;
  static CollectiveDeviceList FromProto(const CollectiveDeviceListProto& proto);
  static CollectiveDeviceList FromProto(const HloInstructionProto& proto);

 private:
  // Construct collective device list from protobuf replica group start and end
  // iterators.
  CollectiveDeviceList(
      tsl::protobuf::RepeatedPtrField<ReplicaGroup>::const_iterator start,
      tsl::protobuf::RepeatedPtrField<ReplicaGroup>::const_iterator end)
      : replica_groups_(
            std::make_shared<std::vector<ReplicaGroup>>(start, end)) {};

  static std::shared_ptr<std::vector<ReplicaGroup>> ToReplicaGroupVector(
      absl::Span<const std::vector<int64_t>> replica_groups) {
    std::shared_ptr<std::vector<ReplicaGroup>> result =
        std::make_shared<std::vector<ReplicaGroup>>();
    result->reserve(replica_groups.size());
    for (const std::vector<int64_t>& g : replica_groups) {
      auto& group = result->emplace_back();
      group.mutable_replica_ids()->Add(g.begin(), g.end());
    }
    return result;
  }

  // Load replica groups from iota tile assignment if not already done so.
  void MaybeMaterializeFullReplicaGroupList() const;

  std::optional<IotaReplicaGroupList> iota_replica_group_list_;
  // shared_ptr for fast copy.
  mutable std::shared_ptr<std::vector<ReplicaGroup>> replica_groups_ = nullptr;
};

}  // namespace xla

#endif  // XLA_HLO_IR_COLLECTIVE_DEVICE_LIST_H_
