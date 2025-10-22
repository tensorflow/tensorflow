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

#ifndef XLA_HLO_IR_REPLICA_GROUP_H_
#define XLA_HLO_IR_REPLICA_GROUP_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "xla/array.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/printer.h"
#include "xla/service/hlo.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {

class MeshAxesReplicaGroupList {
 public:
  explicit MeshAxesReplicaGroupList(Mesh mesh, std::vector<AxisRef> axes)
      : mesh_(std::move(mesh)), axes_(std::move(axes)) {}

  bool operator==(const MeshAxesReplicaGroupList& other) const {
    return mesh_ == other.mesh_ && axes_ == other.axes_;
  }

  template <typename H>
  friend H AbslHashValue(H h, const MeshAxesReplicaGroupList& c) {
    return H::combine(std::move(h), c.mesh_, c.axes_);
  }

  int64_t num_replica_groups() const;
  int64_t num_devices_per_group() const;

  void Print(Printer* printer) const;

  std::string ToString() const;

  MeshAxesReplicaGroupListProto ToProto() const;

  static MeshAxesReplicaGroupList FromProto(
      const MeshAxesReplicaGroupListProto& proto);

 private:
  Mesh mesh_;
  std::vector<AxisRef> axes_;
};

class CartesianProduct {
 public:
  explicit CartesianProduct(std::vector<int64_t> limits)
      : limits_(std::move(limits)) {}

  struct Iterator {
    std::vector<int64_t> current;  // The current state, e.g., {0, 1, 0}
    const std::vector<int64_t>* limits;
    bool at_end;

    explicit Iterator(const std::vector<int64_t>* p_limits, bool is_end = false)
        : current(p_limits->size(), 0), limits(p_limits), at_end(is_end) {
      // If any of the limits are <= 0, (or the list is empty) then the entire
      // product is empty. In this case there is nothing to iterate over.
      if (!at_end) {
        at_end = limits->empty() || std::any_of(limits->begin(), limits->end(),
                                                [](int l) { return l <= 0; });
      }
    }

    std::vector<int64_t> operator*() const { return current; }

    bool operator!=(const Iterator& other) const {
      return at_end != other.at_end;
    }

    Iterator& operator++() {
      // Increment tuple using odometer logic (from right to left or least to
      // most significant digits).
      for (int i = limits->size() - 1; i >= 0; --i) {
        if (++current[i] < (*limits)[i]) {
          return *this;
        }
        current[i] = 0;
      }
      at_end = true;
      return *this;
    }
  };
  Iterator begin() const { return Iterator(&limits_); }
  Iterator end() const { return Iterator(&limits_, true); }

 private:
  std::vector<int64_t> limits_;
};

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

  template <typename H>
  friend H AbslHashValue(H h, const IotaReplicaGroupList& c) {
    return H::combine(std::move(h), c.num_replica_groups_,
                      c.num_devices_per_group_, c.reshape_dims(),
                      c.transpose_perm());
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
  std::vector<std::vector<int64_t>> flattened_replica_groups() const;

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

  bool operator==(const CollectiveDeviceList& other) const {
    if (iota_replica_group_list_.has_value() &&
        other.iota_replica_group_list_.has_value()) {
      return *iota_replica_group_list_ == *other.iota_replica_group_list_;
    }
    const auto& this_groups = replica_groups();
    const auto& other_groups = other.replica_groups();
    if (this_groups.size() != other_groups.size()) {
      return false;
    }
    for (size_t i = 0; i < this_groups.size(); ++i) {
      if (!tsl::protobuf::util::MessageDifferencer::Equals(this_groups[i],
                                                           other_groups[i])) {
        return false;
      }
    }
    return true;
  }

  template <typename H>
  friend H AbslHashValue(H h, const CollectiveDeviceList& c) {
    const auto& groups = c.replica_groups();
    h = H::combine(std::move(h), groups.size());
    for (const auto& group : groups) {
      h = H::combine_contiguous(std::move(h), group.replica_ids().data(),
                                group.replica_ids().size());
    }
    return h;
  }

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

#endif  // XLA_HLO_IR_REPLICA_GROUP_H_
