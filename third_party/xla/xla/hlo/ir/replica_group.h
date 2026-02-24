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

#include "absl/container/flat_hash_map.h"
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

class IotaReplicaGroupList;
class CollectiveDeviceList;

enum class CollectiveDeviceListVersion { kListOfLists, kIota, kMeshAxes };

// Base class providing the interface for all collective device list
// representations.
class CollectiveDeviceListBase {
 public:
  virtual ~CollectiveDeviceListBase() = default;
  CollectiveDeviceListBase() = default;
  CollectiveDeviceListBase(const CollectiveDeviceListBase&) = default;
  CollectiveDeviceListBase& operator=(const CollectiveDeviceListBase&) =
      default;
  CollectiveDeviceListBase(CollectiveDeviceListBase&&) = default;
  CollectiveDeviceListBase& operator=(CollectiveDeviceListBase&&) = default;

  std::optional<IotaReplicaGroupList> MaybeConvertToIotaReplicaGroupList()
      const;
  // This is strict equality, which means that two different types
  // can't be compared for functional equality (i.e. even though an
  // IotaReplicaGroup and a CollectiveDeviceList may correspond to the same
  // underlying set of device groups, they will compare as unequal).
  friend bool operator==(const CollectiveDeviceListBase& lhs,
                         const CollectiveDeviceListBase& rhs) {
    if (typeid(lhs) != typeid(rhs)) {
      return false;
    }
    // If types are the same, delegate to the derived implementation
    return lhs.isEqual(rhs);
  }

  virtual int64_t num_replica_groups() const = 0;
  virtual int64_t num_devices_per_group() const = 0;
  int64_t num_total_devices() const {
    return num_replica_groups() * num_devices_per_group();
  }
  virtual std::vector<std::vector<int64_t>> flattened_replica_groups()
      const = 0;

  virtual const std::vector<ReplicaGroup>& replica_groups() const {
    if (replica_groups_ != nullptr) {
      return *replica_groups_;
    }
    replica_groups_ = std::make_shared<std::vector<ReplicaGroup>>();
    replica_groups_->reserve(num_replica_groups());
    for (const auto& group : flattened_replica_groups()) {
      ReplicaGroup replica_group;
      replica_group.mutable_replica_ids()->Add(group.begin(), group.end());
      replica_groups_->push_back(std::move(replica_group));
    }
    return *replica_groups_;
  };

  virtual void Print(Printer* printer) const = 0;
  virtual void Print(Printer* printer,
                     bool print_full_replica_group_list) const {
    Print(printer);
  }
  virtual std::string ToString() const = 0;
  virtual std::string ToString(bool print_full_replica_group_list) const {
    return ToString();
  };

  static std::unique_ptr<CollectiveDeviceListBase> DeviceListFromProto(
      const HloInstructionProto& proto);

  virtual std::unique_ptr<CollectiveDeviceListBase> Clone() const = 0;
  virtual CollectiveDeviceListVersion version() const = 0;

 protected:
  // Used by operator== to check equality of derived types.
  virtual bool isEqual(const CollectiveDeviceListBase& other) const = 0;

  // shared_ptr for fast copy and lazy materialization.
  mutable std::shared_ptr<std::vector<ReplicaGroup>> replica_groups_ = nullptr;
};

// Compact representation using Mesh and Axis indices.
class MeshAxesReplicaGroupList : public CollectiveDeviceListBase {
 public:
  explicit MeshAxesReplicaGroupList(Mesh mesh, std::vector<AxisRef> axes);

  bool operator==(const MeshAxesReplicaGroupList& other) const {
    return mesh_ == other.mesh_ && axes_ == other.axes_;
  }

  template <typename H>
  friend H AbslHashValue(H h, const MeshAxesReplicaGroupList& c) {
    return H::combine(std::move(h), c.mesh_, c.axes_);
  }

  // Overrides
  int64_t num_replica_groups() const override;
  int64_t num_devices_per_group() const override;
  std::vector<std::vector<int64_t>> flattened_replica_groups() const override;
  void Print(Printer* printer) const override;
  std::string ToString() const override;
  MeshAxesReplicaGroupListProto ToProto() const;

  std::unique_ptr<CollectiveDeviceListBase> Clone() const override {
    return std::make_unique<MeshAxesReplicaGroupList>(*this);
  }
  CollectiveDeviceListVersion version() const override {
    return CollectiveDeviceListVersion::kMeshAxes;
  }

  // Conversion and Serialization
  static MeshAxesReplicaGroupList FromProto(
      const MeshAxesReplicaGroupListProto& proto);
  IotaReplicaGroupList ToIotaReplicaGroupList() const;
  CollectiveDeviceList ToCollectiveDeviceList() const;

 protected:
  bool isEqual(const CollectiveDeviceListBase& other) const override {
    return *this == static_cast<const MeshAxesReplicaGroupList&>(other);
  }

 private:
  struct ReshapeAndAggregateAxes {
    std::vector<int64_t> reshape_dims;
    std::vector<int64_t> aggregate_axes;
  };

  // Internal helpers for computing device groups.
  absl::flat_hash_map<int64_t, ReshapeAndAggregateAxes>
  GetDimToReshapeAndAggregateAxes() const;
  std::pair<std::vector<int64_t>, std::vector<int64_t>> ComputeReindexedAxes()
      const;

  Mesh mesh_;
  std::vector<AxisRef> axes_;
};

// Representation using Iota patterns (reshaping/transposing linear ranges).
class IotaReplicaGroupList : public CollectiveDeviceListBase {
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

  explicit IotaReplicaGroupList(int64_t num_replica_groups,
                                int64_t num_devices_per_group,
                                const IotaTileAssignment& iota_tile_assignment)
      : iota_tile_assignment_(iota_tile_assignment),
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

  int64_t num_replica_groups() const override;
  int64_t num_devices_per_group() const override;
  absl::Span<const int64_t> reshape_dims() const {
    return iota_tile_assignment_.reshape_dims();
  }
  absl::Span<const int> transpose_perm() const {
    return iota_tile_assignment_.transpose_perm();
  }
  Array<int64_t> ToArray() const { return iota_tile_assignment_.ToArray(); }
  std::vector<std::vector<int64_t>> flattened_replica_groups() const override;

  void Print(Printer* printer) const override;
  std::string ToString() const override;

  std::unique_ptr<CollectiveDeviceListBase> Clone() const override {
    return std::make_unique<IotaReplicaGroupList>(*this);
  }
  CollectiveDeviceListVersion version() const override {
    return CollectiveDeviceListVersion::kIota;
  }

  IotaReplicaGroupListProto ToProto() const;
  static IotaReplicaGroupList FromProto(const IotaReplicaGroupListProto& proto);

 protected:
  bool isEqual(const CollectiveDeviceListBase& other) const override {
    return *this == static_cast<const IotaReplicaGroupList&>(other);
  }

 private:
  IotaTileAssignment iota_tile_assignment_;
  int64_t num_replica_groups_ = -1;
  int64_t num_devices_per_group_ = -1;
};

// Legacy/Explicit representation using an explicit list of ReplicaGroups.
class CollectiveDeviceList : public CollectiveDeviceListBase {
 public:
  explicit CollectiveDeviceList() {
    replica_groups_ = std::make_shared<std::vector<ReplicaGroup>>();
  };

  explicit CollectiveDeviceList(std::vector<ReplicaGroup> replica_groups) {
    replica_groups_ =
        std::make_shared<std::vector<ReplicaGroup>>(std::move(replica_groups));
  };

  explicit CollectiveDeviceList(absl::Span<const ReplicaGroup> replica_groups) {
    replica_groups_ = std::make_shared<std::vector<ReplicaGroup>>(
        replica_groups.begin(), replica_groups.end());
  };

  explicit CollectiveDeviceList(
      absl::Span<const std::vector<int64_t>> replica_groups) {
    replica_groups_ = ToReplicaGroupVector(replica_groups);
  };

  bool operator==(const CollectiveDeviceList& other) const {
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

  // Overrides
  const std::vector<ReplicaGroup>& replica_groups() const override;
  std::vector<std::vector<int64_t>> flattened_replica_groups() const override;

  int64_t num_replica_groups() const override {
    return replica_groups_->size();
  }

  int64_t num_devices_per_group() const override {
    return replica_groups_->empty()
               ? 0
               : replica_groups_->begin()->replica_ids_size();
  }

  void Print(Printer* printer) const override;
  void Print(Printer* printer,
             bool print_full_replica_group_list) const override;
  std::string ToString() const override;
  std::string ToString(bool print_full_replica_group_list) const override;

  CollectiveDeviceListVersion version() const override {
    return CollectiveDeviceListVersion::kListOfLists;
  }

  CollectiveDeviceListProto ToProto() const;
  static CollectiveDeviceList FromProto(const CollectiveDeviceListProto& proto);
  static CollectiveDeviceList FromProto(const HloInstructionProto& proto);

  std::unique_ptr<CollectiveDeviceListBase> Clone() const override {
    return std::make_unique<CollectiveDeviceList>(*this);
  };

 protected:
  bool isEqual(const CollectiveDeviceListBase& other) const override {
    return *this == static_cast<const CollectiveDeviceList&>(other);
  }

 private:
  CollectiveDeviceList(
      tsl::protobuf::RepeatedPtrField<ReplicaGroup>::const_iterator start,
      tsl::protobuf::RepeatedPtrField<ReplicaGroup>::const_iterator end) {
    replica_groups_ = std::make_shared<std::vector<ReplicaGroup>>(start, end);
  };

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

  void MaybeMaterializeFullReplicaGroupList() const;
};

std::string ReplicaGroupsToString(
    absl::Span<const ReplicaGroup> replica_groups);
CollectiveDeviceList ConvertToV1CollectiveDeviceList(
    const CollectiveDeviceListBase& device_list);

}  // namespace xla

#endif  // XLA_HLO_IR_REPLICA_GROUP_H_
