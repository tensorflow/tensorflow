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

#include "xla/hlo/ir/collective_device_list.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/service/hlo.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "tsl/platform/protobuf.h"

namespace xla {

std::string ReplicaGroupsToString(
    absl::Span<const ReplicaGroup> replica_groups) {
  std::vector<std::string> replica_group_str;
  replica_group_str.reserve(replica_groups.size());
  for (const ReplicaGroup& group : replica_groups) {
    replica_group_str.push_back(
        absl::StrCat("{", absl::StrJoin(group.replica_ids(), ","), "}"));
  }
  return absl::StrCat("{", absl::StrJoin(replica_group_str, ","), "}");
}

int64_t IotaReplicaGroupList::num_replica_groups() const {
  DCHECK_GE(num_replica_groups_, 0);
  return num_replica_groups_;
}

int64_t IotaReplicaGroupList::num_devices_per_group() const {
  DCHECK_GE(num_devices_per_group_, 0);
  return num_devices_per_group_;
}

std::string IotaReplicaGroupList::ToString() const {
  return iota_tile_assignment_.ToString();
}

IotaReplicaGroupListProto IotaReplicaGroupList::ToProto() const {
  IotaReplicaGroupListProto proto;
  proto.set_num_replica_groups(num_replica_groups_);
  proto.set_num_devices_per_group(num_devices_per_group_);
  proto.mutable_iota_reshape_dims()->Assign(
      iota_tile_assignment_.reshape_dims().begin(),
      iota_tile_assignment_.reshape_dims().end());
  proto.mutable_iota_transpose_perm()->Assign(
      iota_tile_assignment_.transpose_perm().begin(),
      iota_tile_assignment_.transpose_perm().end());
  return proto;
}

IotaReplicaGroupList IotaReplicaGroupList::FromProto(
    const IotaReplicaGroupListProto& proto) {
  return IotaReplicaGroupList(
      proto.num_replica_groups(), proto.num_devices_per_group(),
      std::vector<int64_t>(proto.iota_reshape_dims().begin(),
                           proto.iota_reshape_dims().end()),
      std::vector<int>(proto.iota_transpose_perm().begin(),
                       proto.iota_transpose_perm().end()));
}

CollectiveDeviceList::CollectiveDeviceList(
    tsl::protobuf::RepeatedPtrField<ReplicaGroup>::const_iterator start,
    tsl::protobuf::RepeatedPtrField<ReplicaGroup>::const_iterator end) {
  replica_groups_shared_ =
      std::make_shared<std::vector<ReplicaGroup>>(start, end);
  replica_groups_ = replica_groups_shared_.get();
}

CollectiveDeviceList::CollectiveDeviceList(
    absl::Span<const ReplicaGroup> replica_groups) {
  replica_groups_shared_ = std::make_shared<std::vector<ReplicaGroup>>(
      replica_groups.begin(), replica_groups.end());
  replica_groups_ = replica_groups_shared_.get();
}

CollectiveDeviceList::CollectiveDeviceList(
    absl::Span<const std::vector<int64_t>> replica_groups) {
  auto rg_list = std::make_shared<std::vector<ReplicaGroup>>();
  rg_list->reserve(replica_groups.size());
  for (auto g : replica_groups) {
    auto& group = rg_list->emplace_back();
    *group.mutable_replica_ids() = {g.begin(), g.end()};
  }
  replica_groups_shared_ = std::move(rg_list);
  replica_groups_ = replica_groups_shared_.get();
}

CollectiveDeviceList::CollectiveDeviceList() {
  replica_groups_shared_ = std::make_shared<std::vector<ReplicaGroup>>();
  replica_groups_ = replica_groups_shared_.get();
}

void CollectiveDeviceList::MaybeMaterializeFullReplicaGroupList() const {
  if (replica_groups_ != nullptr) {
    VLOG(10) << "Replica group list already materialized.";
    return;
  }

  DCHECK(iota_replica_group_list_.has_value());
  VLOG(10) << "Materializing full replica group list";

  auto rg_list = std::make_shared<std::vector<ReplicaGroup>>();
  const int64_t num_replica_groups =
      iota_replica_group_list_->num_replica_groups();
  rg_list->reserve(num_replica_groups);

  auto array = iota_replica_group_list_->ToArray();
  // Iota replica group list array must only have 2 dimensions.
  DCHECK_EQ(array.num_dimensions(), 2);
  const int64_t num_devices_per_group =
      iota_replica_group_list_->num_devices_per_group();
  DCHECK_EQ(array.end() - array.begin(),
            num_devices_per_group * num_replica_groups);
  for (auto it = array.begin(), end = array.end(); it != end;
       it += num_devices_per_group) {
    *rg_list->emplace_back().mutable_replica_ids() = {
        it, it + num_devices_per_group};
  }

  replica_groups_shared_ = std::move(rg_list);
  replica_groups_ = replica_groups_shared_.get();
}

const std::vector<ReplicaGroup>& CollectiveDeviceList::replica_groups() const {
  MaybeMaterializeFullReplicaGroupList();
  return *replica_groups_;
}

std::string CollectiveDeviceList::ToString(
    bool print_full_replica_group_list) const {
  if (iota_replica_group_list_.has_value() && !print_full_replica_group_list) {
    return iota_replica_group_list_->ToString();
  }

  return ReplicaGroupsToString(replica_groups());
}

CollectiveDeviceListProto CollectiveDeviceList::ToProto() const {
  CollectiveDeviceListProto proto;
  if (iota_replica_group_list_.has_value()) {
    *(proto.mutable_iota_replica_group_list()) =
        iota_replica_group_list_->ToProto();
    return proto;
  }

  proto.mutable_replica_groups()->Assign(replica_groups().begin(),
                                         replica_groups().end());
  return proto;
}

CollectiveDeviceList CollectiveDeviceList::FromProto(
    const CollectiveDeviceListProto& proto) {
  if (proto.has_iota_replica_group_list()) {
    return CollectiveDeviceList(
        IotaReplicaGroupList::FromProto(proto.iota_replica_group_list()));
  }

  if (proto.replica_groups_size() > 0) {
    return CollectiveDeviceList(proto.replica_groups().begin(),
                                proto.replica_groups().end());
  }

  return CollectiveDeviceList();
}

CollectiveDeviceList CollectiveDeviceList::FromProto(
    const HloInstructionProto& proto) {
  // Create CollectiveDeviceList from legacy field (replica_groups) if it is
  // populated.
  if (proto.replica_groups_size() > 0) {
    VLOG(10) << "Creating collective device list from proto using legacy "
                "replica groups field.";
    return CollectiveDeviceList(proto.replica_groups().begin(),
                                proto.replica_groups().end());
  }

  if (!proto.has_collective_device_list()) {
    return CollectiveDeviceList();
  }

  // Create CollectiveDeviceList from non-legacy field (collective_device_list).
  return FromProto(proto.collective_device_list());
}

}  // namespace xla
