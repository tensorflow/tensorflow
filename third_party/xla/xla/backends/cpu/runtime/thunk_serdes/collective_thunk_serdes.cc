/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/thunk_serdes/collective_thunk_serdes.h"

#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/cpu/runtime/all_gather_thunk.h"
#include "xla/backends/cpu/runtime/all_reduce_thunk.h"
#include "xla/backends/cpu/runtime/all_to_all_thunk.h"
#include "xla/backends/cpu/runtime/collective_permute_thunk.h"
#include "xla/backends/cpu/runtime/collective_thunk.h"
#include "xla/backends/cpu/runtime/reduce_scatter_thunk.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/backends/cpu/runtime/thunk_proto_serdes.h"
#include "xla/backends/cpu/runtime/thunk_proto_serdes_utils.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"

namespace xla::cpu {
namespace {

absl::StatusOr<CollectiveThunk::OpParams> OpParamsFromProto(
    const OpParamsProto& proto) {
  CollectiveThunk::OpParams op_params;
  op_params.has_channel_id = proto.has_channel_id();
  if (proto.use_global_device_ids().contains_value()) {
    op_params.use_global_device_ids = proto.use_global_device_ids().value();
  }
  op_params.op_id = proto.op_id();
  for (const auto& replica_group_proto : proto.replica_group()) {
    ReplicaGroup replica_group;
    for (const auto& replica_id : replica_group_proto.replica_ids()) {
      replica_group.add_replica_ids(replica_id);
    }
    op_params.group.push_back(replica_group);
  }
  return op_params;
}

absl::StatusOr<OpParamsProto> OpParamsToProto(
    const CollectiveThunk::OpParams& op_params) {
  OpParamsProto proto;
  proto.set_has_channel_id(op_params.has_channel_id);

  proto.mutable_use_global_device_ids()->set_contains_value(
      op_params.use_global_device_ids.has_value());
  if (op_params.use_global_device_ids) {
    proto.mutable_use_global_device_ids()->set_value(
        *op_params.use_global_device_ids);
  }

  proto.set_op_id(op_params.op_id);
  for (const auto& group : op_params.group) {
    ReplicaGroup* replica_group = proto.add_replica_group();
    for (const auto& device : group.replica_ids()) {
      replica_group->add_replica_ids(device);
    }
  }
  return proto;
}

absl::StatusOr<std::tuple<CollectiveThunk::OpParams, CollectiveThunk::OpBuffers,
                          CollectiveThunk::OpResources>>
GetCollectiveThunkParamsFromProto(
    const CollectiveThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations,
    const std::vector<std::shared_ptr<Resource>>& resources) {
  TF_ASSIGN_OR_RETURN(CollectiveThunk::OpParams op_params,
                      OpParamsFromProto(proto.op_params()));

  CollectiveThunk::OpBuffers op_buffers;
  for (const auto& shape_buffer_slice_proto :
       proto.op_buffers().source_shapes_buffer_slices()) {
    TF_ASSIGN_OR_RETURN(auto slice_shape,
                        DeserializeSliceShapeFromProto(shape_buffer_slice_proto,
                                                       buffer_allocations));
    const auto& [slice, shape] = slice_shape;
    op_buffers.source_buffers.push_back(slice);
    op_buffers.source_shapes.push_back(shape);
  }

  for (const auto& shape_buffer_slice_proto :
       proto.op_buffers().destination_shapes_buffer_slices()) {
    TF_ASSIGN_OR_RETURN(auto slice_shape,
                        DeserializeSliceShapeFromProto(shape_buffer_slice_proto,
                                                       buffer_allocations));

    const auto& [slice, shape] = slice_shape;
    op_buffers.destination_buffers.push_back(slice);
    op_buffers.destination_shapes.push_back(shape);
  }

  CollectiveThunk::OpResources op_resources;
  if (proto.op_resources().communicator_resource().has_value()) {
    if (resources.size() != 1) {
      return Internal(
          "Expected exactly one resource for collective thunk, but got %d "
          "resources.",
          resources.size());
    }

    op_resources.communicator_resource = resources[0];

    // Validate that the serialized resource has the same type as the
    // resource we are setting.
    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<Resource> communicator_resource_from_proto,
        CreateResourceFromProto(
            proto.op_resources().communicator_resource().value()));

    if (communicator_resource_from_proto->kind() !=
        op_resources.communicator_resource->kind()) {
      return Internal(
          "Resource kind mismatch between global resource state %d and "
          "serialized resource %d.",
          op_resources.communicator_resource->kind(),
          communicator_resource_from_proto->kind());
    }
  } else {
    op_resources.communicator_resource = nullptr;
  }

  return std::make_tuple(op_params, op_buffers, op_resources);
}

absl::Status AllGatherToProto(const AllGatherThunk& thunk,
                              AllGatherThunkProto& proto) {
  return absl::OkStatus();
}

absl::Status AllReduceToProto(const AllReduceThunk& thunk,
                              AllReduceThunkProto& proto) {
  std::string reduction_kind_as_string = absl::StrCat(thunk.reduction_kind());
  proto.set_reduction_kind(reduction_kind_as_string);
  proto.set_single_replica(thunk.single_replica());
  return absl::OkStatus();
}

absl::Status AllToAllToProto(const AllToAllThunk& thunk,
                             AllToAllThunkProto& proto) {
  return absl::OkStatus();
}

absl::Status ReduceScatterToProto(const ReduceScatterThunk& thunk,
                                  ReduceScatterThunkProto& proto) {
  std::string reduction_kind_as_string = absl::StrCat(thunk.reduction_kind());
  proto.set_reduction_kind(reduction_kind_as_string);
  return absl::OkStatus();
}

absl::Status CollectivePermuteToProto(const CollectivePermuteThunk& thunk,
                                      CollectivePermuteThunkProto& proto) {
  for (const auto& source_target_pair : thunk.source_target_pairs()) {
    CollectivePermuteThunkProto::SourceTargetPairProto*
        source_target_pair_proto = proto.add_source_target_pairs();
    source_target_pair_proto->set_source(source_target_pair.first);
    source_target_pair_proto->set_target(source_target_pair.second);
  }
  return absl::OkStatus();
}

absl::Status CollectiveThunkToProto(const Thunk& thunk, ThunkProto& proto) {
  const auto& collective_thunk = absl::down_cast<const CollectiveThunk&>(thunk);
  CollectiveThunkProto* collective_thunk_proto =
      proto.mutable_collective_thunk();

  TF_ASSIGN_OR_RETURN(*collective_thunk_proto->mutable_op_params(),
                      OpParamsToProto(collective_thunk.op_params()));

  collective_thunk_proto->mutable_op_resources()
      ->mutable_communicator_resource()
      ->set_contains_value(
          collective_thunk.op_resources().communicator_resource != nullptr);
  if (collective_thunk.op_resources().communicator_resource != nullptr) {
    TF_ASSIGN_OR_RETURN(
        *collective_thunk_proto->mutable_op_resources()
             ->mutable_communicator_resource()
             ->mutable_value(),
        ToProto(*collective_thunk.op_resources().communicator_resource));
  }

  for (size_t i = 0; i < collective_thunk.op_buffers().source_buffers.size();
       ++i) {
    TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
        collective_thunk.op_buffers().source_buffers[i],
        collective_thunk.op_buffers().source_shapes[i],
        collective_thunk_proto->mutable_op_buffers()
            ->add_source_shapes_buffer_slices()));
  }

  for (size_t i = 0;
       i < collective_thunk.op_buffers().destination_buffers.size(); ++i) {
    TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
        collective_thunk.op_buffers().destination_buffers[i],
        collective_thunk.op_buffers().destination_shapes[i],
        collective_thunk_proto->mutable_op_buffers()
            ->add_destination_shapes_buffer_slices()));
  }

  switch (collective_thunk.collective_kind()) {
    case CollectiveThunk::CollectiveKind::kAllGather:
      TF_RETURN_IF_ERROR(AllGatherToProto(
          absl::down_cast<const AllGatherThunk&>(collective_thunk),
          *collective_thunk_proto->mutable_all_gather_thunk()));
      break;
    case CollectiveThunk::CollectiveKind::kAllReduce:
      TF_RETURN_IF_ERROR(AllReduceToProto(
          absl::down_cast<const AllReduceThunk&>(collective_thunk),
          *collective_thunk_proto->mutable_all_reduce_thunk()));
      break;
    case CollectiveThunk::CollectiveKind::kAllToAll:
      TF_RETURN_IF_ERROR(AllToAllToProto(
          absl::down_cast<const AllToAllThunk&>(collective_thunk),
          *collective_thunk_proto->mutable_all_to_all_thunk()));
      break;
    case CollectiveThunk::CollectiveKind::kReduceScatter:
      TF_RETURN_IF_ERROR(ReduceScatterToProto(
          absl::down_cast<const ReduceScatterThunk&>(collective_thunk),
          *collective_thunk_proto->mutable_reduce_scatter_thunk()));
      break;
    case CollectiveThunk::CollectiveKind::kCollectivePermute:
      TF_RETURN_IF_ERROR(CollectivePermuteToProto(
          absl::down_cast<const CollectivePermuteThunk&>(collective_thunk),
          *collective_thunk_proto->mutable_collective_permute_thunk()));
      break;
  }

  return absl::OkStatus();
}

absl::StatusOr<CollectiveThunk::CollectiveKind>
ProtoCollectiveThunkToCollectiveThunkKind(const CollectiveThunkProto& proto) {
  switch (proto.impl_case()) {
    case CollectiveThunkProto::ImplCase::kAllGatherThunk:
      return CollectiveThunk::CollectiveKind::kAllGather;
    case CollectiveThunkProto::ImplCase::kAllReduceThunk:
      return CollectiveThunk::CollectiveKind::kAllReduce;
    case CollectiveThunkProto::ImplCase::kAllToAllThunk:
      return CollectiveThunk::CollectiveKind::kAllToAll;
    case CollectiveThunkProto::ImplCase::kCollectivePermuteThunk:
      return CollectiveThunk::CollectiveKind::kCollectivePermute;
    case CollectiveThunkProto::ImplCase::kReduceScatterThunk:
      return CollectiveThunk::CollectiveKind::kReduceScatter;
    case CollectiveThunkProto::ImplCase::IMPL_NOT_SET:
      return Internal("Collective thunk kind not set.");
  }
}

absl::StatusOr<std::unique_ptr<AllGatherThunk>> AllGatherThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations,
    const std::vector<std::shared_ptr<Resource>>& resources) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  TF_ASSIGN_OR_RETURN(
      auto collective_thunk_params,
      GetCollectiveThunkParamsFromProto(proto.collective_thunk(),
                                        buffer_allocations, resources));

  const auto& [op_params, op_buffers, op_resources] = collective_thunk_params;
  return AllGatherThunk::Create(info, op_params, op_buffers, op_resources);
}

absl::StatusOr<std::unique_ptr<AllReduceThunk>> AllReduceThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations,
    const std::vector<std::shared_ptr<Resource>>& resources) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  TF_ASSIGN_OR_RETURN(
      auto collective_thunk_params,
      GetCollectiveThunkParamsFromProto(proto.collective_thunk(),
                                        buffer_allocations, resources));

  const auto& [op_params, op_buffers, op_resources] = collective_thunk_params;
  TF_ASSIGN_OR_RETURN(
      ReductionKind reduction_kind,
      ParseReductionKind(
          proto.collective_thunk().all_reduce_thunk().reduction_kind()));

  return AllReduceThunk::Create(
      info, reduction_kind, op_params, op_buffers, op_resources,
      proto.collective_thunk().all_reduce_thunk().single_replica());
}

absl::StatusOr<std::unique_ptr<AllToAllThunk>> AllToAllThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations,
    const std::vector<std::shared_ptr<Resource>>& resources) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));
  TF_ASSIGN_OR_RETURN(
      auto collective_thunk_params,
      GetCollectiveThunkParamsFromProto(proto.collective_thunk(),
                                        buffer_allocations, resources));

  const auto& [op_params, op_buffers, op_resources] = collective_thunk_params;
  return AllToAllThunk::Create(info, op_params, op_buffers, op_resources);
}

absl::StatusOr<std::unique_ptr<CollectivePermuteThunk>>
CollectivePermuteThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations,
    const std::vector<std::shared_ptr<Resource>>& resources) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  TF_ASSIGN_OR_RETURN(
      auto collective_thunk_params,
      GetCollectiveThunkParamsFromProto(proto.collective_thunk(),
                                        buffer_allocations, resources));

  const auto& [op_params, op_buffers, op_resources] = collective_thunk_params;
  std::vector<CollectivePermuteThunk::SourceTargetPair> source_target_pairs;
  for (const auto& source_target_pair_proto : proto.collective_thunk()
                                                  .collective_permute_thunk()
                                                  .source_target_pairs()) {
    source_target_pairs.push_back(
        {source_target_pair_proto.source(), source_target_pair_proto.target()});
  }
  return CollectivePermuteThunk::Create(info, op_params, op_buffers,
                                        op_resources, source_target_pairs);
}

absl::StatusOr<std::unique_ptr<ReduceScatterThunk>> ReduceScatterThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations,
    const std::vector<std::shared_ptr<Resource>>& resources) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  TF_ASSIGN_OR_RETURN(
      auto collective_thunk_params,
      GetCollectiveThunkParamsFromProto(proto.collective_thunk(),
                                        buffer_allocations, resources));

  const auto& [op_params, op_buffers, op_resources] = collective_thunk_params;

  TF_ASSIGN_OR_RETURN(
      ReductionKind reduction_kind,
      ParseReductionKind(
          proto.collective_thunk().reduce_scatter_thunk().reduction_kind()));
  return ReduceScatterThunk::Create(info, reduction_kind, op_params, op_buffers,
                                    op_resources);
}

absl::StatusOr<std::unique_ptr<Thunk>> CollectiveThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations,
    const HloModule* hlo_module,
    const std::vector<std::shared_ptr<Resource>>* resources) {
  if (resources == nullptr) {
    return Internal("CollectiveThunk serdes requires resources.");
  }

  TF_ASSIGN_OR_RETURN(
      CollectiveThunk::CollectiveKind collective_kind,
      ProtoCollectiveThunkToCollectiveThunkKind(proto.collective_thunk()));

  switch (collective_kind) {
    case CollectiveThunk::CollectiveKind::kAllGather:
      return AllGatherThunkFromProto(proto, buffer_allocations, *resources);
    case CollectiveThunk::CollectiveKind::kAllReduce:
      return AllReduceThunkFromProto(proto, buffer_allocations, *resources);
    case CollectiveThunk::CollectiveKind::kAllToAll:
      return AllToAllThunkFromProto(proto, buffer_allocations, *resources);
    case CollectiveThunk::CollectiveKind::kCollectivePermute:
      return CollectivePermuteThunkFromProto(proto, buffer_allocations,
                                             *resources);
    case CollectiveThunk::CollectiveKind::kReduceScatter:
      return ReduceScatterThunkFromProto(proto, buffer_allocations, *resources);
  }
}

}  // namespace

void RegisterCollectiveThunkSerDes() {
  CHECK_OK(ThunkSerDesRegistry::Get().Register(Thunk::Kind::kCollective,
                                               CollectiveThunkToProto,
                                               CollectiveThunkFromProto));
}

// Statically registers the CollectiveThunk serialization/deserialization logic.
static bool collective_thunk_serdes_registered = [] {
  RegisterCollectiveThunkSerDes();
  return true;
}();

}  // namespace xla::cpu
