/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/collectives/all_reduce_blueconnect.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/global_device_id.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

std::vector<HloInstruction*> GetOutputs(HloInstruction& instruction) {
  if (!instruction.shape().IsTuple()) {
    return {&instruction};
  }

  std::vector<HloInstruction*> outputs;
  outputs.reserve(instruction.shape().tuple_shapes().size());

  HloComputation& computation = *instruction.parent();  // never null
  for (int i = 0; i < instruction.shape().tuple_shapes().size(); ++i) {
    outputs.push_back(computation.AddInstruction(
        HloInstruction::CreateGetTupleElement(&instruction, i)));
  }
  return outputs;
}

struct DecomposedReplicaGroups {
  std::vector<ReplicaGroup> scatter_gather_groups;
  std::vector<ReplicaGroup> new_all_reduce_groups;
};

// Returns the global device id for the given replica id. Returns nullopt if
// if the replica id can refer to multiple devices, or if the pass does not
// support the CollectiveOpGroupMode.
std::optional<GlobalDeviceId> TryConvertingReplicaIdToDeviceId(
    int64_t replica_id, const DeviceAssignment& device_assignment,
    CollectiveOpGroupMode collective_group_mode) {
  if (collective_group_mode == CollectiveOpGroupMode::kCrossReplica) {
    if (device_assignment.computation_count() != 1) {
      // If there are multiple partitions, the replica_id may refer to multiple
      // devices on different partitions.
      return std::nullopt;
    }
    return GlobalDeviceId{device_assignment(replica_id, /*computation_id=*/0)};
  } else if (collective_group_mode == CollectiveOpGroupMode::kFlattenedID) {
    int partition_count = device_assignment.computation_count();
    int64_t actual_replica_id = replica_id / partition_count;
    int64_t partition_id = replica_id % partition_count;
    return GlobalDeviceId{device_assignment(actual_replica_id, partition_id)};
  }

  // kCrossPartition and kCrossReplicaAndPartition are unsupported.
  VLOG(1) << "Skip AllReduceBlueConnect because of unsupported "
             "CollectiveOpGroupMode "
          << CollectiveOpGroupModeToString(collective_group_mode);
  return std::nullopt;
}

absl::StatusOr<std::optional<DecomposedReplicaGroups>> TryDecomposeReplicaGroup(
    const ReplicaGroup& replica_group,
    const DeviceAssignment& device_assignment, size_t num_devices_per_host,
    CollectiveOpGroupMode collective_group_mode) {
  int group_size = replica_group.replica_ids_size();
  TF_RET_CHECK(group_size > 0);

  absl::btree_map<int, std::vector<int64_t>> replica_ids_by_host;
  for (int64_t replica_id : replica_group.replica_ids()) {
    std::optional<GlobalDeviceId> device_id = TryConvertingReplicaIdToDeviceId(
        replica_id, device_assignment, collective_group_mode);
    if (!device_id.has_value()) {
      return {std::nullopt};
    }
    TF_RET_CHECK(*device_id >= 0);
    // We assume that devices are ordered by host.
    int host_id = device_id->value() / num_devices_per_host;
    replica_ids_by_host[host_id].push_back(replica_id);
  }

  size_t num_local_devices = replica_ids_by_host.begin()->second.size();
  bool same_num_devices_on_each_host =
      absl::c_all_of(replica_ids_by_host, [&](const auto& entry) {
        return entry.second.size() == num_local_devices;
      });

  if (!same_num_devices_on_each_host) {
    return {std::nullopt};
  }

  std::vector<int64_t> sorted_replica_group;
  sorted_replica_group.reserve(group_size);
  for (const auto& entry : replica_ids_by_host) {
    absl::c_copy(entry.second, std::back_inserter(sorted_replica_group));
  }

  size_t scatter_group_size = std::max(num_local_devices, size_t(2));
  size_t num_scatter_groups = group_size / scatter_group_size;

  if ((group_size % scatter_group_size != 0) || (num_scatter_groups < 2)) {
    return {std::nullopt};
  }

  std::vector<ReplicaGroup> scatter_gather_groups(num_scatter_groups);
  std::vector<ReplicaGroup> new_all_reduce_groups(scatter_group_size);

  for (size_t i = 0; i < group_size; ++i) {
    int64_t replica_id = sorted_replica_group[i];
    scatter_gather_groups[i / scatter_group_size].add_replica_ids(replica_id);
    new_all_reduce_groups[i % scatter_group_size].add_replica_ids(replica_id);
  }

  return {DecomposedReplicaGroups{std::move(scatter_gather_groups),
                                  std::move(new_all_reduce_groups)}};
}

absl::StatusOr<std::optional<DecomposedReplicaGroups>>
TryDecomposeReplicaGroups(const HloAllReduceInstruction& all_reduce,
                          size_t num_devices_per_host) {
  const DeviceAssignment& device_assignment =
      all_reduce.GetModule()->config().static_device_assignment();

  absl::Span<const ReplicaGroup> replica_groups = all_reduce.replica_groups();

  ReplicaGroup all_replicas;  // only populated if replica groups not present.
  if (replica_groups.empty()) {
    for (int i = 0; i < device_assignment.replica_count(); ++i) {
      all_replicas.add_replica_ids(i);
    }
    replica_groups = absl::MakeSpan(&all_replicas, 1);
  }

  TF_ASSIGN_OR_RETURN(
      CollectiveOpGroupMode collective_op_group_mode,
      GetCollectiveOpGroupMode(all_reduce.channel_id().has_value(),
                               all_reduce.use_global_device_ids()));

  std::vector<ReplicaGroup> scatter_gather_groups;
  std::vector<ReplicaGroup> new_all_reduce_groups;

  // Try to find a valid decomposition for each replica group.
  for (const ReplicaGroup& replica_group : replica_groups) {
    TF_ASSIGN_OR_RETURN(
        std::optional<DecomposedReplicaGroups> decomposed_groups,
        TryDecomposeReplicaGroup(replica_group, device_assignment,
                                 num_devices_per_host,
                                 collective_op_group_mode));

    if (!decomposed_groups) return {std::nullopt};

    int scatter_group_size =
        decomposed_groups->scatter_gather_groups[0].replica_ids_size();

    if (scatter_gather_groups.empty()) {
      // Check that every operand is exactly divisible by scatter group sizes.
      for (const HloInstruction* operand : all_reduce.operands()) {
        TF_RET_CHECK(operand->shape().IsArray());
        int64_t num_elements = ShapeUtil::ElementsIn(operand->shape());
        if (num_elements % scatter_group_size != 0) {
          return {std::nullopt};
        }
      }

      scatter_gather_groups.reserve(
          replica_groups.size() *
          decomposed_groups->scatter_gather_groups.size());
      new_all_reduce_groups.reserve(
          replica_groups.size() *
          decomposed_groups->new_all_reduce_groups.size());
    } else if (scatter_group_size !=
               scatter_gather_groups[0].replica_ids_size()) {
      // Reduce-scatter would have different output shapes on different devices.
      return {std::nullopt};
    }

    absl::c_move(decomposed_groups->scatter_gather_groups,
                 std::back_inserter(scatter_gather_groups));
    absl::c_move(decomposed_groups->new_all_reduce_groups,
                 std::back_inserter(new_all_reduce_groups));
  }

  return {DecomposedReplicaGroups{std::move(scatter_gather_groups),
                                  std::move(new_all_reduce_groups)}};
}

// Attempts to decompose all-reduces as described by the BlueConnect paper.
//
// If possible, the all-reduce will be transformed into:
// 1. reduce-scatter
// 2. all-reduce
// 3. all-gather
//
// If the all-reduce replica groups have more than one device within the same
// host, the reduce-scatter will be performed over all devices with each host.
// Otherwise, the reduce-scatter will be performed between pairs of devices on
// different hosts.
//
// When applied repeatedly, this transformation will reproduce the same pattern
// as described in the BlueConnect paper.
absl::StatusOr<bool> TryDecomposeAllReduce(HloAllReduceInstruction* all_reduce,
                                           size_t num_devices_per_host) {
  TF_RET_CHECK(all_reduce);
  TF_RET_CHECK(!all_reduce->has_sharding());

  HloComputation& computation = *all_reduce->parent();  // never null
  PrimitiveType element_type = all_reduce->operand(0)->shape().element_type();

  TF_ASSIGN_OR_RETURN(
      std::optional<DecomposedReplicaGroups> decomposed_groups,
      TryDecomposeReplicaGroups(*all_reduce, num_devices_per_host));

  if (!decomposed_groups) return false;

  // Bitcast operands to 1D to guarantee that first dimension is divisible by
  // scatter group size (we checked num elements was divisible above).
  std::vector<HloInstruction*> flat_operands;
  flat_operands.reserve(all_reduce->operand_count());
  std::vector<Shape> flat_shapes;
  flat_shapes.reserve(all_reduce->operand_count());
  std::vector<Shape> scattered_shapes;
  scattered_shapes.reserve(all_reduce->operand_count());

  int scatter_group_size =
      decomposed_groups->scatter_gather_groups[0].replica_ids_size();

  for (HloInstruction* operand : all_reduce->operands()) {
    TF_RET_CHECK(operand->shape().IsArray());
    int64_t num_elements = ShapeUtil::ElementsIn(operand->shape());
    Shape flat_shape = ShapeUtil::MakeShape(element_type, {num_elements});
    flat_operands.push_back(computation.AddInstruction(
        HloInstruction::CreateBitcast(flat_shape, operand)));
    flat_shapes.push_back(std::move(flat_shape));
    scattered_shapes.push_back(ShapeUtil::MakeShape(
        element_type, {num_elements / scatter_group_size}));
  }

  Shape reduce_scatter_shape = ShapeUtil::MakeMaybeTupleShape(scattered_shapes);

  int64_t next_channel_id = hlo_query::NextChannelId(*computation.parent());
  auto get_channel_id = [&]() -> std::optional<int64_t> {
    if (all_reduce->channel_id().has_value()) {
      return next_channel_id++;
    }
    return std::nullopt;
  };

  HloInstruction* reduce_scatter =
      computation.AddInstruction(HloInstruction::CreateReduceScatter(
          reduce_scatter_shape, flat_operands, all_reduce->to_apply(),
          CollectiveDeviceList(decomposed_groups->scatter_gather_groups),
          /*constrain_layout=*/false, get_channel_id(),
          all_reduce->use_global_device_ids(),
          /*scatter_dimension=*/0));

  HloInstruction* new_all_reduce =
      computation.AddInstruction(HloInstruction::CreateAllReduce(
          reduce_scatter_shape, GetOutputs(*reduce_scatter),
          all_reduce->to_apply(),
          CollectiveDeviceList(decomposed_groups->new_all_reduce_groups),
          /*constrain_layout=*/false, all_reduce->channel_id(),
          all_reduce->use_global_device_ids()));

  HloInstruction* all_gather =
      computation.AddInstruction(HloInstruction::CreateAllGather(
          ShapeUtil::MakeMaybeTupleShape(flat_shapes),
          GetOutputs(*new_all_reduce),
          /*all_gather_dimension=*/0,
          CollectiveDeviceList(decomposed_groups->scatter_gather_groups),
          /*constrain_layout=*/false, get_channel_id(),
          all_reduce->use_global_device_ids()));

  // Bitcast back to the original shapes and replace all-reduce with decomposed
  // implementation.
  std::vector<HloInstruction*> outputs = GetOutputs(*all_gather);
  for (int64_t i = 0; i < outputs.size(); ++i) {
    outputs[i] = computation.AddInstruction(HloInstruction::CreateBitcast(
        all_reduce->operand(i)->shape(), outputs[i]));
  }
  HloInstruction* replacement = MaybeMakeTuple(outputs);

  TF_RETURN_IF_ERROR(
      all_reduce->CopyAllControlDepsTo(reduce_scatter, replacement));

  TF_RETURN_IF_ERROR(all_reduce->DropAllControlDeps());
  TF_RETURN_IF_ERROR(computation.ReplaceInstruction(all_reduce, replacement));

  // Try to apply decomposition recursively.
  TF_RETURN_IF_ERROR(
      TryDecomposeAllReduce(Cast<HloAllReduceInstruction>(new_all_reduce),
                            num_devices_per_host)
          .status());
  return true;
}

}  // namespace

absl::StatusOr<bool> AllReduceBlueConnect::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(1) << "Running AllReduceBlueConnect";

  if (hlo_query::ContainsLayoutConstrainedAllReduce(*module)) {
    VLOG(1)
        << "Skip AllReduceBlueConnect because the module contains all-reduce "
           "with constrained layouts";
    return false;
  }
  if (!module->config().has_static_device_assignment()) {
    VLOG(1)
        << "Skip AllReduceBlueConnect because the module doesn't have static "
           "device assignment";
    return false;
  }

  std::vector<HloAllReduceInstruction*> all_reduces;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (HloPredicateIsOp<HloOpcode::kAllReduce>(instruction)) {
        all_reduces.push_back(Cast<HloAllReduceInstruction>(instruction));
      }
    }
  }

  bool changed = false;
  for (HloAllReduceInstruction* all_reduce : all_reduces) {
    TF_ASSIGN_OR_RETURN(
        bool all_reduce_changed,
        TryDecomposeAllReduce(all_reduce, num_devices_per_host_));
    changed |= all_reduce_changed;
  }

  return changed;
}

}  // namespace xla
