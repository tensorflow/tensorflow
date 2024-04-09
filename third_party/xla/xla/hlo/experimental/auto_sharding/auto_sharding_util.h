/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_UTIL_H_
#define XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_UTIL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/service/call_graph.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/statusor.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status.h"

namespace xla {
namespace spmd {

inline constexpr absl::string_view kPipelineMarker = "xla_pipeline_marker";
inline constexpr absl::string_view kIdentityMarker = "identity";
inline constexpr absl::string_view kPipelineMarkerStartType = "start";
inline constexpr absl::string_view kPipelineMarkerEndType = "end";

inline bool IsManualShardingBoundaryCustomCall(const HloInstruction* ins) {
  return ins->IsCustomCall("SPMDFullToShardShape") ||
         ins->IsCustomCall("SPMDShardToFullShape");
}

inline std::pair<int, int> ParseMeshDims(const std::string& strategy_name) {
  if (absl::StrContains(strategy_name, "{0,1}")) {
    return {0, 1};
  }
  return {1, 0};
}

inline std::string ToAdaptiveString(const HloInstruction* ins) {
  bool is_large_instruction =
      ins->shape().IsTuple() && ins->shape().tuple_shapes_size() > 500;
  if (!is_large_instruction) {
    for (const auto& operand : ins->operands()) {
      is_large_instruction =
          is_large_instruction || (operand->shape().IsTuple() &&
                                   operand->shape().tuple_shapes_size() > 500);
    }
  }
  return is_large_instruction ? ins->ToShortString() : ins->ToString();
}

// Return whether the tensor shape is divisible by
// the number of devices along multiple dimensions.
bool IsDivisible(const HloInstruction* ins, const Array<int64_t>& device_mesh,
                 absl::Span<const int64_t> tensor_dims,
                 absl::Span<const int64_t> mesh_dims);

// Array/Vector/Matrix Utility

// Append elements of `array` to `result`. The `indices` is a generalized
// multi-dimensional index that can index a whole row (use -1 to indicate this).
template <typename T>
void AppendFlattenElements(std::vector<T>* result, const Array<T>& array,
                           absl::Span<const int64_t> indices, int cur_depth,
                           std::vector<int64_t> cur_indices) {
  if (cur_depth == array.num_dimensions() - 1) {
    result->push_back(array(cur_indices));
  } else {
    int next_depth = cur_depth + 1;
    int64_t index = indices[next_depth];

    if (index == -1) {
      for (int64_t i = 0; i < array.dim(next_depth); ++i) {
        cur_indices[next_depth] = i;
        AppendFlattenElements(result, array, indices, next_depth, cur_indices);
      }
    } else {
      cur_indices[next_depth] = index;
      AppendFlattenElements(result, array, indices, next_depth, cur_indices);
    }
  }
}

// Return the index of key in a span. -1 means not found.
template <typename T>
int64_t GetIndex(absl::Span<const T> v, const T& key) {
  auto iter = std::find(v.cbegin(), v.cend(), key);

  if (iter != v.cend()) {
    return std::distance(v.cbegin(), iter);
  }
  return -1;
}

// Print a vector as string.
template <typename T>
std::string ToString(const std::vector<T>& vector) {
  return absl::StrCat("[", absl::StrJoin(vector, ", "), "]");
}

template <typename T>
std::string ToString(const StableHashMap<std::string, T>& map) {
  std::string result;
  for (const auto& v : map) {
    result = absl::StrCat(result, "\n[", v.first, "->", v.second, "]");
  }
  return result;
}

template <typename T>
std::string ToString(const std::vector<std::vector<T>>& vector) {
  std::string result;
  for (const auto& v : vector) {
    result = absl::StrCat(result, "\n[", absl::StrJoin(v, ", "), "]");
  }
  return result;
}

// Print a span as string.
template <typename T>
std::string ToString(absl::Span<T> span) {
  return absl::StrCat("[", absl::StrJoin(span, ", "), "]");
}

// Shape Utility

// Get the bytes of an array shape without checking its layout.
// This is modified from ShapeUtil::ByteSizeOfElements (shape_util.cc).
inline int64_t ByteSizeOfElementsNoCheck(const Shape& shape) {
  TF_DCHECK_OK(ShapeUtil::ValidateShape(shape));
  CHECK(shape.IsArray());
  int64_t allocated_element_count;

  // Disable this check. Otherwise, it raises a fatal error on HloOpcode::kIota
  // generated by jax dropout.
  // CHECK(LayoutUtil::IsDenseArray(shape)) << shape.ShortDebugString();
  allocated_element_count = ShapeUtil::ElementsIn(shape);
  return allocated_element_count *
         ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
}

// Get the number of bytes of a shape.
inline double GetBytes(const Shape& shape) {
  if (shape.IsArray()) {
    return ByteSizeOfElementsNoCheck(shape);
  }
  return ShapeUtil::ByteSizeOf(shape, /*pointer_size=*/8);
}

// Return whether two shapes are equal in dimension.
// The element type and layout are ignored.
inline bool DimensionsEqual(const Shape& a, const Shape& b) {
  return Shape::Equal().IgnoreLayout().IgnoreElementType()(a, b);
}

/*
 * HloInstruction Utility
 */
// Get the space dimensions of a dot instruction.
inline std::pair<tsl::protobuf::RepeatedField<int64_t>,
                 tsl::protobuf::RepeatedField<int64_t>>
GetSpaceDims(const Shape& lhs_shape, const Shape& rhs_shape,
             const DotDimensionNumbers& dnums) {
  tsl::protobuf::RepeatedField<int64_t> lhs_space_dims, rhs_space_dims;

  for (int64_t i = 0; i < lhs_shape.rank(); ++i) {
    if (absl::c_linear_search(dnums.lhs_batch_dimensions(), i) ||
        absl::c_linear_search(dnums.lhs_contracting_dimensions(), i)) {
      continue;
    }
    lhs_space_dims.Add(i);
  }

  for (int64_t i = 0; i < rhs_shape.rank(); ++i) {
    if (absl::c_linear_search(dnums.rhs_batch_dimensions(), i) ||
        absl::c_linear_search(dnums.rhs_contracting_dimensions(), i)) {
      continue;
    }
    rhs_space_dims.Add(i);
  }
  return std::make_pair(std::move(lhs_space_dims), std::move(rhs_space_dims));
}

// Replace old operand with the new one.
inline void ReplaceOperand(HloInstruction* inst,
                           const HloInstruction* old_operand,
                           HloInstruction* new_operand) {
  for (int i = 0; i < inst->operand_count(); ++i) {
    if (inst->operand(i) == old_operand) {
      TF_CHECK_OK(inst->ReplaceOperandWith(i, new_operand));
    }
  }
}

// Return whether this instruction is a custom call marker introduced by us.
inline bool IsCustomCallMarker(const HloInstruction* inst) {
  return inst->IsCustomCall({kPipelineMarker, kIdentityMarker});
}

// Return whether this instruction is a TopK custom call.
inline bool IsTopKCustomCall(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kCustomCall &&
         inst->custom_call_target() == "TopK";
}

// Pass through the custom call marker and get the source instruction
inline const HloInstruction* PassThroughCustomCallMarkerGetSource(
    const HloInstruction* ins) {
  while (ins->opcode() == HloOpcode::kGetTupleElement &&
         IsCustomCallMarker(ins->operand(0))) {
    const HloInstruction* custom_call = ins->operand(0);
    const HloInstruction* tuple = custom_call->operand(0);
    while (IsCustomCallMarker(tuple)) {
      tuple = tuple->operand(0);
    }
    ins = tuple->operand(ins->tuple_index());
  }
  return ins;
}

// Pass through the custom call marker and get the acutal operand.
inline HloInstruction* PassThroughCustomCallMarkerOperand(
    HloInstruction* raw_operand, const HloInstruction* inst) {
  if (!IsCustomCallMarker(raw_operand)) {
    return raw_operand;
  }

  CHECK_EQ(inst->opcode(), HloOpcode::kGetTupleElement);

  int index = inst->tuple_index();
  return raw_operand->mutable_operand(0)->mutable_operand(index);
}

// Return whether the tuple is only used by a custom call marker.
inline bool IsCustomCallMarkerTuple(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kTuple && inst->users().size() == 1 &&
         IsCustomCallMarker(inst->users().front());
}

// Pass through the custom call marker and get the actual user.
inline HloInstruction* PassThroughCustomCallMarkerUser(
    HloInstruction* raw_user, const HloInstruction* inst) {
  if (!IsCustomCallMarkerTuple(raw_user)) {
    return raw_user;
  }

  const HloInstruction* custom_call = raw_user->users().front();

  int index = -1;
  for (int i = 0; i < raw_user->operand_count(); i++) {
    if (raw_user->operand(i) == inst) {
      index = i;
      break;
    }
  }
  CHECK_NE(index, -1);

  HloInstruction* ret = nullptr;
  for (HloInstruction* user : custom_call->users()) {
    CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
    if (user->tuple_index() == index) {
      CHECK_EQ(ret, nullptr);
      ret = user;
    }
  }

  return ret == nullptr ? raw_user : ret;
}

// Return the users of an instruction and its alias,
// excluding the final output tuple.
inline StableHashSet<HloInstruction*> UsersWithAlias(
    const HloInstruction* inst, const AliasMap& alias_map,
    const HloInstruction* output) {
  StableHashSet<HloInstruction*> users;

  for (HloInstruction* user : inst->users()) {
    users.insert(PassThroughCustomCallMarkerUser(user, inst));
  }

  auto iter = alias_map.find(inst);
  if (iter != alias_map.end()) {
    for (HloInstruction* user : iter->second->users()) {
      users.insert(PassThroughCustomCallMarkerUser(user, iter->second));
    }
  }

  users.erase(output);
  return users;
}

// Return whether this instruction is a convert on a parameter.
bool IsParameterConvert(const HloInstruction* inst);

// Return whether the instruction is always replicated.
// (e.g., constant, broadcasted constant, scalar)
bool IsAlwaysReplicated(const HloInstruction* inst);

// Try to reduce the boundary set to its common ancestor
void TryReduceWithCommonAncestor(StableHashSet<HloInstruction*>& replicated_set,
                                 StableHashSet<HloInstruction*>& boundary_set,
                                 StableHashSet<HloInstruction*>& consumer_set,
                                 const AliasMap& alias_map);

// Return whether all users of an instruction is reduce.
bool AllUsersAreReduce(const HloInstruction* inst);

void UseAllReduceForGradAcc(StableHashSet<HloInstruction*>& replicated_set,
                            const HloInstruction* inst);

void SetSharding(HloInstruction* to_split, const HloSharding& output_spec,
                 const HloInstruction* ref_inst,
                 const HloInstruction* shape_inst,
                 StableHashSet<const HloInstruction*>& modified);

template <typename T>
inline std::vector<int> Argsort(const std::vector<T>& scores) {
  std::vector<int> index;
  index.reserve(scores.size());
  for (size_t i = 0; i < scores.size(); ++i) {
    index.push_back(i);
  }
  auto cmp = [&scores](int l, int r) { return scores[l] > scores[r]; };
  std::sort(index.begin(), index.end(), cmp);
  return index;
}

// Given the sharding for an instruction, invoke the sharding propagation pass
// to infer appropriate shardings for its operands.
std::optional<HloSharding> GetInputSharding(const HloInstruction* ins,
                                            int64_t op_index,
                                            const HloSharding& output_sharding,
                                            const xla::CallGraph& call_graph,
                                            int64_t num_devices);

// Return whether the instruction is an activation from another pipeline stage.
bool IsActivationFromAnotherStage(const HloInstruction* inst,
                                  const InstructionBatchDimMap& batch_dim_map);

// Depth analysis (breadth first search) that compute the depth of each
// instruction. We also assign a much larger distance to heavy operators (e.g.,
// dot, convolution).
InstructionDepthMap BuildInstructionDepthMap(
    const HloInstructionSequence& sequence,
    const InstructionBatchDimMap& batch_dim_map);

std::string GetBatchDimMapKey(const HloInstruction* ins, int64_t idx = -1);

// Batch dimension analysis that finds the batch dimension of each instruction.
InstructionBatchDimMap BuildInstructionBatchDimMap(
    const HloInstructionSequence& sequence);

/*
 * HloSharding Utility
 */
// We reuse "Manual" to represent "Undefined" sharding strategy.
// If an op has an"Undefined" strategy, it means auto-sharding pass does not
// decide the sharding strategy for this op.
// We rely on the later sharding propagation pass to assign strategies to them.
inline HloSharding Undefined() { return HloSharding::Manual(); }

inline bool IsUndefined(const HloSharding& hlo_sharding) {
  return hlo_sharding.IsManual();
}

// Pretty print a HloSharding in a simplified form
inline std::string ToStringSimple(const HloSharding& spec) {
  if (spec.IsReplicated()) {
    return "R";
  }
  return ToString(spec.tile_assignment().dimensions());
}

// Insert a copy of the operand to force the sharding of the operand.
inline void ForceOperandSharding(HloInstruction* inst, int operand_num,
                                 const HloSharding& sharding) {
  HloInstruction* operand = inst->mutable_operand(operand_num);
  if (operand->sharding() == sharding) {
    return;
  }
  HloInstruction* replace_with = inst->parent()->AddInstruction(
      HloInstruction::CreateReshape(operand->shape(), operand));
  replace_with->set_sharding(sharding);
  TF_CHECK_OK(inst->ReplaceOperandWith(operand_num, replace_with));
}

// Return whether the sharding is fully tiled.
inline bool IsFullyTiled(const HloSharding& sharding) {
  return sharding.NumTiles() == sharding.tile_assignment().num_elements();
}

// Propagate sharding for dim-wise operations (e.g., slice, pad) which works
// independently on each dimension.
// The sharding can successfully propagate if the operation only happens on
// tensor dimensions that are not tiled.
std::optional<HloSharding> PropagateDimwiseSharding(
    const HloSharding& input_spec, const Shape& old_shape,
    const Shape& new_shape);

HloSharding PropagateDimwiseShardingSlice(const HloSharding& input_spec,
                                          const Shape& old_shape,
                                          const Shape& new_shape,
                                          const Array<int64_t>& device_mesh);

// Propagate sharding for ReduceWindow-like operations.
// The sharding can successfully propagate if the window operation only happens
// on tensor dimensions that are not tiled.
std::optional<HloSharding> PropagateReduceWindowSharding(
    const HloSharding& input_spec, const Shape& old_shape,
    const Window& window);

// Check whether the tile assignment of a HloSharding is valid for our system.
// Definition of validity:
// For every tile dimension, the device id sequence along that dimension has to
// be an arithmetic sequence.
// e.g., we don't allow specs like sharding={devices=[8,1] 0,4,1,5,2,7,3,8}
bool IsValidTileAssignment(const HloSharding& spec);

// Get number of tile dimensions that are not 1. For example, for sharding spec
// {devices=[2,1,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
// spec.tile_assignment.num_dimensions() = [2,1,1,4]. This function returns 2.
// -1 means the tensor is replicated on the whole the mesh.
int64_t NumTileDimensions(const HloSharding& spec);

// When fixing mixed mesh resharding (see below), compute the correct
// intermediate shape in order to insert copies.
Shape ComputeIntermediateShape(const HloSharding& src_sharding,
                               const HloSharding& dst_sharding,
                               const Shape& shape,
                               const Array<int64_t>& device_mesh);

// Forcibly set the sharding of the operand of inst.
// Also fix the resharding between 1d and 2d logical mesh.
absl::Status FixMixedMeshShapeReshardingGetTupleElement(
    HloInstruction* inst, const HloSharding& dst_sharding,
    const Array<int64_t>& device_mesh,
    absl::flat_hash_map<std::string, std::vector<HloSharding>>&
        preserve_shardings);

absl::Status FixMixedMeshShapeReshardingGetTupleElementWithTupleOutput(
    HloInstruction* inst,
    const std::vector<std::optional<HloSharding>>& dst_sharding,
    const Array<int64_t>& device_mesh);

absl::Status FixMixedMeshShapeResharding(HloInstruction* inst, int operand_num,
                                         const HloSharding& dst_sharding,
                                         const Array<int64_t>& device_mesh,
                                         ReshardingCache* resharding_cache);

/*
 * Gradient accumulation
 */
// Find all instructions that compute gradients in gradient accumulation.
// This is done by using the hint from pipeline_marker (gradient marker).
inline std::vector<const HloInstruction*> GetGradientComputationInstructions(
    const std::vector<HloInstruction*>& instructions) {
  std::vector<const HloInstruction*> ret;

  for (size_t i = 0; i < instructions.size(); ++i) {
    const HloInstruction* ins = instructions[i];
    if (ins->IsCustomCall(kPipelineMarker) &&
        (absl::StrContains(ins->metadata().op_name(), "compute_grad") ||
         absl::StrContains(ins->metadata().op_name(), "backward")) &&
        ins->metadata().op_type() == kPipelineMarkerEndType) {
      const HloInstruction* tuple = ins->operand(0);
      for (size_t j = 0; j < tuple->operand_count(); ++j) {
        const HloInstruction* add = tuple->operand(j);
        while (add->opcode() == HloOpcode::kAdd) {
          ret.push_back(add->operand(0));
          ret.push_back(add->operand(1));

          if (add->operand(0)->opcode() == HloOpcode::kAdd) {
            add = add->operand(0);
          } else {
            add = add->operand(1);
          }
        }
      }
    }
  }

  return ret;
}

// Gets the mapping vector from dim_from to dim_to.
// Example: GetDimensionMapping([2], 3) = [0, 1, -1]
std::vector<int64_t> GetDimensionMapping(
    absl::Span<const int64_t> reduced_dimensions, int64_t op_count);

// Checks whether numerator is divisible by denominator.
bool IsDivisible(int64_t numerator, int64_t denominator);

// Generate all replica groups along one device_mesh dimension. Device_mesh can
// be any number of dimensions. |communication_dim| has to be one of
// |device_mesh|'s dimension.
std::vector<std::vector<int64_t>> GetReplicaGroupsAlongOneDimension(
    const Array<int64_t>& device_mesh, int32_t communication_dim);

// Gets values in |array| along |dim| while keeping indices at other
// dimensions at 0, e.g., array is 2D and dim = 1, this returns array[0, 1],
// array[1, 1], array [2, 1], ....
// Returns error status if dim >= array.num_dimensions().
absl::StatusOr<std::vector<int64_t>> GetValuesAlongOneDim(
    const Array<int64_t>& array, int dim);

absl::StatusOr<int64_t> CheckArithmeticSequence(
    absl::Span<const int64_t> sequence);

// Checks if the number of sharded dimensions in the tile assignment matches the
// device mesh.
bool TileAssignmentMatchesMesh(const HloSharding& spec,
                               const Array<int64_t>& mesh);

// Get the mapped mesh dimension for every tensor dimension.
// The returned value maps ith tensor dim to one mesh dim. -1 means the tensor
// is replicated on that dimension.
// For example, returned value [1,2] means the 0th tensor dim maps to the 1st
// mesh dim, and 1st tensor dim maps to the 2nd mesh dim.
std::vector<int64_t> GetTensorDimToMeshDim(
    int64_t tensor_shape_rank, const HloSharding& spec,
    const Array<int64_t>& device_mesh,
    bool consider_reverse_device_meshes = false);

absl::StatusOr<std::vector<int64_t>> GetTensorDimToMeshDimNoCrash(
    int64_t tensor_shape_rank, const HloSharding& spec,
    const Array<int64_t>& device_mesh,
    bool consider_reverse_device_meshes = false);

HloSharding Tile(const Shape& tensor_shape,
                 absl::Span<const int64_t> tensor_dims,
                 absl::Span<const int64_t> mesh_dims,
                 const Array<int64_t>& device_mesh);

AliasMap BuildAliasMap(const HloModule* module,
                       const HloInputOutputAliasConfig& alias_config);

AliasSet BuildAliasSet(const HloModule* module,
                       const HloInputOutputAliasConfig& alias_config,
                       const StrategyMap& strategy_map);

// Transpose an array of any number of dimensions given any axes order.
// Similar to numpy.transpose(array, axes=()) function.
template <typename T>
Array<T> Transpose(const Array<T> array, std::vector<int64_t> axes) {
  // Computes transposed array's size.
  std::vector<int64_t> transposed_array_dimensions(array.dimensions().begin(),
                                                   array.dimensions().end());
  for (size_t i = 0; i < axes.size(); i++) {
    transposed_array_dimensions[i] = array.dimensions()[axes[i]];
  }
  Array<T> transposed_array(transposed_array_dimensions);
  std::vector<int64_t> transposed_array_indices(axes.size());
  array.Each([&](absl::Span<const int64_t> indices, T value) {
    for (int i = 0; i < axes.size(); ++i) {
      transposed_array_indices[i] = indices[axes[i]];
    }
    transposed_array(transposed_array_indices) = value;
  });
  return transposed_array;
}

// Used to determine whether a sharding or mesh shape is 1D, 2D, or 3D.
size_t VectorGreaterThanOneElementCount(absl::Span<const int64_t> span,
                                        bool omit_last_dim = false);

// This functions returns the indices of all vector elements larger than 1, in
// order.
std::vector<int64_t> VectorGreaterThanOneElementIndices(
    absl::Span<const int64_t> span, bool omit_last_dim = false);

int64_t GetInstructionSize(const Shape& shape);

int64_t GetShardedInstructionSize(
    const Shape& shape, int64_t num_devices,
    std::optional<HloSharding> sharding = std::nullopt);

HloInstruction* FindInstruction(
    const std::vector<HloInstruction*>& instructions, absl::string_view name);

// When a complete mesh shape is [1, 8, 4], [1, 8, 1] is its partial mesh shape.
// If a sharding is [8, 4] for the complete mesh shape, we convert it to [8, 1]
// given [1, 8, 1] as the partial mesh shape.
// total_num_devices should equal to the product of mesh_shape elements.
absl::StatusOr<bool> AdjustShardingsWithPartialMeshShape(
    const std::vector<HloInstruction*>& instructions,
    const std::vector<int64_t>& mesh_shape, int64_t total_num_devices,
    bool crash_on_error);

inline bool AdjustShardingsWithPartialMeshShape(
    const std::vector<HloInstruction*>& instructions,
    const std::vector<int64_t>& mesh_shape, int64_t total_num_devices) {
  auto result = AdjustShardingsWithPartialMeshShape(instructions, mesh_shape,
                                                    total_num_devices, true);
  CHECK_OK(result);
  return *result;
}

// Decompose mesh shapes into partial mesh shapes so that we can solve the auto
// sharding problem iteratively. Returns partial mesh shapes with larger
// dimensions first. For example, input [1, 4, 2] returns [1, 4, 1] and [1, 4,
// 2]; input [4, 8, 2] returns [1, 8, 1], [4, 8, 1] and [ 4, 8, 2].
std::vector<std::vector<int64_t>> DecomposeMeshShapes(
    std::vector<int64_t> mesh_shape);

bool OutputInputSameShapes(const HloInstruction* ins);

bool IsEntryComputationInputOrOutput(const HloModule* module,
                                     const HloInstruction* ins);

// Statically estimate the execution counts of HLO ops. This matters for while
// loops, and we use a constant iteration count for all while loops for this
// approximation.
absl::flat_hash_map<const HloInstruction*, int64_t>
ComputeInstructionExecutionCounts(const HloModule* module,
                                  int64_t loop_iteration_count_estimate);

// Generates a set of mesh shapes to try for a given module based on
// pre-existing sharding annotations. If not such annotations exist, it will
// enumerate and return all possible mesh shapes for a given number of devices
// and mesh dimensions.
std::vector<std::vector<int64_t>> InferOrEnumerateMeshShapesToTry(
    const HloModule& module, int64_t num_devices, int num_mesh_dims,
    bool symmetrical_mesh_dims);

// Check if the sharding is "misaligned" wrt the shape. This is true if there is
// at least one dimension of the tensor that is sharded over a number of devices
// that do not complete divide the size of the tensor dimension.
bool IsShardingMisaligned(const HloSharding& sharding, const Shape& shape);

// In a given tuple sharding, replace certain leaves with
// HloSharding::Unknown()
HloSharding ReplaceGivenShardingsWithUnknownForTuple(
    const HloSharding& sharding, const Shape& shape,
    absl::Span<const bool> to_replace_sharding_ids);

}  // namespace spmd
}  // namespace xla

#endif  // XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_UTIL_H_
