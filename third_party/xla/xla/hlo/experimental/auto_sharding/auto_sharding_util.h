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
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_device_mesh.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/service/call_graph.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace spmd {

inline constexpr absl::string_view kIdentityMarker = "identity";

inline constexpr int64_t kAutoShardingPointerSize = 8;

inline bool IsSPMDFullToShardShapeCustomCall(const HloInstruction* ins) {
  return ins->IsCustomCall("SPMDFullToShardShape");
}

inline bool IsSPMDShardToFullShapeCustomCall(const HloInstruction* ins) {
  return ins->IsCustomCall("SPMDShardToFullShape");
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
bool IsDivisible(const HloInstruction* ins, const DeviceMesh& device_mesh,
                 absl::Span<const int64_t> tensor_dims,
                 absl::Span<const int64_t> mesh_dims);

// Array/Vector/Matrix Utility

// Append elements of `array` to `result`. The `indices` is a generalized
// multi-dimensional index that can index a whole row (use -1 to indicate this).
template <typename T>
void AppendFlattenElementsInternal(std::vector<T>* result,
                                   const Array<T>& array,
                                   absl::Span<const int64_t> indices,
                                   int cur_depth,
                                   std::vector<int64_t> cur_indices) {
  if (cur_depth == array.num_dimensions() - 1) {
    result->push_back(array(cur_indices));
  } else {
    int next_depth = cur_depth + 1;
    int64_t index = indices[next_depth];

    if (index == -1) {
      for (int64_t i = 0; i < array.dim(next_depth); ++i) {
        cur_indices[next_depth] = i;
        AppendFlattenElementsInternal(result, array, indices, next_depth,
                                      cur_indices);
      }
    } else {
      cur_indices[next_depth] = index;
      AppendFlattenElementsInternal(result, array, indices, next_depth,
                                    cur_indices);
    }
  }
}

template <typename T>
void AppendFlattenElements(std::vector<T>* result, const Array<T>& array,
                           absl::Span<const int64_t> indices) {
  std::vector<int64_t> tmp_indices(array.num_dimensions(), 0);
  AppendFlattenElementsInternal(result, array, indices,
                                /*cur_depth=*/-1, tmp_indices);
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

template <typename K, typename V>
std::string ToString(const StableMap<K, V>& map) {
  std::string result;
  for (const auto& [k, v] : map) {
    result = absl::StrCat(result, " [", k, "->", v, "]");
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

  for (int64_t i = 0; i < lhs_shape.dimensions_size(); ++i) {
    if (absl::c_linear_search(dnums.lhs_batch_dimensions(), i) ||
        absl::c_linear_search(dnums.lhs_contracting_dimensions(), i)) {
      continue;
    }
    lhs_space_dims.Add(i);
  }

  for (int64_t i = 0; i < rhs_shape.dimensions_size(); ++i) {
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

// Return whether this instruction is a TopK custom call.
inline bool IsTopKCustomCall(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kCustomCall &&
         inst->custom_call_target() == "TopK";
}

// Return whether this instruction is a TopK custom call.
inline bool IsPartialReduceCustomCall(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kCustomCall &&
         inst->custom_call_target() == "PartialReduce";
}

// Return the users of an instruction and its alias,
// excluding the final output tuple.
inline InstructionSet UsersWithAlias(const HloInstruction* inst,
                                     const AliasMap& alias_map,
                                     const HloInstruction* output) {
  InstructionSet users;
  for (HloInstruction* user : inst->users()) {
    HloInstruction* pass_through_user = user;
    if (pass_through_user == output) {
      continue;
    }
    users.insert(pass_through_user);
  }

  auto iter = alias_map.find(inst);
  if (iter != alias_map.end()) {
    for (HloInstruction* user : iter->second->users()) {
      HloInstruction* pass_through_user = user;
      if (pass_through_user == output) {
        continue;
      }
      users.insert(pass_through_user);
    }
  }

  return users;
}

// Return whether this instruction is a convert on a parameter.
bool IsParameterConvert(const HloInstruction* inst);

// Return whether the instruction is always replicated.
// (e.g., constant, broadcasted constant, scalar)
bool IsAlwaysReplicated(const HloInstruction* inst);

// Try to reduce the boundary set to its common ancestor
void TryReduceWithCommonAncestor(InstructionSet& replicated_set,
                                 InstructionSet& boundary_set,
                                 InstructionSet& consumer_set,
                                 const AliasMap& alias_map);

// Return whether all users of an instruction is reduce.
bool AllUsersAreReduce(const HloInstruction* inst);

void UseAllReduceForGradAcc(InstructionSet& replicated_set,
                            const HloInstruction* inst);

void SetSharding(HloInstruction* to_split, const HloSharding& output_spec,
                 const HloInstruction* ref_inst,
                 const HloInstruction* shape_inst,
                 ConstInstructionSet& modified);

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

// Depth analysis (breadth first search) that compute the depth of each
// instruction. We also assign a much larger distance to heavy operators (e.g.,
// dot, convolution).
InstructionDepthMap BuildInstructionDepthMap(
    const HloInstructionSequence& sequence);

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

// The sharding is replicated or the total number of tiles is over or equal to
// the total number of devices. If returns true, this sharding is likely
// provided by users.
inline bool ShardingIsComplete(const HloSharding& sharding,
                               size_t total_num_devices) {
  return sharding.TotalNumTiles() >= total_num_devices ||
         sharding.IsReplicated();
}

// Checks if the argument instruction is a producer for a SPMDFullToShardShape
// custom call.
inline bool IsInstructionBeforeSPMDFullToShardShapeCustomCall(
    const HloInstruction* ins) {
  for (const HloInstruction* user : ins->users()) {
    if (spmd::IsSPMDFullToShardShapeCustomCall(user)) {
      return true;
    }
  }
  return false;
}

// Computes the cartesian product of N vectors
template <typename T>
void ForEachInCartesianProduct(
    const std::vector<std::vector<T>>& sets,
    absl::FunctionRef<void(const std::vector<T>&)> fn) {
  std::vector<std::vector<T>> elements(1, std::vector<T>());
  std::vector<std::vector<T>> temp_elements;
  for (int i = 0; i < sets.size(); i++) {
    temp_elements.clear();
    for (const std::vector<T>& product : elements) {
      for (const T& element : sets[i]) {
        std::vector<T> product_copy = product;
        product_copy.push_back(element);
        temp_elements.push_back(product_copy);
      }
    }
    std::swap(elements, temp_elements);
  }
  for (const std::vector<T>& element : elements) {
    fn(element);
  }
}

// Propagate sharding for dim-wise operations (e.g., slice, pad) which works
// independently on each dimension.
// The sharding can successfully propagate if the operation only happens on
// tensor dimensions that are not tiled.
std::optional<HloSharding> PropagateDimwiseSharding(
    const HloSharding& input_spec, const Shape& old_shape,
    const Shape& new_shape);

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
bool IsValidTileAssignment(const HloSharding& sharding);

// Get number of tile dimensions that are not 1. For example, for sharding
// {devices=[2,1,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
// sharding.tile_assignment.num_dimensions() = [2,1,1,4]. This function
// returns 2. -1 means the tensor is replicated on the whole the mesh.
int64_t NumTileDimensions(const HloSharding& sharding);

// When fixing mixed mesh resharding (see below), compute the correct
// intermediate shape in order to insert copies.
absl::StatusOr<Shape> ComputeIntermediateShape(const HloSharding& src_sharding,
                                               const HloSharding& dst_sharding,
                                               const Shape& shape,
                                               const DeviceMesh& device_mesh);

// Forcibly set the sharding of the operand of inst.
// Also fix the resharding between 1d and 2d logical mesh.
absl::Status FixMixedMeshShapeReshardingGetTupleElement(
    HloInstruction* inst, const HloSharding& dst_sharding,
    const DeviceMesh& device_mesh,
    absl::flat_hash_map<std::string, std::vector<HloSharding>>&
        preserve_shardings);

absl::Status FixMixedMeshShapeReshardingGetTupleElementWithTupleOutput(
    HloInstruction* inst,
    const std::vector<std::optional<HloSharding>>& dst_sharding,
    const DeviceMesh& device_mesh);

absl::Status FixMixedMeshShapeResharding(HloInstruction* inst, int operand_num,
                                         const HloSharding& dst_sharding,
                                         const DeviceMesh& device_mesh,
                                         ReshardingCache* resharding_cache);

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
    const DeviceMesh& device_mesh, int32_t communication_dim);

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
bool TileAssignmentMatchesMesh(const HloSharding& sharding,
                               const DeviceMesh& mesh);

absl::StatusOr<std::vector<int64_t>> GetMeshDimPermutationOrderInShardingSpec(
    const HloSharding& spec, const Array<int64_t>& device_mesh,
    bool consider_reverse_device_meshes);

absl::StatusOr<std::vector<absl::btree_set<int64_t>>>
GetTensorDimToMeshDimMixedMeshSharding(
    int64_t tensor_shape_rank, const HloSharding& sharding,
    const DeviceMesh& device_mesh, bool consider_reverse_device_meshes = false);

// Get the mapped mesh dimension for every tensor dimension.
// The returned value maps ith tensor dim to one mesh dim. -1 means the tensor
// is replicated on that dimension.
// For example, returned value [1,2] means the 0th tensor dim maps to the 1st
// mesh dim, and 1st tensor dim maps to the 2nd mesh dim.
std::vector<int64_t> GetTensorDimToMeshDim(
    int64_t tensor_shape_rank, const HloSharding& spec,
    const DeviceMesh& device_mesh, bool consider_reverse_device_meshes = false);

absl::StatusOr<std::vector<int64_t>> GetTensorDimToMeshDimNoCrash(
    int64_t tensor_shape_rank, const HloSharding& spec,
    const DeviceMesh& device_mesh, bool consider_reverse_device_meshes = false);

HloSharding Tile(const Shape& tensor_shape,
                 absl::Span<const int64_t> tensor_dims,
                 const std::vector<std::vector<int64_t>>& mesh_dims,
                 const DeviceMesh& device_mesh);

HloSharding Tile(const Shape& tensor_shape,
                 absl::Span<const int64_t> tensor_dims,
                 absl::Span<const int64_t> mesh_dims,
                 const DeviceMesh& device_mesh);

HloSharding Tile(const Shape& tensor_shape,
                 absl::Span<const int64_t> tensor_dims,
                 std::initializer_list<int64_t> mesh_dims,
                 const DeviceMesh& device_mesh);

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

std::vector<int64_t> VectorGreaterThanOneElements(
    absl::Span<const int64_t> span, bool omit_last_dim = false);

// Computes bytes size of a shape recursively if it is sharded according to an
// optionally provided sharding
int64_t ByteSizeOfShapeWithSharding(const Shape& shape,
                                    std::optional<HloSharding> sharding);

// Computes bytes size of a shape recursively
inline int64_t ByteSizeOfShape(const Shape& shape) {
  return ByteSizeOfShapeWithSharding(shape, /*sharding=*/std::nullopt);
}

// Computes the byte size of a shape recursively if it is sharded across a given
// number of devices per an optionally provided sharding. If the sharding is
// provided, this function behaves the same as ByteSizeOfShapeWithSharding
// above. If not, it will give a lower bound on the bytes size of the shape if
// sharded across `num_devices` devices.
int64_t ByteSizeOfShapeIfShardedAcrossDevices(
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
    const absl::flat_hash_set<const HloInstruction*>& instructions_to_shard,
    const std::vector<int64_t>& mesh_shape,
    const DeviceMesh& original_device_mesh, bool crash_on_error);

inline bool AdjustShardingsWithPartialMeshShape(
    const std::vector<HloInstruction*>& instructions,
    const absl::flat_hash_set<const HloInstruction*>& instructions_to_shard,
    const std::vector<int64_t>& mesh_shape,
    const DeviceMesh& original_device_mesh) {
  absl::StatusOr<bool> result = AdjustShardingsWithPartialMeshShape(
      instructions, instructions_to_shard, mesh_shape, original_device_mesh,
      /*crash_on_error=*/true);
  CHECK_OK(result);
  return *result;
}

// Decompose mesh shapes into partial mesh shapes so that we can solve the auto
// sharding problem iteratively. Returns partial mesh shapes with larger
// dimensions and more expensive collective costs first. For example, if all
// mesh axes all have collective costs, input [1, 4, 2] returns [1, 4, 1] and
// [1, 4, 2]; input [4, 8, 2] returns [1, 8, 1], [4, 8, 1] and [ 4, 8, 2].
std::vector<std::vector<int64_t>> DecomposeMeshShapes(
    const std::vector<int64_t>& mesh_shape,
    const std::vector<double>& mesh_alpha,
    const std::vector<double>& mesh_beta);

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

// Extract the reduction_dim of a PartialReduce custom call
absl::StatusOr<int64_t> GetPartialReduceReductionDim(const HloInstruction* ins);

// Returns true if an HLO op flows to a SPMDShardToFullShape custom call without
// encountering a SPMDFullToShardShape custom call on the call.
bool OpEncountersShardToFull(const HloInstruction* op);

// Ensures that the modules entry_computation_layout has input/output shapes
// with layouts. If this is not the case, this function will add the layout
// information by extracting it from the HLO ops.
absl::Status EnsureEntryComputationLayoutHasShapeLayouts(HloModule* module);

}  // namespace spmd
}  // namespace xla

#endif  // XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_UTIL_H_
