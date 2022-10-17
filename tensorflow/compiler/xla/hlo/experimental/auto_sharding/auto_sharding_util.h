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

#ifndef TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_UTIL_H_

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

namespace xla {
namespace spmd {
// Type alias

template <typename Key, typename Value>
using StableHashMap = ::absl::flat_hash_map<Key, Value>;
template <typename Key>
using StableHashSet = ::absl::flat_hash_set<Key>;

// Map an instruction to its depth.
using InstructionDepthMap = StableHashMap<const HloInstruction*, int64_t>;
// Map an instruction to its batch dimension.
using InstructionBatchDimMap = StableHashMap<std::string, int>;
// Map an instruction to its alias source parameter.
using AliasMap = StableHashMap<const HloInstruction*, HloInstruction*>;
// Map an instruction to its resharding cache.
using ReshardingCache =
    StableHashMap<const HloInstruction*,
                  std::vector<std::pair<HloSharding, HloInstruction*>>>;

inline constexpr absl::string_view kPipelineMarker = "xla_pipeline_marker";
inline constexpr absl::string_view kIdentityMarker = "identity";
inline constexpr absl::string_view kPipelineMarkerStartType = "start";
inline constexpr absl::string_view kPipelineMarkerEndType = "end";

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

// A simple matrix class to store and manipulate the cost matrices on edges.
// It can create a view for matrix transpose without copying the memory.
// TODO (zhuohan): Inherit from Array2D and add Transpose and operator+ (See
// tensorflow/compiler/xla/array2d.h;l=39)
class Matrix {
 public:
  Matrix() : n_(0), m_(0), transpose_(false), data_(nullptr) {}

  Matrix(size_t n, size_t m) {
    this->n_ = n;
    this->m_ = m;
    transpose_ = false;
    data_ = std::make_shared<std::vector<double>>(n * m, 0.0);
  }

  Matrix(size_t n, size_t m, bool transpose,
         std::shared_ptr<std::vector<double>> data) {
    this->n_ = n;
    this->m_ = m;
    this->transpose_ = transpose;
    this->data_ = data;
  }

  Matrix Transpose() { return Matrix(m_, n_, !transpose_, data_); }

  double operator()(size_t i, size_t j) const {
    size_t idx;
    if (transpose_) {
      idx = j * n_ + i;
    } else {
      idx = i * m_ + j;
    }
    CHECK(data_ != nullptr) << n_ << " , " << m_;
    CHECK(idx < n_ * m_) << idx << " , " << n_ << " , " << m_;
    return (*data_)[idx];
  }

  double& operator()(size_t i, size_t j) {
    size_t idx;
    if (transpose_) {
      idx = j * n_ + i;
    } else {
      idx = i * m_ + j;
    }
    CHECK(data_ != nullptr) << n_ << " , " << m_;
    CHECK(idx < n_ * m_) << idx << " , " << n_ << " , " << m_;
    return (*data_)[idx];
  }

  Matrix operator+(const Matrix& other) {
    CHECK_EQ(n_, other.n_);
    CHECK_EQ(m_, other.m_);
    Matrix ret = Matrix(n_, m_);
    for (size_t i = 0; i < n_; ++i) {
      for (size_t j = 0; j < m_; ++j) {
        ret(i, j) = operator()(i, j) + other(i, j);
      }
    }
    return ret;
  }

  std::string ToString() const {
    std::string str;

    for (size_t i = 0; i < n_; ++i) {
      for (size_t j = 0; j < m_; ++j) {
        absl::StrAppend(&str, operator()(i, j), " ");
      }
      absl::StrAppend(&str, "\n");
    }

    return str;
  }

  size_t n_;
  size_t m_;
  bool transpose_;
  std::shared_ptr<std::vector<double>> data_;
};

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
inline std::pair<std::vector<int64_t>, std::vector<int64_t>> GetSpaceDims(
    const Shape& lhs_shape, const Shape& rhs_shape,
    const DotDimensionNumbers& dnums) {
  std::vector<int64_t> lhs_space_dims;
  std::vector<int64_t> rhs_space_dims;

  for (int64_t i = 0; i < lhs_shape.rank(); ++i) {
    if (absl::c_linear_search(dnums.lhs_batch_dimensions(), i) ||
        absl::c_linear_search(dnums.lhs_contracting_dimensions(), i)) {
      continue;
    }
    lhs_space_dims.push_back(i);
  }

  for (int64_t i = 0; i < rhs_shape.rank(); ++i) {
    if (absl::c_linear_search(dnums.rhs_batch_dimensions(), i) ||
        absl::c_linear_search(dnums.rhs_contracting_dimensions(), i)) {
      continue;
    }
    rhs_space_dims.push_back(i);
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

// Return whether the reshape is a special reshape that switches the batch dim
// of a dot.
bool IsBatchDimSwitchReshape(const HloInstruction* inst);

// Return whether the instruction is followed by a broadcast.
bool IsFollowedByBroadcast(const HloInstruction* inst);

// Return whether the instruction is followed by a reduce.
bool IsFollowedByReduce(const HloInstruction* inst);

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

// Remove all custom call makers in an HloModule.
void RemoveCustomCallMarker(HloModule* module);

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

// Propagate sharding for broadcast.
// The output will be tiled along the broadcasted dimension the same way
// as the input for the broadcast while the other dimensions are kept
// non-tiled.
HloSharding BroadcastSharding(const HloSharding& input_spec,
                              const Shape& new_shape,
                              absl::Span<const int64_t> dimensions);

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
bool IsValidTileAssignment(const HloSharding& spec);

// Get number of tile dimensions that are not 1. For example, for sharding spec
// {devices=[2,1,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
// spec.tile_assignment.num_dimensions() = [2,1,1,4]. This function returns 2.
// -1 means the tensor is replicated on the whole the mesh.
int64_t NumTileDimensions(const HloSharding& spec);

// Forcibly set the sharding of the operand of inst.
// Also fix the resharding between 1d and 2d logical mesh.
void FixMixedMeshShapeResharding(HloInstruction* inst, int operand_num,
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
    const absl::Span<const int64_t> reduced_dimensions, const int64_t op_count);

// Checks whether denominator is divisible by numerator.
bool IsDivisible(int64_t denominator, int64_t numerator);

// Generate all replica groups along one device_mesh dimension. Device_mesh can
// be any number of dimensions. |communication_dim| has to be one of
// |device_mesh|'s dimension.
std::vector<std::vector<int64_t>> GetReplicaGroupsAlongOneDimension(
    const Array<int64_t>& device_mesh, int32_t communication_dim);

// Gets values in |array| along |dim| while keeping indices at other
// dimensions at 0, e.g., array is 2D and dim = 1, this returns array[0, 1],
// array[1, 1], array [2, 1], ....
// Returns error status if dim >= array.num_dimensions().
StatusOr<std::vector<int64_t>> GetValuesAlongOneDim(const Array<int64_t>& array,
                                                    int dim);

StatusOr<int64_t> CheckArithmeticSequence(absl::Span<const int64_t> sequence);

// Checks if the number of sharded dimensions in the tile assignment matches the
// device mesh.
bool TileAssignmentMatchesMesh(const HloSharding& spec,
                               const Array<int64_t>& mesh);

// Get the mapped mesh dimension for every tensor dimension.
// The returned value maps ith tensor dim to one mesh dim. -1 means the tensor
// is replicated on that dimension.
// For example, returned value [1,2] means the 0th tensor dim maps to the 1st
// mesh dim, and 1st tensor dim maps to the 2nd mesh dim.
std::vector<int64_t> GetTensorDimToMeshDim(const int64_t tensor_shape_rank,
                                           const HloSharding& spec,
                                           const Array<int64_t>& device_mesh);

HloSharding Tile(const Shape& tensor_shape,
                 absl::Span<const int64_t> tensor_dims,
                 absl::Span<const int64_t> mesh_dims,
                 const Array<int64_t>& device_mesh);

// Transpose an array of any number of dimensions given any axes order.
// Similar to numpy.transpose(array, axes=()) function.
template <typename T>
Array<T> Transpose(const Array<T> array, std::vector<int64_t> axes) {
  // Computes transposed array's size.
  std::vector<int64_t> transposed_array_dimensions(array.dimensions());
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
size_t VectorGreaterThanOneElementCount(const std::vector<int64_t>& vector,
                                        bool omit_last_dim = false);

std::vector<int64_t> VectorGreaterThanOneElementIndices(
    const std::vector<int64_t>& vector, bool omit_last_dim = false);

int64_t GetInstructionSize(const Shape& shape);

int64_t GetShardedInstructionSize(
    const Shape& shape, int64_t num_devices,
    std::optional<HloSharding> sharding = std::nullopt);

HloInstruction* FindInstruction(
    const std::vector<HloInstruction*>& instructions, absl::string_view name);
double AllToAllCostUtil(double num_bytes, int mesh_dim, int64_t num_devices,
                        const std::vector<double>& mesh_alpha,
                        const std::vector<double>& mesh_beta);

double ReshardingCostMixedMeshShape(
    const Shape& shape, std::vector<int64_t> src_tensor_dim_to_mesh_dim,
    std::vector<int64_t> dst_tensor_dim_to_mesh_dim, int64_t num_devices,
    const std::vector<double>& mesh_alpha,
    const std::vector<double>& mesh_beta);

// When a complete mesh shape is [1, 8, 4], [1, 8, 1] is its partial mesh shape.
// If a sharding is [8, 4] for the complete mesh shape, we convert it to [8, 1]
// given [1, 8, 1] as the partial mesh shape.
// total_num_devices should equal to the product of mesh_shape elements.
bool AdjustShardingsWithPartialMeshShape(
    const std::vector<HloInstruction*>& instructions,
    const std::vector<int64_t>& mesh_shape, int64_t total_num_devices);

// Decompose mesh shapes into partial mesh shapes so that we can solve the auto
// sharding problem iteratively. Returns partial mesh shapes with larger
// dimensions first. For example, input [1, 4, 2] returns [1, 4, 1] and [1, 4,
// 2]; input [4, 8, 2] returns [1, 8, 1], [4, 8, 1] and [ 4, 8, 2].
std::vector<std::vector<int64_t>> DecomposeMeshShapes(
    std::vector<int64_t> mesh_shape);
}  // namespace spmd
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_UTIL_H_
