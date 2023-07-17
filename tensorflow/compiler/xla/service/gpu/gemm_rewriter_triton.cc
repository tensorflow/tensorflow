/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter_triton.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <queue>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/autotuning.pb.h"
#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_schedule.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_query.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_padding_requirements.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_types.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/instruction_fusion.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

bool TensorIterationSpec::operator==(const TensorIterationSpec& other) const {
  for (int dim = 0; dim < TensorIterationSpec::kMaxDimsPerTensor; ++dim) {
    if (dim_iteration_specs_[dim].size() != other[dim].size()) {
      return false;
    }
    for (int fragment = 0; fragment < dim_iteration_specs_[dim].size();
         ++fragment) {
      if (dim_iteration_specs_[dim][fragment].stride !=
              other[dim][fragment].stride ||
          dim_iteration_specs_[dim][fragment].count !=
              other[dim][fragment].count) {
        return false;
      }
    }
  }
  return true;
}

namespace {

// Batch dimensions of an operand of a dot instruction.
// Just an unified accessor to lhs_batch_dimensions and rhs_batch_dimensions.
const tsl::protobuf::RepeatedField<int64_t>& BatchDimensionsForOperand(
    const HloInstruction& dot, const int operand_number) {
  const DotDimensionNumbers& dimension_numbers = dot.dot_dimension_numbers();
  if (operand_number == 0) {
    return dimension_numbers.lhs_batch_dimensions();
  }
  return dimension_numbers.rhs_batch_dimensions();
}

// Index of the only contracting dimension of dot instruction operand.
int64_t ContractingDimensionIndex(const HloInstruction& dot,
                                  const int operand_number) {
  const DotDimensionNumbers& dimension_numbers = dot.dot_dimension_numbers();
  if (operand_number == 0) {
    CHECK_EQ(dimension_numbers.lhs_contracting_dimensions().size(), 1);
    return dimension_numbers.lhs_contracting_dimensions(0);
  }
  CHECK_EQ(dimension_numbers.rhs_contracting_dimensions().size(), 1);
  return dimension_numbers.rhs_contracting_dimensions(0);
}

// Index of the only non-contracting dimension of dot instruction operand.
int64_t NonContractingDimensionIndex(const HloInstruction& dot,
                                     const int operand_number) {
  StatusOr<std::vector<int64_t>> non_contracting_dims =
      GetNonContractingDims(dot.operand(operand_number)->shape(),
                            BatchDimensionsForOperand(dot, operand_number),
                            {ContractingDimensionIndex(dot, operand_number)});
  TF_CHECK_OK(non_contracting_dims.status());
  CHECK_EQ(non_contracting_dims->size(), 1);
  return non_contracting_dims->front();
}

// Data types that are tested to work in the triton GEMM emitter.
bool IsSupportedDataType(PrimitiveType type, GpuVersion gpu_version) {
  auto cuda_compute_capability =
      std::get<se::CudaComputeCapability>(gpu_version);
  switch (type) {
    case PRED:
    case S8:
    case S16:
    case S32:
    case F16:
    case F32:
      return true;
    case BF16:
      return cuda_compute_capability.IsAtLeast(
          stream_executor::CudaComputeCapability::AMPERE);
    default:
      return false;
  }
}

// Tells if f(a+b) == f(a) + f(b).
bool IsDistributiveOverAddition(const HloInstruction& hlo) {
  // The list is most likely incomplete.
  // For example division can be added too but only for operand #0.
  if (hlo.opcode() == HloOpcode::kMultiply ||
      hlo.opcode() == HloOpcode::kNegate ||
      hlo.opcode() == HloOpcode::kBitcast ||
      hlo.opcode() == HloOpcode::kReshape || hlo.opcode() == HloOpcode::kCopy ||
      hlo.opcode() == HloOpcode::kTranspose ||
      hlo.opcode() == HloOpcode::kConvert ||
      hlo.opcode() == HloOpcode::kBroadcast ||
      hlo.opcode() == HloOpcode::kSlice) {
    return true;
  }
  return false;
}

FusionDecision RequireTritonFusibleConvert(const HloInstruction* input,
                                           GpuVersion gpu_version) {
  // TODO(b/266862494): Can pick up almost any
  // convert, but if it's reducing the data volume it should rather be fused
  // to the output of the producer kernel. However not all operations support
  // output fusion - then it should be fused here anyway!
  if (ShapeUtil::ByteSizeOf(input->operand(0)->shape()) >
      ShapeUtil::ByteSizeOf(input->shape())) {
    return "Narrowing conversion.";
  }
  return FusionDecision{};
}

// Handles numbers of dimensions of a target HLO instruction
// projected onto source one.
// Used to calculate cumulative index transformations done by non-elementwise
// instructions between source and target.
class DimensionOrder {
 public:
  // Description of one dimension of HLO shape.
  struct DimDescription {
    int64_t target_dim_number;
    int subdim_number;
    int64_t size;
    bool operator==(const DimDescription& other) const {
      return target_dim_number == other.target_dim_number &&
             subdim_number == other.subdim_number && size == other.size;
    }
    std::string ToString() const {
      return absl::StrCat(target_dim_number, ":", subdim_number, ":", size);
    }
  };
  // Sequence describing all dimensions of HLO's output shape
  // in layout minor-to-major (physical) order.
  using DimOrderVector = std::vector<DimDescription>;

  DimensionOrder(const DimensionOrder&) = default;
  // Dimension order constructed for the output shape of `hlo`.
  // `hlo` is currently supposed to be an operand of dot();
  // dimension indices describing the operand
  // are stored along with the dimension order for later analysis.
  explicit DimensionOrder(
      const HloInstruction* hlo, const int64_t splittable_dimension_index = -1,
      const int64_t splittable_dimension_supported_major_size = 0)
      : splittable_dimension_index_(splittable_dimension_index),
        splittable_dimension_supported_major_part_size_(
            splittable_dimension_supported_major_size) {
    dim_order_.reserve(hlo->shape().rank());
    for (const int64_t i : hlo->shape().layout().minor_to_major()) {
      dim_order_.push_back({i, 0, hlo->shape().dimensions(i)});
    }
  }

  // Create dimension order describing a dot operand according to
  // the currently supported configurations.
  static DimensionOrder FromDotOperand(const HloInstruction& dot,
                                       int operand_number, int64_t split_k = 1);

  // Create dimension order describing dot's output.
  static DimensionOrder FromDotOutput(
      const HloInstruction& dot, int64_t split_k = 1,
      int64_t splittable_dimension_supported_major_part_size = 0);

  enum class TransformDirection { kInputToOutput, kOutputToInput };

  // Transforms the DimensionOrder so that from a description one side
  // of `hlo` it becomes a description of the other side of `hlo`.
  FusionDecision HandleInstruction(const HloInstruction* hlo,
                                   TransformDirection direction) {
    VLOG(7) << hlo->ToString();
    if (hlo->opcode() == HloOpcode::kParameter ||
        hlo_query::IsScalarConstant(hlo)) {
      return FusionDecision{};
    } else if (hlo->opcode() == HloOpcode::kTranspose ||
               hlo->opcode() == HloOpcode::kCopy) {
      return HandleCopyOrTransposeOrBroadcast(hlo, direction);
    } else if (hlo->opcode() == HloOpcode::kBroadcast) {
      if (direction != TransformDirection::kOutputToInput) {
        return "Unsupported broadcast direction.";
      }
      return HandleCopyOrTransposeOrBroadcast(hlo, direction);
    } else if (hlo->operand_count() > 0 &&
               IsTritonSupportedElementwise(
                   hlo->opcode(), hlo->operand(0)->shape().element_type())) {
      return FusionDecision{};
    } else if (hlo->opcode() == HloOpcode::kBitcast) {
      return HandleBitcast(hlo, direction);
    } else if (hlo->opcode() == HloOpcode::kReshape) {
      if (!ShapeUtil::ReshapeIsBitcast(hlo->operand(0)->shape(),
                                       hlo->shape())) {
        return "Non-bitcast reshape.";
      }
      return HandleBitcast(hlo, direction);
    }
    return "Unimplemented instruction.";
  }

  // Get the raw data of the dimension order.
  const DimOrderVector& GetDimOrderVector() const { return dim_order_; }

  // Index of dot dimension that can be split.
  // Currently typically LHS non-contracting one.
  int64_t SplittableDimensionIndex() const {
    return splittable_dimension_index_;
  }

  // Tells whether `size` major part of a dimension can be physically split.
  bool IsSupportedSplittableDimensionMajorPartSize(int64_t size) const {
    // 0 means no specific size requirement.
    return splittable_dimension_supported_major_part_size_ == 0 ||
           splittable_dimension_supported_major_part_size_ == size;
  }

  // Tells that two dimension orders describe the same tensor physical layout.
  bool IsPhysicallyEquivalent(const DimensionOrder& other) const;

  std::string ToString() const {
    return absl::StrJoin(dim_order_, "-",
                         [](std::string* out, const DimDescription& d) {
                           absl::StrAppend(out, d.ToString());
                         });
  }

 private:
  // See HandleInstruction() for the general description of Handle*().
  FusionDecision HandleBitcast(const HloInstruction*, TransformDirection);
  FusionDecision HandleCopyOrTransposeOrBroadcast(const HloInstruction*,
                                                  TransformDirection);

  DimOrderVector dim_order_;
  const int64_t splittable_dimension_index_;
  const int64_t splittable_dimension_supported_major_part_size_;
};

using DimIterationSpec = TensorIterationSpec::DimIterationSpec;

TensorIterationSpec DimensionOrderToTensorIterationSpec(
    const DimensionOrder& order) {
  const DimensionOrder::DimOrderVector& dim_order_vector =
      order.GetDimOrderVector();
  TensorIterationSpec tensor_spec;
  int64_t accumulated_stride = 1;
  for (int dim_order_index = 0; dim_order_index < dim_order_vector.size();
       ++dim_order_index) {
    const DimensionOrder::DimDescription& dim =
        dim_order_vector[dim_order_index];
    VLOG(6) << dim.target_dim_number << "\t" << dim.subdim_number << "\t"
            << dim.size;

    if (dim.size == 1) {
      continue;
    }

    DimIterationSpec& dim_spec = tensor_spec[dim.target_dim_number];
    if (dim_order_index > 0 &&
        dim_order_vector[dim_order_index - 1].target_dim_number ==
            dim.target_dim_number) {
      if (dim_spec.empty()) {
        // Previous parts of this dimension were degenerate -
        // so create the dimension here.
        dim_spec.push_back({accumulated_stride, dim.size, {dim.size}});
      } else {
        // Contiguous dimension, split only logically. Merge it back.
        dim_spec.back().count *= dim.size;
        dim_spec.back().subfragments.push_back(dim.size);
      }
    } else {
      dim_spec.push_back({accumulated_stride, dim.size, {dim.size}});
    }

    accumulated_stride *= dim.size;
  }
  // Create all absent dimensions as degenerate ones to simplify later queries.
  for (DimIterationSpec& dim_spec : tensor_spec) {
    if (dim_spec.empty()) {
      dim_spec.push_back({/*stride=*/0, /*count=*/1, /*subfragments=*/{1}});
    }
  }
  return tensor_spec;
}

bool DimensionOrder::IsPhysicallyEquivalent(const DimensionOrder& other) const {
  return DimensionOrderToTensorIterationSpec(*this) ==
         DimensionOrderToTensorIterationSpec(other);
}

DimensionOrder DimensionOrder::FromDotOperand(const HloInstruction& dot,
                                              const int operand_number,
                                              const int64_t split_k) {
  const HloInstruction* operand = dot.operand(operand_number);
  // There can be either none or one split-K batch dimension.
  const int num_split_k_batch_dims = split_k > 1;
  // LHS non-contracting dimension can be split if non-splitK batch is absent.
  if (operand_number == 0 &&
      dot.dot_dimension_numbers().lhs_batch_dimensions_size() -
              num_split_k_batch_dims ==
          0) {
    return DimensionOrder(
        operand, /*splittable_dimension_index=*/NonContractingDimensionIndex(
            dot, operand_number));
  }
  return DimensionOrder(operand);
}

DimensionOrder DimensionOrder::FromDotOutput(
    const HloInstruction& dot, const int64_t split_k,
    const int64_t splittable_dimension_supported_major_part_size) {
  // Allow non-contracting dimension originating from LHS to split if
  // this dimension is split at the output at the same ratio as
  // at the input.
  int64_t splittable_dimension_index = -1;
  if (splittable_dimension_supported_major_part_size > 1) {
    // Split-K dimension is the first one in the output if present;
    // LHS non-contracting follows (batch is absent in this case).
    splittable_dimension_index = (split_k > 1) ? 1 : 0;
  }
  return DimensionOrder(&dot, splittable_dimension_index,
                        splittable_dimension_supported_major_part_size);
}

FusionDecision DimensionOrder::HandleBitcast(const HloInstruction* hlo,
                                             TransformDirection direction) {
  const Shape& target_shape = (direction == TransformDirection::kOutputToInput)
                                  ? hlo->operand(0)->shape()
                                  : hlo->shape();
  DimOrderVector target_dim_order;
  target_dim_order.reserve(dim_order_.size());
  // Size of not yet assigned part of current target dimension.
  int64_t target_remaining_size = 1;
  // Iterate in parallel over source dimension order and target dimensions
  // in minor_to_major order. Find groups of dimensions of equal size
  // and project the source dimension order onto the target.
  auto target_dim_iter = target_shape.layout().minor_to_major().cbegin();
  for (auto src_dim = dim_order_.cbegin(); src_dim != dim_order_.cend();
       ++src_dim) {
    if (target_remaining_size >= src_dim->size) {
      if (target_remaining_size % src_dim->size) {
        return "Unsupported bitcast";
      }
      // Source dimension fragment completely fits into the target one:
      // just copy it as is.
      target_dim_order.push_back(*src_dim);
      // Update the size of the remaining part of the target that is
      // carried over to next source dimensions.
      target_remaining_size /= src_dim->size;
    } else {
      // Source is larger than target. Assign further target dimensions.
      // Size of the not yet assigned part of the source dimension.
      int64_t src_remaining_size = src_dim->size;
      // Subdimension index tracking dimension splits.
      int subdim_index = src_dim->subdim_number;
      if (target_remaining_size > 1) {
        // If there is a remaining fragment of a previous target dimension
        // assign it first.
        if (src_remaining_size % target_remaining_size) {
          return "Unsupported bitcast";
        }
        target_dim_order.push_back(
            {src_dim->target_dim_number, subdim_index, target_remaining_size});
        ++subdim_index;
        // Update the size of the fragment remaining to assign.
        src_remaining_size /= target_remaining_size;
        target_remaining_size = 1;
      }
      while (src_remaining_size > 1) {
        // Assign target dimensions until the source remainder is covered.
        int64_t target_dim_size = target_shape.dimensions(*target_dim_iter);
        int64_t new_fragment_size = target_dim_size;
        if (target_dim_size > src_remaining_size) {
          // If adding the next target dimension exceeds source fragment size
          // assign the remainder of the source and carry over the remainder
          // of the target.
          if (target_dim_size % src_remaining_size) {
            return "Unsupported bitcast";
          }
          target_remaining_size = target_dim_size / src_remaining_size;
          new_fragment_size = src_remaining_size;
        }
        target_dim_order.push_back(
            {src_dim->target_dim_number, subdim_index, new_fragment_size});
        src_remaining_size /= new_fragment_size;
        ++target_dim_iter;
        ++subdim_index;
      }
    }
  }
  CHECK_EQ(target_remaining_size, 1);

  // Handle remaining major dimensions of the target. Call all degenerate
  // ones subdimensions of the most-major non-degenerate one. Otherwise
  // give up.
  int subdim_index = target_dim_order.back().subdim_number + 1;
  while (target_dim_iter != target_shape.layout().minor_to_major().cend()) {
    if (target_shape.dimensions(*target_dim_iter) != 1) {
      return "Unsupported bitcast";
    }
    target_dim_order.push_back(
        {target_dim_order.back().target_dim_number, subdim_index, 1});
    ++subdim_index;
    ++target_dim_iter;
  }

  dim_order_ = target_dim_order;
  return FusionDecision{};
}

FusionDecision DimensionOrder::HandleCopyOrTransposeOrBroadcast(
    const HloInstruction* hlo, const TransformDirection direction) {
  // Every HLO dimension can correspond to a group of subdimensions in
  // dim_order_. For the easier handling of permutations: group dim_order_ by
  // dimension, apply permutations, then finally remove the grouping.
  // Group subdimensions by iterating over them in the same order as over
  // dimensions and matching by total size.
  const HloInstruction* src =
      (direction == TransformDirection::kOutputToInput) ? hlo : hlo->operand(0);
  const HloInstruction* dst =
      (direction == TransformDirection::kOutputToInput) ? hlo->operand(0) : hlo;
  std::vector<DimOrderVector> src_physical;
  src_physical.reserve(src->shape().rank());
  auto dim_order_it = dim_order_.cbegin();
  for (int64_t dim_index : src->shape().layout().minor_to_major()) {
    const int64_t dim_size = src->shape().dimensions(dim_index);
    int64_t subdim_size_accumulator = 1;
    DimOrderVector subdim_group;
    do {
      subdim_size_accumulator *= dim_order_it->size;
      subdim_group.push_back(*dim_order_it);
      ++dim_order_it;
    } while (subdim_size_accumulator < dim_size);
    CHECK_EQ(subdim_size_accumulator, dim_size);
    src_physical.push_back(subdim_group);
  }
  // Source physical -> source logical.
  std::vector<DimOrderVector> src_logical;
  src_logical.resize(src_physical.size());
  for (int i = 0; i < src_physical.size(); ++i) {
    src_logical[src->shape().layout().minor_to_major(i)] = src_physical[i];
  }
  // Source logical -> destination logical.
  std::vector<DimOrderVector> dst_logical;
  if (hlo->opcode() == HloOpcode::kTranspose) {
    const auto transpose = Cast<HloTransposeInstruction>(hlo);
    std::vector<int64_t> permutation(transpose->dimensions().cbegin(),
                                     transpose->dimensions().cend());
    if (direction == TransformDirection::kInputToOutput) {
      permutation = InversePermutation(permutation);
    }
    dst_logical.resize(permutation.size());
    for (int i = 0; i < permutation.size(); ++i) {
      dst_logical[permutation[i]] = src_logical[i];
    }
  } else if (hlo->opcode() == HloOpcode::kBroadcast) {
    const auto broadcast = Cast<HloBroadcastInstruction>(hlo);
    dst_logical.resize(broadcast->dimensions().size());
    for (int i = 0; i < broadcast->dimensions().size(); ++i) {
      dst_logical[i] = src_logical[broadcast->dimensions()[i]];
    }
  } else {
    // Copy preserves the logical shape, just permutes the layout.
    CHECK(ShapeUtil::SameDimensions(src->shape(), dst->shape()));
    dst_logical = src_logical;
  }
  // Destination logical -> destination physical and ungroup subdimensions.
  const Layout& dst_layout = dst->shape().layout();
  dim_order_.clear();
  for (int64_t dim_idx : dst_layout.minor_to_major()) {
    for (const DimDescription& subdim : dst_logical[dim_idx]) {
      dim_order_.push_back(subdim);
    }
  }
  return FusionDecision{};
}

// Tells if the dimension order is supported by the triton GEMM emitter.
// Only the dimension indicated by SplittableDimensionIndex() can be split
// physically once by other dimensions. Other ones can be only split logically.
// All subdimensions within a dimension have to be ordered.
FusionDecision RequireTritonGemmSupportedDimOrder(const DimensionOrder& order) {
  std::array<int, TensorIterationSpec::kMaxDimsPerTensor> subdim_counters = {
      -1, -1, -1, -1};
  std::array<int, TensorIterationSpec::kMaxDimsPerTensor> split_counters = {
      -1, -1, -1, -1};
  const DimensionOrder::DimOrderVector& dim_order_vector =
      order.GetDimOrderVector();
  VLOG(8) << order.ToString();
  for (int i = 0; i < dim_order_vector.size(); i++) {
    const auto [dim_number, subdim_number, size] = dim_order_vector[i];
    if (subdim_counters[dim_number] != subdim_number - 1) {
      return "Transpose within a dimension.";
    }
    ++subdim_counters[dim_number];
    if (size == 1) {
      continue;
    }
    if (i == 0 || dim_order_vector[i - 1].target_dim_number != dim_number) {
      ++split_counters[dim_number];
      if (dim_number == order.SplittableDimensionIndex() &&
          order.IsSupportedSplittableDimensionMajorPartSize(size)) {
        if (split_counters[dim_number] > 1) {
          return "2nd split of a splittable dimension.";
        }
      } else if (split_counters[dim_number] > 0) {
        return "Split of a non-splittable dimension.";
      }
    }
  }
  return FusionDecision{};
}

// Difference of input and output data volumes of an instruction.
int64_t InputMinusOutputBytes(const HloInstruction& hlo) {
  CHECK(!hlo.shape().IsTuple());
  int64_t input_size = 0;
  for (const HloInstruction* operand : hlo.operands()) {
    CHECK(!operand->shape().IsTuple());
    input_size += ShapeUtil::ByteSizeOf(operand->shape());
  }
  return input_size - ShapeUtil::ByteSizeOf(hlo.shape());
}

// Tells if an instruction has no user into which it could be fused.
// More cases should be added here.
bool CanNotBeFusedIntoAUser(const HloInstruction& hlo) {
  return hlo.IsRoot() || (hlo.user_count() == 1 && hlo.users()[0]->IsRoot() &&
                          hlo.users()[0]->opcode() == HloOpcode::kTuple);
}

// Tells if an instruction has no input into which it could be fused.
// More cases should be added here.
bool CanNotBeFusedIntoAProducer(const HloInstruction& hlo) {
  return hlo_query::AllOperandsAreParametersOrConstants(hlo);
}

// Let input and output data volumes of a fusion grow by small amounts.
constexpr int kIoToleranceBytes = 1024;

// Tells that fusing an instruction as an input is efficient.
bool IsInputWorthFusing(const HloInstruction& hlo) {
  if (hlo.user_count() > 1) {
    return false;
  }
  return hlo_query::AllOperandsAreParametersOrConstants(hlo) ||
         InputMinusOutputBytes(hlo) <= kIoToleranceBytes;
}

// Tells that fusing an instruction as an output is efficient.
bool IsOutputWorthFusing(const HloInstruction& hlo) {
  return CanNotBeFusedIntoAUser(hlo) ||
         InputMinusOutputBytes(hlo) >= -kIoToleranceBytes;
}

// Checks if the instruction is possible and profitable to fuse.
// If so tries to transform dim_order describing one side of `hlo` into a
// description of its other side if it is supported by the triton GEMM emitter.
FusionDecision CanFuse(const HloInstruction& hlo, bool as_input,
                       DimensionOrder& dim_order,
                       absl::flat_hash_map<const HloInstruction*,
                                           HloInstruction*>& old_to_new_mapping,
                       const GpuVersion gpu_version) {
  if (hlo.opcode() == HloOpcode::kTuple ||
      hlo.opcode() == HloOpcode::kGetTupleElement) {
    return "Unsupported instruction.";
  }
  for (const HloInstruction* operand : hlo.operands()) {
    if (!IsSupportedDataType(operand->shape().element_type(), gpu_version)) {
      return "Unsupported input data type.";
    }
  }
  if (!IsSupportedDataType(hlo.shape().element_type(), gpu_version)) {
    return "Unsupported output data type.";
  }
  if (hlo.opcode() == HloOpcode::kBroadcast &&
      !hlo_query::IsScalarConstant(hlo.operand(0))) {
    return "Skipping unsupported broadcast.";
  }
  if (as_input) {
    if (hlo.GetModule()
            ->config()
            .debug_options()
            .xla_gpu_triton_fusion_level() < 2) {
      if (hlo.opcode() == HloOpcode::kConvert) {
        if (FusionDecision decision =
                RequireTritonFusibleConvert(&hlo, gpu_version);
            !decision) {
          return decision;
        }
      } else if (hlo.IsElementwise() && hlo.opcode() != HloOpcode::kCopy) {
        return "Ignored elementwise operation";
      }
    } else {
      if (!CanNotBeFusedIntoAProducer(hlo) && !IsInputWorthFusing(hlo)) {
        return "Not obviously profitable to fuse as input.";
      }
    }
  } else {
    if (hlo.GetModule()
            ->config()
            .debug_options()
            .xla_gpu_triton_fusion_level() < 2) {
      return "Skipping fusing outputs at low fusion levels.";
    }
    for (const HloInstruction* operand : hlo.operands()) {
      // Skip already fused operands.
      if (old_to_new_mapping.contains(operand)) {
        continue;
      }
      // Currently only broadcasts of scalar constants or parameters
      // are accepted as other inputs of non-unary operations
      // in the output fusion.
      if (hlo_query::IsBroadcastOfScalarConstant(*operand) ||
          operand->opcode() == HloOpcode::kParameter) {
        continue;
      }
      return "Has multiple inputs - not properly analyzed yet.";
    }
    if (!IsOutputWorthFusing(hlo)) {
      return "Not obviously profitable to fuse as output.";
    }
  }

  if (FusionDecision decision = dim_order.HandleInstruction(
          &hlo, as_input ? DimensionOrder::TransformDirection::kOutputToInput
                         : DimensionOrder::TransformDirection::kInputToOutput);
      !decision) {
    return decision;
  }

  return RequireTritonGemmSupportedDimOrder(dim_order);
}

// Clone an instruction into the fusion.
void Fuse(HloInstruction& hlo,
          absl::flat_hash_map<const HloInstruction*, HloInstruction*>&
              old_to_new_mapping,
          std::vector<HloInstruction*>& fusion_inputs,
          HloComputation::Builder& builder) {
  if (old_to_new_mapping.contains(&hlo)) {
    return;
  }
  VLOG(3) << "Fusing " << hlo.ToString();
  auto get_or_add_parameter = [&](HloInstruction& instr) {
    if (auto it = old_to_new_mapping.find(&instr);
        it != old_to_new_mapping.end()) {
      return it->second;
    }
    fusion_inputs.push_back(&instr);
    return old_to_new_mapping
        .insert({&instr,
                 builder.AddInstruction(HloInstruction::CreateParameter(
                     fusion_inputs.size() - 1, instr.shape(),
                     absl::StrCat("parameter_", fusion_inputs.size() - 1)))})
        .first->second;
  };
  if (hlo.opcode() == HloOpcode::kParameter ||
      hlo.opcode() == HloOpcode::kGetTupleElement) {
    get_or_add_parameter(hlo);
  } else {
    std::vector<HloInstruction*> hlo_new_operands;
    for (HloInstruction* operand : hlo.operands()) {
      hlo_new_operands.push_back(get_or_add_parameter(*operand));
    }
    old_to_new_mapping[&hlo] = builder.AddInstruction(
        hlo.CloneWithNewOperands(hlo.shape(), hlo_new_operands));
  }
}

// Tells how many new parameters does a fusion gain by fusing the operation as
// an input.
int64_t NumAddedParameters(const HloInstruction& hlo) {
  // Non-scalar constant is equivalent to a parameter: one input, one output.
  if (hlo.opcode() == HloOpcode::kConstant &&
      !ShapeUtil::IsScalar(hlo.shape())) {
    return 0;
  }
  // All other instructions add all own inputs and remove own single output.
  return hlo.operand_count() - 1;
}

// Fuse an instruction with all its fusible inputs.
// If an input is not fusible stop there and make a parameter of the new
// fusion, otherwise put it onto stack and check its own inputs first.
void FuseWithInputsRecursively(
    HloInstruction* root, DimensionOrder root_dim_order,
    // Dimension orders describing inputs of corresponding instructions.
    absl::flat_hash_map<const HloInstruction*, DimensionOrder>& dim_orders,
    const GpuVersion gpu_version,
    absl::flat_hash_map<const HloInstruction*, HloInstruction*>&
        old_to_new_mapping,
    std::vector<HloInstruction*>& fusion_inputs,
    HloComputation::Builder& builder) {
  absl::flat_hash_set<const HloInstruction*> visited;
  std::stack<HloInstruction*> to_fuse;
  // Instructions at the edge 'to_fuse' that can either get fused too or
  // become parameters of the fusion. Used to track the number of parameters
  // of the fusion.
  absl::flat_hash_set<const HloInstruction*> inputs;
  // Currently only one physically unique dim order per scope is supported.
  // Let it change while the scope has one input; afterwards require all
  // of them to be physically compatible.
  const HloInstruction* reference_dim_order_hlo = nullptr;
  if (CanFuse(*root, /*as_input=*/true, root_dim_order, old_to_new_mapping,
              gpu_version)) {
    to_fuse.push(root);
    inputs.insert(root->operands().begin(), root->operands().end());
    // root_dim_order went through output -> input transformation here.
    CHECK(dim_orders.insert({root, root_dim_order}).second) << root->ToString();
  }
  visited.insert(root);
  while (!to_fuse.empty()) {
    bool top_is_ready_to_fuse = true;
    HloInstruction* hlo = to_fuse.top();
    if (reference_dim_order_hlo == nullptr && hlo->operand_count() > 1) {
      reference_dim_order_hlo = hlo;
    }
    for (HloInstruction* operand : hlo->mutable_operands()) {
      if (visited.insert(operand).second) {
        // Stop adding new parameters.
        if (inputs.size() >= DotFusionAnalysis::kMaxParameterPerScope &&
            NumAddedParameters(*operand) > 0) {
          continue;
        }
        // Operand's output is described by its consumer's input.
        DimensionOrder operand_dim_order(dim_orders.at(hlo));
        // CanFuse() makes output -> input transformation of
        // operand_dim_order if succeeds.
        if (CanFuse(*operand, /*as_input=*/true, operand_dim_order,
                    old_to_new_mapping, gpu_version)) {
          if (reference_dim_order_hlo != nullptr &&
              !operand_dim_order.IsPhysicallyEquivalent(
                  dim_orders.at(reference_dim_order_hlo))) {
            continue;
          }
          to_fuse.push(operand);
          if (operand->opcode() != HloOpcode::kParameter) {
            inputs.erase(operand);
          }
          inputs.insert(operand->operands().begin(), operand->operands().end());
          // Save the dimension order description of operand's input.
          CHECK(dim_orders.insert({operand, operand_dim_order}).second)
              << operand->ToString();
          top_is_ready_to_fuse = false;
        }
      }
    }
    if (top_is_ready_to_fuse) {
      Fuse(*hlo, old_to_new_mapping, fusion_inputs, builder);
      to_fuse.pop();
    }
  }
}

// Fuses dot and the compatible and profitable to fuse operations around it
// into a new fusion computation constructed using the builder. fusion_inputs
// get populated with the non-fused instructions that become operands of the
// call to this fusion. fusion_output_ptr (if not nullptr) gets assigned the
// original instruction that has to be replaced by the call to the fusion.
StatusOr<FusionDecision> FuseDot(HloInstruction& dot,
                                 const GpuVersion gpu_version,
                                 HloComputation::Builder& builder,
                                 std::vector<HloInstruction*>& fusion_inputs,
                                 HloInstruction** fusion_output_ptr) {
  VLOG(5) << dot.ToString();
  if (FusionDecision can_handle = CanTritonHandleGEMM(dot, gpu_version);
      !can_handle) {
    VLOG(3) << can_handle.Explain();
    return can_handle;
  }

  // Original instruction -> fused one.
  absl::flat_hash_map<const HloInstruction*, HloInstruction*>
      old_to_new_mapping;

  // Separate traversal from LHS and RHS inputs of the dot: they use
  // differently shaped tiles but may go through same HLO graph nodes.
  // Direct dot inputs have well defined dimension orders.

  auto fuse_inputs = [&](int operand_number)
      -> StatusOr<absl::flat_hash_map<const HloInstruction*, DimensionOrder>> {
    absl::flat_hash_map<const HloInstruction*, DimensionOrder> dim_orders;
    int operand_count_before = fusion_inputs.size();
    // Direct dot inputs have well defined dimension orders.
    FuseWithInputsRecursively(
        dot.mutable_operand(operand_number),
        DimensionOrder::FromDotOperand(dot, operand_number), dim_orders,
        gpu_version, old_to_new_mapping, fusion_inputs, builder);
    TF_RET_CHECK(fusion_inputs.size() - operand_count_before <=
                 DotFusionAnalysis::kMaxParameterPerScope);
    return dim_orders;
  };
  // Check if non-contracting dimension originating from LHS operand in the
  // output can be split. This currently requires this dimension being split
  // in the operand the same way.
  int64_t lhs_nc_split_major_part = -1;
  {
    TF_ASSIGN_OR_RETURN(const auto lhs_dim_orders, fuse_inputs(0));
    // Looking at first LHS parameter to find split non-contracting dimension
    // is sufficient because currently all parameters of one scope have to use
    // the same tiling.
    auto first_lhs_parameter_it = lhs_dim_orders.cbegin();
    while (first_lhs_parameter_it != lhs_dim_orders.cend()) {
      if (first_lhs_parameter_it->first->opcode() == HloOpcode::kParameter) {
        break;
      }
      ++first_lhs_parameter_it;
    }
    if (first_lhs_parameter_it != lhs_dim_orders.cend()) {
      const auto lhs_nc_iter_spec = DimensionOrderToTensorIterationSpec(
          first_lhs_parameter_it->second)[NonContractingDimensionIndex(dot, 0)];
      if (lhs_nc_iter_spec.size() > 1) {
        lhs_nc_split_major_part = lhs_nc_iter_spec.at(1).count;
      }
    }
  }
  TF_RET_CHECK(fuse_inputs(1).ok());

  Fuse(dot, old_to_new_mapping, fusion_inputs, builder);

  // Fusion at dot's output.

  // These describe _outputs_ of corresponding HLOs.
  absl::flat_hash_map<const HloInstruction*, DimensionOrder> out_dim_orders;
  out_dim_orders.insert(
      {&dot, DimensionOrder::FromDotOutput(dot, /*split_k=*/1,
                                           lhs_nc_split_major_part)});
  HloInstruction* fusion_output = &dot;
  bool output_changed = true;
  while (output_changed) {
    output_changed = false;
    if (fusion_output->user_count() != 1) {
      break;
    }
    HloInstruction* user = fusion_output->users()[0];
    if (!IsDistributiveOverAddition(*user)) {
      break;
    }
    // Describes the output of `current_output` = input of `user`.
    DimensionOrder dim_order(out_dim_orders.at(fusion_output));
    if (CanFuse(*user, /*as_input=*/false, dim_order, old_to_new_mapping,
                gpu_version)) {
      // Now it describes the output of the user.
      CHECK(out_dim_orders.insert({user, dim_order}).second);
      for (HloInstruction* operand : user->operands()) {
        if (!old_to_new_mapping.contains(operand)) {
          // Here we need again a dim order describing inputs of the user.
          FuseWithInputsRecursively(
              operand, DimensionOrder(out_dim_orders.at(fusion_output)),
              out_dim_orders, gpu_version, old_to_new_mapping, fusion_inputs,
              builder);
        }
      }
      Fuse(*user, old_to_new_mapping, fusion_inputs, builder);
      fusion_output = user;
      output_changed = true;
    }
  }
  if (fusion_output_ptr != nullptr) {
    *fusion_output_ptr = fusion_output;
  }
  if (dot.GetModule()->config().debug_options().xla_gpu_triton_gemm_any()) {
    return FusionDecision{};
  }
  for (const auto& iter : old_to_new_mapping) {
    if (iter.second->opcode() == HloOpcode::kConvert ||
        iter.second->opcode() == HloOpcode::kTranspose) {
      return FusionDecision{};
    }
  }
  return "No profitable operations to fuse.";
}

// Extracts into fused computations parts of HLO graph including dot()
// operations that can target the triton GEMM emitter.
class GemmRewriterTritonVisitor : public DfsHloRewriteVisitor {
 public:
  explicit GemmRewriterTritonVisitor(const GpuVersion gpu_version)
      : gpu_version_(gpu_version) {}
  // Checks that a dot() should be targeting the triton GEMM emitter;
  // if so - fuses all its compatible inputs and outputs as a new computation
  // and replaces the original dot() with a call to the computation.
  Status HandleDot(HloInstruction* dot) override {
    std::string fusion_name = absl::StrCat("triton_gemm_", dot->name());
    HloComputation::Builder builder(absl::StrCat(fusion_name, "_computation"));
    std::vector<HloInstruction*> fusion_inputs;
    HloInstruction* fusion_output = nullptr;
    TF_ASSIGN_OR_RETURN(
        const FusionDecision should_fuse,
        FuseDot(*dot, gpu_version_, builder, fusion_inputs, &fusion_output));
    if (builder.last_added_instruction() == nullptr) {
      return OkStatus();
    }
    // If a GEMM requiring padding for cuBLAS is encountered here this
    // happened because earlier ShouldTritonHandleGEMM() accepted it and padding
    // was skipped. Accept it ignoring profitability checks.
    if (!CublasRequiresPadding(
            *Cast<HloDotInstruction>(dot),
            std::get<se::CudaComputeCapability>(gpu_version_)) &&
        !should_fuse) {
      return OkStatus();
    }

    HloComputation* computation =
        dot->GetModule()->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                            /*is_entry=*/false);
    HloInstruction* dot_fusion =
        dot->parent()->AddInstruction(HloInstruction::CreateFusion(
            computation->root_instruction()->shape(),
            HloInstruction::FusionKind::kCustom, fusion_inputs, computation));
    dot_fusion->GetModule()->SetAndUniquifyInstrName(dot_fusion, fusion_name);

    TF_ASSIGN_OR_RETURN(auto backend_config,
                        dot_fusion->backend_config<FusionBackendConfig>());
    backend_config.set_kind(std::string(kTritonGemmFusionKind));
    TF_RETURN_IF_ERROR(dot_fusion->set_backend_config(backend_config));

    if (fusion_output->IsRoot()) {
      fusion_output->parent()->set_root_instruction(dot_fusion);
      TF_RETURN_IF_ERROR(
          fusion_output->parent()->RemoveInstructionAndUnusedOperands(
              fusion_output));
      MarkAsChanged();
    } else {
      TF_RETURN_IF_ERROR(ReplaceInstruction(fusion_output, dot_fusion));
    }
    XLA_VLOG_LINES(5, computation->ToString());
    return OkStatus();
  }

 private:
  GpuVersion gpu_version_;
};

StatusOr<bool> RunOnComputation(HloComputation* computation,
                                GpuVersion gpu_version) {
  GemmRewriterTritonVisitor visitor(gpu_version);
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}

// Copy source values into destination incrementing those >= threshold by 1.
void CopyIncrementingAboveThreshold(
    const tsl::protobuf::RepeatedField<int64_t>& source,
    tsl::protobuf::RepeatedField<int64_t>& destination, const int threshold) {
  destination.Reserve(source.size());
  for (int64_t x : source) {
    if (x >= threshold) {
      ++x;
    }
    destination.Add(x);
  }
}

Status UncompilableMatmul(absl::string_view explanation) {
  Status s = absl::CancelledError(explanation);
  s.SetPayload(kUncompilableFusion, absl::Cord(explanation));
  return s;
}

StatusOr<HloInstruction*> MakeSplitKOperand(
    HloInstruction& dot, const DotFusionAnalysis& analysis,
    const AutotuneResult::TritonGemmKey& tiling,
    const int64_t contracting_dim_idx, const int operand_number) {
  const Shape& shape = dot.operand(operand_number)->shape();
  Shape new_shape(shape.element_type(), {}, {}, {});

  // TODO(b/274775195): implement split-K with padding.
  if (tiling.split_k() > shape.dimensions(contracting_dim_idx)) {
    return UncompilableMatmul("Too small total contracting dimension size.");
  }
  DotFusionAnalysis::Scope scope = (operand_number == 0)
                                       ? DotFusionAnalysis::Scope::LHS
                                       : DotFusionAnalysis::Scope::RHS;
  for (const HloInstruction* param : analysis.ScopeParameters(scope)) {
    // If an operand of dot does not read any parameters its K dimension
    // does not need analysis for fragmentation.
    const DimIterationSpec* spec =
        analysis.IterSpec(scope, param, contracting_dim_idx);
    // Split contracting dimension is not implemented yet.
    CHECK_EQ(spec->size(), 1);
    auto fragment = spec->at(0).subfragments.crbegin();
    int64_t size_to_split = tiling.split_k();
    while (size_to_split > *fragment) {
      if (size_to_split % *fragment) {
        return UncompilableMatmul("Contracting dimension is too fragmented.");
      }
      size_to_split /= *fragment;
      ++fragment;
    }
    if (*fragment % size_to_split) {
      return UncompilableMatmul("Contracting dimension is too fragmented.");
    }
    if (tiling.split_k() > ceil(1.0 * spec->at(0).count / tiling.block_k())) {
      return UncompilableMatmul(
          "Too small divisible part of the contracting dimension.");
    }
  }

  for (int i = 0; i < shape.rank(); ++i) {
    const int64_t dimension_size = shape.dimensions(i);
    if (i == contracting_dim_idx) {
      new_shape.add_dimensions(tiling.split_k());
      new_shape.add_dimensions(dimension_size / tiling.split_k());
    } else {
      new_shape.add_dimensions(dimension_size);
    }
  }

  Layout* new_layout = new_shape.mutable_layout();
  // Iterate through the logical dimension numbers in their physical order;
  // copy them into the new layout incrementing by one those that get shifted
  // by the insertion of the new batch dimension.
  for (int64_t logical_dim_idx : shape.layout().minor_to_major()) {
    // When 'logical_dim_idx' == 'contracting_dim_idx' add both
    // 'logical_dim_idx'+1 and 'logical_dim_idx' because it gets split into two.
    if (logical_dim_idx >= contracting_dim_idx) {
      new_layout->add_minor_to_major(logical_dim_idx + 1);
    }
    if (logical_dim_idx <= contracting_dim_idx) {
      new_layout->add_minor_to_major(logical_dim_idx);
    }
  }
  return MakeBitcastHlo(dot.mutable_operand(operand_number), new_shape);
}

// Apply split K configuration from the tiling to the fused dot() computation:
// bitcast the operands, change the output shape and the dot dimensions.
Status MakeDotComputationSplitKBatch(
    HloComputation* computation, const AutotuneResult::TritonGemmKey& tiling,
    bool disable_reduced_precision_reduction) {
  HloInstruction* dot =
      hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
  const DotFusionAnalysis analysis(computation);
  const DotDimensionNumbers& old_dim_numbers = dot->dot_dimension_numbers();
  DotDimensionNumbers new_dim_numbers;

  const int64_t lhs_contracting_idx = ContractingDimensionIndex(*dot, 0);
  CopyIncrementingAboveThreshold(
      old_dim_numbers.lhs_contracting_dimensions(),
      *new_dim_numbers.mutable_lhs_contracting_dimensions(),
      lhs_contracting_idx);
  new_dim_numbers.mutable_lhs_batch_dimensions()->Add(lhs_contracting_idx);
  CopyIncrementingAboveThreshold(
      old_dim_numbers.lhs_batch_dimensions(),
      *new_dim_numbers.mutable_lhs_batch_dimensions(), lhs_contracting_idx);

  const int64_t rhs_contracting_idx = ContractingDimensionIndex(*dot, 1);
  CopyIncrementingAboveThreshold(
      old_dim_numbers.rhs_contracting_dimensions(),
      *new_dim_numbers.mutable_rhs_contracting_dimensions(),
      rhs_contracting_idx);
  new_dim_numbers.mutable_rhs_batch_dimensions()->Add(rhs_contracting_idx);
  CopyIncrementingAboveThreshold(
      old_dim_numbers.rhs_batch_dimensions(),
      *new_dim_numbers.mutable_rhs_batch_dimensions(), rhs_contracting_idx);

  // Collect HLOs to transform between dot output and root. These will
  // get a new major most batch dimension sized as split K factor. Other inputs
  // of these HLOs will get broadcasted.
  std::stack<HloInstruction*> to_process;
  // Store the same HLOs also in a hash set for quick lookups.
  absl::flat_hash_set<HloInstruction*> to_process_set;
  HloInstruction* current = dot;
  do {
    to_process.push(current);
    CHECK(to_process_set.insert(current).second);
    if (current->users().empty()) {
      break;
    }
    CHECK_EQ(current->user_count(), 1);
    current = current->users()[0];
    if (!IsDistributiveOverAddition(*current)) {
      return Cancelled("Operation non-distributive over addition after dot.");
    }
  } while (true);

  // Process the collected HLOs from computation root to dot.
  while (!to_process.empty()) {
    HloInstruction* current = to_process.top();
    to_process.pop();
    // Add split-K dimension to `current`.
    HloInstruction* expanded;
    if (current == dot) {
      TF_ASSIGN_OR_RETURN(
          HloInstruction * lhs,
          MakeSplitKOperand(*dot, analysis, tiling, lhs_contracting_idx, 0));
      TF_ASSIGN_OR_RETURN(
          HloInstruction * rhs,
          MakeSplitKOperand(*dot, analysis, tiling, rhs_contracting_idx, 1));
      expanded = MakeDotHlo(lhs, rhs, new_dim_numbers, dot->precision_config(),
                            dot->shape().element_type())
                     .value();
      dot->SetupDerivedInstruction(expanded);
    } else {
      expanded = computation->AddInstruction(
          current->CloneWithNewShape(ShapeUtil::PrependMajorDimension(
              tiling.split_k(), current->shape())));
    }
    TF_RETURN_IF_ERROR(current->ReplaceAllUsesWithDifferentShape(expanded));
    TF_RETURN_IF_ERROR(computation->RemoveInstruction(current));
    // Broadcast operands.
    if (current == dot) {
      continue;
    }
    for (int i = 0; i < expanded->operands().size(); ++i) {
      HloInstruction* operand = expanded->mutable_operand(i);
      if (!to_process_set.contains(operand)) {
        std::vector<int64_t> broadcast_dimensions(operand->shape().rank());
        absl::c_iota(broadcast_dimensions, 1);
        TF_RETURN_IF_ERROR(expanded->ReplaceOperandWithDifferentShape(
            i, MakeBroadcastHlo(operand, broadcast_dimensions,
                                ShapeUtil::PrependMajorDimension(
                                    tiling.split_k(), operand->shape()))));
      }
    }
  }

  if (disable_reduced_precision_reduction) {
    PrimitiveType output_type =
        computation->root_instruction()->shape().element_type();
    PrimitiveType accumulator_type = output_type == PrimitiveType::F64
                                         ? PrimitiveType::F64
                                         : PrimitiveType::F32;

    computation->root_instruction()->mutable_shape()->set_element_type(
        accumulator_type);
  }
  return OkStatus();
}

// Propagate dimension orders in consumer->producer direction starting at
// `origin` with input `origin_dim_order` till parameters of the computation.
// Store the found parameters and their iteration specs.
Status PropagateDimensionOrdersToParameters(
    const HloInstruction& origin, const DimensionOrder& origin_dim_order,
    absl::flat_hash_set<const HloInstruction*>& parameters,
    absl::flat_hash_map<const HloInstruction*, TensorIterationSpec>&
        iter_specs) {
  absl::flat_hash_set<const HloInstruction*> visited;
  std::queue<const HloInstruction*> to_process;
  // Dimension orders describing inputs of corresponding instructions.
  absl::flat_hash_map<const HloInstruction*, DimensionOrder> dim_orders;
  TF_RET_CHECK(RequireTritonGemmSupportedDimOrder(origin_dim_order));
  dim_orders.insert({&origin, origin_dim_order});
  visited.insert(&origin);
  to_process.push(&origin);
  while (!to_process.empty()) {
    const HloInstruction* hlo = to_process.front();
    to_process.pop();
    if (hlo->opcode() == HloOpcode::kParameter) {
      // One parameter corresponds to one iteration spec in the results of the
      // analysis. This describes well situations when a parameter has one or
      // more elementwise users - they share the same tiling. Situations when
      // one instruction is read differently by different users in the same
      // scope of the dot are currently prevented during the fusion.
      TF_RET_CHECK(parameters.insert(hlo).second);
      VLOG(5) << hlo->ToString();
    }
    for (const HloInstruction* operand : hlo->operands()) {
      if (!visited.insert(operand).second) {
        continue;
      }
      if (operand->opcode() == HloOpcode::kDot) {
        // Encountering the dot itself happens during the processing of the
        // output fusion. The propagation should stop at it.
        continue;
      }
      // Operand's output is described by its consumer's input.
      auto [it, inserted] =
          dim_orders.insert({operand, DimensionOrder(dim_orders.at(hlo))});
      TF_RET_CHECK(inserted);
      DimensionOrder& hlo_operand_dim_order = it->second;
      TF_RET_CHECK(hlo_operand_dim_order.HandleInstruction(
          operand, DimensionOrder::TransformDirection::kOutputToInput))
          << operand->ToString();
      TF_RET_CHECK(RequireTritonGemmSupportedDimOrder(hlo_operand_dim_order));
      to_process.push(operand);
    }
  }
  // For now all parameters of one scope have to use the same tiling.
  for (const HloInstruction* parameter : parameters) {
    TF_RET_CHECK(dim_orders.at(parameter).IsPhysicallyEquivalent(
        dim_orders.at(*parameters.cbegin())));
    iter_specs[parameter] =
        DimensionOrderToTensorIterationSpec(dim_orders.at(parameter));
  }
  return OkStatus();
}

}  // anonymous namespace

// BF16 is supported in a sense that all operations on it are implemented
// through F32 and converts have to be inserted into the HLO graph, but
// they can be missing during fusion.

std::vector<HloOpcode> TritonSupportedUnaryElementwise(
    PrimitiveType element_type) {
  std::vector<HloOpcode> ret = {HloOpcode::kConvert};
  if (element_type == PrimitiveType::PRED) {
    ret.push_back(HloOpcode::kNot);
    return ret;
  }
  ret.push_back(HloOpcode::kAbs);
  ret.push_back(HloOpcode::kNegate);
  if (element_type == PrimitiveType::F32 ||
      element_type == PrimitiveType::BF16 ||
      element_type == PrimitiveType::F64) {
    absl::c_copy(std::vector<HloOpcode>{HloOpcode::kCos, HloOpcode::kExp,
                                        HloOpcode::kExpm1, HloOpcode::kLog,
                                        HloOpcode::kLog1p, HloOpcode::kRsqrt,
                                        HloOpcode::kSin, HloOpcode::kSqrt,
                                        HloOpcode::kCbrt, HloOpcode::kTan,
                                        HloOpcode::kTanh},
                 std::back_inserter(ret));
  }
  return ret;
}

std::vector<HloOpcode> TritonSupportedBinaryElementwise(
    PrimitiveType element_type) {
  if (element_type == PrimitiveType::PRED) {
    return {HloOpcode::kAnd, HloOpcode::kOr, HloOpcode::kXor,
            HloOpcode::kCompare};
  }
  std::vector<HloOpcode> ret = {HloOpcode::kAdd,      HloOpcode::kCompare,
                                HloOpcode::kMaximum,  HloOpcode::kMinimum,
                                HloOpcode::kMultiply, HloOpcode::kSubtract};
  if (element_type == PrimitiveType::F32 ||
      element_type == PrimitiveType::BF16 ||
      element_type == PrimitiveType::F64) {
    ret.push_back(HloOpcode::kAtan2);
    ret.push_back(HloOpcode::kDivide);
    ret.push_back(HloOpcode::kPower);
  }
  return ret;
}

std::vector<HloOpcode> TritonSupportedTernaryElementwise(
    PrimitiveType element_type) {
  return {HloOpcode::kSelect};
}

bool IsTritonSupportedElementwise(HloOpcode opcode,
                                  PrimitiveType element_type) {
  return absl::c_linear_search(TritonSupportedUnaryElementwise(element_type),
                               opcode) ||
         absl::c_linear_search(TritonSupportedBinaryElementwise(element_type),
                               opcode) ||
         absl::c_linear_search(TritonSupportedTernaryElementwise(element_type),
                               opcode);
}

Status MakeDotSplitKBatch(HloInstruction* dot_fusion,
                          const AutotuneResult::TritonGemmKey& tiling) {
  CHECK_EQ(dot_fusion->opcode(), HloOpcode::kFusion);

  if (dot_fusion->shape().IsTuple()) {
    return Unimplemented("Tuple output is not supported with split-K yet.");
  }

  const bool disable_reduced_precision_reduction =
      dot_fusion->GetModule()
          ->config()
          .debug_options()
          .xla_gpu_triton_gemm_disable_reduced_precision_reduction();
  const PrimitiveType output_type = dot_fusion->shape().element_type();
  const Layout output_layout = dot_fusion->shape().layout();

  TF_RETURN_IF_ERROR(MakeDotComputationSplitKBatch(
      dot_fusion->fused_instructions_computation(), tiling,
      disable_reduced_precision_reduction));
  const HloInstruction* root = dot_fusion->fused_expression_root();

  *dot_fusion->mutable_shape() = root->shape();
  HloInstruction* zero =
      dot_fusion->parent()->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(root->shape().element_type())));
  // The batch dimension to reduce is the first one by construction.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * reduce,
      MakeReduceHlo(dot_fusion, zero, /*dimensions=*/{0}, HloOpcode::kAdd));

  // If the original dot had non-standard layout, this reduce should have that
  // too.
  *reduce->mutable_shape()->mutable_layout() = output_layout;

  if (dot_fusion->IsRoot()) {
    dot_fusion->parent()->set_root_instruction(reduce,
                                               /*accept_different_shape=*/true);
  } else {
    TF_RETURN_IF_ERROR(dot_fusion->ReplaceAllUsesWithDifferentShape(reduce));
  }

  if (disable_reduced_precision_reduction) {
    HloInstruction* convert = MakeConvertToHlo(reduce, output_type);
    if (reduce->IsRoot()) {
      reduce->parent()->set_root_instruction(convert,
                                             /*accept_different_shape=*/true);
    } else {
      TF_RETURN_IF_ERROR(reduce->ReplaceAllUsesWithDifferentShape(convert));
    }
  }

  return OkStatus();
}

DotFusionAnalysis::DotFusionAnalysis(const HloComputation* dot_computation,
                                     const int64_t split_k) {
  VLOG(5) << dot_computation->ToString();

  const HloInstruction* dot = hlo_query::GetFirstInstructionWithOpcode(
      *dot_computation, HloOpcode::kDot);

  for (const Scope scope : {Scope::LHS, Scope::RHS}) {
    const int operand_number = static_cast<int>(scope);
    const HloInstruction* operand = dot->operand(operand_number);
    DimensionOrder dot_operand_dim_order =
        DimensionOrder::FromDotOperand(*dot, operand_number, split_k);
    CHECK(dot_operand_dim_order.HandleInstruction(
        operand, DimensionOrder::TransformDirection::kOutputToInput));
    CHECK_OK(PropagateDimensionOrdersToParameters(
        *operand, dot_operand_dim_order, parameters_[scope],
        iter_specs_[scope]));
  }

  int64_t lhs_nc_split_major_part_size = -1;
  if (!ScopeParameters(Scope::LHS).empty()) {
    const TensorIterationSpec::DimIterationSpec* lhs_nc_iter_spec =
        IterSpec(Scope::LHS, *ScopeParameters(Scope::LHS).cbegin(),
                 NonContractingDimensionIndex(*dot, 0));
    if (lhs_nc_iter_spec->size() > 1) {
      lhs_nc_split_major_part_size = lhs_nc_iter_spec->at(1).count;
    }
  }
  DimensionOrder dim_order = DimensionOrder::FromDotOutput(
      *dot, split_k, lhs_nc_split_major_part_size);
  const HloInstruction* output = dot;
  // Currently supported is one fusion output and one path from dot to it.
  // Propagate dimension order from dot to root.
  while (!output->IsRoot()) {
    CHECK_EQ(output->user_count(), 1);
    output = output->users()[0];
    CHECK(dim_order.HandleInstruction(
        output, DimensionOrder::TransformDirection::kInputToOutput));
    CHECK(RequireTritonGemmSupportedDimOrder(dim_order));
  }
  CHECK(iter_specs_[Scope::OUTPUT]
            .insert({output, DimensionOrderToTensorIterationSpec(dim_order)})
            .second);
  if (output != dot) {
    // Propagate back to parameters of the output fusion.
    CHECK(dim_order.HandleInstruction(
        output, DimensionOrder::TransformDirection::kOutputToInput));
    CHECK_OK(PropagateDimensionOrdersToParameters(*output, dim_order,
                                                  parameters_[Scope::OUTPUT],
                                                  iter_specs_[Scope::OUTPUT]));
  }
}

const DimIterationSpec* DotFusionAnalysis::IterSpec(
    const DotFusionAnalysis::Scope scope, const HloInstruction* hlo,
    const int dimension) const {
  auto ret = iter_specs_.at(scope).find(hlo);
  if (ret != iter_specs_.at(scope).end()) {
    return &ret->second[dimension];
  }
  return nullptr;
}

FusionDecision CanTritonHandleGEMM(const HloInstruction& dot,
                                   const GpuVersion gpu_version) {
  if (dot.opcode() != HloOpcode::kDot ||
      absl::c_any_of(dot.precision_config().operand_precision(),
                     [](int x) { return x != PrecisionConfig::DEFAULT; })) {
    return "Non-default precision.";
  }

  auto supported_output_type = [&](const PrimitiveType t) {
    const auto cuda_compute_capability =
        std::get<se::CudaComputeCapability>(gpu_version);
    switch (t) {
      case F16:
      case F32:
        return true;
      case BF16:
        return cuda_compute_capability.IsAtLeast(
            stream_executor::CudaComputeCapability::AMPERE);
      default:
        return false;
    }
  };

  // TODO(b/266862493): Support more output types.
  if (!supported_output_type(dot.shape().element_type())) {
    return "Unsupported output data type.";
  }

  if (!IsSupportedDataType(dot.operand(0)->shape().element_type(),
                           gpu_version) ||
      !IsSupportedDataType(dot.operand(1)->shape().element_type(),
                           gpu_version)) {
    return "Unsupported input data type.";
  }

  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();

  // TODO(b/269580541): support multiple batch dimensions.
  if (dim_numbers.lhs_batch_dimensions().size() > 1) {
    return "Multiple batch dimensions.";
  }

  // Cases where lhs or rhs have no non-contracting dims are not handled.
  if (dim_numbers.lhs_batch_dimensions().size() +
              dim_numbers.lhs_contracting_dimensions().size() ==
          dot.operand(0)->shape().rank() ||
      dim_numbers.rhs_batch_dimensions().size() +
              dim_numbers.rhs_contracting_dimensions().size() ==
          dot.operand(1)->shape().rank()) {
    return "No non-contracting dimensions.";
  }

  return FusionDecision{};
}

bool ShouldTritonHandleGEMM(HloInstruction& dot, const GpuVersion gpu_version) {
  std::vector<HloInstruction*> fusion_inputs;
  HloComputation::Builder builder("disposable");
  return FuseDot(dot, gpu_version, builder, fusion_inputs,
                 /*fusion_output_ptr=*/nullptr)
      ->CanFuse();
}

StatusOr<bool> GemmRewriterTriton::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result,
                        RunOnComputation(computation, gpu_version_));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
