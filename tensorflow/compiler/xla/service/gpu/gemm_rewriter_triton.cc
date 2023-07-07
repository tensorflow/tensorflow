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
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/autotuning.pb.h"
#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_query.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
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
bool IsSupportedDataType(PrimitiveType t, GpuVersion gpu_version) {
  auto cuda_compute_capability =
      std::get<se::CudaComputeCapability>(gpu_version);
  switch (t) {
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

Status RequireTritonFusibleConvert(const HloInstruction* input,
                                   GpuVersion gpu_version) {
  if (!IsSupportedDataType(input->operand(0)->shape().element_type(),
                           gpu_version)) {
    return Unimplemented("unsupported data type");
  }
  // TODO(b/266862494): Can pick up almost any
  // convert, but if it's reducing the data volume it should rather be fused
  // to the output of the producer kernel. However not all operations support
  // output fusion - then it should be fused here anyway!
  if (ShapeUtil::ByteSizeOf(input->operand(0)->shape()) >
      ShapeUtil::ByteSizeOf(input->shape())) {
    return FailedPrecondition("narrowing conversion");
  }
  return OkStatus();
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
  };
  // Sequence describing all dimensions of HLO's output shape
  // in layout minor-to-major (physical) order.
  using DimOrderVector = std::vector<DimDescription>;

  DimensionOrder(const DimensionOrder&) = default;
  // Dimension order constructed for the output shape of `hlo`.
  // `hlo` is currently supposed to be an operand of dot();
  // dimension indices describing the operand
  // are stored along with the dimension order for later analysis.
  explicit DimensionOrder(const HloInstruction* hlo,
                          const int64_t splittable_dimension_index = -1)
      : splittable_dimension_index_(splittable_dimension_index) {
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
  static DimensionOrder FromDotOutput(const HloInstruction& dot);

  // Transforms the DimensionOrder so that from a description of the output
  // of `hlo` it becomes a description of the input of `hlo`.
  Status HandleInstruction(const HloInstruction* hlo) {
    VLOG(7) << hlo->ToString();
    if (hlo->opcode() == HloOpcode::kParameter) {
      return OkStatus();
    } else if (hlo->opcode() == HloOpcode::kTranspose ||
               hlo->opcode() == HloOpcode::kCopy) {
      return HandleCopyOrTranspose(hlo);
    } else if (hlo->operand_count() > 0 &&
               IsTritonSupportedElementwise(
                   hlo->opcode(), hlo->operand(0)->shape().element_type())) {
      return OkStatus();
    } else if (hlo->opcode() == HloOpcode::kBitcast) {
      return HandleBitcast(hlo);
    } else if (hlo->opcode() == HloOpcode::kReshape) {
      if (!ShapeUtil::ReshapeIsBitcast(hlo->operand(0)->shape(),
                                       hlo->shape())) {
        return Unimplemented("Non-bitcast reshape.");
      }
      return HandleBitcast(hlo);
    } else if (hlo_query::IsScalarConstant(hlo) ||
               hlo_query::IsBroadcastOfScalarConstant(*hlo)) {
      // Dimension order collapses on a scalar, for simplicity leave it equal
      // to the output one for now.
      return OkStatus();
    } else {
      return Unimplemented("Instruction: %s", hlo->ToString());
    }
    return OkStatus();
  }

  // Get the raw data of the dimension order.
  const DimOrderVector& GetDimOrderVector() const { return dim_order_; }

  // Index of dot dimension that can be split.
  // Currently typically LHS non-contracting one.
  int64_t SplittableDimensionIndex() const {
    return splittable_dimension_index_;
  }

 private:
  // See HandleInstruction() for the general description of Handle*().
  Status HandleBitcast(const HloInstruction* hlo);
  Status HandleCopyOrTranspose(const HloInstruction* hlo);

  DimOrderVector dim_order_;
  int64_t splittable_dimension_index_;
};

DotFusionAnalysis::TensorIterationSpec DimensionOrderToTensorIterationSpec(
    const DimensionOrder& order) {
  const DimensionOrder::DimOrderVector& dim_order_vector =
      order.GetDimOrderVector();
  DotFusionAnalysis::TensorIterationSpec tensor_spec;
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

    DotFusionAnalysis::DimIterationSpec& dim_spec =
        tensor_spec[dim.target_dim_number];
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
  for (DotFusionAnalysis::DimIterationSpec& dim_spec : tensor_spec) {
    if (dim_spec.empty()) {
      dim_spec.push_back({/*stride=*/0, /*count=*/1, /*subfragments=*/{1}});
    }
  }
  return tensor_spec;
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

DimensionOrder DimensionOrder::FromDotOutput(const HloInstruction& dot) {
  return DimensionOrder(&dot);
}

Status DimensionOrder::HandleBitcast(const HloInstruction* hlo) {
  const Shape& operand_shape = hlo->operand(0)->shape();
  DimOrderVector operand_dim_order;
  operand_dim_order.reserve(dim_order_.size());
  // Size of not yet assigned part of current operand dimension.
  int64_t operand_remaining_size = 1;
  // Iterate in parallel over output dimension order and operand dimensions
  // in minor_to_major order. Find groups of dimensions of equal size
  // and project the output dimension order onto the operand.
  auto operand_dim_iter = operand_shape.layout().minor_to_major().cbegin();
  for (auto out_dim = dim_order_.cbegin(); out_dim != dim_order_.cend();
       ++out_dim) {
    if (operand_remaining_size >= out_dim->size) {
      if (operand_remaining_size % out_dim->size) {
        return Unimplemented("Unsupported bitcast: %s", hlo->ToString());
      }
      // Output dimension fragment completely fits into the operand one:
      // just copy it as is.
      operand_dim_order.push_back(*out_dim);
      // Update the size of the remaining part of the operand that is
      // carried over to next output dimensions.
      operand_remaining_size /= out_dim->size;
    } else {
      // Output is larger than input. Assign further operand dimensions.
      // Size of the not yet assigned part of the output dimension.
      int64_t out_remaining_size = out_dim->size;
      // Subdimension index tracking dimension splits.
      int subdim_index = out_dim->subdim_number;
      if (operand_remaining_size > 1) {
        // If there is a remaining fragment of a previous operand dimension
        // assign it first.
        if (out_remaining_size % operand_remaining_size) {
          return Unimplemented("Unsupported bitcast: %s", hlo->ToString());
        }
        operand_dim_order.push_back(
            {out_dim->target_dim_number, subdim_index, operand_remaining_size});
        ++subdim_index;
        // Update the size of the fragment remaining to assign.
        out_remaining_size /= operand_remaining_size;
        operand_remaining_size = 1;
      }
      while (out_remaining_size > 1) {
        // Assign operand dimensions until the output remainder is covered.
        int64_t operand_dim_size = operand_shape.dimensions(*operand_dim_iter);
        int64_t new_fragment_size = operand_dim_size;
        if (operand_dim_size > out_remaining_size) {
          // If adding the next operand dimension exceeds output fragment size
          // assign the remainder of the output and carry over the remainder
          // of the operand.
          if (operand_dim_size % out_remaining_size) {
            return Unimplemented("Unsupported bitcast: %s", hlo->ToString());
          }
          operand_remaining_size = operand_dim_size / out_remaining_size;
          new_fragment_size = out_remaining_size;
        }
        operand_dim_order.push_back(
            {out_dim->target_dim_number, subdim_index, new_fragment_size});
        out_remaining_size /= new_fragment_size;
        ++operand_dim_iter;
        ++subdim_index;
      }
    }
  }
  CHECK_EQ(operand_remaining_size, 1);

  // Handle remaining major dimensions of the operand. Call all degenerate
  // ones subdimensions of the most-major non-degenerate one. Otherwise
  // give up.
  int subdim_index = operand_dim_order.back().subdim_number + 1;
  while (operand_dim_iter != operand_shape.layout().minor_to_major().cend()) {
    if (operand_shape.dimensions(*operand_dim_iter) != 1) {
      return Unimplemented("Unsupported bitcast: %s", hlo->ToString());
    }
    operand_dim_order.push_back(
        {operand_dim_order.back().target_dim_number, subdim_index, 1});
    ++subdim_index;
    ++operand_dim_iter;
  }

  dim_order_ = operand_dim_order;
  return OkStatus();
}

Status DimensionOrder::HandleCopyOrTranspose(const HloInstruction* hlo) {
  // Every HLO dimension can correspond to a group of subdimensions in
  // dim_order_. For the easier handling of permutations: group dim_order_ by
  // dimension, apply permutations, then finally remove the grouping.
  // Group subdimensions by iterating over them in the same order as over
  // dimensions and matching by total size.
  std::vector<DimOrderVector> out_physical;
  out_physical.reserve(hlo->shape().rank());
  auto dim_order_it = dim_order_.cbegin();
  for (int64_t dim_index : hlo->shape().layout().minor_to_major()) {
    const int64_t dim_size = hlo->shape().dimensions(dim_index);
    int64_t subdim_size_accumulator = 1;
    DimOrderVector subdim_group;
    do {
      subdim_size_accumulator *= dim_order_it->size;
      subdim_group.push_back(*dim_order_it);
      ++dim_order_it;
    } while (subdim_size_accumulator < dim_size);
    CHECK_EQ(subdim_size_accumulator, dim_size);
    out_physical.push_back(subdim_group);
  }
  // Out physical -> out logical.
  std::vector<DimOrderVector> out_logical;
  out_logical.resize(out_physical.size());
  for (int i = 0; i < out_physical.size(); ++i) {
    out_logical[hlo->shape().layout().minor_to_major(i)] = out_physical[i];
  }
  // Out logical -> operand logical.
  std::vector<DimOrderVector> operand_logical;
  if (hlo->opcode() == HloOpcode::kTranspose) {
    auto transpose = ::xla::Cast<HloTransposeInstruction>(hlo);
    operand_logical.resize(out_logical.size());
    for (int i = 0; i < out_logical.size(); ++i) {
      operand_logical[transpose->dimensions()[i]] = out_logical[i];
    }
  } else {
    // Copy preserves the logical shape, just permutes the layout.
    const Shape& operand_shape = hlo->operand(0)->shape();
    CHECK(ShapeUtil::SameDimensions(hlo->shape(), operand_shape));
    operand_logical = out_logical;
  }
  // Operand logical -> operand physical and ungroup subdimensions.
  const Layout& operand_layout = hlo->operand(0)->shape().layout();
  dim_order_.clear();
  for (int64_t dim_idx : operand_layout.minor_to_major()) {
    for (const DimDescription& subdim : operand_logical[dim_idx]) {
      dim_order_.push_back(subdim);
    }
  }
  return OkStatus();
}

// Tells if the dimension order is supported by the triton GEMM emitter.
// Only the dimension indicated by SplittableDimensionIndex() can be split
// physically once by other dimensions. Other ones can be only split logically.
// All subdimensions within a dimension have to be ordered.
Status RequireTritonGemmSupportedDimOrder(const DimensionOrder& order) {
  std::array<int, DotFusionAnalysis::kMaxDimsPerTensor> subdim_counters = {
      -1, -1, -1, -1};
  std::array<int, DotFusionAnalysis::kMaxDimsPerTensor> split_counters = {
      -1, -1, -1, -1};
  const DimensionOrder::DimOrderVector& dim_order_vector =
      order.GetDimOrderVector();
  for (int i = 0; i < dim_order_vector.size(); i++) {
    const auto [dim_number, subdim_number, size] = dim_order_vector[i];
    VLOG(8) << dim_number << "\t" << subdim_number << "\t" << size;
    if (subdim_counters[dim_number] != subdim_number - 1) {
      return Unimplemented("Transpose within a dimension.");
    }
    ++subdim_counters[dim_number];
    if (size == 1) {
      continue;
    }
    if (i == 0 || dim_order_vector[i - 1].target_dim_number != dim_number) {
      ++split_counters[dim_number];
      if (dim_number == order.SplittableDimensionIndex()) {
        if (split_counters[dim_number] > 1) {
          return Unimplemented("2nd split of a splittable dimension.");
        }
      } else if (split_counters[dim_number] > 0) {
        return Unimplemented("Split of a non-splittable dimension.");
      }
    }
  }
  return OkStatus();
}

// Transforms dim_order describing the output of `hlo` into a
// description of its input if it is supported by the triton GEMM emitter.
Status CanFuse(const HloInstruction* hlo, DimensionOrder& dim_order,
               const GpuVersion gpu_version) {
  if (hlo->opcode() == HloOpcode::kConvert) {
    return RequireTritonFusibleConvert(hlo, gpu_version);
  } else if (hlo->IsElementwise() && hlo->opcode() != HloOpcode::kCopy) {
    // Temporarily forbid fusing elementwise operations
    // other than copy and convert.
    return Unimplemented("Unsupported elementwise operation");
  }
  TF_RETURN_IF_ERROR(dim_order.HandleInstruction(hlo));
  return RequireTritonGemmSupportedDimOrder(dim_order);
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
    VLOG(5) << dot->ToString();
    if (!IsTritonHandledGEMM(*dot, gpu_version_)) {
      return OkStatus();
    }

    // TODO(b/266857789): also fuse convert(dot()) at output if present:
    // seen on s8xf32->bf16
    std::string suggested_name = absl::StrCat("triton_gemm_", dot->name());
    HloComputation::Builder builder(
        absl::StrCat(suggested_name, "_computation"));
    // Original instruction -> fused one.
    absl::flat_hash_map<const HloInstruction*, HloInstruction*>
        old_to_new_mapping;
    absl::flat_hash_set<const HloInstruction*> visited;
    std::vector<HloInstruction*> call_operands;
    // Traverse and fuse dot() inputs bottom-up starting from direct operands.
    // If an input is not fusible stop there and make it a parameter of the new
    // fusion, otherwise put it onto stack and check its own inputs first.
    std::stack<HloInstruction*> to_fuse;
    // Dimension orders describing inputs of corresponding instructions.
    absl::flat_hash_map<const HloInstruction*, DimensionOrder> dim_orders;
    to_fuse.push(dot);
    while (!to_fuse.empty()) {
      bool top_is_ready_to_fuse = true;
      HloInstruction* hlo = to_fuse.top();
      for (HloInstruction* operand : hlo->mutable_operands()) {
        if (visited.insert(operand).second) {
          DimensionOrder operand_dim_order = [&] {
            // Direct dot inputs are described by default dimension orders.
            if (operand == dot->operand(0)) {
              return DimensionOrder::FromDotOperand(*dot, 0);
            } else if (operand == dot->operand(1)) {
              return DimensionOrder::FromDotOperand(*dot, 1);
            }
            // Otherwise operand's output is described by its consumer's input.
            return DimensionOrder(dim_orders.at(hlo));
          }();
          // CanFuse() makes output -> input transformation of
          // operand_dim_order if succeeds.
          if (CanFuse(operand, operand_dim_order, gpu_version_).ok()) {
            VLOG(3) << "Fusing " << operand->ToString();
            to_fuse.push(operand);
            // Save the dimension order description of operand's input.
            dim_orders.insert({operand, operand_dim_order});
            top_is_ready_to_fuse = false;
          }
        }
      }
      if (top_is_ready_to_fuse) {
        if (hlo->opcode() == HloOpcode::kParameter ||
            hlo->opcode() == HloOpcode::kGetTupleElement) {
          old_to_new_mapping[hlo] =
              builder.AddInstruction(HloInstruction::CreateParameter(
                  call_operands.size(), hlo->shape(),
                  absl::StrCat("parameter_", call_operands.size())));
          call_operands.push_back(hlo);
        } else {
          std::vector<HloInstruction*> hlo_new_operands;
          for (HloInstruction* operand : hlo->operands()) {
            const auto iter = old_to_new_mapping.find(operand);
            if (iter != old_to_new_mapping.end()) {
              hlo_new_operands.push_back(iter->second);
            } else {
              hlo_new_operands.push_back(
                  builder.AddInstruction(HloInstruction::CreateParameter(
                      call_operands.size(), operand->shape(),
                      absl::StrCat("parameter_", call_operands.size()))));
              call_operands.push_back(operand);
            }
          }
          old_to_new_mapping[hlo] = builder.AddInstruction(
              hlo->CloneWithNewOperands(hlo->shape(), hlo_new_operands));
        }
        to_fuse.pop();
      }
    }
    HloComputation* computation =
        dot->GetModule()->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                            /*is_entry=*/false);
    HloInstruction* dot_fusion =
        dot->parent()->AddInstruction(HloInstruction::CreateFusion(
            computation->root_instruction()->shape(),
            HloInstruction::FusionKind::kCustom, call_operands, computation));
    dot_fusion->GetModule()->SetAndUniquifyInstrName(dot_fusion,
                                                     suggested_name);

    TF_ASSIGN_OR_RETURN(auto backend_config,
                        dot_fusion->backend_config<FusionBackendConfig>());
    backend_config.set_kind(std::string(kTritonGemmFusionKind));
    TF_RETURN_IF_ERROR(dot_fusion->set_backend_config(backend_config));

    if (dot->IsRoot()) {
      dot->parent()->set_root_instruction(dot_fusion);
      TF_RETURN_IF_ERROR(
          dot->parent()->RemoveInstructionAndUnusedOperands(dot));
      MarkAsChanged();
    } else {
      TF_RETURN_IF_ERROR(ReplaceInstruction(dot, dot_fusion));
    }
    VLOG(5) << computation->ToString();
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

StatusOr<HloInstruction*> MakeSplitKOperand(
    HloInstruction& dot, const DotFusionAnalysis& analysis,
    const AutotuneResult::TritonGemmKey& tiling,
    const int64_t contracting_dim_idx, const int operand_number) {
  const Shape& shape = dot.operand(operand_number)->shape();
  Shape new_shape(shape.element_type(), {}, {}, {});

  // TODO(b/274775195): implement split-K with padding.
  if (tiling.split_k() > shape.dimensions(contracting_dim_idx)) {
    return Cancelled("Too small total contracting dimension size.");
  }
  DotFusionAnalysis::Scope scope = (operand_number == 0)
                                       ? DotFusionAnalysis::Scope::LHS
                                       : DotFusionAnalysis::Scope::RHS;
  for (const HloInstruction* param : analysis.ScopeParameters(scope)) {
    // If an operand of dot does not read any parameters its K dimension
    // does not need analysis for fragmentation.
    const DotFusionAnalysis::DimIterationSpec* spec =
        analysis.IterSpec(scope, param, contracting_dim_idx);
    // Split contracting dimension is not implemented yet.
    CHECK_EQ(spec->size(), 1);
    auto fragment = spec->at(0).subfragments.crbegin();
    int64_t size_to_split = tiling.split_k();
    while (size_to_split > *fragment) {
      if (size_to_split % *fragment) {
        return Cancelled("Contracting dimension is too fragmented.");
      }
      size_to_split /= *fragment;
      ++fragment;
    }
    if (*fragment % size_to_split) {
      return Cancelled("Contracting dimension is too fragmented.");
    }
    if (tiling.split_k() > ceil(1.0 * spec->at(0).count / tiling.block_k())) {
      return Cancelled(
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
  TF_ASSIGN_OR_RETURN(
      HloInstruction * lhs,
      MakeSplitKOperand(*dot, analysis, tiling, lhs_contracting_idx, 0));
  CopyIncrementingAboveThreshold(
      old_dim_numbers.lhs_contracting_dimensions(),
      *new_dim_numbers.mutable_lhs_contracting_dimensions(),
      lhs_contracting_idx);
  new_dim_numbers.mutable_lhs_batch_dimensions()->Add(lhs_contracting_idx);
  CopyIncrementingAboveThreshold(
      old_dim_numbers.lhs_batch_dimensions(),
      *new_dim_numbers.mutable_lhs_batch_dimensions(), lhs_contracting_idx);

  const int64_t rhs_contracting_idx = ContractingDimensionIndex(*dot, 1);
  TF_ASSIGN_OR_RETURN(
      HloInstruction * rhs,
      MakeSplitKOperand(*dot, analysis, tiling, rhs_contracting_idx, 1));
  CopyIncrementingAboveThreshold(
      old_dim_numbers.rhs_contracting_dimensions(),
      *new_dim_numbers.mutable_rhs_contracting_dimensions(),
      rhs_contracting_idx);
  new_dim_numbers.mutable_rhs_batch_dimensions()->Add(rhs_contracting_idx);
  CopyIncrementingAboveThreshold(
      old_dim_numbers.rhs_batch_dimensions(),
      *new_dim_numbers.mutable_rhs_batch_dimensions(), rhs_contracting_idx);

  HloInstruction* new_dot =
      MakeDotHlo(lhs, rhs, new_dim_numbers, dot->precision_config(),
                 dot->shape().element_type())
          .value();
  // `new_dot` will have default output layout even if `dot` had a custom one.
  // We will set the original output layout on the reduce operation.

  dot->SetupDerivedInstruction(new_dot);
  TF_RETURN_IF_ERROR(dot->ReplaceAllUsesWithDifferentShape(new_dot));
  TF_RETURN_IF_ERROR(dot->parent()->RemoveInstruction(dot));
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
    const HloInstruction* dot_operand = dot->operand(operand_number);
    absl::flat_hash_set<const HloInstruction*> visited;
    std::queue<const HloInstruction*> to_process;
    // Dimension orders describing inputs of corresponding instructions.
    absl::flat_hash_map<const HloInstruction*, DimensionOrder> dim_orders;
    DimensionOrder dot_operand_dim_order =
        DimensionOrder::FromDotOperand(*dot, operand_number, split_k);
    TF_CHECK_OK(dot_operand_dim_order.HandleInstruction(dot_operand));
    TF_CHECK_OK(RequireTritonGemmSupportedDimOrder(dot_operand_dim_order))
        << dot_computation->ToString();
    dim_orders.insert({dot_operand, dot_operand_dim_order});
    visited.insert(dot_operand);
    to_process.push(dot_operand);
    while (!to_process.empty()) {
      const HloInstruction* hlo = to_process.front();
      to_process.pop();
      if (hlo->opcode() == HloOpcode::kParameter) {
        CHECK(parameters_[scope].insert(hlo).second);
        VLOG(5) << hlo->ToString();
      }
      for (const HloInstruction* hlo_operand : hlo->operands()) {
        if (!visited.insert(hlo_operand).second) {
          continue;
        }
        // Operand's output is described by its consumer's input.
        auto [it, inserted] = dim_orders.insert(
            {hlo_operand, DimensionOrder(dim_orders.at(hlo))});
        CHECK(inserted);
        DimensionOrder& hlo_operand_dim_order = it->second;
        TF_CHECK_OK(hlo_operand_dim_order.HandleInstruction(hlo_operand));
        TF_CHECK_OK(RequireTritonGemmSupportedDimOrder(hlo_operand_dim_order))
            << " " << dot_computation->ToString();
        to_process.push(hlo_operand);
      }
    }

    for (const HloInstruction* parameter : parameters_[scope]) {
      iter_specs_[scope][parameter] =
          DimensionOrderToTensorIterationSpec(dim_orders.at(parameter));
    }
  }

  DimensionOrder dim_order = DimensionOrder::FromDotOutput(*dot);
  CHECK(iter_specs_[Scope::OUTPUT]
            .insert({dot, DimensionOrderToTensorIterationSpec(dim_order)})
            .second);
}

const DotFusionAnalysis::DimIterationSpec* DotFusionAnalysis::IterSpec(
    const DotFusionAnalysis::Scope scope, const HloInstruction* hlo,
    const int dimension) const {
  auto ret = iter_specs_.at(scope).find(hlo);
  if (ret != iter_specs_.at(scope).end()) {
    return &ret->second.at(dimension);
  }
  return nullptr;
}

bool IsTritonHandledGEMM(const HloInstruction& dot,
                         const GpuVersion gpu_version) {
  if (dot.opcode() != HloOpcode::kDot ||
      absl::c_any_of(dot.precision_config().operand_precision(),
                     [](int x) { return x != PrecisionConfig::DEFAULT; })) {
    return false;
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
    return false;
  }

  if (!IsSupportedDataType(dot.operand(0)->shape().element_type(),
                           gpu_version) ||
      !IsSupportedDataType(dot.operand(1)->shape().element_type(),
                           gpu_version)) {
    return false;
  }

  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();

  // TODO(b/269580541): support multiple batch dimensions.
  if (dim_numbers.lhs_batch_dimensions().size() > 1) {
    return false;
  }

  // Cases where lhs or rhs have no non-contracting dims are not handled.
  if (dim_numbers.lhs_batch_dimensions().size() +
              dim_numbers.lhs_contracting_dimensions().size() ==
          dot.operand(0)->shape().rank() ||
      dim_numbers.rhs_batch_dimensions().size() +
              dim_numbers.rhs_contracting_dimensions().size() ==
          dot.operand(1)->shape().rank()) {
    return false;
  }

  if (dot.GetModule()->config().debug_options().xla_gpu_triton_gemm_any()) {
    return true;
  }

  // Traverse HLO graph part checking that it both can be fused
  // and is worth fusing.
  auto has_triton_fusible_inputs = [&gpu_version](const HloInstruction& dot,
                                                  const int operand_number) {
    DimensionOrder dim_order =
        DimensionOrder::FromDotOperand(dot, operand_number);
    std::queue<const HloInstruction*> queue;
    queue.push(dot.operand(operand_number));
    while (!queue.empty()) {
      const HloInstruction* current = queue.front();
      queue.pop();
      if (!CanFuse(current, dim_order, gpu_version).ok()) {
        continue;
      }
      // Stop as soon as a profitable operation is fused.
      if (current->opcode() == HloOpcode::kConvert ||
          current->opcode() == HloOpcode::kTranspose) {
        return true;
      }
      for (const HloInstruction* operand : current->operands()) {
        queue.push(operand);
      }
    }
    return false;
  };

  return has_triton_fusible_inputs(dot, 0) || has_triton_fusible_inputs(dot, 1);

  // TODO(b/266857789): either check that no output fusion (axpy, relu etc)
  // is expected or actually support it.
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
