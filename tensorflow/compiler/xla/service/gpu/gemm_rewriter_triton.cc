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
#include <cstdint>
#include <stack>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
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

// Index of first batch dimension of dot instruction operand; -1 if none exist.
int64_t FirstBatchDimensionForOperand(const HloInstruction& dot,
                                      const int operand_number) {
  tsl::protobuf::RepeatedField<int64_t> dimensions =
      BatchDimensionsForOperand(dot, operand_number);
  return dimensions.empty() ? -1 : dimensions[0];
}

// Index of first contracting dimension of dot instruction operand.
int64_t FirstContractingDimensionIndex(const HloInstruction& dot,
                                       const int operand_number) {
  const DotDimensionNumbers& dimension_numbers = dot.dot_dimension_numbers();
  if (operand_number == 0) {
    return dimension_numbers.lhs_contracting_dimensions(0);
  }
  return dimension_numbers.rhs_contracting_dimensions(0);
}

// Data types that are tested to work in the triton GEMM emitter.
bool IsTritonSupportedInputType(
    PrimitiveType t, se::CudaComputeCapability cuda_compute_capability) {
  switch (t) {
    case PRED:
    case S8:
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

Status RequireTritonFusibleConvert(
    const HloInstruction* input,
    se::CudaComputeCapability cuda_compute_capability) {
  if (!IsTritonSupportedInputType(input->operand(0)->shape().element_type(),
                                  cuda_compute_capability)) {
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
                          const int64_t batch_dimension_index,
                          const int64_t splittable_dimension_index)
      : batch_dimension_index_(batch_dimension_index),
        splittable_dimension_index_(splittable_dimension_index) {
    dim_order_.reserve(hlo->shape().rank());
    for (const int64_t i : hlo->shape().layout().minor_to_major()) {
      dim_order_.push_back({i, 0, hlo->shape().dimensions(i)});
    }
  }

  // Create dimension order describing a dot operand according to
  // the currently supported configurations.
  static DimensionOrder FromDotOperand(const HloInstruction& dot,
                                       int operand_number, int64_t split_k = 1);

  // Transforms the DimensionOrder so that from a description of the output
  // of `hlo` it becomes a description of the input of `hlo`.
  Status HandleInstruction(const HloInstruction* hlo) {
    VLOG(7) << hlo->ToString();
    if (hlo->opcode() == HloOpcode::kBitcast) {
      return HandleBitcast(hlo);
    } else if (hlo->opcode() == HloOpcode::kReshape) {
      if (!ShapeUtil::ReshapeIsBitcast(hlo->operand(0)->shape(),
                                       hlo->shape())) {
        return Unimplemented("non-bitcast reshape");
      }
      return HandleBitcast(hlo);
    } else if (hlo->opcode() == HloOpcode::kTranspose ||
               hlo->opcode() == HloOpcode::kCopy) {
      return HandleCopyOrTranspose(hlo);
    } else if (hlo->opcode() == HloOpcode::kConvert) {
      return OkStatus();
    } else {
      return Unimplemented("other instruction type");
    }
    return OkStatus();
  }

  const DimOrderVector& GetDimOrderVector() const { return dim_order_; }

  int64_t BatchDimensionIndex() const { return batch_dimension_index_; }

  int64_t SplittableDimensionIndex() const {
    return splittable_dimension_index_;
  }

 private:
  // See HandleInstruction() for the general description of Handle*().
  Status HandleBitcast(const HloInstruction* hlo);
  Status HandleCopyOrTranspose(const HloInstruction* hlo);

  DimOrderVector dim_order_;
  int64_t batch_dimension_index_;
  int64_t splittable_dimension_index_;
};

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
        operand, /*batch_dimension_index=*/-1,
        GetNonContractingDims(operand->shape(), /*batch_dims=*/{},
                              {FirstContractingDimensionIndex(dot, 0)})
            .value()[0]);
  }
  return DimensionOrder(operand,
                        FirstBatchDimensionForOperand(dot, operand_number),
                        /*splittable_dimension_index=*/-1);
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
        return Unimplemented("Unsupported bitcast.");
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
          return Unimplemented("Unsupported bitcast.");
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
            return Unimplemented("Unsupported bitcast.");
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
      return Unimplemented("Unsupported bitcast.");
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
  // At most: contracting, non-contracting, split-K, another batch.
  std::array<int, 4> subdim_counters = {-1, -1, -1, -1};
  std::array<int, 4> split_counters = {-1, -1, -1, -1};
  const DimensionOrder::DimOrderVector& dim_order_vector =
      order.GetDimOrderVector();
  for (int i = 0; i < dim_order_vector.size(); i++) {
    const auto [dim_number, subdim_number, size] = dim_order_vector[i];
    VLOG(8) << dim_number << "\t" << subdim_number << "\t" << size;
    if (subdim_counters[dim_number] != subdim_number - 1) {
      return Unimplemented("transpose within a dimension");
    }
    ++subdim_counters[dim_number];
    if (size == 1) {
      continue;
    }
    if (i == 0 || dim_order_vector[i - 1].target_dim_number != dim_number) {
      ++split_counters[dim_number];
      if (dim_number == order.SplittableDimensionIndex()) {
        if (split_counters[dim_number] > 1) {
          return Unimplemented("2nd split of a splittable dimension");
        }
      } else if (split_counters[dim_number] > 0) {
        return Unimplemented("split of a non-splittable dimension");
      }
    }
  }
  return OkStatus();
}

// Tries to transform dim_order describing the output of `hlo` into a
// description of its input if it is supported by the triton GEMM emitter.
Status TryToFuse(const HloInstruction* hlo, DimensionOrder& dim_order,
                 const se::CudaComputeCapability cuda_compute_capability) {
  if (hlo->opcode() == HloOpcode::kConvert) {
    return RequireTritonFusibleConvert(hlo, cuda_compute_capability);
  }
  TF_RETURN_IF_ERROR(dim_order.HandleInstruction(hlo));
  return RequireTritonGemmSupportedDimOrder(dim_order);
}

// Extracts into fused computations parts of HLO graph including dot()
// operations that can target the triton GEMM emitter.
class GemmRewriterTritonVisitor : public DfsHloRewriteVisitor {
 public:
  explicit GemmRewriterTritonVisitor(const se::CudaComputeCapability cc)
      : cuda_compute_capability_(cc) {}
  // Checks that a dot() should be targeting the triton GEMM emitter;
  // if so - fuses all its compatible inputs and outputs as a new computation
  // and replaces the original dot() with a call to the computation.
  Status HandleDot(HloInstruction* dot) override {
    VLOG(5) << dot->ToString();
    if (!IsTritonHandledGEMM(*dot, cuda_compute_capability_)) {
      return OkStatus();
    }

    // TODO(b/266857789): also fuse convert(dot()) at output if present:
    // seen on s8xf32->bf16
    std::string suggested_name = absl::StrCat("triton_gemm_", dot->name());
    HloComputation::Builder builder(suggested_name);
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
          // TryToFuse() makes output -> input transformation of
          // operand_dim_order if succeeds.
          if (TryToFuse(operand, operand_dim_order, cuda_compute_capability_)
                  .ok()) {
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
            dot->shape(), HloInstruction::FusionKind::kCustom, call_operands,
            computation));
    dot_fusion->GetModule()->SetAndUniquifyInstrName(dot_fusion,
                                                     suggested_name);
    dot_fusion->set_raw_backend_config_string(
        std::string(kTritonGemmBackendConfig));
    if (dot->IsRoot()) {
      dot->parent()->set_root_instruction(dot_fusion);
      TF_RETURN_IF_ERROR(
          dot->parent()->RemoveInstructionAndUnusedOperands(dot));
    } else {
      TF_RETURN_IF_ERROR(dot->parent()->ReplaceInstruction(dot, dot_fusion));
    }
    VLOG(5) << dot_fusion->ToString();
    MarkAsChanged();
    return OkStatus();
  }

 private:
  se::CudaComputeCapability cuda_compute_capability_;
};

StatusOr<bool> RunOnComputation(
    HloComputation* computation,
    se::CudaComputeCapability cuda_compute_capability) {
  GemmRewriterTritonVisitor visitor(cuda_compute_capability);
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
    HloInstruction& dot,
    const tensorflow::AutotuneResult::TritonGemmKey& tiling,
    const int64_t contracting_dim_idx, const int operand_number) {
  const Shape& shape = dot.operand(operand_number)->shape();
  Shape new_shape(shape.element_type(), {}, {}, {});

  // TODO(b/274775195): implement split-K with padding.
  if (shape.dimensions(contracting_dim_idx) % tiling.split_k()) {
    return Unimplemented("K dimension requires padding for split-K.");
  }

  if (tiling.split_k() >
      ceil(1.0 * shape.dimensions(contracting_dim_idx) / tiling.block_k())) {
    return Cancelled("Too small contracting dimension.");
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

  absl::Span<const int64_t> physical_dim_order =
      shape.layout().minor_to_major();
  const int contracting_dim_physical_idx =
      absl::c_find(physical_dim_order, contracting_dim_idx) -
      physical_dim_order.begin();
  Layout* batch_dot_layout = new_shape.mutable_layout();
  for (int64_t physical_dim_idx : physical_dim_order) {
    // When physical_dim_idx == contracting_dim_physical_idx add both
    // physical_dim_idx+1 and physical_dim_idx because it gets split into two.
    if (physical_dim_idx >= contracting_dim_physical_idx) {
      batch_dot_layout->add_minor_to_major(physical_dim_idx + 1);
    }
    if (physical_dim_idx <= contracting_dim_physical_idx) {
      batch_dot_layout->add_minor_to_major(physical_dim_idx);
    }
  }
  return MakeBitcastHlo(dot.mutable_operand(operand_number), new_shape);
}

}  // anonymous namespace

Status MakeDotComputationSplitKBatch(
    HloComputation* computation,
    const tensorflow::AutotuneResult::TritonGemmKey& tiling) {
  HloInstruction* dot = computation->root_instruction();
  CHECK_EQ(dot->opcode(), HloOpcode::kDot);
  const DotDimensionNumbers& old_dim_numbers = dot->dot_dimension_numbers();
  DotDimensionNumbers new_dim_numbers;

  const int64_t lhs_contracting_idx = FirstContractingDimensionIndex(*dot, 0);
  TF_ASSIGN_OR_RETURN(HloInstruction * lhs,
                      MakeSplitKOperand(*dot, tiling, lhs_contracting_idx, 0));
  CopyIncrementingAboveThreshold(
      old_dim_numbers.lhs_contracting_dimensions(),
      *new_dim_numbers.mutable_lhs_contracting_dimensions(),
      lhs_contracting_idx);
  CopyIncrementingAboveThreshold(
      old_dim_numbers.lhs_batch_dimensions(),
      *new_dim_numbers.mutable_lhs_batch_dimensions(), lhs_contracting_idx);
  new_dim_numbers.mutable_lhs_batch_dimensions()->Add(lhs_contracting_idx);

  const int64_t rhs_contracting_idx = FirstContractingDimensionIndex(*dot, 1);
  TF_ASSIGN_OR_RETURN(HloInstruction * rhs,
                      MakeSplitKOperand(*dot, tiling, rhs_contracting_idx, 1));
  CopyIncrementingAboveThreshold(
      old_dim_numbers.rhs_contracting_dimensions(),
      *new_dim_numbers.mutable_rhs_contracting_dimensions(),
      rhs_contracting_idx);
  CopyIncrementingAboveThreshold(
      old_dim_numbers.rhs_batch_dimensions(),
      *new_dim_numbers.mutable_rhs_batch_dimensions(), rhs_contracting_idx);
  new_dim_numbers.mutable_rhs_batch_dimensions()->Add(rhs_contracting_idx);

  HloInstruction* new_dot =
      MakeDotHlo(lhs, rhs, new_dim_numbers, dot->precision_config(),
                 dot->shape().element_type())
          .value();
  dot->SetupDerivedInstruction(new_dot);
  TF_RETURN_IF_ERROR(dot->ReplaceAllUsesWithDifferentShape(new_dot));
  TF_RETURN_IF_ERROR(dot->parent()->RemoveInstruction(dot));
  return OkStatus();
}

Status MakeDotSplitKBatch(
    HloInstruction* dot_fusion,
    const tensorflow::AutotuneResult::TritonGemmKey& tiling) {
  CHECK_EQ(dot_fusion->opcode(), HloOpcode::kFusion);
  TF_RETURN_IF_ERROR(MakeDotComputationSplitKBatch(
      dot_fusion->fused_instructions_computation(), tiling));
  const HloInstruction* dot = dot_fusion->fused_expression_root();

  *dot_fusion->mutable_shape() = dot->shape();
  HloInstruction* zero =
      dot_fusion->parent()->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(dot->shape().element_type())));
  const int new_batch_dim_idx =
      dot->dot_dimension_numbers().lhs_batch_dimensions().size() - 1;
  HloInstruction* reduce =
      MakeReduceHlo(dot_fusion, zero, {new_batch_dim_idx}, HloOpcode::kAdd)
          .value();

  if (dot_fusion->IsRoot()) {
    dot_fusion->parent()->set_root_instruction(reduce, true);
  } else {
    TF_RETURN_IF_ERROR(dot_fusion->ReplaceAllUsesWithDifferentShape(reduce));
  }
  return OkStatus();
}

DotFusionAnalysis::DotFusionAnalysis(const HloInstruction* root,
                                     const int64_t split_k) {
  VLOG(5) << root->parent()->ToString();

  while (root->opcode() != HloOpcode::kDot) {
    CHECK_EQ(root->operand_count(), 1);
    root = root->operand(0);
  }

  for (int64_t operand_number = 0; operand_number < root->operand_count();
       ++operand_number) {
    const HloInstruction* parameter = root->operand(operand_number);
    DimensionOrder dim_order =
        DimensionOrder::FromDotOperand(*root, operand_number, split_k);
    while (parameter->opcode() != HloOpcode::kParameter) {
      CHECK_EQ(parameter->operand_count(), 1);
      TF_CHECK_OK(dim_order.HandleInstruction(parameter));
      TF_CHECK_OK(RequireTritonGemmSupportedDimOrder(dim_order))
          << " " << root->parent()->ToString();
      parameter = parameter->operand(0);
    }
    operand_to_parameter_[operand_number] = parameter;
    VLOG(5) << parameter->ToString();

    const DimensionOrder::DimOrderVector& dim_order_vector =
        dim_order.GetDimOrderVector();
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

      IterationSpec& iter_spec =
          iter_specs_[operand_number][dim.target_dim_number];
      if (dim_order_index > 0 &&
          dim_order_vector[dim_order_index - 1].target_dim_number ==
              dim.target_dim_number) {
        if (iter_spec.empty()) {
          // Previous parts of this dimension were degenerate -
          // so create the dimension here.
          iter_spec.push_back({accumulated_stride, dim.size});
        } else {
          // Contiguous dimension, split only logically. Merge it back.
          iter_spec.back().count *= dim.size;
        }
      } else {
        iter_spec.push_back({accumulated_stride, dim.size});
      }

      accumulated_stride *= dim.size;
    }
  }
}

bool IsTritonHandledGEMM(
    const HloInstruction& dot,
    const se::CudaComputeCapability cuda_compute_capability) {
  if (dot.opcode() != HloOpcode::kDot ||
      absl::c_any_of(dot.precision_config().operand_precision(),
                     [](int x) { return x != PrecisionConfig::DEFAULT; })) {
    return false;
  }

  auto supported_output_type = [&](const PrimitiveType t) {
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

  if (!IsTritonSupportedInputType(dot.operand(0)->shape().element_type(),
                                  cuda_compute_capability) ||
      !IsTritonSupportedInputType(dot.operand(1)->shape().element_type(),
                                  cuda_compute_capability)) {
    return false;
  }

  // TODO(b/269580541): support multiple batch dimensions.
  if (dot.dot_dimension_numbers().lhs_batch_dimensions().size() > 1) {
    return false;
  }

  if (dot.GetModule()->config().debug_options().xla_gpu_triton_gemm_any()) {
    return true;
  }

  // Traverse HLO graph part checking that it both can be fused
  // and is worth fusing.
  auto has_triton_fusible_inputs = [&](const int operand_number) {
    const HloInstruction* input = dot.operand(operand_number);
    DimensionOrder dim_order =
        DimensionOrder::FromDotOperand(dot, operand_number);
    while (TryToFuse(input, dim_order, cuda_compute_capability).ok()) {
      if (input->opcode() == HloOpcode::kConvert ||
          input->opcode() == HloOpcode::kTranspose) {
        return true;
      }
      input = input->operand(0);
    }
    return false;
  };

  return has_triton_fusible_inputs(0) || has_triton_fusible_inputs(1);

  // TODO(b/266857789): either check that no output fusion (axpy, relu etc)
  // is expected or actually support it.
}

StatusOr<bool> GemmRewriterTriton::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(
        bool result, RunOnComputation(computation, cuda_compute_capability_));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
