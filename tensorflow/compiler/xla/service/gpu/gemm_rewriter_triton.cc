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
#include <stack>
#include <tuple>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

int FirstBatchDimensionIndex(const DotDimensionNumbers& dimension_numbers,
                             const int operand_number) {
  if (operand_number == 0) {
    return dimension_numbers.lhs_batch_dimensions_size()
               ? dimension_numbers.lhs_batch_dimensions(0)
               : -1;
  }
  return dimension_numbers.rhs_batch_dimensions_size()
             ? dimension_numbers.rhs_batch_dimensions(0)
             : -1;
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

Status DimensionOrder::HandleBitcast(const HloInstruction* hlo) {
  const Shape& operand_shape = hlo->operand(0)->shape();
  DimOrderVector operand_dim_order;
  operand_dim_order.reserve(operand_shape.rank());
  // Subdimension index tracking dimension splits.
  int subdim_index = 0;
  // Iterate in parallel over output and operand dimensions
  // in minor_to_major order. Find groups of dimensions of equal size
  // and project the output dimension order onto the operand.
  auto operand_dim_iter = operand_shape.layout().minor_to_major().cbegin();
  for (int64_t out_dim_index = 0; out_dim_index < hlo->shape().rank();
       ++out_dim_index) {
    int64_t out_dim_size = hlo->shape().dimensions_minor(out_dim_index);
    if (operand_dim_iter == operand_shape.layout().minor_to_major().cend()) {
      // Out of dimensions of the operand -> output should only have
      // degenerate dimensions from here.
      if (out_dim_size == 1) {
        continue;
      }
      // Otherwise this is an arbitrary transformation like
      // [2, 3] -> [3, 2] which is not supported yet
      return Unimplemented("general bitcast");
    }
    int64_t operand_dim_size = operand_shape.dimensions(*operand_dim_iter);
    VLOG(9) << hlo->shape().layout().minor_to_major(out_dim_index) << " "
            << *operand_dim_iter;
    VLOG(9) << out_dim_size << " " << operand_dim_size;
    subdim_index = 0;
    if (out_dim_size == operand_dim_size) {
      // 1:1 matching dimensions.
      operand_dim_order.push_back(dim_order_[out_dim_index]);
    } else if (out_dim_size < operand_dim_size) {
      // Multiple output dimensions <- one operand dimension:
      //  just keep their order.
      do {
        operand_dim_order.push_back(dim_order_[out_dim_index]);
        ++out_dim_index;
        if (out_dim_index == hlo->shape().rank()) {
          return Unimplemented("general bitcast");
        }
        out_dim_size *= hlo->shape().dimensions_minor(out_dim_index);
      } while (out_dim_size != operand_dim_size);
      operand_dim_order.push_back(dim_order_[out_dim_index]);
    } else {
      // One output dimension <- multiple operand dimensions:
      //  create new sub-dimensions.
      do {
        if (dim_order_[out_dim_index].subdim_number != 0) {
          return Unimplemented("split of subdimension");
        }
        operand_dim_order.push_back(
            {dim_order_[out_dim_index].target_dim_number, subdim_index,
             operand_shape.dimensions(*operand_dim_iter)});
        ++subdim_index;
        ++operand_dim_iter;
        if (operand_dim_iter ==
            operand_shape.layout().minor_to_major().cend()) {
          return Unimplemented("general bitcast");
        }
        operand_dim_size *= operand_shape.dimensions(*operand_dim_iter);
      } while (out_dim_size != operand_dim_size);
      operand_dim_order.push_back(
          {dim_order_[out_dim_index].target_dim_number, subdim_index,
           operand_shape.dimensions(*operand_dim_iter)});
    }
    ++operand_dim_iter;
  }
  // Handle remaining major dimensions of the operand. Call all degenerate
  // ones subdimensions of the most-major non-degenerate one. Otherwise
  // give up.
  while (operand_dim_iter != operand_shape.layout().minor_to_major().cend()) {
    ++subdim_index;
    if (operand_shape.dimensions(*operand_dim_iter) != 1) {
      return Unimplemented("general bitcast");
    }
    operand_dim_order.push_back(
        {dim_order_[hlo->shape().rank() - 1].target_dim_number, subdim_index,
         1});
    ++operand_dim_iter;
  }
  dim_order_ = operand_dim_order;
  return OkStatus();
}

Status DimensionOrder::HandleCopyOrTranspose(const HloInstruction* hlo) {
  const Layout& operand_layout = hlo->operand(0)->shape().layout();
  // Out physical -> out logical.
  DimOrderVector out_logical;
  out_logical.resize(dim_order_.size());
  for (int i = 0; i < dim_order_.size(); ++i) {
    out_logical[hlo->shape().layout().minor_to_major(i)] = dim_order_[i];
  }
  // Out logical -> operand logical.
  DimOrderVector operand_logical;
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
  // Operand logical -> operand physical.
  for (int i = 0; i < dim_order_.size(); ++i) {
    dim_order_[i] = operand_logical[operand_layout.minor_to_major(i)];
  }
  return OkStatus();
}

// Tells if the dimension order is supported by the triton GEMM emitter.
// Only the dimension indicated by SplittableDimensionIndex() can be split
// physically once by other dimensions. Other ones can be only split logically.
// All subdimensions within a dimension have to be ordered.
Status RequireTritonGemmSupportedDimOrder(const DimensionOrder& order) {
  std::array<int, 3> subdim_counters = {-1, -1, -1};
  std::array<int, 3> split_counters = {0, 0, 0};
  int previous_dim_number = -1;
  for (int i = 0; i < order.GetDimOrderVector().size(); i++) {
    const auto [dim_number, subdim_number, size] = order.GetDimOrderVector()[i];
    VLOG(8) << dim_number << " " << subdim_number << " " << size;
    if (dim_number == order.BatchDimensionIndex() &&
        i != order.GetDimOrderVector().size() - 1) {
      return Unimplemented("non-major-most batch dimension");
    }
    if (subdim_counters[dim_number] != subdim_number - 1) {
      return Unimplemented("transpose within a dimension");
    }
    ++subdim_counters[dim_number];
    if (previous_dim_number >= 0 && previous_dim_number != dim_number) {
      ++split_counters[previous_dim_number];
      if (dim_number == order.SplittableDimensionIndex()) {
        if (split_counters[dim_number] > 1) {
          return Unimplemented("2nd split of a splittable dimension");
        }
      } else if (split_counters[dim_number] > 0) {
        return Unimplemented("split of a non-splittable dimension");
      }
    }
    previous_dim_number = dim_number;
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
    HloComputation::Builder builder(dot->name());
    // Original instruction -> fused one.
    absl::flat_hash_map<const HloInstruction*, HloInstruction*>
        old_to_new_mapping;
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
        if (!old_to_new_mapping.contains(operand)) {
          DimensionOrder operand_dim_order = [&] {
            // Direct dot inputs are described by default dimension orders.
            if (operand == dot->operand(0)) {
              if (dot->dot_dimension_numbers().lhs_batch_dimensions_size()) {
                return DimensionOrder(
                    operand,
                    dot->dot_dimension_numbers().lhs_batch_dimensions_size(),
                    -1);
              }
              // Non-contracting dimension can be split if batch is absent.
              return DimensionOrder(
                  operand, -1,
                  NoncontractingDimensionIndex(
                      dot->dot_dimension_numbers().lhs_contracting_dimensions(
                          0),
                      -1));
            } else if (operand == dot->operand(1)) {
              return DimensionOrder(
                  operand,
                  FirstBatchDimensionIndex(dot->dot_dimension_numbers(), 1),
                  -1);
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
    HloComputation* custom_call_computation =
        dot->GetModule()->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                            /*is_entry=*/false);
    HloInstruction* dot_custom_call =
        dot->parent()->AddInstruction(HloInstruction::CreateCustomCall(
            dot->shape(), call_operands, custom_call_computation,
            kTritonCallTarget));
    if (dot->IsRoot()) {
      dot->parent()->set_root_instruction(dot_custom_call);
      TF_RETURN_IF_ERROR(
          dot->parent()->RemoveInstructionAndUnusedOperands(dot));
    } else {
      TF_RETURN_IF_ERROR(
          dot->parent()->ReplaceInstruction(dot, dot_custom_call));
    }
    VLOG(5) << dot_custom_call->ToString();
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

}  // anonymous namespace

int NoncontractingDimensionIndex(const int contracting_dimension_index,
                                 const int batch_dimension_index) {
  // Sum of all indices is 0 + 1 = 1 if only two dimensions are present.
  int ret = 1 - contracting_dimension_index;
  if (batch_dimension_index >= 0) {
    // Sum of all indices is 0 + 1 + 2 = 3 if three dimensions are present.
    ret += (2 - batch_dimension_index);
  }
  return ret;
}

DotFusionAnalysis::DotFusionAnalysis(const HloInstruction* root) {
  VLOG(5) << root->parent()->ToString();

  while (root->opcode() != HloOpcode::kDot) {
    CHECK_EQ(root->operand_count(), 1);
    root = root->operand(0);
  }

  for (int64_t operand_number = 0; operand_number < root->operand_count();
       ++operand_number) {
    const HloInstruction* parameter = root->operand(operand_number);
    DimensionOrder dim_order(parameter, -1, -1);
    while (parameter->opcode() != HloOpcode::kParameter) {
      CHECK_EQ(parameter->operand_count(), 1);
      dim_order.HandleInstruction(parameter).ok();
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
      VLOG(6) << dim.target_dim_number << " " << dim.subdim_number << " "
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
  if (dot.opcode() != HloOpcode::kDot) {
    return false;
  }
  const DotDimensionNumbers& dimension_numbers = dot.dot_dimension_numbers();

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
  if (dimension_numbers.lhs_batch_dimensions().size() > 1) {
    return false;
  }

  if (dot.GetModule()->config().debug_options().xla_gpu_triton_gemm_any()) {
    return true;
  }

  // Traverse HLO graph part checking that it both can be fused
  // and is worth fusing.
  auto has_triton_fusible_inputs = [&](const HloInstruction* input,
                                       int64_t batch_dimension_index,
                                       int64_t contracting_dimension_index) {
    DimensionOrder dim_order(input, batch_dimension_index,
                             contracting_dimension_index);
    while (TryToFuse(input, dim_order, cuda_compute_capability).ok()) {
      if (input->opcode() == HloOpcode::kConvert ||
          input->opcode() == HloOpcode::kTranspose) {
        return true;
      }
      input = input->operand(0);
    }
    return false;
  };

  return has_triton_fusible_inputs(
             dot.operand(0), FirstBatchDimensionIndex(dimension_numbers, 0),
             dimension_numbers.lhs_contracting_dimensions(0)) ||
         has_triton_fusible_inputs(
             dot.operand(1), FirstBatchDimensionIndex(dimension_numbers, 1),
             dimension_numbers.rhs_contracting_dimensions(0));

  // TODO(b/266857789): either check that no output fusion (axpy, relu etc)
  // is expected or actually support it.
}

StatusOr<bool> GemmRewriterTriton::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    if (computation->IsCustomCallComputation() &&
        IsTritonCustomCall(*computation->CustomCallInstruction())) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(
        bool result, RunOnComputation(computation, cuda_compute_capability_));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
