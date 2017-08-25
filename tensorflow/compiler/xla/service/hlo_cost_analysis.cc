/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"

#include <cmath>

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

constexpr char HloCostAnalysis::kFlopsKey[];
constexpr char HloCostAnalysis::kTranscendentalsKey[];
constexpr char HloCostAnalysis::kBytesAccessedKey[];
constexpr char HloCostAnalysis::kSecondsKey[];

HloCostAnalysis::HloCostAnalysis(const ShapeSizeFunction& shape_size)
    : HloCostAnalysis(shape_size, {}) {}

HloCostAnalysis::HloCostAnalysis(const ShapeSizeFunction& shape_size,
                                 const Properties& per_second_rates)
    : shape_size_(shape_size), per_second_rates_(per_second_rates) {}

Status HloCostAnalysis::Preprocess(HloInstruction* hlo) {
  // Set current instruction cost values to reasonable default values. Each
  // handler can overwrite these values. In Postprocess, these values are
  // accumulated and written to the per-instruction maps.
  current_properties_.clear();
  current_should_compute_bottleneck_time_ = true;

  // The default number of bytes accessed for an instruction is the sum of the
  // sizes of the inputs and outputs. The default ShapeUtil::ByteSizeOf does not
  // handle opaque types.
  float bytes_accessed = shape_size_(hlo->shape());
  for (const HloInstruction* operand : hlo->operands()) {
    bytes_accessed += shape_size_(operand->shape());
  }
  current_properties_[kBytesAccessedKey] = bytes_accessed;

  return Status::OK();
}

Status HloCostAnalysis::Postprocess(HloInstruction* hlo) {
  if (current_should_compute_bottleneck_time_) {
    // Compute the time as the time of the bottleneck, i.e. the slowest property
    // given the per-second rate of each property.
    float max_seconds = 0.0f;
    for (const auto& property : current_properties_) {
      if (property.first != kSecondsKey) {
        max_seconds = std::max(
            max_seconds,
            property.second /
                GetProperty(property.first, per_second_rates_, INFINITY));
      }
    }
    current_properties_[kSecondsKey] = max_seconds;
  }

  TF_RET_CHECK(hlo_properties_.emplace(hlo, current_properties_).second);
  for (const auto& property : current_properties_) {
    properties_sum_[property.first] += property.second;
  }

  return Status::OK();
}

Status HloCostAnalysis::HandleElementwiseOp(HloInstruction* hlo_instruction) {
  const auto& shape = hlo_instruction->shape();
  // For element-wise operations, the number of computations is the same as the
  // number of elements in the output shape.
  auto computation_count = ShapeUtil::ElementsIn(shape);
  auto opcode = hlo_instruction->opcode();
  // We treat transcendental operations separately since one transcendental
  // operation can correspond to several floating point ops.
  if (opcode == HloOpcode::kExp || opcode == HloOpcode::kPower ||
      opcode == HloOpcode::kTanh || opcode == HloOpcode::kSin ||
      opcode == HloOpcode::kCos) {
    current_properties_[kTranscendentalsKey] = computation_count;
  } else {
    // Note: transcendental operations are considered a separate category from
    // FLOPs.
    current_properties_[kFlopsKey] = computation_count;
  }
  return Status::OK();
}

/*static*/ float HloCostAnalysis::GetProperty(const string& key,
                                              const Properties& properties,
                                              const float default_value) {
  auto key_value = properties.find(key);
  return key_value == properties.end() ? default_value : key_value->second;
}

/*static*/ float HloCostAnalysis::GetPropertyForHlo(
    const HloInstruction& hlo, const string& key,
    const HloToProperties& hlo_to_properties) {
  auto it = hlo_to_properties.find(&hlo);
  if (it == hlo_to_properties.end()) {
    return 0.0f;
  } else {
    return GetProperty(key, it->second);
  }
}

Status HloCostAnalysis::HandleElementwiseUnary(HloInstruction* hlo,
                                               HloOpcode opcode) {
  return HandleElementwiseOp(hlo);
}

Status HloCostAnalysis::HandleElementwiseBinary(HloInstruction* hlo,
                                                HloOpcode opcode) {
  return HandleElementwiseOp(hlo);
}

Status HloCostAnalysis::HandleCompare(HloInstruction* compare, HloOpcode opcode,
                                      HloInstruction* lhs,
                                      HloInstruction* rhs) {
  return HandleElementwiseOp(compare);
}

Status HloCostAnalysis::HandleClamp(HloInstruction* clamp,
                                    HloInstruction* min_instruction,
                                    HloInstruction* arg_instruction,
                                    HloInstruction* max_instruction) {
  return HandleElementwiseOp(clamp);
}

Status HloCostAnalysis::HandleReducePrecision(HloInstruction* hlo) {
  return HandleElementwiseOp(hlo);
}

Status HloCostAnalysis::HandleParameter(HloInstruction* parameter) {
  current_properties_[kBytesAccessedKey] = 0;
  return Status::OK();
}

Status HloCostAnalysis::HandleConstant(HloInstruction* constant,
                                       const Literal& literal) {
  current_properties_[kBytesAccessedKey] = 0;
  return Status::OK();
}

Status HloCostAnalysis::HandleGetTupleElement(HloInstruction* get_tuple_element,
                                              HloInstruction* operand) {
  // GetTupleElement forwards a pointer and does not touch each element in the
  // output.
  current_properties_[kBytesAccessedKey] = 0;
  return Status::OK();
}

Status HloCostAnalysis::HandleSelect(HloInstruction* select,
                                     HloInstruction* pred,
                                     HloInstruction* on_true,
                                     HloInstruction* on_false) {
  return Status::OK();
}

Status HloCostAnalysis::HandleReverse(HloInstruction* reverse,
                                      HloInstruction* operand_instruction) {
  return Status::OK();
}

Status HloCostAnalysis::HandleSlice(HloInstruction* slice,
                                    HloInstruction* operand_instruction) {
  return Status::OK();
}

Status HloCostAnalysis::HandleDynamicSlice(HloInstruction* dynamic_slice,
                                           HloInstruction* operand,
                                           HloInstruction* start_indices) {
  return Status::OK();
}

Status HloCostAnalysis::HandleDynamicUpdateSlice(
    HloInstruction* dynamic_update, HloInstruction* operand,
    HloInstruction* update, HloInstruction* start_indices) {
  return Status::OK();
}

Status HloCostAnalysis::HandleTuple(
    HloInstruction* tuple,
    tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  // The tuple instruction only gathers pointers from inputs (it doesn't iterate
  // through them). The memory touched is then only the size of the output
  // index table of the tuple.

  current_properties_[kBytesAccessedKey] = shape_size_(tuple->shape());
  return Status::OK();
}

Status HloCostAnalysis::HandleConcatenate(
    HloInstruction* concatenate,
    tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  return Status::OK();
}

Status HloCostAnalysis::HandleConvert(HloInstruction* convert) {
  return HandleElementwiseOp(convert);
}

Status HloCostAnalysis::HandleCopy(HloInstruction* copy) {
  return Status::OK();
}

Status HloCostAnalysis::HandleDot(HloInstruction* dot,
                                  HloInstruction* lhs_instruction,
                                  HloInstruction* rhs_instruction) {
  const Shape& lhs_shape = lhs_instruction->shape();
  const Shape& rhs_shape = rhs_instruction->shape();
  // Count of elements along the reduction dimension (last dimension for the
  // rhs).
  int64 reduction_width = lhs_shape.dimensions(ShapeUtil::Rank(lhs_shape) - 1);

  // First divide by reduction width before multiplying by rhs elements to avoid
  // overflow.
  int64 fma_count;
  if (reduction_width == 0) {
    fma_count = 0;
  } else {
    fma_count = (ShapeUtil::ElementsIn(lhs_shape) / reduction_width) *
                ShapeUtil::ElementsIn(rhs_shape);
  }

  // We count an FMA operation as 2 floating point operations.
  current_properties_[kFlopsKey] = kFmaFlops * fma_count;
  return Status::OK();
}

Status HloCostAnalysis::HandleInfeed(HloInstruction* infeed) {
  return Status::OK();
}

Status HloCostAnalysis::HandleOutfeed(HloInstruction* outfeed) {
  return Status::OK();
}

Status HloCostAnalysis::HandleMap(
    HloInstruction* map, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    HloComputation* function,
    tensorflow::gtl::ArraySlice<HloInstruction*> /*static_operands*/) {
  // Compute properties of the mapped function.
  TF_ASSIGN_OR_RETURN(const Properties sub_properties,
                      ProcessSubcomputation(function));

  // Compute the cost of all elements for this Map operation.
  const int64 element_count = ShapeUtil::ElementsIn(map->shape());
  for (const auto& property : sub_properties) {
    if (property.first != kBytesAccessedKey) {
      current_properties_[property.first] = property.second * element_count;
    }
  }
  return Status::OK();
}

Status HloCostAnalysis::HandleReduce(
    HloInstruction* reduce, HloInstruction* arg, HloInstruction* init_value,
    tensorflow::gtl::ArraySlice<int64> dimensions, HloComputation* function) {
  // Compute the cost of the user function.
  TF_ASSIGN_OR_RETURN(const Properties sub_properties,
                      ProcessSubcomputation(function));

  // Compute the cost of all elements for this Reduce operation.
  int64 reduction_count = ShapeUtil::ElementsIn(arg->shape()) -
                          ShapeUtil::ElementsIn(reduce->shape());
  for (const auto& property : sub_properties) {
    if (property.first != kBytesAccessedKey) {
      current_properties_[property.first] = property.second * reduction_count;
    }
  }
  return Status::OK();
}

Status HloCostAnalysis::HandleReduceWindow(HloInstruction* reduce_window,
                                           HloInstruction* operand,
                                           const Window& window,
                                           HloComputation* function) {
  // Compute the properties of the reduction function.
  TF_ASSIGN_OR_RETURN(const Properties sub_properties,
                      ProcessSubcomputation(function));

  // Compute the cost of all elements for this ReduceWindow operation. For each
  // output element there are window_size - 1 reductions to perform.
  int64 window_element_count = 1;
  for (const auto& dimension : window.dimensions()) {
    window_element_count *= dimension.size();
  }
  const int64 output_element_count =
      ShapeUtil::ElementsIn(reduce_window->shape());
  const int64 reduction_count =
      (window_element_count - 1) * output_element_count;
  for (const auto& property : sub_properties) {
    if (property.first != kBytesAccessedKey) {
      current_properties_[property.first] = property.second * reduction_count;
    }
  }
  return Status::OK();
}

Status HloCostAnalysis::HandleSelectAndScatter(HloInstruction* instruction) {
  // Compute the properties of the select and scatter function.
  // Compute the properties of the reduction function.
  TF_ASSIGN_OR_RETURN(const Properties select_properties,
                      ProcessSubcomputation(instruction->select()));
  TF_ASSIGN_OR_RETURN(const Properties scatter_properties,
                      ProcessSubcomputation(instruction->scatter()));

  // Compute the cost of all elements for this operation. For each scatter
  // source element there are window_size - 1 select computations to perform and
  // 1 scatter computation to perform.
  const auto source = instruction->operand(1);
  const auto source_element_count = ShapeUtil::ElementsIn(source->shape());
  int64 window_element_count = 1;
  for (const auto& dimension : instruction->window().dimensions()) {
    window_element_count *= dimension.size();
  }
  const int64 select_count = source_element_count * (window_element_count - 1);
  for (const auto& property : select_properties) {
    if (property.first != kBytesAccessedKey) {
      current_properties_[property.first] += property.second * select_count;
    }
  }
  for (const auto& property : scatter_properties) {
    if (property.first != kBytesAccessedKey) {
      current_properties_[property.first] +=
          property.second * source_element_count;
    }
  }
  return Status::OK();
}

Status HloCostAnalysis::HandleBitcast(HloInstruction* bitcast) {
  // A bitcast does no computation and touches no memory.
  current_properties_[kBytesAccessedKey] = 0;
  return Status::OK();
}

Status HloCostAnalysis::HandleBroadcast(HloInstruction* broadcast) {
  return Status::OK();
}

Status HloCostAnalysis::HandlePad(HloInstruction* pad) { return Status::OK(); }

Status HloCostAnalysis::HandleSend(HloInstruction* send) {
  return Status::OK();
}

Status HloCostAnalysis::HandleRecv(HloInstruction* recv) {
  return Status::OK();
}

Status HloCostAnalysis::HandleReshape(HloInstruction* reshape) {
  return Status::OK();
}

Status HloCostAnalysis::HandleBatchNormTraining(
    HloInstruction* batchNormTraining) {
  // TODO(b/62294698): Implement cost analysis for batch-norm-training.
  return Status::OK();
}

Status HloCostAnalysis::HandleBatchNormInference(
    HloInstruction* batchNormInference) {
  // TODO(b/62294698): Implement cost analysis for batch-norm-inference.
  return Status::OK();
}

Status HloCostAnalysis::HandleBatchNormGrad(HloInstruction* batchNormGrad) {
  // TODO(b/62294698): Implement cost analysis for batch-norm-grad.
  return Status::OK();
}

Status HloCostAnalysis::HandleTranspose(HloInstruction* transpose) {
  return Status::OK();
}

Status HloCostAnalysis::HandleConvolution(HloInstruction* convolution,
                                          HloInstruction* lhs_instruction,
                                          HloInstruction* rhs_instruction,
                                          const Window& window) {
  const auto& dnums = convolution->convolution_dimension_numbers();
  const int64 output_features =
      convolution->shape().dimensions(dnums.feature_dimension());

  // For each output element, we do one fma per element in the kernel at some
  // given output feature index.
  const int64 fmas_per_output_element =
      ShapeUtil::ElementsIn(rhs_instruction->shape()) / output_features;
  const int64 output_elements = ShapeUtil::ElementsIn(convolution->shape());
  current_properties_[kFlopsKey] =
      output_elements * fmas_per_output_element * kFmaFlops;
  return Status::OK();
}

Status HloCostAnalysis::HandleCrossReplicaSum(HloInstruction* crs) {
  // We assume 2 replicas, so that each output element is the sum of two input
  // elements.
  //
  // TODO(b/33004697): Compute correct cost here, taking the actual number of
  // replicas into account.
  current_properties_[kFlopsKey] = ShapeUtil::ElementsIn(crs->shape());
  return Status::OK();
}

Status HloCostAnalysis::HandleRng(HloInstruction* random,
                                  RandomDistribution distribution) {
  // TODO(b/26346211): Implement better estimates for the RNG cost, since the
  // cost changes with the implementation and the distribution. For now, assume
  // the cost of each RNG is same as a transcendental operation.
  current_properties_[kTranscendentalsKey] =
      ShapeUtil::ElementsIn(random->shape());
  return Status::OK();
}

Status HloCostAnalysis::HandleFusion(HloInstruction* fusion) {
  // Compute the properties of the fused expression and attribute them to the
  // fusion node. Use a dummy shape_size to avoid any errors from trying to
  // calculate the size of a shape that does not have a layout, since nodes
  // inside fusion nodes do not necessarily have a layout assigned.
  ShapeSizeFunction shape_size = [](const Shape& shape) { return 0; };
  TF_ASSIGN_OR_RETURN(
      current_properties_,
      ProcessSubcomputation(fusion->fused_instructions_computation(),
                            &shape_size));

  // Fusion nodes that produce a tuple also produce the entries in the tuple.
  // Ignore the memory accessed inside fused ops, since fusion is supposed to
  // prevent intermediate data from touching slow memory.
  current_properties_[kBytesAccessedKey] = 0;
  ShapeUtil::ForEachSubshape(
      fusion->shape(),
      [this](const Shape& subshape, const ShapeIndex& /*shape_index*/) {
        current_properties_[kBytesAccessedKey] += shape_size_(subshape);
      });

  for (const HloInstruction* operand : fusion->operands()) {
    current_properties_[kBytesAccessedKey] += shape_size_(operand->shape());
  }

  return Status::OK();
}

Status HloCostAnalysis::HandleCall(HloInstruction* call) {
  TF_ASSIGN_OR_RETURN(current_properties_,
                      ProcessSubcomputation(call->to_apply()));
  current_should_compute_bottleneck_time_ = false;
  return Status::OK();
}

Status HloCostAnalysis::HandleCustomCall(
    HloInstruction* custom_call,
    tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    tensorflow::StringPiece custom_call_target) {
  return Unimplemented("Custom-call is not implemented for HLO cost analysis.");
}

Status HloCostAnalysis::HandleSort(HloInstruction* sort,
                                   HloInstruction* operand_instruction) {
  // This assumes a comparison based N*log(N) algorithm. As for all ops, the
  // actual properties of the op depend on the backend implementation.
  int64 elements = ShapeUtil::ElementsIn(operand_instruction->shape());
  current_properties_[kFlopsKey] = elements * tensorflow::Log2Ceiling(elements);
  return Status::OK();
}

Status HloCostAnalysis::HandleWhile(HloInstruction* xla_while) {
  // Since the number of iterations of the while node will not always be
  // something that we can statically analyze, we cannot precisely compute the
  // cost of a while node. For now compute the cost of a single iteration.
  //
  // TODO(b/26346211): Improve the cost analysis for while nodes.
  TF_ASSIGN_OR_RETURN(const Properties body_properties,
                      ProcessSubcomputation(xla_while->while_body()));

  TF_ASSIGN_OR_RETURN(const Properties condition_properties,
                      ProcessSubcomputation(xla_while->while_condition()));

  current_properties_.clear();
  for (const auto& property : body_properties) {
    current_properties_[property.first] += property.second;
  }
  for (const auto& property : condition_properties) {
    current_properties_[property.first] += property.second;
  }
  current_should_compute_bottleneck_time_ = false;

  return Status::OK();
}

Status HloCostAnalysis::FinishVisit(HloInstruction* root) {
  return Status::OK();
}

float HloCostAnalysis::flop_count() const {
  return GetProperty(kFlopsKey, properties_sum_);
}

float HloCostAnalysis::transcendental_count() const {
  return GetProperty(kTranscendentalsKey, properties_sum_);
}

float HloCostAnalysis::bytes_accessed() const {
  return GetProperty(kBytesAccessedKey, properties_sum_);
}

float HloCostAnalysis::seconds() const {
  return GetProperty(kSecondsKey, properties_sum_);
}

int64 HloCostAnalysis::flop_count(const HloInstruction& hlo) const {
  return GetPropertyForHlo(hlo, kFlopsKey, hlo_properties_);
}

int64 HloCostAnalysis::transcendental_count(const HloInstruction& hlo) const {
  return GetPropertyForHlo(hlo, kTranscendentalsKey, hlo_properties_);
}

int64 HloCostAnalysis::bytes_accessed(const HloInstruction& hlo) const {
  return GetPropertyForHlo(hlo, kBytesAccessedKey, hlo_properties_);
}

float HloCostAnalysis::seconds(const HloInstruction& hlo) const {
  return GetPropertyForHlo(hlo, kSecondsKey, hlo_properties_);
}

StatusOr<HloCostAnalysis::Properties> HloCostAnalysis::ProcessSubcomputation(
    HloComputation* computation, const ShapeSizeFunction* shape_size) {
  if (shape_size == nullptr) {
    shape_size = &shape_size_;
  }
  HloCostAnalysis visitor(*shape_size, per_second_rates_);
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.properties();
}

}  // namespace xla
