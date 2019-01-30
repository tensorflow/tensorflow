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
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace xla {

constexpr char HloCostAnalysis::kFlopsKey[];
constexpr char HloCostAnalysis::kTranscendentalsKey[];
constexpr char HloCostAnalysis::kBytesAccessedKey[];
constexpr char HloCostAnalysis::kOptimalSecondsKey[];

HloCostAnalysis::HloCostAnalysis(const ShapeSizeFunction& shape_size)
    : HloCostAnalysis(shape_size, {}) {}

HloCostAnalysis::HloCostAnalysis(const ShapeSizeFunction& shape_size,
                                 const Properties& per_second_rates)
    : shape_size_(shape_size), per_second_rates_(per_second_rates) {}

Status HloCostAnalysis::Preprocess(const HloInstruction* hlo) {
  // Set current instruction cost values to reasonable default values. Each
  // handler can overwrite these values. In Postprocess, these values are
  // accumulated and written to the per-instruction maps.
  current_properties_.clear();
  current_should_compute_bottleneck_time_ = true;

  // The default number of bytes accessed for an instruction is the sum of the
  // sizes of the inputs and outputs. The default ShapeUtil::ByteSizeOf does not
  // handle opaque types.
  float bytes_accessed = GetShapeSize(hlo->shape());
  for (const HloInstruction* operand : hlo->operands()) {
    bytes_accessed += GetShapeSize(operand->shape());
  }
  current_properties_[kBytesAccessedKey] = bytes_accessed;

  return Status::OK();
}

Status HloCostAnalysis::Postprocess(const HloInstruction* hlo) {
  if (current_should_compute_bottleneck_time_) {
    // Compute the time as the time of the bottleneck, i.e. the slowest property
    // given the per-second rate of each property.
    float optimal_seconds = 0.0f;
    for (const auto& property : current_properties_) {
      if (property.first != kOptimalSecondsKey) {
        optimal_seconds = std::max(
            optimal_seconds,
            property.second /
                GetProperty(property.first, per_second_rates_, INFINITY));
      }
    }
    current_properties_[kOptimalSecondsKey] = optimal_seconds;
  }

  TF_RET_CHECK(hlo_properties_.emplace(hlo, current_properties_).second);
  for (const auto& property : current_properties_) {
    properties_sum_[property.first] += property.second;
  }

  return Status::OK();
}

Status HloCostAnalysis::HandleElementwiseOp(
    const HloInstruction* hlo_instruction) {
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

int64 HloCostAnalysis::GetShapeSize(const Shape& shape) const {
  if (!LayoutUtil::HasLayout(shape)) {
    return 0;
  }
  return shape_size_(shape);
}

Status HloCostAnalysis::HandleElementwiseUnary(const HloInstruction* hlo) {
  return HandleElementwiseOp(hlo);
}

Status HloCostAnalysis::HandleElementwiseBinary(const HloInstruction* hlo) {
  return HandleElementwiseOp(hlo);
}

Status HloCostAnalysis::HandleCompare(const HloInstruction* compare) {
  return HandleElementwiseOp(compare);
}

Status HloCostAnalysis::HandleClamp(const HloInstruction* clamp) {
  return HandleElementwiseOp(clamp);
}

Status HloCostAnalysis::HandleReducePrecision(const HloInstruction* hlo) {
  return HandleElementwiseOp(hlo);
}

Status HloCostAnalysis::HandleParameter(const HloInstruction*) {
  current_should_compute_bottleneck_time_ = false;
  current_properties_[kBytesAccessedKey] = 0;
  current_properties_[kOptimalSecondsKey] = 0;
  return Status::OK();
}

Status HloCostAnalysis::HandleConstant(const HloInstruction*) {
  current_should_compute_bottleneck_time_ = false;
  current_properties_[kBytesAccessedKey] = 0;
  current_properties_[kOptimalSecondsKey] = 0;
  return Status::OK();
}

Status HloCostAnalysis::HandleIota(const HloInstruction*) {
  return Status::OK();
}

Status HloCostAnalysis::HandleGetTupleElement(const HloInstruction*) {
  // GetTupleElement forwards a pointer and does not touch each element in the
  // output.
  current_should_compute_bottleneck_time_ = false;
  current_properties_[kBytesAccessedKey] = 0;
  current_properties_[kOptimalSecondsKey] = 0;
  return Status::OK();
}

Status HloCostAnalysis::HandleSelect(const HloInstruction* hlo) {
  return HandleElementwiseOp(hlo);
}

Status HloCostAnalysis::HandleTupleSelect(const HloInstruction*) {
  return Status::OK();
}

Status HloCostAnalysis::HandleReverse(const HloInstruction*) {
  return Status::OK();
}

Status HloCostAnalysis::HandleSlice(const HloInstruction* slice) {
  current_properties_[kBytesAccessedKey] = GetShapeSize(slice->shape()) * 2;
  return Status::OK();
}

Status HloCostAnalysis::HandleDynamicSlice(
    const HloInstruction* dynamic_slice) {
  current_properties_[kBytesAccessedKey] =
      GetShapeSize(dynamic_slice->shape()) * 2;
  return Status::OK();
}

Status HloCostAnalysis::HandleDynamicUpdateSlice(
    const HloInstruction* dynamic_update_slice) {
  current_properties_[kBytesAccessedKey] =
      GetShapeSize(dynamic_update_slice->operand(1)->shape()) * 2;
  return Status::OK();
}

Status HloCostAnalysis::HandleTuple(const HloInstruction* tuple) {
  // The tuple instruction only gathers pointers from inputs (it doesn't iterate
  // through them). The memory touched is then only the size of the output
  // index table of the tuple.

  current_properties_[kBytesAccessedKey] = GetShapeSize(tuple->shape());
  return Status::OK();
}

Status HloCostAnalysis::HandleConcatenate(const HloInstruction*) {
  return Status::OK();
}

Status HloCostAnalysis::HandleConvert(const HloInstruction* convert) {
  return HandleElementwiseOp(convert);
}

Status HloCostAnalysis::HandleCopy(const HloInstruction*) {
  return Status::OK();
}

Status HloCostAnalysis::HandleDomain(const HloInstruction* domain) {
  // Domain does not have any computation or data transfer.
  current_should_compute_bottleneck_time_ = false;
  current_properties_[kBytesAccessedKey] = 0;
  current_properties_[kOptimalSecondsKey] = 0;
  return Status::OK();
}

Status HloCostAnalysis::HandleDot(const HloInstruction* dot) {
  const Shape& lhs_shape = dot->operand(0)->shape();
  const Shape& dot_shape = dot->shape();
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  // Count of elements along the reduction dimension (last dimension for the
  // rhs).
  int64 reduction_width = 1;
  for (auto dim : dnums.lhs_contracting_dimensions()) {
    reduction_width *= lhs_shape.dimensions(dim);
  }
  // Each output elment requires reduction_widht FMA operations.
  current_properties_[kFlopsKey] =
      kFmaFlops * ShapeUtil::ElementsIn(dot_shape) * reduction_width;
  return Status::OK();
}

Status HloCostAnalysis::HandleInfeed(const HloInstruction*) {
  return Status::OK();
}

Status HloCostAnalysis::HandleOutfeed(const HloInstruction*) {
  return Status::OK();
}

Status HloCostAnalysis::HandleMap(const HloInstruction* map) {
  // Compute properties of the mapped function.
  TF_ASSIGN_OR_RETURN(const Properties sub_properties,
                      ProcessNestedSubcomputation(map->to_apply()));

  // Compute the cost of all elements for this Map operation.
  const int64 element_count = ShapeUtil::ElementsIn(map->shape());
  for (const auto& property : sub_properties) {
    if (property.first != kBytesAccessedKey) {
      current_properties_[property.first] = property.second * element_count;
    }
  }
  return Status::OK();
}

Status HloCostAnalysis::HandleReduce(const HloInstruction* reduce) {
  HloComputation* function = reduce->to_apply();
  // Compute the cost of the user function.
  TF_ASSIGN_OR_RETURN(const Properties sub_properties,
                      ProcessNestedSubcomputation(function));

  // Compute the cost of all elements for this Reduce operation.
  // This counts the number of times the reduction function is applied, so it
  // does not need to be multiplied by the number of input tensors - that's
  // already "priced in" by the sub-computation doing more work.
  auto arg = reduce->operand(0);
  auto output_shape = reduce->shape().IsArray()
                          ? reduce->shape()
                          : reduce->shape().tuple_shapes(0);
  int64 reduction_count =
      ShapeUtil::ElementsIn(arg->shape()) - ShapeUtil::ElementsIn(output_shape);
  for (const auto& property : sub_properties) {
    if (property.first != kBytesAccessedKey) {
      current_properties_[property.first] = property.second * reduction_count;
    }
  }
  return Status::OK();
}

Status HloCostAnalysis::HandleReduceWindow(
    const HloInstruction* reduce_window) {
  const Window& window = reduce_window->window();
  auto function = reduce_window->to_apply();
  // Compute the properties of the reduction function.
  TF_ASSIGN_OR_RETURN(const Properties sub_properties,
                      ProcessNestedSubcomputation(function));

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

Status HloCostAnalysis::HandleSelectAndScatter(
    const HloInstruction* instruction) {
  // Compute the properties of the select and scatter function.
  // Compute the properties of the reduction function.
  TF_ASSIGN_OR_RETURN(const Properties select_properties,
                      ProcessNestedSubcomputation(instruction->select()));
  TF_ASSIGN_OR_RETURN(const Properties scatter_properties,
                      ProcessNestedSubcomputation(instruction->scatter()));

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

Status HloCostAnalysis::HandleBitcast(const HloInstruction*) {
  // A bitcast does no computation and touches no memory.
  current_properties_[kBytesAccessedKey] = 0;
  current_properties_[kOptimalSecondsKey] = 0;
  return Status::OK();
}

Status HloCostAnalysis::HandleBroadcast(const HloInstruction*) {
  return Status::OK();
}

Status HloCostAnalysis::HandlePad(const HloInstruction*) {
  return Status::OK();
}

Status HloCostAnalysis::HandleSend(const HloInstruction*) {
  return Status::OK();
}

Status HloCostAnalysis::HandleSendDone(const HloInstruction*) {
  return Status::OK();
}

Status HloCostAnalysis::HandleRecv(const HloInstruction*) {
  return Status::OK();
}

Status HloCostAnalysis::HandleRecvDone(const HloInstruction*) {
  return Status::OK();
}

Status HloCostAnalysis::HandleReshape(const HloInstruction*) {
  return Status::OK();
}

Status HloCostAnalysis::HandleBatchNormTraining(const HloInstruction*) {
  // TODO(b/62294698): Implement cost analysis for batch-norm-training.
  return Status::OK();
}

Status HloCostAnalysis::HandleBatchNormInference(const HloInstruction*) {
  // TODO(b/62294698): Implement cost analysis for batch-norm-inference.
  return Status::OK();
}

Status HloCostAnalysis::HandleBatchNormGrad(const HloInstruction*) {
  // TODO(b/62294698): Implement cost analysis for batch-norm-grad.
  return Status::OK();
}

Status HloCostAnalysis::HandleTranspose(const HloInstruction*) {
  return Status::OK();
}

Status HloCostAnalysis::HandleAfterAll(const HloInstruction*) {
  // This instruction is used to enforce ordering at compile time. No code is
  // emitted.
  current_should_compute_bottleneck_time_ = false;
  current_properties_[kBytesAccessedKey] = 0;
  current_properties_[kOptimalSecondsKey] = 0;
  return Status::OK();
}

Status HloCostAnalysis::HandleAddDependency(
    const HloInstruction* add_dependency) {
  // This instruction is used to enforce ordering at compile time. No code is
  // emitted.
  current_should_compute_bottleneck_time_ = false;
  current_properties_[kBytesAccessedKey] = 0;
  current_properties_[kOptimalSecondsKey] = 0;
  return Status::OK();
}

Status HloCostAnalysis::HandleConvolution(const HloInstruction* convolution) {
  auto lhs = convolution->operand(0);
  auto rhs = convolution->operand(1);
  Window window = convolution->window();
  const auto& result_shape = convolution->shape();
  const Shape& lhs_shape = lhs->shape();
  const Shape& rhs_shape = rhs->shape();

  const auto& dnums = convolution->convolution_dimension_numbers();

  const int64 input_batch_dim = dnums.input_batch_dimension();
  const int64 input_feature_dim = dnums.input_feature_dimension();
  const int64 output_feature_dim = dnums.output_feature_dimension();
  const int64 input_feature =
      ShapeUtil::GetDimension(lhs_shape, input_feature_dim);
  const int64 output_feature =
      ShapeUtil::GetDimension(result_shape, output_feature_dim);
  const int64 batch = ShapeUtil::GetDimension(lhs_shape, input_batch_dim);

  DimensionVector kernel_limits;
  DimensionVector output_limits;
  DimensionVector input_limits;
  if (window.dimensions().empty()) {
    window = window_util::MakeWindow({1});
    kernel_limits.push_back(1);
    output_limits.push_back(1);
    input_limits.push_back(1);
  } else {
    for (int64 spatial_dimension = 0;
         spatial_dimension < window.dimensions_size(); ++spatial_dimension) {
      // Spatial dimension number for kernel (rhs).
      const int64 kernel_spatial_dim =
          dnums.kernel_spatial_dimensions(spatial_dimension);
      const int64 kernel_limit = rhs_shape.dimensions(kernel_spatial_dim);
      kernel_limits.push_back(kernel_limit);

      // Spatial dimension number for output.
      const int64 output_spatial_dim =
          dnums.output_spatial_dimensions(spatial_dimension);
      const int64 output_limit = result_shape.dimensions(output_spatial_dim);
      output_limits.push_back(output_limit);

      // Spatial dimension number for input (lhs).
      const int64 input_spatial_dim =
          dnums.input_spatial_dimensions(spatial_dimension);
      const int64 input_limit = lhs_shape.dimensions(input_spatial_dim);
      input_limits.push_back(input_limit);
    }
  }

  DimensionVector valid_position_counts;

  // Loop over each spatial dimension.
  for (int64 spatial_dimension = 0;
       spatial_dimension < window.dimensions_size(); ++spatial_dimension) {
    int64 valid_position_count = 0;
    // Loop over each point in the kernel.
    for (int64 kernel_idx = 0; kernel_idx < kernel_limits[spatial_dimension];
         ++kernel_idx) {
      // Loop over each point in the output.
      for (int64 output_idx = 0; output_idx < output_limits[spatial_dimension];
           ++output_idx) {
        // Calculate lhs (input) index without taking base dilation into
        // account.
        const auto& window_dim = window.dimensions(spatial_dimension);
        const int64 undilated_index = output_idx * window_dim.stride() -
                                      window_dim.padding_low() +
                                      kernel_idx * window_dim.window_dilation();

        // Calculate the actual lhs (input) index after dilation. Avoid the
        // division as an optimization.
        const int64 lhs_spatial_index =
            window_dim.base_dilation() > 1
                ? undilated_index / window_dim.base_dilation()
                : undilated_index;

        // Skip if the lhs (input) index is to be dilated.
        if (undilated_index != lhs_spatial_index * window_dim.base_dilation()) {
          continue;
        }

        // Skip if input index is not in bound.
        if (lhs_spatial_index < 0 ||
            lhs_spatial_index >= input_limits[spatial_dimension]) {
          continue;
        }

        valid_position_count += 1;
      }
    }
    valid_position_counts.push_back(valid_position_count);
  }

  const int64 fma_count = (input_feature / convolution->feature_group_count()) *
                          output_feature *
                          (batch / convolution->batch_group_count()) *
                          Product(valid_position_counts);
  current_properties_[kFlopsKey] = fma_count * kFmaFlops;
  return Status::OK();
}

Status HloCostAnalysis::HandleFft(const HloInstruction* fft) {
  auto real_shape =
      fft->operand(0)->shape().IsTuple()
          ? ShapeUtil::GetTupleElementShape(fft->operand(0)->shape(), 0)
          : fft->operand(0)->shape();
  constexpr int kFmaPerComplexMul = 4;
  int64 log_factors = 1;
  for (int64 dim : fft->fft_length()) {
    log_factors *= tensorflow::Log2Floor(dim);
  }
  current_properties_[kFlopsKey] = kFmaFlops * kFmaPerComplexMul * log_factors *
                                   ShapeUtil::ElementsIn(real_shape);
  return Status::OK();
}

Status HloCostAnalysis::HandleAllReduce(const HloInstruction* crs) {
  // We assume 2 replicas, so that each output element is the sum of two input
  // elements.
  //
  // TODO(b/33004697): Compute correct cost here, taking the actual number of
  // replicas into account.
  double flops = 0.0;
  ShapeUtil::ForEachSubshape(crs->shape(),
                             [&](const Shape& subshape, const ShapeIndex&) {
                               if (subshape.IsArray()) {
                                 flops += ShapeUtil::ElementsIn(subshape);
                               }
                             });
  current_properties_[kFlopsKey] = flops;
  return Status::OK();
}

Status HloCostAnalysis::HandleAllToAll(const HloInstruction* hlo) {
  return Status::OK();
}

Status HloCostAnalysis::HandleCollectivePermute(const HloInstruction* /*hlo*/) {
  return Status::OK();
}

Status HloCostAnalysis::HandleReplicaId(const HloInstruction* /*hlo*/) {
  return Status::OK();
}

Status HloCostAnalysis::HandleRng(const HloInstruction* random) {
  // TODO(b/26346211): Implement better estimates for the RNG cost, since the
  // cost changes with the implementation and the distribution. For now, assume
  // the cost of each RNG is same as a transcendental operation.
  current_properties_[kTranscendentalsKey] =
      ShapeUtil::ElementsIn(random->shape());
  return Status::OK();
}

Status HloCostAnalysis::HandleFusion(const HloInstruction* fusion) {
  TF_ASSIGN_OR_RETURN(
      current_properties_,
      ProcessNestedSubcomputation(fusion->fused_instructions_computation()));

  // Fusion nodes that produce a tuple also produce the entries in the tuple.
  // Ignore the memory accessed inside fused ops, since fusion is supposed to
  // prevent intermediate data from touching slow memory.
  current_properties_[kBytesAccessedKey] = 0;
  ShapeUtil::ForEachSubshape(
      fusion->shape(),
      [this](const Shape& subshape, const ShapeIndex& /*shape_index*/) {
        current_properties_[kBytesAccessedKey] += GetShapeSize(subshape);
      });

  for (const HloInstruction* operand : fusion->operands()) {
    current_properties_[kBytesAccessedKey] += GetShapeSize(operand->shape());
  }

  return Status::OK();
}

Status HloCostAnalysis::HandleCall(const HloInstruction* call) {
  TF_ASSIGN_OR_RETURN(current_properties_,
                      ProcessUnnestedSubcomputation(call->to_apply()));
  current_should_compute_bottleneck_time_ = false;
  return Status::OK();
}

Status HloCostAnalysis::HandleCustomCall(const HloInstruction*) {
  // Mark applicable fields as "unknown", since we don't know what CustomCall
  // does.  This is better than returning an error, which would stop iteration,
  // and therefore would prevent us from getting *any* stats for a computation
  // which contains a CustomCall.
  current_properties_[kOptimalSecondsKey] = -1;
  current_properties_[kBytesAccessedKey] = -1;
  current_properties_[kFlopsKey] = -1;
  current_should_compute_bottleneck_time_ = false;
  return Status::OK();
}

Status HloCostAnalysis::HandleSort(const HloInstruction* sort) {
  // This assumes a comparison based N*log(N) algorithm. As for all ops, the
  // actual properties of the op depend on the backend implementation.
  int64 elements = ShapeUtil::ElementsIn(sort->operand(0)->shape());
  current_properties_[kFlopsKey] = elements * tensorflow::Log2Ceiling(elements);
  return Status::OK();
}

Status HloCostAnalysis::HandleWhile(const HloInstruction* xla_while) {
  // Since the number of iterations of the while node will not always be
  // something that we can statically analyze, we cannot precisely compute the
  // cost of a while node. For now compute the cost of a single iteration.
  TF_ASSIGN_OR_RETURN(const Properties body_properties,
                      ProcessUnnestedSubcomputation(xla_while->while_body()));

  TF_ASSIGN_OR_RETURN(
      const Properties condition_properties,
      ProcessUnnestedSubcomputation(xla_while->while_condition()));

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

Status HloCostAnalysis::HandleConditional(const HloInstruction* conditional) {
  // Compute the cost of the true and false computations and take the maximum
  // from those for each property.
  TF_ASSIGN_OR_RETURN(
      const Properties true_computation_properties,
      ProcessUnnestedSubcomputation(conditional->true_computation()));
  TF_ASSIGN_OR_RETURN(
      const Properties false_computation_properties,
      ProcessUnnestedSubcomputation(conditional->false_computation()));
  current_properties_ = true_computation_properties;
  for (const auto& property : false_computation_properties) {
    if (!tensorflow::gtl::InsertIfNotPresent(&current_properties_, property)) {
      current_properties_[property.first] =
          std::max(current_properties_[property.first], property.second);
    }
  }
  current_should_compute_bottleneck_time_ = false;

  return Status::OK();
}

Status HloCostAnalysis::HandleGather(const HloInstruction* gather) {
  // Gather doesn't read the whole input buffer, it's equivalent to a copy the
  // size of the output shape and a read of the gather indices.
  current_properties_[kBytesAccessedKey] =
      GetShapeSize(gather->shape()) * 2 +
      GetShapeSize(gather->operand(1)->shape());
  // Gather does not issue any flops.
  return Status::OK();
}

Status HloCostAnalysis::HandleScatter(const HloInstruction* scatter) {
  current_properties_[kBytesAccessedKey] =
      GetShapeSize(scatter->operand(2)->shape()) * 2 +
      GetShapeSize(scatter->operand(1)->shape());
  const int64 element_count =
      ShapeUtil::ElementsIn(scatter->operand(2)->shape());
  TF_ASSIGN_OR_RETURN(const Properties sub_properties,
                      ProcessNestedSubcomputation(scatter->to_apply()));
  for (const auto& property : sub_properties) {
    if (property.first != kBytesAccessedKey) {
      current_properties_[property.first] = property.second * element_count;
    }
  }
  return Status::OK();
}

Status HloCostAnalysis::HandleGetDimensionSize(
    const HloInstruction* /*get_size*/) {
  return Status::OK();
}

Status HloCostAnalysis::FinishVisit(const HloInstruction*) {
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

float HloCostAnalysis::optimal_seconds() const {
  return GetProperty(kOptimalSecondsKey, properties_sum_);
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

float HloCostAnalysis::optimal_seconds(const HloInstruction& hlo) const {
  return GetPropertyForHlo(hlo, kOptimalSecondsKey, hlo_properties_);
}

StatusOr<HloCostAnalysis::Properties>
HloCostAnalysis::ProcessNestedSubcomputation(HloComputation* computation) {
  HloCostAnalysis visitor(shape_size_, per_second_rates_);
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.properties();
}

StatusOr<HloCostAnalysis::Properties>
HloCostAnalysis::ProcessUnnestedSubcomputation(HloComputation* computation) {
  HloCostAnalysis visitor(shape_size_, per_second_rates_);
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  hlo_properties_.insert(visitor.hlo_properties_.begin(),
                         visitor.hlo_properties_.end());
  return visitor.properties();
}

}  // namespace xla
