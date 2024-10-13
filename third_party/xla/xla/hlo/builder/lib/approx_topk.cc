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

#include "xla/hlo/builder/lib/approx_topk.h"

#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/lib/approx_topk_shape.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

// Used by rank 2+ operands
const uint64_t kTpuLaneTiling = 128;
// Used by rank 1 operands.
const uint64_t kTpuChunkTiling = 1024;

namespace xla {

namespace {
absl::StatusOr<std::vector<PrimitiveType>> GetOperandTypes(
    XlaBuilder* builder, absl::Span<const XlaOp> operands,
    absl::Span<const XlaOp> init_values) {
  std::vector<PrimitiveType> op_types;
  auto num_operands = operands.size();
  auto operands_shapes = builder->GetOperandShapes(operands).value();
  auto init_values_shapes = builder->GetOperandShapes(init_values).value();
  for (int i = 0; i < num_operands; ++i) {
    const auto& op_shape = operands_shapes[i];
    const auto& init_shape = init_values_shapes[i];
    if (op_shape.rank() == 0) {
      return InvalidArgument("ApproxTopK operands must have rank 1+.");
    }
    if (!ShapeUtil::CompatibleIgnoringElementType(operands_shapes[0],
                                                  op_shape)) {
      return InvalidArgument("operands shape mismatch: %s vs %s",
                             operands_shapes[0].DebugString(),
                             op_shape.DebugString());
    }
    if (op_shape.element_type() != init_shape.element_type()) {
      return InvalidArgument("operands type mismatch: %s vs %s",
                             op_shape.DebugString(), init_shape.DebugString());
    }
    op_types.push_back(op_shape.element_type());
  }
  return op_types;
}
}  // namespace

// Converts a comparator to a combiner computation that can be fed to reduce or
// partial reduce ops.
XlaComputation BuildReductionComputation(
    XlaBuilder* builder, absl::Span<const PrimitiveType> op_types,
    const XlaComputation& comparator) {
  auto num_operands = op_types.size();
  std::vector<XlaOp> lhs_params;
  std::vector<XlaOp> rhs_params;
  int64_t param_number = 0;
  lhs_params.reserve(num_operands);
  rhs_params.reserve(num_operands);
  auto reduction_builder = builder->CreateSubBuilder("ReductionFn");
  for (const auto& op_type : op_types) {
    lhs_params.push_back(Parameter(reduction_builder.get(), param_number,
                                   ShapeUtil::MakeScalarShape(op_type),
                                   absl::StrFormat("lhs.%d", param_number)));
    param_number++;
  }
  for (const auto& op_type : op_types) {
    rhs_params.push_back(Parameter(reduction_builder.get(), param_number,
                                   ShapeUtil::MakeScalarShape(op_type),
                                   absl::StrFormat("rhs.%d", param_number)));
    param_number++;
  }

  std::vector<XlaOp> comparator_args;
  comparator_args.reserve(num_operands * 2);
  for (int i = 0; i < num_operands; ++i) {
    comparator_args.push_back(lhs_params[i]);
    comparator_args.push_back(rhs_params[i]);
  }
  auto pred = Call(reduction_builder.get(), comparator, comparator_args);
  std::vector<XlaOp> results;
  results.reserve(num_operands);
  for (int i = 0; i < num_operands; ++i) {
    results.push_back(Select(pred, lhs_params[i], rhs_params[i]));
  }
  Tuple(reduction_builder.get(), results);
  return reduction_builder->BuildAndNoteError();
}

XlaOp AggregateToTopKBuilder(XlaBuilder* builder,
                             absl::Span<const XlaOp> operands,
                             absl::Span<const XlaOp> init_values, int64_t top_k,
                             int64_t reduction_dim,
                             const XlaComputation& comparator) {
  auto operands_shapes = builder->GetOperandShapes(operands).value();
  int64_t rank = operands_shapes[0].rank();
  int64_t num_operands = operands.size();

  if (top_k == 1) {
    auto status_or_optypes = GetOperandTypes(builder, operands, init_values);
    if (!status_or_optypes.ok()) {
      return builder->ReportError(status_or_optypes.status());
    }
    auto op_types = status_or_optypes.value();

    auto reduction_computation =
        BuildReductionComputation(builder, op_types, comparator);
    auto val_args = Reduce(builder, operands, init_values,
                           reduction_computation, {reduction_dim});
    Shape op_shape = operands_shapes[0];
    op_shape.mutable_dimensions()[reduction_dim] = 1;
    auto top1_vals =
        Reshape(GetTupleElement(val_args, 0), op_shape.dimensions());
    auto top1_args =
        Reshape(GetTupleElement(val_args, 1), op_shape.dimensions());
    return Tuple(builder, {top1_vals, top1_args});
  }

  auto sorted_results = Sort(operands, comparator, reduction_dim);
  std::vector<int64_t> slice_start_indices(rank, 0);
  std::vector<int64_t> slice_limit_indices;
  std::vector<int64_t> slice_strides(rank, 1);
  slice_limit_indices.insert(slice_limit_indices.begin(),
                             operands_shapes[0].dimensions().begin(),
                             operands_shapes[0].dimensions().end());
  slice_limit_indices[reduction_dim] = top_k;

  std::vector<XlaOp> sliced_results;
  sliced_results.reserve(num_operands);
  for (int i = 0; i < num_operands; ++i) {
    sliced_results.push_back(Slice(GetTupleElement(sorted_results, i),
                                   slice_start_indices, slice_limit_indices,
                                   slice_strides));
  }
  return Tuple(builder, sliced_results);
}

XlaOp ApproxTopK(XlaBuilder* builder, absl::Span<const XlaOp> operands,
                 absl::Span<const XlaOp> init_values, int64_t top_k,
                 int64_t reduction_dim, const XlaComputation& comparator,
                 float recall_target, bool aggregate_to_topk,
                 int64_t reduction_input_size_override) {
  // Validates shapes and ranks
  if (operands.size() != init_values.size()) {
    return builder->ReportError(
        InvalidArgument("operands and init_values size mismatch: %d vs %d",
                        operands.size(), init_values.size()));
  }
  auto num_operands = operands.size();
  auto operands_shapes = builder->GetOperandShapes(operands).value();
  auto init_values_shapes = builder->GetOperandShapes(init_values).value();
  auto status_or_optypes = GetOperandTypes(builder, operands, init_values);
  if (!status_or_optypes.ok()) {
    return builder->ReportError(status_or_optypes.status());
  }
  auto op_types = status_or_optypes.value();
  int64_t rank = operands_shapes[0].rank();
  if (reduction_dim < 0 || reduction_dim >= rank) {
    return builder->ReportError(
        InvalidArgument("reduction_dim should range in [0,%d)", rank));
  }

  auto reduction_computation =
      BuildReductionComputation(builder, op_types, comparator);

  uint64_t tpu_tiling = rank == 1 ? kTpuChunkTiling : kTpuLaneTiling;
  uint64_t n = operands_shapes[0].dimensions(reduction_dim);
  // ApproxTopK can only reduce elements larger than the tiling.
  if (n <= tpu_tiling) {
    if (aggregate_to_topk) {
      return AggregateToTopKBuilder(builder, operands, init_values, top_k,
                                    reduction_dim, comparator);
    }
    return Tuple(builder, operands);
  }

  auto status_or_approx_output_size = ApproxTopKReductionOutputSize(
      n, rank, top_k, recall_target, /*aggregate_to_topk=*/false,
      reduction_input_size_override);
  if (!status_or_approx_output_size.status().ok()) {
    return builder->ReportError(status_or_approx_output_size.status());
  }

  int64_t approx_output_size, log2_reduction;
  std::tie(approx_output_size, log2_reduction) =
      status_or_approx_output_size.value();

  if (log2_reduction == 0) {
    if (aggregate_to_topk) {
      return AggregateToTopKBuilder(builder, operands, init_values, top_k,
                                    reduction_dim, comparator);
    }
    return Tuple(builder, operands);
  }

  std::vector<XlaOp> partial_reduce_args;
  partial_reduce_args.reserve(operands.size() + init_values.size());
  for (const auto& op : operands) {
    partial_reduce_args.push_back(op);
  }
  for (const auto& op : init_values) {
    partial_reduce_args.push_back(op);
  }
  std::vector<const Shape*> approx_output_shapes;
  approx_output_shapes.reserve(operands_shapes.size());
  for (auto& op_shape : operands_shapes) {
    op_shape.mutable_dimensions()[reduction_dim] = approx_output_size;
    approx_output_shapes.push_back(&op_shape);
  }
  auto approx_output_shape =
      ShapeUtil::MakeTupleShapeWithPtrs(approx_output_shapes);
  // PartialReduce options in the JSON form.
  auto partial_reduce_option = absl::StrFormat(
      "{\"log2_reduction\": %d, "
      "\"reduction_dim\": %d, "
      "\"to_apply_type\": \"comparator\", "
      "\"top_k\": %d, "
      "\"recall_target\": %f}",
      log2_reduction, reduction_dim, top_k, recall_target);

  auto approx_topk = CustomCallWithComputation(
      builder, "PartialReduce", partial_reduce_args, comparator,
      approx_output_shape, partial_reduce_option);

  if (aggregate_to_topk) {
    std::vector<XlaOp> approx_topk_results;
    approx_topk_results.reserve(num_operands);
    for (int i = 0; i < num_operands; ++i) {
      approx_topk_results.push_back(GetTupleElement(approx_topk, i));
    }
    return AggregateToTopKBuilder(builder, approx_topk_results, init_values,
                                  top_k, reduction_dim, comparator);
  }
  return approx_topk;
}

XlaOp ApproxTopKFallback(XlaBuilder* builder, absl::Span<const XlaOp> operands,
                         absl::Span<const XlaOp> init_values, int64_t top_k,
                         int64_t reduction_dim,
                         const XlaComputation& comparator, float recall_target,
                         bool aggregate_to_topk,
                         int64_t reduction_input_size_override) {
  auto operands_shapes = builder->GetOperandShapes(operands).value();
  int64_t rank = operands_shapes[0].rank();
  uint64_t n = operands_shapes[0].dimensions(reduction_dim);
  // Align the output size with ApproxTopK.
  auto status_or_approx_output_size = ApproxTopKReductionOutputSize(
      n, rank, top_k, recall_target, aggregate_to_topk,
      reduction_input_size_override);
  if (!status_or_approx_output_size.ok()) {
    return builder->ReportError(status_or_approx_output_size.status());
  }
  auto output_size = status_or_approx_output_size.value().first;
  return AggregateToTopKBuilder(builder, operands, init_values, output_size,
                                reduction_dim, comparator);
}

}  // namespace xla
