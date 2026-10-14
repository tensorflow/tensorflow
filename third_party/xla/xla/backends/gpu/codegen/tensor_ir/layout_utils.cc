/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/tensor_ir/layout_utils.h"

#include <cstdint>
#include <string>
#include <vector>

#include "tensor_ir/Dialect/TensorIR.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/layout_util.h"
#include "xla/shape.h"

namespace xla::gpu::tensor_ir {

std::string ComputeStrideString(const Shape& shape) {
  int64_t rank = shape.dimensions().size();
  if (rank == 0) {
    return "()";
  }
  std::vector<int64_t> strides(rank, 1);
  int64_t stride = 1;
  for (int64_t dim : shape.layout().minor_to_major()) {
    strides[dim] = stride;
    stride *= shape.dimensions(dim);
  }
  return absl::StrCat("(", absl::StrJoin(strides, ","), ")");
}

bool HasDefaultLayout(const Shape& shape) {
  if (!shape.IsArray()) {
    return false;
  }
  // Scalars have only one layout, and a shape without an explicit layout is
  // treated as having the default one (what `LayoutUtil::SetToDefaultLayout`
  // would assign); `shape.layout()` must not be called on such a shape.
  if (shape.dimensions().empty() || !shape.has_layout()) {
    return true;
  }
  return LayoutUtil::IsMonotonicWithDim0Major(shape.layout());
}

namespace {

// Returns the shapes of the graph results `ImportAndLegalizeComputation`
// produces for `computation`: the elements of a root tuple, or the root shape
// itself otherwise.
absl::StatusOr<absl::Span<const Shape>> GetResultShapes(
    const HloComputation& computation) {
  const Shape& root_shape = computation.root_instruction()->shape();
  absl::Span<const Shape> shapes =
      root_shape.IsTuple() ? absl::MakeConstSpan(root_shape.tuple_shapes())
                           : absl::MakeConstSpan(&root_shape, 1);
  for (const Shape& shape : shapes) {
    if (!shape.IsArray()) {
      return absl::InvalidArgumentError(
          absl::StrCat("AttachLayoutStrides: non-array result of computation '",
                       computation.name(), "': ", shape.ToString()));
    }
  }
  return shapes;
}

}  // namespace

absl::Status AttachLayoutStrides(mlir::nv_tensor_ir::GraphOp graph,
                                 const HloComputation& computation) {
  int64_t num_parameters = computation.num_parameters();
  if (graph.getNumArguments() != num_parameters) {
    return absl::InternalError(absl::StrCat(
        "AttachLayoutStrides: graph '", graph.getSymName().str(), "' has ",
        graph.getNumArguments(), " arguments, expected ", num_parameters,
        " for computation '", computation.name(), "'"));
  }

  mlir::MLIRContext* context = graph.getContext();
  mlir::StringAttr stride_attr_name = mlir::StringAttr::get(
      context, mlir::nv_tensor_ir::TensorIRDialect::getStrideAttrName());

  for (int64_t i = 0; i < num_parameters; ++i) {
    const Shape& shape = computation.parameter_instruction(i)->shape();
    if (!shape.IsArray()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "AttachLayoutStrides: non-array parameter at index ", i,
          " of computation '", computation.name(), "': ", shape.ToString()));
    }
    if (!HasDefaultLayout(shape)) {
      graph.setArgAttr(
          i, stride_attr_name,
          mlir::StringAttr::get(context, ComputeStrideString(shape)));
    }
  }

  ABSL_ASSIGN_OR_RETURN(absl::Span<const Shape> result_shapes,
                   GetResultShapes(computation));
  if (graph.getNumResults() != result_shapes.size()) {
    return absl::InternalError(absl::StrCat(
        "AttachLayoutStrides: graph '", graph.getSymName().str(), "' has ",
        graph.getNumResults(), " results, expected ", result_shapes.size(),
        " for computation '", computation.name(), "'"));
  }
  for (int64_t i = 0; i < result_shapes.size(); ++i) {
    if (!HasDefaultLayout(result_shapes[i])) {
      graph.setResultAttr(i, stride_attr_name,
                          mlir::StringAttr::get(
                              context, ComputeStrideString(result_shapes[i])));
    }
  }

  return absl::OkStatus();
}

absl::Status AttachBufferAlignments(
    mlir::nv_tensor_ir::GraphOp graph,
    absl::Span<const emitters::KernelArgument> kernel_args) {
  int64_t num_inputs = graph.getNumArguments();
  int64_t num_results = graph.getNumResults();
  if (kernel_args.size() != num_inputs + num_results) {
    return absl::InternalError(absl::StrCat(
        "AttachBufferAlignments: expected ", num_inputs + num_results,
        " kernel arguments for graph '", graph.getSymName().str(), "' with ",
        num_inputs, " arguments and ", num_results, " results, but got ",
        kernel_args.size()));
  }

  mlir::MLIRContext* context = graph.getContext();
  mlir::StringAttr alignment_attr_name = mlir::StringAttr::get(
      context, mlir::nv_tensor_ir::TensorIRDialect::getAlignmentAttrName());
  mlir::IntegerType i64_type = mlir::IntegerType::get(context, 64);

  for (int64_t i = 0; i < num_inputs; ++i) {
    graph.setArgAttr(
        i, alignment_attr_name,
        mlir::IntegerAttr::get(i64_type, kernel_args[i].alignment()));
  }
  for (int64_t i = 0; i < num_results; ++i) {
    graph.setResultAttr(i, alignment_attr_name,
                        mlir::IntegerAttr::get(
                            i64_type, kernel_args[num_inputs + i].alignment()));
  }

  return absl::OkStatus();
}

}  // namespace xla::gpu::tensor_ir
