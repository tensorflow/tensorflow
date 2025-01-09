/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/xnnpack/xnn_dot_thunk.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xnnpack.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/dot_lib.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/xnnpack/xnn_fusion_thunk.h"
#include "xla/backends/cpu/runtime/xnnpack/xnn_interop.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla::cpu {

absl::StatusOr<xnn_subgraph_t> XnnDotThunk::BuildDotSubgraph(
    absl::Span<const Argument> arguments, absl::Span<const Result> results) {
  xnn_subgraph_t subgraph = nullptr;
  XNN_RETURN_IF_ERROR(xnn_create_subgraph(/*external_value_ids=*/3,
                                          /*flags=*/0, &subgraph));

  uint32_t lhs_id = XNN_INVALID_VALUE_ID;
  uint32_t rhs_id = XNN_INVALID_VALUE_ID;
  uint32_t out_id = XNN_INVALID_VALUE_ID;

  auto dims = [](absl::Span<const int64_t> dims) -> std::vector<size_t> {
    return {dims.begin(), dims.end()};
  };

  std::vector<size_t> lhs_dims = dims(dot_slices_.lhs_shape.dimensions());
  std::vector<size_t> rhs_dims = dims(dot_slices_.rhs_shape.dimensions());
  std::vector<size_t> out_dims = dims(dot_slices_.out_shape.dimensions());

  XNN_RETURN_IF_ERROR(xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, lhs_dims.size(), lhs_dims.data(), nullptr,
      /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &lhs_id));

  XNN_RETURN_IF_ERROR(xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, rhs_dims.size(), rhs_dims.data(), nullptr,
      /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_INPUT, &rhs_id));

  XNN_RETURN_IF_ERROR(xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, out_dims.size(), out_dims.data(), nullptr,
      /*external_id=*/2, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id));

  XNN_RETURN_IF_ERROR(xnn_define_batch_matrix_multiply(
      subgraph, lhs_id, rhs_id, out_id,
      /*flags=*/dot_canonical_dims_.rhs_canonical ? 0 : XNN_FLAG_TRANSPOSE_B));

  return subgraph;
}

absl::StatusOr<bool> XnnDotThunk::IsSupported(
    const DotDimensionNumbers& dot_dimensions, const Shape& lhs_shape,
    const Shape& rhs_shape, const Shape& out_shape) {
  // TODO(ezhulenev): Support other element types.
  if (lhs_shape.element_type() != F32 || rhs_shape.element_type() != F32 ||
      out_shape.element_type() != F32) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(DotShape dot_shape, GetDotShape(dot_dimensions, lhs_shape,
                                                      rhs_shape, out_shape));

  TF_ASSIGN_OR_RETURN(DotCanonicalDims dot_canonical_dims,
                      GetDotCanonicalDims(dot_dimensions, dot_shape));

  // XNNPACK does not support transposing LHS or col-major layouts.
  return dot_canonical_dims.lhs_canonical &&
         !dot_canonical_dims.lhs_column_major &&
         !dot_canonical_dims.rhs_column_major;
}

absl::StatusOr<std::unique_ptr<XnnDotThunk>> XnnDotThunk::Create(
    Info info, DotDimensionNumbers dot_dimensions,
    BufferAllocation::Slice lhs_buffer, Shape lhs_shape,
    BufferAllocation::Slice rhs_buffer, Shape rhs_shape,
    BufferAllocation::Slice out_buffer, Shape out_shape) {
  TF_RETURN_IF_ERROR(InitializeXnnPack());

  TF_ASSIGN_OR_RETURN(DotShape dot_shape, GetDotShape(dot_dimensions, lhs_shape,
                                                      rhs_shape, out_shape));

  TF_ASSIGN_OR_RETURN(DotCanonicalDims dot_canonical_dims,
                      GetDotCanonicalDims(dot_dimensions, dot_shape));

  DotSlices dot_slices{lhs_buffer, std::move(lhs_shape),
                       rhs_buffer, std::move(rhs_shape),
                       out_buffer, std::move(out_shape)};

  return absl::WrapUnique(
      new XnnDotThunk(info, std::move(dot_dimensions), std::move(dot_slices),
                      std::move(dot_shape), std::move(dot_canonical_dims)));
}

static std::vector<XnnFusionThunk::Argument> DotArguments(
    const DotSlices& slices) {
  return {XnnFusionThunk::Argument{slices.lhs_buffer, slices.lhs_shape},
          XnnFusionThunk::Argument{slices.rhs_buffer, slices.rhs_shape}};
}

static std::vector<XnnFusionThunk::Result> DotResults(const DotSlices& slices) {
  return {XnnFusionThunk::Result{slices.out_buffer, slices.out_shape}};
}

XnnDotThunk::XnnDotThunk(Info info, DotDimensionNumbers dot_dimensions,
                         DotSlices dot_slices, DotShape dot_shape,
                         DotCanonicalDims dot_canonical_dims)
    : XnnFusionThunk(std::move(info), DotArguments(dot_slices),
                     DotResults(dot_slices),
                     std::bind(&XnnDotThunk::BuildDotSubgraph, this,
                               std::placeholders::_1, std::placeholders::_2)),
      dot_dimensions_(std::move(dot_dimensions)),
      dot_slices_(std::move(dot_slices)),
      dot_shape_(std::move(dot_shape)),
      dot_canonical_dims_(std::move(dot_canonical_dims)) {}

std::string XnnDotThunk::fusion_kind() const { return "dot"; }

std::string XnnDotThunk::fusion_description() const {
  return absl::StrFormat(
      "lhs_batch_dims=[%s], rhs_batch_dims=[%s], "
      "lhs_contract_dims=[%s], rhs_contract_dims=[%s]",
      absl::StrJoin(dot_dimensions_.lhs_batch_dimensions(), ","),
      absl::StrJoin(dot_dimensions_.rhs_batch_dimensions(), ","),
      absl::StrJoin(dot_dimensions_.lhs_contracting_dimensions(), ","),
      absl::StrJoin(dot_dimensions_.rhs_contracting_dimensions(), ","));
}

std::vector<std::string> XnnDotThunk::fusion_details() const {
  return {
      absl::StrFormat("  matmul shape: batch_size=%d, lhs=%s, rhs=%s, out=%s",
                      dot_shape_.batch_size,
                      dot_shape_.lhs_matmul_shape.ToString(true),
                      dot_shape_.rhs_matmul_shape.ToString(true),
                      dot_shape_.out_matmul_shape.ToString(true)),
      absl::StrFormat("  matmul dims: m=%d, k=%d, n=%d, lhs_column_major=%v, "
                      "lhs_canonical=%v rhs_column_major=%v, rhs_canonical=%v",
                      dot_canonical_dims_.m, dot_canonical_dims_.k,
                      dot_canonical_dims_.n,
                      dot_canonical_dims_.lhs_column_major,
                      dot_canonical_dims_.lhs_canonical,
                      dot_canonical_dims_.rhs_column_major,
                      dot_canonical_dims_.rhs_canonical),
  };
}

std::string XnnDotThunk::argument_name(size_t index) const {
  return index == 0 ? "lhs" : "rhs";
}

std::string XnnDotThunk::result_name(size_t index) const { return "out"; }

}  // namespace xla::cpu
