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

#include "xla/backends/cpu/runtime/dot_thunk.h"

#include <complex>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xla/backends/cpu/runtime/dot_lib.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<DotThunk>> DotThunk::Create(
    Info info, DotDimensionNumbers dot_dimensions,
    BufferAllocation::Slice lhs_buffer, Shape lhs_shape,
    BufferAllocation::Slice rhs_buffer, Shape rhs_shape,
    BufferAllocation::Slice out_buffer, Shape out_shape) {
  TF_ASSIGN_OR_RETURN(DotShape dot_shape, GetDotShape(dot_dimensions, lhs_shape,
                                                      rhs_shape, out_shape));

  TF_ASSIGN_OR_RETURN(DotCanonicalDims dot_canonical_dims,
                      GetDotCanonicalDims(dot_dimensions, dot_shape));

  DotSlices dot_slices{lhs_buffer, std::move(lhs_shape),
                       rhs_buffer, std::move(rhs_shape),
                       out_buffer, std::move(out_shape)};

  return absl::WrapUnique(
      new DotThunk(info, std::move(dot_dimensions), std::move(dot_slices),
                   std::move(dot_shape), std::move(dot_canonical_dims)));
}

DotThunk::DotThunk(Info info, DotDimensionNumbers dot_dimensions,
                   DotSlices dot_slices, DotShape dot_shape,
                   DotCanonicalDims dot_canonical_dims)
    : Thunk(Kind::kDot, info),
      dot_dimensions_(std::move(dot_dimensions)),
      dot_slices_(std::move(dot_slices)),
      dot_shape_(std::move(dot_shape)),
      dot_canonical_dims_(std::move(dot_canonical_dims)) {}

tsl::AsyncValueRef<DotThunk::ExecuteEvent> DotThunk::Execute(
    const ExecuteParams& params) {

  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase lhs_data,
      params.buffer_allocations->GetDeviceAddress(dot_slices_.lhs_buffer));

  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase rhs_data,
      params.buffer_allocations->GetDeviceAddress(dot_slices_.rhs_buffer));

  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase out_data,
      params.buffer_allocations->GetDeviceAddress(dot_slices_.out_buffer));

  VLOG(3) << absl::StreamFormat(
      "Dot operation: lhs_batch_dims=[%s], rhs_batch_dims=[%s], "
      "lhs_contract_dims=[%s], rhs_contract_dims=[%s]",
      absl::StrJoin(dot_dimensions_.lhs_batch_dimensions(), ","),
      absl::StrJoin(dot_dimensions_.rhs_batch_dimensions(), ","),
      absl::StrJoin(dot_dimensions_.lhs_contracting_dimensions(), ","),
      absl::StrJoin(dot_dimensions_.rhs_contracting_dimensions(), ","));

  VLOG(3) << absl::StreamFormat(
      "  lhs: %s in slice %s (%p)", dot_slices_.lhs_shape.ToString(true),
      dot_slices_.lhs_buffer.ToString(), lhs_data.opaque());
  VLOG(3) << absl::StreamFormat(
      "  rhs: %s in slice %s (%p)", dot_slices_.rhs_shape.ToString(true),
      dot_slices_.rhs_buffer.ToString(), rhs_data.opaque());
  VLOG(3) << absl::StreamFormat(
      "  out: %s in slice %s (%p)", dot_slices_.out_shape.ToString(true),
      dot_slices_.out_buffer.ToString(), out_data.opaque());

  VLOG(3) << absl::StreamFormat(
      "  matmul shape: batch_size=%d, lhs=%s, rhs=%s, out=%s",
      dot_shape_.batch_size, dot_shape_.lhs_matmul_shape.ToString(true),
      dot_shape_.rhs_matmul_shape.ToString(true),
      dot_shape_.out_matmul_shape.ToString(true));

  VLOG(3) << absl::StreamFormat(
      "  matmul dims: m=%d, k=%d, n=%d, lhs_column_major=%v, lhs_canonical=%v, "
      "rhs_column_major=%v, rhs_canonical=%v",
      dot_canonical_dims_.m, dot_canonical_dims_.k, dot_canonical_dims_.n,
      dot_canonical_dims_.lhs_column_major, dot_canonical_dims_.lhs_canonical,
      dot_canonical_dims_.rhs_column_major, dot_canonical_dims_.rhs_canonical);

  // Eigen expects column-major layout. If the matrices are row major, then use
  // the following identity to compute the product:
  //
  //   (A x B)^T = B^T x A^T
  //
  // The connection between this identity and memory layout is that the
  // transpose operation can also be considered as an operation that changes the
  // memory layout of a matrix from row-major to column-major or vice versa.
  //
  // Effectively this involves swapping the 'lhs' with 'rhs' and 'm' with 'n'.

  void* out = out_data.opaque();
  void* lhs = lhs_data.opaque();
  void* rhs = rhs_data.opaque();

  int64_t m = dot_canonical_dims_.m;
  int64_t n = dot_canonical_dims_.n;
  int64_t k = dot_canonical_dims_.k;

  // Decide if a transpose is required based on an XOR of the canonical and
  // column major flags.
  bool transpose_lhs = (dot_canonical_dims_.lhs_canonical !=
                        dot_canonical_dims_.lhs_column_major);
  bool transpose_rhs = (dot_canonical_dims_.rhs_canonical !=
                        dot_canonical_dims_.rhs_column_major);

  if (!dot_canonical_dims_.output_column_major) {
    std::swap(m, n);
    std::swap(lhs, rhs);
    std::swap(transpose_lhs, transpose_rhs);
    transpose_lhs = !transpose_lhs;
    transpose_rhs = !transpose_rhs;
  }

  PrimitiveType element_type = dot_shape_.lhs_matmul_shape.element_type();
  int64_t byte_width = primitive_util::ByteWidth(element_type);

  int64_t lhs_stride = m * k * byte_width;
  int64_t rhs_stride = k * n * byte_width;
  int64_t out_stride = m * n * byte_width;

  auto batch_ptr = [&](void* ptr, int64_t stride, int64_t index) -> void* {
    return static_cast<uint8_t*>(ptr) + stride * index;
  };

  tsl::CountDownAsyncValueRef<ExecuteEvent> state(dot_shape_.batch_size);

  auto dispatch = [&](auto type_tag) {
    for (int64_t i = 0; i < dot_shape_.batch_size; ++i) {
      TypedMatMul<decltype(type_tag)>(
          params.intra_op_threadpool, batch_ptr(out, out_stride, i),
          batch_ptr(lhs, lhs_stride, i), batch_ptr(rhs, rhs_stride, i), m, n, k,
          transpose_lhs, transpose_rhs,
          [state]() mutable { state.CountDown(); });
    }
  };

  switch (element_type) {
    case BF16:
      dispatch(bfloat16{});  // Enable Eigen BF16 kernel for fallback.
      break;
    case F16:
      dispatch(half{});
      break;
    case F32:
      dispatch(float{});
      break;
    case F64:
      dispatch(double{});
      break;
    case S32:
      dispatch(int32_t{});
      break;
    case C64:
      dispatch(std::complex<float>{});
      break;
    case C128:
      dispatch(std::complex<double>{});
      break;
    default:
      return Unimplemented(
          "Unsupported element type for DotThunk::Execute: %s",
          primitive_util::LowercasePrimitiveTypeName(element_type));
  }

  return state.AsRef();
}

}  // namespace xla::cpu
