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

#include "xla/service/cpu/runtime/dot_thunk.h"

#include <atomic>
#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {
namespace {

// Dot operation is implemented as a matrix-matrix multiply (row-major x
// rowm-major or col-major x col-major). For batched dot operations, it is
// implemented as multiple matrix multiplications repeated for each batch
// element.
//
// We rely on col-major Eigen contraction and figure out how to represent dot
// operation as a contraction based on the dot dimension numbers.
struct MatMulDims {
  // The number of rows in the LHS.
  int64_t m;

  // The number of columns in the LHS, which also must be equal to the
  // number of rows in the RHS.
  int64_t k;

  // The number of columns in the RHS.
  int64_t n;

  // True if the LHS matrix is column major.
  bool lhs_column_major;

  // True if the LHS contraction dimension is 1.
  bool lhs_canonical;

  // True if the RHS matrix is column major.
  bool rhs_column_major;

  // True if the RHS contraction dimension is 0.
  bool rhs_canonical;
};

// Keep track of pending matrix multiplications and an event that signals
// completion of Dot operation to the caller.
struct ExecuteState {
  explicit ExecuteState(int64_t batch_size)
      : pending_matmuls(batch_size),
        event(tsl::MakeConstructedAsyncValueRef<Thunk::ExecuteEvent>()) {}

  void Notify() {
    if (pending_matmuls.load(std::memory_order_relaxed) == 1 ||
        pending_matmuls.fetch_sub(1, std::memory_order_relaxed) == 1) {
      event.SetStateConcrete();
    }
  }

  std::atomic<int64_t> pending_matmuls;
  tsl::AsyncValueRef<Thunk::ExecuteEvent> event;
};

}  // namespace

static MatMulDims GetMatMulDims(
    const Shape& lhs_shape, absl::Span<const int64_t> lhs_contracting_dims,
    const Shape& rhs_shape, absl::Span<const int64_t> rhs_contracting_dims) {
  // Non-contracting dots should never make it here.
  CHECK_EQ(lhs_contracting_dims.size(), 1);
  CHECK_EQ(rhs_contracting_dims.size(), 1);
  CHECK_LT(lhs_contracting_dims[0], 2);
  CHECK_LT(rhs_contracting_dims[0], 2);

  auto is_column_major = [](const Shape& shape) {
    return shape.rank() > 1 && LayoutUtil::Minor(shape.layout(), 0) == 0;
  };

  return MatMulDims{
      /*m=*/lhs_shape.rank() <= 1
          ? 1LL
          : lhs_shape.dimensions(1LL - lhs_contracting_dims[0]),
      /*k=*/lhs_shape.dimensions(lhs_contracting_dims[0]),
      /*n=*/rhs_shape.rank() <= 1
          ? 1LL
          : rhs_shape.dimensions(1LL - rhs_contracting_dims[0]),
      /*lhs_column_major=*/is_column_major(lhs_shape),
      /*lhs_canonical=*/lhs_shape.rank() <= 1 || lhs_contracting_dims[0] == 1,
      /*rhs_column_major=*/is_column_major(rhs_shape),
      /*rhs_canonical=*/rhs_contracting_dims[0] == 0};
}

absl::StatusOr<std::unique_ptr<DotThunk>> DotThunk::Create(
    Info info, DotDimensionNumbers dot_dimensions,
    BufferAllocation::Slice lhs_buffer, Shape lhs_shape,
    BufferAllocation::Slice rhs_buffer, Shape rhs_shape,
    BufferAllocation::Slice out_buffer, Shape out_shape) {
  // All shapes must be in dim0-major layout.
  if (!LayoutUtil::IsMonotonicWithDim0Major(lhs_shape.layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(rhs_shape.layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(out_shape.layout())) {
    return InvalidArgument(
        "DotThunk requires all operands and outputs to be in "
        "dim0-major layout: lhs_shape=[%s], rhs_shape=[%s], out_shape=[%s]",
        lhs_shape.ToString(true), rhs_shape.ToString(true),
        out_shape.ToString(true));
  }

  // Batch dimensions must be contiguous and start at 0.
  std::vector<int64_t> batch_dims(dot_dimensions.lhs_batch_dimensions().size());
  absl::c_iota(batch_dims, 0);

  if (!absl::c_equal(dot_dimensions.lhs_batch_dimensions(), batch_dims) ||
      !absl::c_equal(dot_dimensions.rhs_batch_dimensions(), batch_dims)) {
    return InvalidArgument(
        "Batch dimensions must be contiguous and start at 0: "
        "lhs_batch_dims=[%s], rhs_batch_dims=[%s]",
        absl::StrJoin(dot_dimensions.lhs_batch_dimensions(), ","),
        absl::StrJoin(dot_dimensions.rhs_batch_dimensions(), ","));
  }

  int64_t num_batch_dims = batch_dims.size();
  int64_t batch_size =
      std::accumulate(out_shape.dimensions().begin(),
                      out_shape.dimensions().begin() + num_batch_dims, 1LL,
                      std::multiplies<int64_t>());

  Shape lhs_matmul_shape = ShapeUtil::DeleteDimensions(batch_dims, lhs_shape);
  Shape rhs_matmul_shape = ShapeUtil::DeleteDimensions(batch_dims, rhs_shape);
  Shape out_matmul_shape = ShapeUtil::DeleteDimensions(batch_dims, out_shape);

  // Check that matmul shapes are rank 2 or less and can be represented as
  // Eigen 2D contraction.
  if (lhs_matmul_shape.rank() > 2 || rhs_matmul_shape.rank() > 2 ||
      out_matmul_shape.rank() > 2) {
    return InvalidArgument(
        "MatMul shape must be rank 2 or less: lhs=%s, rhs=%s, out=%s",
        lhs_matmul_shape.ToString(true), rhs_matmul_shape.ToString(true),
        out_matmul_shape.ToString(true));
  }

  return absl::WrapUnique(new DotThunk(
      info, std::move(dot_dimensions), lhs_buffer, std::move(lhs_shape),
      rhs_buffer, std::move(rhs_shape), out_buffer, std::move(out_shape),
      batch_size, std::move(lhs_matmul_shape), std::move(rhs_matmul_shape),
      std::move(out_matmul_shape)));
}

DotThunk::DotThunk(Info info, DotDimensionNumbers dot_dimensions,
                   BufferAllocation::Slice lhs_buffer, Shape lhs_shape,
                   BufferAllocation::Slice rhs_buffer, Shape rhs_shape,
                   BufferAllocation::Slice out_buffer, Shape out_shape,
                   int64_t batch_size, Shape lhs_matmul_shape,
                   Shape rhs_matmul_shape, Shape out_matmul_shape)
    : Thunk(Kind::kDot, info),
      dot_dimensions_(dot_dimensions),
      lhs_buffer_(lhs_buffer),
      lhs_shape_(lhs_shape),
      rhs_buffer_(rhs_buffer),
      rhs_shape_(rhs_shape),
      out_buffer_(out_buffer),
      out_shape_(out_shape),
      batch_size_(batch_size),
      lhs_matmul_shape_(lhs_matmul_shape),
      rhs_matmul_shape_(rhs_matmul_shape),
      out_matmul_shape_(out_matmul_shape) {
  // Copy from the original dot dimension numbers.
  lhs_matmul_contracting_dims_.assign(
      dot_dimensions_.lhs_contracting_dimensions().begin(),
      dot_dimensions_.lhs_contracting_dimensions().end());
  rhs_matmul_contracting_dims_.assign(
      dot_dimensions_.rhs_contracting_dimensions().begin(),
      dot_dimensions_.rhs_contracting_dimensions().end());

  // Adjust contracting dimensions for leading batch dimensions.
  for (int64_t& dim : lhs_matmul_contracting_dims_)
    dim -= dot_dimensions_.lhs_batch_dimensions_size();
  for (int64_t& dim : rhs_matmul_contracting_dims_)
    dim -= dot_dimensions_.rhs_batch_dimensions_size();
}

tsl::AsyncValueRef<DotThunk::ExecuteEvent> DotThunk::Execute(
    const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase lhs_data,
                      params.buffer_allocations->GetDeviceAddress(lhs_buffer_));

  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase rhs_data,
                      params.buffer_allocations->GetDeviceAddress(rhs_buffer_));

  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase out_data,
                      params.buffer_allocations->GetDeviceAddress(out_buffer_));

  VLOG(3) << absl::StreamFormat(
      "Dot operation: lhs_batch_dims=[%s], rhs_batch_dims=[%s], "
      "lhs_contract_dims=[%s], rhs_contract_dims=[%s]",
      absl::StrJoin(dot_dimensions_.lhs_batch_dimensions(), ","),
      absl::StrJoin(dot_dimensions_.rhs_batch_dimensions(), ","),
      absl::StrJoin(dot_dimensions_.lhs_contracting_dimensions(), ","),
      absl::StrJoin(dot_dimensions_.rhs_contracting_dimensions(), ","));

  VLOG(3) << absl::StreamFormat("  lhs: %s in slice %s (%p)",
                                lhs_shape_.ToString(true),
                                lhs_buffer_.ToString(), lhs_data.opaque());
  VLOG(3) << absl::StreamFormat("  rhs: %s in slice %s (%p)",
                                rhs_shape_.ToString(true),
                                rhs_buffer_.ToString(), rhs_data.opaque());
  VLOG(3) << absl::StreamFormat("  out: %s in slice %s (%p)",
                                out_shape_.ToString(true),
                                out_buffer_.ToString(), out_data.opaque());

  VLOG(3) << absl::StreamFormat(
      "  matmul shape: batch_size=%d, lhs=%s, rhs=%s, out=%s", batch_size_,
      lhs_matmul_shape_.ToString(true), rhs_matmul_shape_.ToString(true),
      out_matmul_shape_.ToString(true));

  MatMulDims matmul_dims =
      GetMatMulDims(lhs_matmul_shape_, lhs_matmul_contracting_dims_,
                    rhs_matmul_shape_, rhs_matmul_contracting_dims_);

  VLOG(3) << absl::StreamFormat(
      "  matmul dims: m=%d, k=%d, n=%d, lhs_column_major=%v, lhs_canonical=%v, "
      "rhs_column_major=%v, rhs_canonical=%v",
      matmul_dims.m, matmul_dims.k, matmul_dims.n, matmul_dims.lhs_column_major,
      matmul_dims.lhs_canonical, matmul_dims.rhs_column_major,
      matmul_dims.rhs_canonical);

  if (params.intra_op_threadpool == nullptr) {
    return InvalidArgument("Intra-op threadpool must be provided for DotThunk");
  }

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

  bool transpose_lhs = !matmul_dims.lhs_canonical;
  bool transpose_rhs = !matmul_dims.rhs_canonical;

  CHECK_EQ(matmul_dims.lhs_column_major, matmul_dims.rhs_column_major);
  if (!matmul_dims.lhs_column_major) {
    std::swap(matmul_dims.m, matmul_dims.n);
    std::swap(lhs, rhs);
    std::swap(transpose_lhs, transpose_rhs);
  }

  PrimitiveType element_type = lhs_matmul_shape_.element_type();
  int64_t byte_width = primitive_util::ByteWidth(element_type);

  int64_t lhs_stride = matmul_dims.m * matmul_dims.k * byte_width;
  int64_t rhs_stride = matmul_dims.k * matmul_dims.n * byte_width;
  int64_t out_stride = matmul_dims.m * matmul_dims.n * byte_width;

  auto batch_ptr = [&](void* ptr, int64_t stride, int64_t index) -> void* {
    return static_cast<uint8_t*>(ptr) + stride * index;
  };

  auto state = std::make_shared<ExecuteState>(batch_size_);

  auto dispatch = [&](auto type_tag) {
    for (int64_t i = 0; i < batch_size_; ++i) {
      TypedMatMul<decltype(type_tag)>(
          params.intra_op_threadpool, batch_ptr(out, out_stride, i),
          batch_ptr(lhs, lhs_stride, i), batch_ptr(rhs, rhs_stride, i),
          matmul_dims.m, matmul_dims.n, matmul_dims.k, transpose_lhs,
          transpose_rhs, [state] { state->Notify(); });
    }
  };

  switch (element_type) {
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

  return state->event;
}

}  // namespace xla::cpu
