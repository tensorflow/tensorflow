/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/stream_executor/gpu/gpu_blas_lt.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/primitive_util.h"
#include "xla/service/algorithm_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/protobuf/dnn.pb.h"
#if GOOGLE_CUDA
#include "tsl/platform/tensor_float_32_utils.h"
#endif

namespace stream_executor {

namespace gpu {

using blas::ComputationType;
using blas::DataType;
using xla::PrimitiveType;

absl::StatusOr<DataType> AsBlasDataType(PrimitiveType dtype) {
  switch (dtype) {
    case PrimitiveType::F8E5M2:
      return DataType::kF8E5M2;
    case PrimitiveType::F8E4M3FN:
      return DataType::kF8E4M3FN;
    case PrimitiveType::F8E5M2FNUZ:
      return DataType::kF8E5M2FNUZ;
    case PrimitiveType::F8E4M3FNUZ:
      return DataType::kF8E4M3FNUZ;
    case PrimitiveType::S8:
      return DataType::kInt8;
    case PrimitiveType::F16:
      return DataType::kHalf;
    case PrimitiveType::BF16:
      return DataType::kBF16;
    case PrimitiveType::F32:
      return DataType::kFloat;
    case PrimitiveType::S32:
      return DataType::kInt32;
    case PrimitiveType::F64:
      return DataType::kDouble;
    case PrimitiveType::C64:
      return DataType::kComplexFloat;
    case PrimitiveType::C128:
      return DataType::kComplexDouble;
    default:
      return xla::Internal(
          "AsBlasDataType: unsupported type: %s",
          xla::primitive_util::LowercasePrimitiveTypeName(dtype));
  }
}

absl::StatusOr<PrimitiveType> AsXlaPrimitiveType(DataType dtype) {
  switch (dtype) {
    case DataType::kF8E5M2:
      return PrimitiveType::F8E5M2;
    case DataType::kF8E4M3FN:
      return PrimitiveType::F8E4M3FN;
    case DataType::kF8E5M2FNUZ:
      return PrimitiveType::F8E5M2FNUZ;
    case DataType::kF8E4M3FNUZ:
      return PrimitiveType::F8E4M3FNUZ;
    case DataType::kInt8:
      return PrimitiveType::S8;
    case DataType::kHalf:
      return PrimitiveType::F16;
    case DataType::kBF16:
      return PrimitiveType::BF16;
    case DataType::kFloat:
      return PrimitiveType::F32;
    case DataType::kInt32:
      return PrimitiveType::S32;
    case DataType::kDouble:
      return PrimitiveType::F64;
    case DataType::kComplexFloat:
      return PrimitiveType::C64;
    case DataType::kComplexDouble:
      return PrimitiveType::C128;
    default:
      return xla::Internal("AsXlaPrimitiveType: unsupported dtype");
  }
}

MatrixLayout::MatrixLayout(xla::PrimitiveType dtype_, int64_t num_rows_,
                           int64_t num_cols_, MatrixLayout::Order order_,
                           int64_t batch_size_,
                           std::optional<int64_t> leading_dim_stride_,
                           std::optional<int64_t> batch_stride_,
                           std::optional<blas::Transpose> transpose_)
    : dtype(dtype_),
      num_rows(num_rows_),
      num_cols(num_cols_),
      order(order_),
      batch_size(batch_size_) {
  if (!leading_dim_stride_) {
    leading_dim_stride = order == Order::kRowMajor ? num_cols : num_rows;
  } else {
    leading_dim_stride = *leading_dim_stride_;
  }
  if (!batch_stride_) {
    batch_stride = (batch_size > 1) ? num_rows * num_cols : 0;
  } else {
    batch_stride = *batch_stride_;
  }
  transpose = transpose_ ? *transpose_ : blas::Transpose::kNoTranspose;
}

void MatrixLayout::Transpose() {
  std::swap(num_rows, num_cols);
  order = (order == Order::kRowMajor) ? Order::kColumnMajor : Order::kRowMajor;
}

absl::StatusOr<ComputationType> GetBlasComputationType(
    xla::PrecisionConfig::Algorithm algorithm, xla::PrimitiveType lhs_dtype,
    xla::PrimitiveType output_dtype, int64_t compute_precision) {
  if (algorithm == xla::PrecisionConfig::ALG_UNSET) {
    switch (output_dtype) {
      case PrimitiveType::F8E5M2:      // fall-through
      case PrimitiveType::F8E4M3FN:    // fall-through
      case PrimitiveType::F8E5M2FNUZ:  // fall-through
      case PrimitiveType::F8E4M3FNUZ:  // fall-through
      case PrimitiveType::F16:         // fall-through
      case PrimitiveType::BF16:
        // Accumulate in f32 precision.
        return ComputationType::kF32;
      case PrimitiveType::F32:  // fall-through
      case PrimitiveType::C64:
#if GOOGLE_CUDA
        if (tsl::tensor_float_32_execution_enabled() &&
            compute_precision <= 1 && lhs_dtype == output_dtype) {
          // CublasLt requires compute type to be F32 for F8 matmul.
          // TF32 should only be chosen for FP32 or C64 gemm
          return ComputationType::kTF32AsF32;
        }
#endif
        return ComputationType::kF32;
      case PrimitiveType::F64:  // fall-through
      case PrimitiveType::C128:
        return ComputationType::kF64;
      case PrimitiveType::S32:
        return ComputationType::kI32;
      default:
        return xla::Internal("GetBlasComputationType: unsupported type");
    }
  }

  return xla::algorithm_util::GetBlasComputationType(algorithm);
}

// BLAS GeMM's output is column-major. If we require row-major, use identity:
// C^T = (A @ B)^T = B^T @ A^T.
bool MakeOutputColumnMajor(MatrixLayout& lhs, MatrixLayout& rhs,
                           MatrixLayout& output, MatrixLayout* c) {
  bool swap_operands = output.order != MatrixLayout::Order::kColumnMajor;
  if (swap_operands) {
    std::swap(lhs, rhs);
    rhs.Transpose();
    // prevent layouts from being swapped two times if they are equal
    if (&lhs != &rhs) {
      lhs.Transpose();
    }
    if (c != nullptr && c != &output) {
      c->Transpose();
    }
    output.Transpose();
  }
  return swap_operands;
}

/*static*/ auto BlasLt::GetMatmulPlan(const Stream* stream,
                                      const GemmConfig& cfg, Epilogue epilogue)
    -> absl::StatusOr<MatmulPlanPtr> {
  auto blas = Get(stream);
  if (blas == nullptr) {
    return xla::Internal("BlasLt is unavailable");
  }
  return blas->GetMatmulPlan(cfg, epilogue);
}

/*static*/ BlasLt* BlasLt::Get(const Stream* stream) {
  auto blas = stream->parent()->AsBlas();
  return (blas != nullptr ? blas->GetBlasLt() : nullptr);
}

DataType GetScaleType(DataType c_type, ComputationType computation_type) {
  return (computation_type == ComputationType::kF32 &&
                  c_type != DataType::kComplexFloat
              ? DataType::kFloat
              : c_type);
}

}  // namespace gpu

}  // namespace stream_executor
