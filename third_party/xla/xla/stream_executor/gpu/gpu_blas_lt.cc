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

#include "xla/stream_executor/gpu/gpu_blas_lt.h"

#include <cstdint>
#include <utility>

#include "xla/primitive_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"
#if GOOGLE_CUDA
#include "tsl/platform/tensor_float_32_utils.h"
#endif

namespace stream_executor::gpu {

tsl::StatusOr<blas::DataType> AsBlasDataType(xla::PrimitiveType dtype) {
  switch (dtype) {
    case xla::PrimitiveType::F8E5M2:
      return blas::DataType::kF8E5M2;
    case xla::PrimitiveType::F8E4M3FN:
      return blas::DataType::kF8E4M3FN;
    case xla::PrimitiveType::S8:
      return blas::DataType::kInt8;
    case xla::PrimitiveType::F16:
      return blas::DataType::kHalf;
    case xla::PrimitiveType::BF16:
      return blas::DataType::kBF16;
    case xla::PrimitiveType::F32:
      return blas::DataType::kFloat;
    case xla::PrimitiveType::S32:
      return blas::DataType::kInt32;
    case xla::PrimitiveType::F64:
      return blas::DataType::kDouble;
    case xla::PrimitiveType::C64:
      return blas::DataType::kComplexFloat;
    case xla::PrimitiveType::C128:
      return blas::DataType::kComplexDouble;
    default:
      return xla::InternalError(
          "AsBlasDataType: unsupported type: %s",
          xla::primitive_util::LowercasePrimitiveTypeName(dtype));
  }
}

tsl::StatusOr<xla::PrimitiveType> AsXlaPrimitiveType(blas::DataType dtype) {
  switch (dtype) {
    case blas::DataType::kF8E5M2:
      return xla::PrimitiveType::F8E5M2;
    case blas::DataType::kF8E4M3FN:
      return xla::PrimitiveType::F8E4M3FN;
    case blas::DataType::kInt8:
      return xla::PrimitiveType::S8;
    case blas::DataType::kHalf:
      return xla::PrimitiveType::F16;
    case blas::DataType::kBF16:
      return xla::PrimitiveType::BF16;
    case blas::DataType::kFloat:
      return xla::PrimitiveType::F32;
    case blas::DataType::kInt32:
      return xla::PrimitiveType::S32;
    case blas::DataType::kDouble:
      return xla::PrimitiveType::F64;
    case blas::DataType::kComplexFloat:
      return xla::PrimitiveType::C64;
    case blas::DataType::kComplexDouble:
      return xla::PrimitiveType::C128;
    default:
      return xla::InternalError("AsXlaPrimitiveType: unsupported dtype");
  }
}

tsl::StatusOr<blas::ComputationType> GetBlasComputationType(
    xla::PrimitiveType lhs_dtype, xla::PrimitiveType output_dtype,
    int64_t compute_precision) {
  switch (output_dtype) {
    case xla::PrimitiveType::F8E5M2:    // fall-through
    case xla::PrimitiveType::F8E4M3FN:  // fall-through
    case xla::PrimitiveType::F16:       // fall-through
    case xla::PrimitiveType::BF16:
      // Accumulate in f32 precision.
      return blas::ComputationType::kF32;
    case xla::PrimitiveType::F32:  // fall-through
    case xla::PrimitiveType::C64:
#if GOOGLE_CUDA
      if (tsl::tensor_float_32_execution_enabled() && compute_precision <= 1 &&
          lhs_dtype == output_dtype) {
        // CublasLt requires compute type to be F32 for F8 matmul.
        // TF32 should only be chosen for FP32 or C64 gemm
        return blas::ComputationType::kTF32AsF32;
      }
#endif
      return blas::ComputationType::kF32;
    case xla::PrimitiveType::F64:  // fall-through
    case xla::PrimitiveType::C128:
      return blas::ComputationType::kF64;
    case xla::PrimitiveType::S32:
      return blas::ComputationType::kI32;
    default:
      return xla::InternalError("GetBlasComputationType: unsupported type");
  }
}

// BLAS GeMM's output is column-major. If we require row-major, use identity:
// C^T = (A @ B)^T = B^T @ A^T.
bool MakeOutputColumnMajor(MatrixLayout& lhs, MatrixLayout& rhs,
                           MatrixLayout& output, MatrixLayout* pC) {
  bool swap_operands = output.order != MatrixLayout::Order::kColumnMajor;
  if (swap_operands) {
    std::swap(lhs, rhs);
    rhs.Transpose();
    lhs.Transpose();
    // prevent pC and output from being swapped two times if they are equal!
    if (pC != nullptr && pC != &output) {
      pC->Transpose();
    }
    output.Transpose();
  }
  return swap_operands;
}

/*static*/ auto BlasLt::GetMatmulPlan(const Stream* stream,
                                      const GemmConfig& cfg, Epilogue epilogue)
    -> tsl::StatusOr<MatmulPlanPtr> {
  auto blas = Get(stream);
  if (blas == nullptr) {
    return xla::InternalError("BlasLt is unavailable");
  }
  return blas->GetMatmulPlan(cfg, epilogue);
}

/*static*/ BlasLt* BlasLt::Get(const Stream* stream) {
  auto blas = stream->parent()->AsBlas();
  return (blas != nullptr ? blas->GetBlasLt() : nullptr);
}

blas::DataType GetScaleType(blas::DataType c_type,
                            blas::ComputationType computation_type) {
  return ((computation_type == blas::ComputationType::kF32) &&
          (c_type != blas::DataType::kComplexFloat))
             ? blas::DataType::kFloat
             : c_type;
}

}  // namespace stream_executor::gpu
