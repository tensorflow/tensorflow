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
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/primitive_util.h"
#include "xla/service/algorithm_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.pb.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
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
    case PrimitiveType::F8E4M3:
      return DataType::kF8E4M3;
    case PrimitiveType::F8E4M3FN:
      return DataType::kF8E4M3FN;
    case PrimitiveType::F8E5M2FNUZ:
      return DataType::kF8E5M2FNUZ;
    case PrimitiveType::F8E4M3FNUZ:
      return DataType::kF8E4M3FNUZ;
    case PrimitiveType::F8E3M4:
      return DataType::kF8E3M4;
    case PrimitiveType::F4E2M1FN:
      return DataType::kF4E2M1FN;
    case PrimitiveType::F8E8M0FNU:
      return DataType::kF8E8M0FNU;
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
    case DataType::kF8E4M3:
      return PrimitiveType::F8E4M3;
    case DataType::kF8E4M3FN:
      return PrimitiveType::F8E4M3FN;
    case DataType::kF8E5M2FNUZ:
      return PrimitiveType::F8E5M2FNUZ;
    case DataType::kF8E4M3FNUZ:
      return PrimitiveType::F8E4M3FNUZ;
    case DataType::kF8E3M4:
      return PrimitiveType::F8E3M4;
    case DataType::kF4E2M1FN:
      return PrimitiveType::F4E2M1FN;
    case DataType::kF8E8M0FNU:
      return PrimitiveType::F8E8M0FNU;
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

absl::StatusOr<MatrixLayout> MatrixLayout::FromProto(
    const xla::GemmConfigProto::MatrixLayout& proto) {
  Order order;
  switch (proto.order()) {
    case xla::GemmConfigProto::MatrixLayout::ORDER_ROW_MAJOR:
      order = Order::kRowMajor;
      break;
    case xla::GemmConfigProto::MatrixLayout::ORDER_COLUMN_MAJOR:
      order = Order::kColumnMajor;
      break;
    case xla::GemmConfigProto::MatrixLayout::ORDER_UNKNOWN:
    default:
      return absl::InvalidArgumentError("Invalid matrix layout order");
  }

  TF_ASSIGN_OR_RETURN(blas::Transpose transpose,
                      blas::FromProto(proto.transpose()));
  return MatrixLayout(proto.dtype(), proto.num_rows(), proto.num_cols(), order,
                      proto.batch_size(), proto.leading_dim_stride(),
                      proto.batch_stride(), transpose);
}

xla::GemmConfigProto::MatrixLayout MatrixLayout::ToProto() const {
  xla::GemmConfigProto::MatrixLayout proto;
  switch (order) {
    case Order::kRowMajor:
      proto.set_order(xla::GemmConfigProto::MatrixLayout::ORDER_ROW_MAJOR);
      break;
    case Order::kColumnMajor:
      proto.set_order(xla::GemmConfigProto::MatrixLayout::ORDER_COLUMN_MAJOR);
      break;
    default: {
      LOG(FATAL) << "Invalid matrix layout order";
    }
  }
  proto.set_num_rows(num_rows);
  proto.set_num_cols(num_cols);
  proto.set_batch_size(batch_size);
  proto.set_leading_dim_stride(leading_dim_stride);
  proto.set_batch_stride(batch_stride);
  proto.set_transpose(blas::ToProto(transpose));
  proto.set_dtype(dtype);
  return proto;
}

absl::StatusOr<ComputationType> GetBlasComputationType(
    xla::PrecisionConfig::Algorithm algorithm, xla::PrimitiveType lhs_dtype,
    xla::PrimitiveType output_dtype, int64_t compute_precision) {
  if (algorithm == xla::PrecisionConfig::ALG_UNSET) {
    switch (output_dtype) {
      case PrimitiveType::F8E5M2:      // fall-through
      case PrimitiveType::F8E4M3:      // fall-through
      case PrimitiveType::F8E4M3FN:    // fall-through
      case PrimitiveType::F8E5M2FNUZ:  // fall-through
      case PrimitiveType::F8E4M3FNUZ:  // fall-through
      case PrimitiveType::F8E3M4:      // fall-through
      case PrimitiveType::F4E2M1FN:    // fall-through
      case PrimitiveType::F8E8M0FNU:   // fall-through
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

absl::StatusOr<BlasLt::MatmulPlan*> BlasLt::GetOrCreateMatmulPlan(
    const std::string& key, PlanCreateFunc create) {
  absl::MutexLock lock(&plan_cache_mu_);  // double mutex ???
  auto res = plan_cache_.emplace(key, MatmulPlanPtr{});
  // New entry inserted: always create a new matmul plan if key is empty,
  // this is used by command_buffer_thunk test.
  if (res.second || key.empty()) {
    VLOG(2) << "Creating a plan for: " << key;
    TF_ASSIGN_OR_RETURN(res.first->second, create());
    VLOG(2) << "Plan created: cache size: " << plan_cache_.size();
  }
  return res.first->second.get();
}

void BlasLt::ClearMatmulPlanCache() {
  absl::MutexLock lock(&plan_cache_mu_);
  plan_cache_.clear();
}

size_t BlasLt::GetMatmulPlanCacheSize() const {
  absl::MutexLock lock(&plan_cache_mu_);
  return plan_cache_.size();
}

absl::StatusOr<GemmConfig> GemmConfig::FromProto(
    const xla::GemmConfigProto& proto) {
  TF_ASSIGN_OR_RETURN(MatrixLayout lhs_layout,
                      MatrixLayout::FromProto(proto.lhs_layout()));
  TF_ASSIGN_OR_RETURN(MatrixLayout rhs_layout,
                      MatrixLayout::FromProto(proto.rhs_layout()));
  TF_ASSIGN_OR_RETURN(MatrixLayout c_layout,
                      MatrixLayout::FromProto(proto.c_layout()));
  TF_ASSIGN_OR_RETURN(MatrixLayout output_layout,
                      MatrixLayout::FromProto(proto.output_layout()));
  std::optional<blas::ComputationType> compute_type =
      blas::FromProto(proto.compute_type());
  return GemmConfig{
      std::move(lhs_layout),
      std::move(rhs_layout),
      std::move(c_layout),
      std::move(output_layout),
      {proto.alpha_real(), proto.alpha_imag()},
      proto.beta(),
      proto.compute_precision(),
      proto.precision_algorithm(),
      proto.has_algorithm() ? std::optional(proto.algorithm()) : std::nullopt,
      proto.grad_x(),
      proto.grad_y(),
      compute_type};
}

xla::GemmConfigProto GemmConfig::ToProto() const {
  xla::GemmConfigProto proto;
  *proto.mutable_lhs_layout() = lhs_layout.ToProto();
  *proto.mutable_rhs_layout() = rhs_layout.ToProto();
  *proto.mutable_c_layout() = c_layout.ToProto();
  *proto.mutable_output_layout() = output_layout.ToProto();
  proto.set_alpha_real(alpha.real());
  proto.set_alpha_imag(alpha.imag());
  proto.set_beta(beta);
  proto.set_compute_precision(compute_precision);
  proto.set_precision_algorithm(precision_algorithm);
  if (algorithm.has_value()) {
    proto.set_algorithm(*algorithm);
  }
  proto.set_grad_x(grad_x);
  proto.set_grad_y(grad_y);
  if (compute_type.has_value()) {
    proto.set_compute_type(blas::ToProto(*compute_type));
  }
  return proto;
}

}  // namespace gpu
}  // namespace stream_executor
