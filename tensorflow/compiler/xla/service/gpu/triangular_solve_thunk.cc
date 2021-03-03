/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/triangular_solve_thunk.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/device_memory.h"

namespace xla {
namespace gpu {

TriangularSolveThunk::TriangularSolveThunk(
    ThunkInfo thunk_info, const TriangularSolveOptions& options,
    const BufferAllocation::Slice& a_buffer,
    const BufferAllocation::Slice& b_buffer, PrimitiveType type,
    int64 batch_size, int64 m, int64 n, int64 a_batch_stride,
    int64 b_batch_stride)
    : Thunk(Kind::kTriangularSolve, thunk_info),
      uplo_(options.lower() ? se::blas::UpperLower::kLower
                            : se::blas::UpperLower::kUpper),
      side_(options.left_side() ? se::blas::Side::kLeft
                                : se::blas::Side::kRight),
      unit_diagonal_(options.unit_diagonal() ? se::blas::Diagonal::kUnit
                                             : se::blas::Diagonal::kNonUnit),
      a_buffer_(a_buffer),
      b_buffer_(b_buffer),
      type_(type),
      batch_size_(batch_size),
      m_(m),
      n_(n),
      a_batch_stride_(a_batch_stride),
      b_batch_stride_(b_batch_stride) {
  transpose_a_ = [&] {
    switch (options.transpose_a()) {
      case TriangularSolveOptions::NO_TRANSPOSE:
        return se::blas::Transpose::kNoTranspose;
      case TriangularSolveOptions::TRANSPOSE:
        return se::blas::Transpose::kTranspose;
      case TriangularSolveOptions::ADJOINT:
        return se::blas::Transpose::kConjugateTranspose;
      default:
        LOG(ERROR) << "Invalid triangular solve transpose value "
                   << options.transpose_a();
        return se::blas::Transpose::kNoTranspose;
    }
  }();
}

Status TriangularSolveThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto& stream = *params.stream;
  auto& buffer_allocations = *params.buffer_allocations;

  VLOG(3) << "uplo=" << se::blas::UpperLowerString(uplo_)
          << " side=" << se::blas::SideString(side_)
          << " diagonal=" << se::blas::DiagonalString(unit_diagonal_)
          << " batch_size=" << batch_size_ << " m=" << m_ << " n=" << n_
          << " a_batch_stride=" << a_batch_stride_
          << " b_batch_stride=" << b_batch_stride_;

  const int lda = side_ == se::blas::Side::kLeft ? m_ : n_;
  const int ldb = m_;

  char* a_base = static_cast<char*>(
      buffer_allocations.GetDeviceAddress(a_buffer_).opaque());
  char* b_base = static_cast<char*>(
      buffer_allocations.GetDeviceAddress(b_buffer_).opaque());
  for (int64 i = 0; i < batch_size_; ++i) {
    bool launch_ok;
    se::DeviceMemoryBase a_data =
        se::DeviceMemoryBase(a_base + i * a_batch_stride_, a_batch_stride_);
    se::DeviceMemoryBase b_data =
        se::DeviceMemoryBase(b_base + i * b_batch_stride_, b_batch_stride_);
    switch (type_) {
      case F32: {
        se::DeviceMemory<float> b_data_typed(b_data);
        launch_ok = stream
                        .ThenBlasTrsm(side_, uplo_, transpose_a_,
                                      unit_diagonal_, m_, n_, /*alpha=*/1.0f,
                                      se::DeviceMemory<float>(a_data), lda,
                                      &b_data_typed, ldb)
                        .ok();
        break;
      }
      case F64: {
        se::DeviceMemory<double> b_data_typed(b_data);
        launch_ok = stream
                        .ThenBlasTrsm(side_, uplo_, transpose_a_,
                                      unit_diagonal_, m_, n_, /*alpha=*/1.0,
                                      se::DeviceMemory<double>(a_data), lda,
                                      &b_data_typed, ldb)
                        .ok();
        break;
      }
      case C64: {
        se::DeviceMemory<std::complex<float>> b_data_typed(b_data);
        launch_ok =
            stream
                .ThenBlasTrsm(side_, uplo_, transpose_a_, unit_diagonal_, m_,
                              n_, /*alpha=*/1.0f,
                              se::DeviceMemory<std::complex<float>>(a_data),
                              lda, &b_data_typed, ldb)
                .ok();
        break;
      }
      case C128: {
        se::DeviceMemory<std::complex<double>> b_data_typed(b_data);
        launch_ok =
            stream
                .ThenBlasTrsm(side_, uplo_, transpose_a_, unit_diagonal_, m_,
                              n_, /*alpha=*/1.0,
                              se::DeviceMemory<std::complex<double>>(a_data),
                              lda, &b_data_typed, ldb)
                .ok();
        break;
      }
      default:
        return InvalidArgument("Invalid type for triangular solve %d", type_);
    }
    if (!launch_ok) {
      return InternalError("Unable to launch triangular solve for thunk %p",
                           this);
    }
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
