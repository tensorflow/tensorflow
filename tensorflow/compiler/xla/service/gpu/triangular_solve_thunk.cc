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
#include "tensorflow/compiler/xla/service/gpu/precompiled_kernels.h"
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
    se::GpuAsmOpts asm_opts,  //
    const BufferAllocation::Slice& a_buffer,
    const BufferAllocation::Slice& b_buffer,
    const BufferAllocation::Slice& temp_buffer,  //
    PrimitiveType type, int64_t batch_size, int64_t m, int64_t n,
    int64_t a_batch_stride, int64_t b_batch_stride)
    : Thunk(Kind::kTriangularSolve, thunk_info),
      asm_opts_(asm_opts),
      uplo_(options.lower() ? se::blas::UpperLower::kLower
                            : se::blas::UpperLower::kUpper),
      side_(options.left_side() ? se::blas::Side::kLeft
                                : se::blas::Side::kRight),
      unit_diagonal_(options.unit_diagonal() ? se::blas::Diagonal::kUnit
                                             : se::blas::Diagonal::kNonUnit),
      a_buffer_(a_buffer),
      b_buffer_(b_buffer),
      temp_buffer_(temp_buffer),
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

  se::DeviceMemoryBase a_data = buffer_allocations.GetDeviceAddress(a_buffer_);
  se::DeviceMemoryBase b_data = buffer_allocations.GetDeviceAddress(b_buffer_);

  bool launch_ok;
  if (batch_size_ == 1) {
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
  } else {
    // cublas trsmBatched requires us to materialize out two arrays of
    // batch_size_ pointers, pointing to the individual `a` and `b` matrices of
    // our input.  batch_pointers_bytes is the size in bytes of one of these
    // arrays.
    int64_t batch_pointers_bytes = sizeof(void*) * batch_size_;
    TF_RET_CHECK(temp_buffer_.size() >= 2 * batch_pointers_bytes);
    void** temp_base = reinterpret_cast<void**>(
        buffer_allocations.GetDeviceAddress(temp_buffer_).opaque());
    se::DeviceMemoryBase a_pointers(temp_base, batch_pointers_bytes);
    se::DeviceMemoryBase b_pointers(temp_base + batch_size_,
                                    batch_pointers_bytes);

    TF_RETURN_IF_ERROR(MakeBatchPointers(
        &stream, asm_opts_, a_data, a_batch_stride_, batch_size_, a_pointers));
    TF_RETURN_IF_ERROR(MakeBatchPointers(
        &stream, asm_opts_, b_data, b_batch_stride_, batch_size_, b_pointers));

    switch (type_) {
      case F32: {
        se::DeviceMemory<float*> typed_b_pointers(b_pointers);
        launch_ok =
            stream
                .ThenBlasTrsmBatched(side_, uplo_, transpose_a_, unit_diagonal_,
                                     m_, n_, /*alpha=*/1.0f,
                                     se::DeviceMemory<float*>(a_pointers), lda,
                                     &typed_b_pointers, ldb, batch_size_)
                .ok();
        break;
      }
      case F64: {
        se::DeviceMemory<double*> typed_b_pointers(b_pointers);
        launch_ok =
            stream
                .ThenBlasTrsmBatched(side_, uplo_, transpose_a_, unit_diagonal_,
                                     m_, n_, /*alpha=*/1.0f,
                                     se::DeviceMemory<double*>(a_pointers), lda,
                                     &typed_b_pointers, ldb, batch_size_)
                .ok();
        break;
      }
      case C64: {
        se::DeviceMemory<std::complex<float>*> typed_b_pointers(b_pointers);
        launch_ok = stream
                        .ThenBlasTrsmBatched(
                            side_, uplo_, transpose_a_, unit_diagonal_, m_, n_,
                            /*alpha=*/1.0f,
                            se::DeviceMemory<std::complex<float>*>(a_pointers),
                            lda, &typed_b_pointers, ldb, batch_size_)
                        .ok();
        break;
      }
      case C128: {
        se::DeviceMemory<std::complex<double>*> typed_b_pointers(b_pointers);
        launch_ok = stream
                        .ThenBlasTrsmBatched(
                            side_, uplo_, transpose_a_, unit_diagonal_, m_, n_,
                            /*alpha=*/1.0f,
                            se::DeviceMemory<std::complex<double>*>(a_pointers),
                            lda, &typed_b_pointers, ldb, batch_size_)
                        .ok();
        break;
      }
      default:
        return InvalidArgument("Invalid type for triangular solve %d", type_);
    }
  }

  if (!launch_ok) {
    return InternalError("Unable to launch triangular solve for thunk %p",
                         this);
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
