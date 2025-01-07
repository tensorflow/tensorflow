/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/service/gpu/runtime/triangular_solve_thunk.h"

#include <complex>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/make_batch_pointers.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace gpu {

TriangularSolveThunk::TriangularSolveThunk(
    ThunkInfo thunk_info, const TriangularSolveOptions& options,
    const BufferAllocation::Slice& a_buffer,
    const BufferAllocation::Slice& b_buffer,
    const BufferAllocation::Slice& temp_buffer,  //
    PrimitiveType type, int64_t batch_size, int64_t m, int64_t n,
    int64_t a_batch_stride, int64_t b_batch_stride)
    : Thunk(Kind::kTriangularSolve, thunk_info),
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

absl::Status TriangularSolveThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  auto& buffer_allocations = *params.buffer_allocations;
  return RunTriangularSolve(buffer_allocations.GetDeviceAddress(a_buffer_),
                            buffer_allocations.GetDeviceAddress(b_buffer_),
                            buffer_allocations.GetDeviceAddress(temp_buffer_),
                            uplo_, side_, unit_diagonal_, transpose_a_, type_,
                            batch_size_, m_, n_, a_batch_stride_,
                            b_batch_stride_, params.stream);
}

absl::Status RunTriangularSolve(se::DeviceMemoryBase a_data,
                                se::DeviceMemoryBase b_data,
                                se::DeviceMemoryBase temp_data,
                                se::blas::UpperLower uplo, se::blas::Side side,
                                se::blas::Diagonal unit_diagonal,
                                se::blas::Transpose transpose_a,
                                PrimitiveType type, int64_t batch_size,
                                int64_t m, int64_t n, int64_t a_batch_stride,
                                int64_t b_batch_stride, se::Stream* stream) {
  VLOG(3) << "uplo=" << se::blas::UpperLowerString(uplo)
          << " side=" << se::blas::SideString(side)
          << " diagonal=" << se::blas::DiagonalString(unit_diagonal)
          << " batch_size=" << batch_size << " m=" << m << " n=" << n
          << " a_batch_stride=" << a_batch_stride
          << " b_batch_stride=" << b_batch_stride;

  const int lda = side == se::blas::Side::kLeft ? m : n;
  const int ldb = m;

  auto blas = stream->parent()->AsBlas();
  if (blas == nullptr) {
    return absl::InternalError("No BLAS support in stream.");
  }
  bool launch_ok;
  if (batch_size == 1) {
    switch (type) {
      case F32: {
        se::DeviceMemory<float> b_data_typed(b_data);
        launch_ok = blas->DoBlasTrsm(
            stream, side, uplo, transpose_a, unit_diagonal, m, n,
            /*alpha=*/1.0f, se::DeviceMemory<float>(a_data), lda, &b_data_typed,
            ldb);
        break;
      }
      case F64: {
        se::DeviceMemory<double> b_data_typed(b_data);
        launch_ok = blas->DoBlasTrsm(
            stream, side, uplo, transpose_a, unit_diagonal, m, n,
            /*alpha=*/1.0, se::DeviceMemory<double>(a_data), lda, &b_data_typed,
            ldb);
        break;
      }
      case C64: {
        se::DeviceMemory<std::complex<float>> b_data_typed(b_data);
        launch_ok = blas->DoBlasTrsm(
            stream, side, uplo, transpose_a, unit_diagonal, m, n,
            /*alpha=*/1.0f, se::DeviceMemory<std::complex<float>>(a_data), lda,
            &b_data_typed, ldb);
        break;
      }
      case C128: {
        se::DeviceMemory<std::complex<double>> b_data_typed(b_data);
        launch_ok = blas->DoBlasTrsm(
            stream, side, uplo, transpose_a, unit_diagonal, m, n,
            /*alpha=*/1.0, se::DeviceMemory<std::complex<double>>(a_data), lda,
            &b_data_typed, ldb);
        break;
      }
      default:
        return InvalidArgument("Invalid type for triangular solve %d", type);
    }
  } else {
    // cublas trsmBatched requires us to materialize out two arrays of
    // batch_size_ pointers, pointing to the individual `a` and `b` matrices of
    // our input.  batch_pointers_bytes is the size in bytes of one of these
    // arrays.
    int64_t batch_pointers_bytes = sizeof(void*) * batch_size;
    TF_RET_CHECK(temp_data.size() >= 2 * batch_pointers_bytes);
    void** temp_base = reinterpret_cast<void**>(temp_data.opaque());
    se::DeviceMemoryBase a_pointers(temp_base, batch_pointers_bytes);
    se::DeviceMemoryBase b_pointers(temp_base + batch_size,
                                    batch_pointers_bytes);

    TF_RETURN_IF_ERROR(MakeBatchPointers(stream, a_data, a_batch_stride,
                                         batch_size, a_pointers));
    TF_RETURN_IF_ERROR(MakeBatchPointers(stream, b_data, b_batch_stride,
                                         batch_size, b_pointers));

    switch (type) {
      case F32: {
        se::DeviceMemory<float*> typed_b_pointers(b_pointers);
        launch_ok = blas->DoBlasTrsmBatched(
            stream, side, uplo, transpose_a, unit_diagonal, m, n,
            /*alpha=*/1.0f, se::DeviceMemory<float*>(a_pointers), lda,
            &typed_b_pointers, ldb, batch_size);
        break;
      }
      case F64: {
        se::DeviceMemory<double*> typed_b_pointers(b_pointers);
        launch_ok = blas->DoBlasTrsmBatched(
            stream, side, uplo, transpose_a, unit_diagonal, m, n,
            /*alpha=*/1.0f, se::DeviceMemory<double*>(a_pointers), lda,
            &typed_b_pointers, ldb, batch_size);
        break;
      }
      case C64: {
        se::DeviceMemory<std::complex<float>*> typed_b_pointers(b_pointers);
        launch_ok = blas->DoBlasTrsmBatched(
            stream, side, uplo, transpose_a, unit_diagonal, m, n,
            /*alpha=*/1.0f, se::DeviceMemory<std::complex<float>*>(a_pointers),
            lda, &typed_b_pointers, ldb, batch_size);
        break;
      }
      case C128: {
        se::DeviceMemory<std::complex<double>*> typed_b_pointers(b_pointers);
        launch_ok = blas->DoBlasTrsmBatched(
            stream, side, uplo, transpose_a, unit_diagonal, m, n,
            /*alpha=*/1.0f, se::DeviceMemory<std::complex<double>*>(a_pointers),
            lda, &typed_b_pointers, ldb, batch_size);
        break;
      }
      default:
        return InvalidArgument("Invalid type for triangular solve %d", type);
    }
  }

  if (!launch_ok) {
    return Internal("Unable to launch triangular solve");
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
