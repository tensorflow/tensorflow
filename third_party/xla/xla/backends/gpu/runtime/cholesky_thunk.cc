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

#include "xla/backends/gpu/runtime/cholesky_thunk.h"

#include <complex>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/runtime/make_batch_pointers.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu_solver_context.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

template <typename T>
absl::Status DoPotrfBatched(CholeskyParams* params, se::Stream* stream,
                            stream_executor::GpuSolverContext& context) {
  T* a_base = static_cast<T*>(params->a_buffer.opaque());
  se::DeviceMemory<int> infos(params->info_buffer);
#if TENSORFLOW_USE_ROCSOLVER
  // hipsolver is not supported so allocate a GPU buffer
  se::ScopedDeviceMemory<T*> ptrs(
      stream->parent(), stream->parent()->AllocateArray<T*>(batch_size_));
  auto as = *ptrs;
#else
  se::DeviceMemory<T*> as(params->workspace_buffer);
#endif

  CHECK_GE(as.size(), params->batch_size);
  CHECK_GE(infos.size(), params->batch_size);

  // Run a kernel that sets as[i] = &a_base[i * stride].
  const int64_t stride_bytes = params->n * params->n * sizeof(T);
  TF_RETURN_IF_ERROR(MakeBatchPointers(
      stream, se::DeviceMemoryBase(a_base), stride_bytes,
      static_cast<int>(params->batch_size), se::DeviceMemoryBase(as)));

  // Now that we've set up the `as` array, we can call cusolver.
  return context.PotrfBatched(params->uplo, params->n, as, params->n, infos,
                              params->batch_size);
}

template <typename T>
absl::Status DoPotrfUnbatched(CholeskyParams* params, se::Stream* stream,
                              stream_executor::GpuSolverContext& context) {
  T* a_base = static_cast<T*>(params->a_buffer.opaque());
  int* info_base = static_cast<int*>(params->info_buffer.opaque());

  int64_t stride = params->n * params->n;
  for (int64_t i = 0; i < params->batch_size; ++i) {
    se::DeviceMemory<T> a_data(
        se::DeviceMemoryBase(&a_base[i * stride], sizeof(T) * stride));
    se::DeviceMemory<int> info_data(
        se::DeviceMemoryBase(&info_base[i], sizeof(int)));
    se::DeviceMemory<T> workspace_data(params->workspace_buffer);
    TF_RETURN_IF_ERROR(context.Potrf(params->uplo, params->n, a_data, params->n,
                                     info_data, workspace_data));
  }
  return absl::OkStatus();
}

absl::Status RunCholesky(PrimitiveType type, CholeskyParams* cholesky_params,
                         se::Stream* stream,
                         stream_executor::GpuSolverContext* local_context) {
  TF_RETURN_IF_ERROR(local_context->SetStream(stream));
  if (cholesky_params->batch_size > 1) {
    switch (type) {
      case F32:
        return DoPotrfBatched<float>(cholesky_params, stream, *local_context);
      case F64:
        return DoPotrfBatched<double>(cholesky_params, stream, *local_context);
      case C64:
        return DoPotrfBatched<std::complex<float>>(cholesky_params, stream,
                                                   *local_context);
      case C128:
        return DoPotrfBatched<std::complex<double>>(cholesky_params, stream,
                                                    *local_context);
      default:
        return InvalidArgument("Invalid type for cholesky %s",
                               PrimitiveType_Name(type));
    }
  } else {
    switch (type) {
      case F32:
        return DoPotrfUnbatched<float>(cholesky_params, stream, *local_context);
      case F64:
        return DoPotrfUnbatched<double>(cholesky_params, stream,
                                        *local_context);
      case C64:
        return DoPotrfUnbatched<std::complex<float>>(cholesky_params, stream,
                                                     *local_context);
      case C128:
        return DoPotrfUnbatched<std::complex<double>>(cholesky_params, stream,
                                                      *local_context);
      default:
        return InvalidArgument("Invalid type for cholesky %s",
                               PrimitiveType_Name(type));
    }
  }
}

}  // namespace

CholeskyThunk::CholeskyThunk(
    ThunkInfo thunk_info, const CholeskyOptions& options,
    BufferAllocation::Slice a_buffer, BufferAllocation::Slice workspace_buffer,
    BufferAllocation::Slice info_buffer, PrimitiveType type, int64_t batch_size,
    int64_t n,
    absl::AnyInvocable<
        absl::StatusOr<std::unique_ptr<stream_executor::GpuSolverContext>>()>
        solver_context_creator)
    : Thunk(Kind::kCholesky, thunk_info),
      uplo_(options.lower() ? se::blas::UpperLower::kLower
                            : se::blas::UpperLower::kUpper),
      a_buffer_(a_buffer),
      workspace_buffer_(workspace_buffer),
      info_buffer_(info_buffer),
      type_(type),
      batch_size_(batch_size),
      n_(n),
      solver_context_creator_(std::move(solver_context_creator)) {}

absl::Status CholeskyThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(3) << "type=" << PrimitiveType_Name(type_)
          << " uplo=" << se::blas::UpperLowerString(uplo_)
          << " batch_size=" << batch_size_ << " n=" << n_
          << " a=" << a_buffer_.ToString()
          << " workspace=" << workspace_buffer_.ToString()
          << " info=" << info_buffer_.ToString();

  se::DeviceMemoryBase a_buffer =
      params.buffer_allocations->GetDeviceAddress(a_buffer_);
  se::DeviceMemoryBase info_buffer =
      params.buffer_allocations->GetDeviceAddress(info_buffer_);
  se::DeviceMemoryBase workspace_buffer =
      params.buffer_allocations->GetDeviceAddress(workspace_buffer_);
  CholeskyParams cholesky_params{n_,       batch_size_,      uplo_,
                                 a_buffer, workspace_buffer, info_buffer};
  thread_local absl::StatusOr<
      std::unique_ptr<stream_executor::GpuSolverContext>>
      context = solver_context_creator_();
  TF_RETURN_IF_ERROR(context.status());
  auto local_context = context.value().get();
  return RunCholesky(type_, &cholesky_params, params.stream, local_context);
}
}  // namespace gpu
}  // namespace xla
