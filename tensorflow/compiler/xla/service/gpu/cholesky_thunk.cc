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

#include "tensorflow/compiler/xla/service/gpu/cholesky_thunk.h"

#include <complex>
#include <memory>
#include <string>
#include <utility>

#include "tensorflow/compiler/xla/service/gpu/cusolver_context.h"
#include "tensorflow/compiler/xla/service/gpu/precompiled_kernels.h"
#include "tensorflow/compiler/xla/stream_executor/blas.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/logging.h"

namespace xla {
namespace gpu {

namespace {

template <typename T>
Status DoPotrfBatched(const se::GpuAsmOpts& asm_opts, CholeskyParams* params,
                      se::Stream* stream, GpuSolverContext& context) {
  T* a_base = static_cast<T*>(params->a_buffer.opaque());
  se::DeviceMemory<int> infos(params->info_buffer);
#if TENSORFLOW_USE_ROCSOLVER
  // hipsolver is not supported so allocate a GPU buffer
  se::ScopedDeviceMemory<T*> ptrs =
      stream->parent()->AllocateOwnedArray<T*>(batch_size_);
  auto as = *ptrs;
#else
  se::DeviceMemory<T*> as(params->workspace_buffer);
#endif

  CHECK_GE(as.size(), params->batch_size);
  CHECK_GE(infos.size(), params->batch_size);

  // Run a kernel that sets as[i] = &a_base[i * stride].
  const int64_t stride_bytes = params->n * params->n * sizeof(T);
  TF_RETURN_IF_ERROR(MakeBatchPointers(
      stream, asm_opts, se::DeviceMemoryBase(a_base), stride_bytes,
      static_cast<int>(params->batch_size), se::DeviceMemoryBase(as)));

  // Now that we've set up the `as` array, we can call cusolver.
  return context.PotrfBatched(params->uplo, params->n, as, params->n, infos,
                              params->batch_size);
}

}  // namespace

CholeskyThunk::CholeskyThunk(ThunkInfo thunk_info,
                             const CholeskyOptions& options,
                             const se::GpuAsmOpts asm_opts,
                             BufferAllocation::Slice a_buffer,
                             BufferAllocation::Slice workspace_buffer,
                             BufferAllocation::Slice info_buffer,
                             PrimitiveType type, int64_t batch_size, int64_t n)
    : Thunk(Kind::kCholesky, thunk_info),
      asm_opts_(asm_opts),
      uplo_(options.lower() ? se::blas::UpperLower::kLower
                            : se::blas::UpperLower::kUpper),
      a_buffer_(a_buffer),
      workspace_buffer_(workspace_buffer),
      info_buffer_(info_buffer),
      type_(type),
      batch_size_(batch_size),
      n_(n) {}

Status CholeskyThunk::ExecuteOnStream(const ExecuteParams& params) {
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
  return RunCholesky(asm_opts_, type_, &cholesky_params, params.stream);
}

Status RunCholesky(const se::GpuAsmOpts& asm_opts, PrimitiveType type,
                   CholeskyParams* cholesky_params, se::Stream* stream) {
  TF_ASSIGN_OR_RETURN(thread_local GpuSolverContext context,
                      GpuSolverContext::Create(stream));

  switch (type) {
    case F32:
      return DoPotrfBatched<float>(asm_opts, cholesky_params, stream, context);
    case F64:
      return DoPotrfBatched<double>(asm_opts, cholesky_params, stream, context);
    case C64:
      return DoPotrfBatched<std::complex<float>>(asm_opts, cholesky_params,
                                                 stream, context);
    case C128:
      return DoPotrfBatched<std::complex<double>>(asm_opts, cholesky_params,
                                                  stream, context);
    default:
      return InvalidArgument("Invalid type for cholesky %s",
                             PrimitiveType_Name(type));
  }
}

}  // namespace gpu
}  // namespace xla
