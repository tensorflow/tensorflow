/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/numeric_options_utils.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/rnn/blas_gemm.h"
namespace tensorflow {

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
namespace {
template <typename T>
se::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory) {
  se::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
  se::DeviceMemory<T> typed(wrapped);
  return typed;
}
}  // namespace
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace functor {
template <typename T>
void TensorCuBlasGemm<T>::operator()(OpKernelContext* ctx, bool transa,
                                     bool transb, uint64 m, uint64 n, uint64 k,
                                     float alpha, const T* a, int lda,
                                     const T* b, int ldb, float beta, T* c,
                                     int ldc) {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  se::blas::Transpose trans[] = {se::blas::Transpose::kNoTranspose,
                                 se::blas::Transpose::kTranspose};

  auto a_ptr = AsDeviceMemory(a);
  auto b_ptr = AsDeviceMemory(b);
  auto c_ptr = AsDeviceMemory(c);
  auto* stream = ctx->op_device_context()->stream();
  auto* blas = stream->parent()->AsBlas();
  OP_REQUIRES(ctx, blas != nullptr, absl::InternalError("No BLAS for stream."));

  OP_REQUIRES_OK(
      ctx, blas->BlasGemm(stream, trans[transa], trans[transb], m, n, k,
                          static_cast<T>(alpha), a_ptr, lda, b_ptr, ldb,
                          static_cast<T>(beta), &c_ptr, ldc,
                          GetNumericOptions(), se::blas::CallContext::kNone));
#else
  ctx->SetStatus(errors::InvalidArgument("CuBlasGemm needs CUDA."));
#endif
}

template struct TensorCuBlasGemm<Eigen::half>;
template struct TensorCuBlasGemm<float>;

}  // end namespace functor
}  // end namespace tensorflow
