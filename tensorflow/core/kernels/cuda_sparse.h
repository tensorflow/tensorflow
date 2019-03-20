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
==============================================================================
*/

#ifndef TENSORFLOW_CORE_KERNELS_CUDA_SPARSE_H_
#define TENSORFLOW_CORE_KERNELS_CUDA_SPARSE_H_

// This header declares the class CudaSparse, which contains wrappers of
// cuSparse libraries for use in TensorFlow kernels.

#ifdef GOOGLE_CUDA

#include <functional>
#include <vector>

#include "cuda/include/cusparse.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/public/version.h"

// Macro that specializes a sparse method for all 4 standard
// numeric types.
// TODO: reuse with cuda_solvers
#define TF_CALL_LAPACK_TYPES(m) \
  m(float, S) m(double, D) m(std::complex<float>, C) m(std::complex<double>, Z)

namespace tensorflow {

inline string ConvertCUSparseErrorToString(const cusparseStatus_t status) {
  switch (status) {
#define STRINGIZE(q) #q
#define RETURN_IF_STATUS(err) \
  case err:                   \
    return STRINGIZE(err);

    RETURN_IF_STATUS(CUSPARSE_STATUS_SUCCESS)
    RETURN_IF_STATUS(CUSPARSE_STATUS_NOT_INITIALIZED)
    RETURN_IF_STATUS(CUSPARSE_STATUS_ALLOC_FAILED)
    RETURN_IF_STATUS(CUSPARSE_STATUS_INVALID_VALUE)
    RETURN_IF_STATUS(CUSPARSE_STATUS_ARCH_MISMATCH)
    RETURN_IF_STATUS(CUSPARSE_STATUS_MAPPING_ERROR)
    RETURN_IF_STATUS(CUSPARSE_STATUS_EXECUTION_FAILED)
    RETURN_IF_STATUS(CUSPARSE_STATUS_INTERNAL_ERROR)
    RETURN_IF_STATUS(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED)

#undef RETURN_IF_STATUS
#undef STRINGIZE
    default:
      return strings::StrCat("Unknown CUSPARSE error: ",
                             static_cast<int>(status));
  }
}

#define TF_RETURN_IF_CUSPARSE_ERROR(expr)                                  \
  do {                                                                     \
    auto status = (expr);                                                  \
    if (TF_PREDICT_FALSE(status != CUSPARSE_STATUS_SUCCESS)) {             \
      return errors::Internal(__FILE__, ":", __LINE__, " (", TF_STR(expr), \
                              "): cuSparse call failed with status ",      \
                              ConvertCUSparseErrorToString(status));       \
    }                                                                      \
  } while (0)

// The CudaSparse class provides a simplified templated API for cuSparse
// (http://docs.nvidia.com/cuda/cusparse/index.html).
// An object of this class wraps static cuSparse instances,
// and will launch Cuda kernels on the stream wrapped by the GPU device
// in the OpKernelContext provided to the constructor.
//
// Notice: All the computational member functions are asynchronous and simply
// launch one or more Cuda kernels on the Cuda stream wrapped by the CudaSparse
// object.

class CudaSparse {
 public:
  // This object stores a pointer to context, which must outlive it.
  explicit CudaSparse(OpKernelContext *context);
  virtual ~CudaSparse() {}

  // This initializes the CudaSparse class if it hasn't
  // been initialized yet.  All following public methods require the
  // class has been initialized.  Can be run multiple times; all
  // subsequent calls after the first have no effect.
  Status Initialize();  // Move to constructor?

  // ====================================================================
  // Wrappers for cuSparse start here.
  //

  // Solves tridiagonal system of equations.
  // See: https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-gtsv
  // Returns Status::OK() if the kernel was launched successfully.
  template <typename Scalar>
  Status Gtsv(int m, int n, const Scalar *dl, const Scalar *d, const Scalar *du,
              Scalar *B, int ldb) const;

 private:
  bool initialized_;
  OpKernelContext *context_;  // not owned.
  cudaStream_t cuda_stream_;
  cusparseHandle_t *cusparse_handle_;  // not owned.

  TF_DISALLOW_COPY_AND_ASSIGN(CudaSparse);
};

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_KERNELS_CUDA_SPARSE_H_
