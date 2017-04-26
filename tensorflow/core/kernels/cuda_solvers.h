/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// This header implements CudaSolverDN and CuBlas, which contain templatized
// wrappers of linear algebra solvers in the cuBlas and cuSolverDN libraries
// for use in TensorFlow kernels.

#ifdef GOOGLE_CUDA

#include "cuda/include/cublas_v2.h"
#include "cuda/include/cusolverDn.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// A class that provides a simplified templated API for the solver methods
// in cuSolverDN (http://docs.nvidia.com/cuda/cusolver).
// An object of this class wraps a cuSolverDN instance, and will launch
// kernels on the cuda stream wrapped by the GPU device in the OpKernelContext
// provided to the constructor. The class methods transparently fetch the output
// status of the solvers (a.k.a. the LAPACK "info" output variable) without
// having to manually synchronize the underlying Cuda stream.
class CudaSolverDN {
 public:
  explicit CudaSolverDN(OpKernelContext* context);
  virtual ~CudaSolverDN();

  // ====================================================================
  // Templated wrappers for cuSolver functions start here.

  // Cholesky factorization.
  // Computes Cholesky factorization A = L * L^T.
  // Returns Status::OK(), if the Cholesky factorization was successful.
  // If info is not nullptr it is used to return the potrf info code:
  // Returns zero if success, returns -i if the
  // i-th parameter is wrong, returns i > 0, if the leading minor of order i is
  // not positive definite, see:
  // http://docs.nvidia.com/cuda/cusolver/#cuds-lt-t-gt-potrf
  template <typename Scalar>
  Status potrf(cublasFillMode_t uplo, int n, Scalar* A, int lda,
               int* info) const;

  /*
  TODO(rmlarsen, volunteers): Implement the kernels below.
  // Uses Cholesky factorization to solve A * X = B.
  // See: http://docs.nvidia.com/cuda/cusolver/#cuds-lt-t-gt-potrs
  template <typename Scalar>
  Status potrs(cublasFillMode_t uplo, int n, int nrhs, const Scalar* A, int lda,
             Scalar* B, int ldb, int* info) const;

  // LU factorization.
  // Computes LU factorization with partial pivoting P * A = L * U.
  // See: http://docs.nvidia.com/cuda/cusolver/#cuds-lt-t-gt-getrf
  template <typename Scalar>
  Status getrf(int m, int n, Scalar* A, int lda, int* devIpiv,
             int* devInfo) const;

  // Uses LU factorization to solve A * X = B.
  // See: http://docs.nvidia.com/cuda/cusolver/#cuds-lt-t-gt-getrs
  template <typename Scalar>
  Status getrs(int n, int nrhs, const Scalar* A, int lda, const int* devIpiv,
             Scalar* B, int ldb, int* devInfo) const;

  // QR factorization.
  // Computes QR factorization A = Q * R.
  // See: http://docs.nvidia.com/cuda/cusolver/#cuds-lt-t-gt-geqrf
  template <typename Scalar>
  Status geqrf(int m, int n, Scalar* A, int lda, Scalar* TAU, int* devInfo)
  const;

  // Multiplies by Q.
  // See: http://docs.nvidia.com/cuda/cusolver/#cuds-lt-t-gt-ormqr
  template <typename Scalar>
  Status mqr(cublasSideMode_t side, cublasOperation_t trans, int m, int n, int
  k, const Scalar* A, int lda, const Scalar* tau, Scalar* C, int ldc, int*
  devInfo const);

  // Materializes Q.
  // See: http://docs.nvidia.com/cuda/cusolver/#cuds-lt-t-gt-orgqr
  template <typename Scalar>
  Status gqr(int m, int n, int k, Scalar* A, int lda, const Scalar* tau,
           int* devInfo) const;

  // Symmetric/Hermitian Eigen decomposition.
  // See: http://docs.nvidia.com/cuda/cusolver/#cuds-lt-t-gt-syevd
  template <typename Scalar>
  Status evd(cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, Scalar* A,
           int lda, Scalar* W, int* devInfo) const;

  // Singular value decomposition.
  // See: http://docs.nvidia.com/cuda/cusolver/#cuds-lt-t-gt-gesvd
  template <typename Scalar>
  Status gesvd(signed char jobu, signed char jobvt, int m, int n, Scalar* A,
             int lda, Scalar* S, Scalar* U, int ldu, Scalar* VT, int ldvt,
             int* devInfo);
*/

 private:
  // Copies dev_info status back from the device to host and uses event manager
  // to wait (with a timeout) until the copy has finished. Returns an error if
  // the copy fails to complete successfully within the timeout period.
  Status GetInfo(const int* dev_info, int* host_info) const;

  OpKernelContext* context_;  // not owned.
  cudaStream_t cuda_stream_;
  cusolverDnHandle_t handle_;
};

/*
  TODO(rmlarsen, volunteers): Implement the kernels below. These are utils and
batched solvers not currently wrapped by stream executor. class CudaBlas {
 public:
  // Initializes a cuSolverDN handle that will launch kernels using the
  // cuda stream wrapped by the GPU device in context.
  explicit CudaBlas(OpKernelContext* context);
  virtual ~CudaBlas();

  // Templatized wrappers for cuBlas functions.

  // Matrix addition, copy and transposition.
  // See: http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-geam
  template <typename Scalar>
  Status geam(cublasOperation_t transa, cublasOperation_t transb, int m, int n,
            const Scalar* alpha, const Scalar* A, int lda, const Scalar* beta,
            const Scalar* B, int ldb, Scalar* C, int ldc) const;

  // Batched LU fatorization.
  // See:
http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-getrfbatched
  template <typename Scalar>
  Status getrfBatched(int n, Scalar* Aarray[], int lda, int* PivotArray,
                    int* infoArray, int batchSize) const;

  // Batched linear solver using LU factorization from getrfBatched.
  // See:
http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-getrsbatched
  template <typename Scalar>
  Status getrsBatched(cublasOperation_t trans, int n, int nrhs,
                    const Scalar* Aarray[], int lda, const int* devIpiv,
                    Scalar* Barray[], int ldb, int* info, int batchSize) const;

  // Batched matrix inverse.
  // See:
http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-getribatched
  template <typename Scalar>
  Status getriBatched(cublasHandle_t handle, int n, Scalar* Aarray[], int lda,
                    int* PivotArray, Scalar* Carray[], int ldc, int* infoArray,
                    int batchSize);

 private:
  // Copies dev_info status back from the device to host and uses event manager
  // to wait (with a timeout) until the copy has finished. Returns an error if
  // the copy fails to complete successfully within the timeout period.
  Status GetInfo(const int* dev_info, int* host_info) const;

  OpKernelContext* context_;  // not owned.
  cudaStream_t cuda_stream_;
  cublasHandle_t handle_;
};
*/

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
