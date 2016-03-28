/* Copyright 2015 Google Inc. All Rights Reserved.

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

// Exposes the family of BLAS routines as pre-canned high performance calls for
// use in conjunction with the StreamExecutor abstraction.
//
// Note that this interface is optionally supported by platforms; see
// StreamExecutor::SupportsBlas() for details.
//
// This abstraction makes it simple to entrain BLAS operations on GPU data into
// a Stream -- users typically will not use this API directly, but will use the
// Stream builder methods to entrain these operations "under the hood". For
// example:
//
//  DeviceMemory<float> x = stream_exec->AllocateArray<float>(1024);
//  DeviceMemory<float> y = stream_exec->AllocateArray<float>(1024);
//  // ... populate x and y ...
//  Stream stream{stream_exec};
//  stream
//    .Init()
//    .ThenBlasAxpy(1024, 5.5, x, 1, &y, 1)
//    .BlockHostUntilDone();
//
// By using stream operations in this manner the user can easily intermix custom
// kernel launches (via StreamExecutor::ThenLaunch()) with these pre-canned BLAS
// routines.

#ifndef TENSORFLOW_STREAM_EXECUTOR_BLAS_H_
#define TENSORFLOW_STREAM_EXECUTOR_BLAS_H_

#include <complex>
#include "tensorflow/stream_executor/platform/port.h"

#include "tensorflow/stream_executor/lib/array_slice.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace perftools {
namespace gputools {

class Stream;
class ScratchAllocator;

template <typename ElemT>
class DeviceMemory;

namespace blas {

// Specifies whether the input matrix will be transposed or
// transposed+conjugated before any BLAS operations.
enum class Transpose { kNoTranspose, kTranspose, kConjugateTranspose };

// Returns a name for t.
string TransposeString(Transpose t);

// Specifies whether the upper or lower triangular part of a
// symmetric/Hermitian matrix is used.
enum class UpperLower { kUpper, kLower };

// Returns a name for ul.
string UpperLowerString(UpperLower ul);

// Specifies whether a matrix is unit triangular.
enum class Diagonal { kUnit, kNonUnit };

// Returns a name for d.
string DiagonalString(Diagonal d);

// Specifies whether a Hermitian matrix appears on the left or right in
// operation.
enum class Side { kLeft, kRight };

// Returns a name for s.
string SideString(Side s);

// BLAS support interface -- this can be derived from a GPU executor when the
// underlying platform has an BLAS library implementation available. See
// StreamExecutor::AsBlas().
//
// Thread-hostile: CUDA associates a CUDA-context with a particular thread in
// the system. Any operation that a user attempts to perform by enqueueing BLAS
// operations on a thread not-associated with the CUDA-context has unknown
// behavior at the current time; see b/13176597
class BlasSupport {
 public:
  virtual ~BlasSupport() {}

  // Computes the sum of magnitudes of the vector elements.
  // result <- |Re x(1)| + |Im x(1)| + |Re  x(2)| + |Im  x(2)|+ ... + |Re  x(n)|
  // + |Im x(n)|.
  // Note that Im x(i) = 0 for real types float/double.
  virtual bool DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *result) = 0;
  virtual bool DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *result) = 0;
  virtual bool DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<float> *result) = 0;
  virtual bool DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<double> *result) = 0;

  // Performs a BLAS y <- ax+y operation.
  virtual bool DoBlasAxpy(Stream *stream, uint64 elem_count, float alpha,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) = 0;
  virtual bool DoBlasAxpy(Stream *stream, uint64 elem_count, double alpha,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) = 0;
  virtual bool DoBlasAxpy(Stream *stream, uint64 elem_count,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) = 0;
  virtual bool DoBlasAxpy(Stream *stream, uint64 elem_count,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) = 0;

  // Copies vector to another vector: y <- x.
  virtual bool DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) = 0;
  virtual bool DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) = 0;
  virtual bool DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) = 0;
  virtual bool DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) = 0;

  // Performs a BLAS dot product result <- x . y.
  virtual bool DoBlasDot(Stream *stream, uint64 elem_count,
                         const DeviceMemory<float> &x, int incx,
                         const DeviceMemory<float> &y, int incy,
                         DeviceMemory<float> *result) = 0;
  virtual bool DoBlasDot(Stream *stream, uint64 elem_count,
                         const DeviceMemory<double> &x, int incx,
                         const DeviceMemory<double> &y, int incy,
                         DeviceMemory<double> *result) = 0;

  // Performs a BLAS dot product result <- conj(x) . y for complex types.
  virtual bool DoBlasDotc(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *result) = 0;
  virtual bool DoBlasDotc(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *result) = 0;

  // Performs a BLAS dot product result <- x . y for complex types. Note that
  // x is unconjugated in this routine.
  virtual bool DoBlasDotu(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *result) = 0;
  virtual bool DoBlasDotu(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *result) = 0;

  // Computes the Euclidean norm of a vector: result <- ||x||.
  // See the following link for more information of Euclidean norm:
  // http://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm
  virtual bool DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *result) = 0;
  virtual bool DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *result) = 0;
  virtual bool DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<float> *result) = 0;
  virtual bool DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<double> *result) = 0;

  // Performs rotation of points in the plane:
  // x(i) = c*x(i) + s*y(i)
  // y(i) = c*y(i) - s*x(i).
  virtual bool DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<float> *x, int incx,
                         DeviceMemory<float> *y, int incy, float c,
                         float s) = 0;
  virtual bool DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<double> *x, int incx,
                         DeviceMemory<double> *y, int incy, double c,
                         double s) = 0;
  virtual bool DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<std::complex<float>> *x, int incx,
                         DeviceMemory<std::complex<float>> *y, int incy,
                         float c, float s) = 0;
  virtual bool DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<std::complex<double>> *x, int incx,
                         DeviceMemory<std::complex<double>> *y, int incy,
                         double c, double s) = 0;

  // Computes the parameters for a Givens rotation.
  // Given the Cartesian coordinates (a, b) of a point, these routines return
  // the parameters c, s, r, and z associated with the Givens rotation. The
  // parameters c and s define a unitary matrix such that:
  //
  //   |  c s |.| a | = | r |
  //   | -s c | | b |   | 0 |
  //
  // The parameter z is defined such that if |a| > |b|, z is s; otherwise if
  // c is not 0 z is 1/c; otherwise z is 1.
  virtual bool DoBlasRotg(Stream *stream, DeviceMemory<float> *a,
                          DeviceMemory<float> *b, DeviceMemory<float> *c,
                          DeviceMemory<float> *s) = 0;
  virtual bool DoBlasRotg(Stream *stream, DeviceMemory<double> *a,
                          DeviceMemory<double> *b, DeviceMemory<double> *c,
                          DeviceMemory<double> *s) = 0;
  virtual bool DoBlasRotg(Stream *stream, DeviceMemory<std::complex<float>> *a,
                          DeviceMemory<std::complex<float>> *b,
                          DeviceMemory<float> *c,
                          DeviceMemory<std::complex<float>> *s) = 0;
  virtual bool DoBlasRotg(Stream *stream, DeviceMemory<std::complex<double>> *a,
                          DeviceMemory<std::complex<double>> *b,
                          DeviceMemory<double> *c,
                          DeviceMemory<std::complex<double>> *s) = 0;

  // Performs modified Givens rotation of points in the plane.
  // Given two vectors x and y, each vector element of these vectors is replaced
  // as follows:
  //
  //   | x(i) | =  H | x(i) |
  //   | y(i) |      | y(i) |
  //
  // for i=1 to n, where H is a modified Givens transformation matrix whose
  // values are stored in the param[1] through param[4] array.
  // For more information please Google this routine.
  virtual bool DoBlasRotm(Stream *stream, uint64 elem_count,
                          DeviceMemory<float> *x, int incx,
                          DeviceMemory<float> *y, int incy,
                          const DeviceMemory<float> &param) = 0;
  virtual bool DoBlasRotm(Stream *stream, uint64 elem_count,
                          DeviceMemory<double> *x, int incx,
                          DeviceMemory<double> *y, int incy,
                          const DeviceMemory<double> &param) = 0;

  // Computes the parameters for a modified Givens rotation.
  // Given Cartesian coordinates (x1, y1) of an input vector, these routines
  // compute the components of a modified Givens transformation matrix H that
  // zeros the y-component of the resulting vector:
  //
  //   | x1 | =  H | x1 * sqrt(d1) |
  //   |  0 |      | y1 * sqrt(d1) |
  //
  // For more information please Google this routine.
  virtual bool DoBlasRotmg(Stream *stream, DeviceMemory<float> *d1,
                           DeviceMemory<float> *d2, DeviceMemory<float> *x1,
                           const DeviceMemory<float> &y1,
                           DeviceMemory<float> *param) = 0;
  virtual bool DoBlasRotmg(Stream *stream, DeviceMemory<double> *d1,
                           DeviceMemory<double> *d2, DeviceMemory<double> *x1,
                           const DeviceMemory<double> &y1,
                           DeviceMemory<double> *param) = 0;

  // Computes the product of a vector by a scalar: x <- a*x.
  virtual bool DoBlasScal(Stream *stream, uint64 elem_count, float alpha,
                          DeviceMemory<float> *x, int incx) = 0;
  virtual bool DoBlasScal(Stream *stream, uint64 elem_count, double alpha,
                          DeviceMemory<double> *x, int incx) = 0;
  virtual bool DoBlasScal(Stream *stream, uint64 elem_count, float alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) = 0;
  virtual bool DoBlasScal(Stream *stream, uint64 elem_count, double alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) = 0;
  virtual bool DoBlasScal(Stream *stream, uint64 elem_count,
                          std::complex<float> alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) = 0;
  virtual bool DoBlasScal(Stream *stream, uint64 elem_count,
                          std::complex<double> alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) = 0;

  // Swaps a vector with another vector.
  virtual bool DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<float> *x, int incx,
                          DeviceMemory<float> *y, int incy) = 0;
  virtual bool DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<double> *x, int incx,
                          DeviceMemory<double> *y, int incy) = 0;
  virtual bool DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<std::complex<float>> *x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) = 0;
  virtual bool DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<std::complex<double>> *x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) = 0;

  // Finds the index of the element with maximum absolute value.
  virtual bool DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<float> &x, int incx,
                           DeviceMemory<int> *result) = 0;
  virtual bool DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<double> &x, int incx,
                           DeviceMemory<int> *result) = 0;
  virtual bool DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<float>> &x, int incx,
                           DeviceMemory<int> *result) = 0;
  virtual bool DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<double>> &x,
                           int incx, DeviceMemory<int> *result) = 0;

  // Finds the index of the element with minimum absolute value.
  virtual bool DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<float> &x, int incx,
                           DeviceMemory<int> *result) = 0;
  virtual bool DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<double> &x, int incx,
                           DeviceMemory<int> *result) = 0;
  virtual bool DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<float>> &x, int incx,
                           DeviceMemory<int> *result) = 0;
  virtual bool DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<double>> &x,
                           int incx, DeviceMemory<int> *result) = 0;

  // Computes a matrix-vector product using a general band matrix:
  //
  //     y <- alpha * a * x + beta * y,
  // or
  //     y <- alpha * a' * x + beta * y,
  // or
  //     y <- alpha * conj(a') * x + beta * y,
  //
  // alpha and beta are scalars; a is an m-by-n general band matrix, with kl
  // sub-diagonals and ku super-diagonals; x is a vector with
  // n(trans==kNoTranspose)/m(otherwise) elements;
  // y is a vector with m(trans==kNoTranspose)/n(otherwise) elements.
  virtual bool DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) = 0;
  virtual bool DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) = 0;
  virtual bool DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) = 0;
  virtual bool DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) = 0;

  // Computes a matrix-vector product using a general matrix.
  //
  //     y <- alpha * a * x + beta * y,
  // or
  //     y <- alpha * a' * x + beta * y,
  // or
  //     y <- alpha * conj(a') * x + beta * y,
  //
  // alpha and beta are scalars; a is an m-by-n general matrix; x is a vector
  // with n(trans==kNoTranspose)/m(otherwise) elements;
  // y is a vector with m(trans==kNoTranspose)/n(otherwise) elements.
  virtual bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) = 0;
  virtual bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, double alpha, const DeviceMemory<double> &a,
                          int lda, const DeviceMemory<double> &x, int incx,
                          double beta, DeviceMemory<double> *y, int incy) = 0;
  virtual bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) = 0;
  virtual bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) = 0;

  // Performs a rank-1 update of a general matrix.
  //
  //     a <- alpha * x * y' + a,
  //
  // alpha is a scalar; x is an m-element vector; y is an n-element vector; a is
  // an m-by-n general matrix.
  virtual bool DoBlasGer(Stream *stream, uint64 m, uint64 n, float alpha,
                         const DeviceMemory<float> &x, int incx,
                         const DeviceMemory<float> &y, int incy,
                         DeviceMemory<float> *a, int lda) = 0;
  virtual bool DoBlasGer(Stream *stream, uint64 m, uint64 n, double alpha,
                         const DeviceMemory<double> &x, int incx,
                         const DeviceMemory<double> &y, int incy,
                         DeviceMemory<double> *a, int lda) = 0;

  // Performs a rank-1 update (conjugated) of a general matrix.
  //
  //     a <- alpha * x * conj(y') + a,
  //
  // alpha is a scalar; x is an m-element vector; y is an n-element vector; a is
  // an m-by-n general matrix.
  virtual bool DoBlasGerc(Stream *stream, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) = 0;
  virtual bool DoBlasGerc(Stream *stream, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) = 0;

  // Performs a rank-1 update (unconjugated) of a general matrix.
  //
  //     a <- alpha * x * y' + a,
  //
  // alpha is a scalar; x is an m-element vector; y is an n-element vector; a is
  // an m-by-n general matrix.
  virtual bool DoBlasGeru(Stream *stream, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) = 0;
  virtual bool DoBlasGeru(Stream *stream, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) = 0;

  // Computes a matrix-vector product using a Hermitian band matrix.
  //
  //     y <- alpha * a * x + beta * y,
  //
  // alpha and beta are scalars; a is an n-by-n Hermitian band matrix, with k
  // super-diagonals; x and y are n-element vectors.
  virtual bool DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) = 0;
  virtual bool DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) = 0;

  // Computes a matrix-vector product using a Hermitian matrix.
  //
  //     y <- alpha * a * x + beta * y,
  //
  // alpha and beta are scalars; a is an n-by-n Hermitian matrix; x and y are
  // n-element vectors.
  virtual bool DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) = 0;
  virtual bool DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) = 0;

  // Performs a rank-1 update of a Hermitian matrix.
  //
  //     a <- alpha * x * conj(x') + a,
  //
  // alpha is a scalar; x is an n-element vector; a is an n-by-n Hermitian
  // matrix.
  virtual bool DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha,
                         const DeviceMemory<std::complex<float>> &x, int incx,
                         DeviceMemory<std::complex<float>> *a, int lda) = 0;
  virtual bool DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha,
                         const DeviceMemory<std::complex<double>> &x, int incx,
                         DeviceMemory<std::complex<double>> *a, int lda) = 0;

  // Performs a rank-2 update of a Hermitian matrix.
  //
  //     a <- alpha * x * conj(x') + conj(alpha) * y * conj(x') + a,
  //
  // alpha is a scalar; x and y are n-element vectors; a is an n-by-n Hermitian
  // matrix.
  virtual bool DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) = 0;
  virtual bool DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) = 0;

  // Computes a matrix-vector product using a Hermitian packed matrix.
  //
  //     y <- alpha * a * x + beta * y,
  //
  // alpha and beta are scalars; a is an n-by-n Hermitian matrix, supplied in
  // packed form; x and y are n-element vectors.
  virtual bool DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &ap,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) = 0;
  virtual bool DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &ap,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) = 0;

  // Performs a rank-1 update of a Hermitian packed matrix.
  //
  //     a <- alpha * x * conj(x') + a,
  //
  // alpha is a scalar; x is an n-element vector; a is an n-by-n Hermitian
  // matrix, supplied in packed form.
  virtual bool DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha,
                         const DeviceMemory<std::complex<float>> &x, int incx,
                         DeviceMemory<std::complex<float>> *ap) = 0;
  virtual bool DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha,
                         const DeviceMemory<std::complex<double>> &x, int incx,
                         DeviceMemory<std::complex<double>> *ap) = 0;

  // Performs a rank-2 update of a Hermitian packed matrix.
  //
  //     a <- alpha * x * conj(x') + conj(alpha) * y * conj(x') + a,
  //
  // alpha is a scalar; x and y are n-element vectors; a is an n-by-n Hermitian
  // matrix, supplied in packed form.
  virtual bool DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *ap) = 0;
  virtual bool DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *ap) = 0;

  // Computes a matrix-vector product using a symmetric band matrix.
  //
  //     y <- alpha * a * x + beta * y,
  //
  // alpha and beta are scalars; a is an n-by-n symmetric band matrix, with k
  // super-diagonals; x and y are n-element vectors.
  virtual bool DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) = 0;
  virtual bool DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, double alpha, const DeviceMemory<double> &a,
                          int lda, const DeviceMemory<double> &x, int incx,
                          double beta, DeviceMemory<double> *y, int incy) = 0;

  // Computes a matrix-vector product using a symmetric packed matrix.
  //
  //     y <- alpha * a * x + beta * y,
  //
  // alpha and beta are scalars; a is an n-by-n symmetric matrix, supplied in
  // packed form; x and y are n-element vectors.
  virtual bool DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &ap,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) = 0;
  virtual bool DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &ap,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) = 0;

  // Performs a rank-1 update of a symmetric packed matrix.
  //
  //     a <- alpha * x * x' + a,
  //
  // alpha is a scalar; x is an n-element vector; a is an n-by-n symmetric
  // matrix, supplied in packed form.
  virtual bool DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha, const DeviceMemory<float> &x, int incx,
                         DeviceMemory<float> *ap) = 0;
  virtual bool DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha, const DeviceMemory<double> &x, int incx,
                         DeviceMemory<double> *ap) = 0;

  // Performs a rank-2 update of a symmetric packed matrix.
  //
  //     a <- alpha * x * x' + alpha * y * x' + a,
  //
  // alpha is a scalar; x and y are n-element vectors; a is an n-by-n symmetric
  // matrix, supplied in packed form.
  virtual bool DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &x, int incx,
                          const DeviceMemory<float> &y, int incy,
                          DeviceMemory<float> *ap) = 0;
  virtual bool DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &x, int incx,
                          const DeviceMemory<double> &y, int incy,
                          DeviceMemory<double> *ap) = 0;

  // Computes a matrix-vector product for a symmetric matrix.
  //
  //     y <- alpha * a * x + beta * y,
  //
  // alpha and beta are scalars; a is an n-by-n symmetric matrix; x and y are
  // n-element vectors.
  virtual bool DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) = 0;
  virtual bool DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) = 0;

  // Performs a rank-1 update of a symmetric matrix.
  //
  //     a <- alpha * x * x' + a,
  //
  // alpha is a scalar; x is an n-element vector; a is an n-by-n symmetric
  // matrix.
  virtual bool DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha, const DeviceMemory<float> &x, int incx,
                         DeviceMemory<float> *a, int lda) = 0;
  virtual bool DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha, const DeviceMemory<double> &x, int incx,
                         DeviceMemory<double> *a, int lda) = 0;

  // Performs a rank-2 update of symmetric matrix.
  //
  //     a <- alpha * x * x' + alpha * y * x' + a,
  //
  // alpha is a scalar; x and y are n-element vectors; a is an n-by-n symmetric
  // matrix.
  virtual bool DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &x, int incx,
                          const DeviceMemory<float> &y, int incy,
                          DeviceMemory<float> *a, int lda) = 0;
  virtual bool DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &x, int incx,
                          const DeviceMemory<double> &y, int incy,
                          DeviceMemory<double> *a, int lda) = 0;

  // Computes a matrix-vector product using a triangular band matrix.
  //
  //     x <- a * x,
  // or
  //     x <- a' * x,
  // or
  //     x <- conj(a') * x,
  //
  // a is an n-by-n unit, or non-unit, upper or lower triangular band matrix,
  // with k+1 diagonals; x is a n-element vector.
  virtual bool DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) = 0;
  virtual bool DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) = 0;
  virtual bool DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<float>> &a,
                          int lda, DeviceMemory<std::complex<float>> *x,
                          int incx) = 0;
  virtual bool DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<double>> &a,
                          int lda, DeviceMemory<std::complex<double>> *x,
                          int incx) = 0;

  // Solves a system of linear equations whose coefficients are in a triangular
  // band matrix as below:
  //
  //     a * x = b,
  // or
  //     a' * x = b,
  // or
  //     conj(a') * x = b,
  //
  // b and x are n-element vectors; a is an n-by-n unit, or non-unit, upper or
  // lower triangular band matrix, with k+1 diagonals.
  virtual bool DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) = 0;
  virtual bool DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) = 0;
  virtual bool DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<float>> &a,
                          int lda, DeviceMemory<std::complex<float>> *x,
                          int incx) = 0;
  virtual bool DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<double>> &a,
                          int lda, DeviceMemory<std::complex<double>> *x,
                          int incx) = 0;

  // Computes a matrix-vector product using a triangular packed matrix.
  //
  //     x <- a * x,
  // or
  //     x <- a' * x,
  // or
  //     x <- conj(a') * x,
  //
  // a is an n-by-n unit, or non-unit, upper or lower triangular matrix,
  // supplied in packed form; x is a n-element vector.
  virtual bool DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &ap, DeviceMemory<float> *x,
                          int incx) = 0;
  virtual bool DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &ap,
                          DeviceMemory<double> *x, int incx) = 0;
  virtual bool DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &ap,
                          DeviceMemory<std::complex<float>> *x, int incx) = 0;
  virtual bool DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &ap,
                          DeviceMemory<std::complex<double>> *x, int incx) = 0;

  // Solves a system of linear equations whose coefficients are in a triangular
  // packed matrix as below:
  //
  //     a * x = b,
  // or
  //     a' * x = b,
  // or
  //     conj(a') * x = b,
  //
  // b and x are n-element vectors; a is an n-by-n unit, or non-unit, upper or
  // lower triangular matrix, supplied in packed form.
  virtual bool DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &ap, DeviceMemory<float> *x,
                          int incx) = 0;
  virtual bool DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &ap,
                          DeviceMemory<double> *x, int incx) = 0;
  virtual bool DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &ap,
                          DeviceMemory<std::complex<float>> *x, int incx) = 0;
  virtual bool DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &ap,
                          DeviceMemory<std::complex<double>> *x, int incx) = 0;

  // Computes a matrix-vector product using a triangular matrix.
  //
  //     x <- a * x,
  // or
  //     x <- a' * x,
  // or
  //     x <- conj(a') * x,
  //
  // a is an n-by-n unit, or non-unit, upper or lower triangular matrix; x is a
  // n-element vector.
  virtual bool DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) = 0;
  virtual bool DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) = 0;
  virtual bool DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) = 0;
  virtual bool DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) = 0;

  // Solves a system of linear equations whose coefficients are in a triangular
  // matrix as below:
  //
  //     a * x = b,
  // or
  //     a' * x = b,
  // or
  //     conj(a') * x = b,
  //
  // b and x are n-element vectors; a is an n-by-n unit, or non-unit, upper or
  // lower triangular matrix.
  virtual bool DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) = 0;
  virtual bool DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) = 0;
  virtual bool DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) = 0;
  virtual bool DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) = 0;

  // Computes a matrix-matrix product with general matrices:
  //
  //     c <- alpha * op(a) * op(b) + beta * c,
  //
  // op(X) is one of op(X) = X, or op(X) = X', or op(X) = conj(X'); alpha and
  // beta are scalars; a, b, and c are matrices; op(a) is an m-by-k matrix;
  // op(b) is a k-by-n matrix; c is an m-by-n matrix.
  virtual bool DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &b, int ldb, float beta,
                          DeviceMemory<float> *c, int ldc) = 0;
  virtual bool DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &b, int ldb, double beta,
                          DeviceMemory<double> *c, int ldc) = 0;
  virtual bool DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) = 0;
  virtual bool DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) = 0;

  // Computes a batch of matrix-matrix product with general matrices.
  // This is a batched version of DoBlasGemm.
  // The batched GEMM computes matrix product for each input/output in a, b,
  // and c, which contain batch_count DeviceMemory objects.
  virtual bool DoBlasGemmBatched(
      Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
      uint64 n, uint64 k, float alpha,
      const port::ArraySlice<DeviceMemory<float> *> &a, int lda,
      const port::ArraySlice<DeviceMemory<float> *> &b, int ldb, float beta,
      const port::ArraySlice<DeviceMemory<float> *> &c, int ldc,
      int batch_count, ScratchAllocator *scratch_allocator) = 0;
  virtual bool DoBlasGemmBatched(
      Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
      uint64 n, uint64 k, double alpha,
      const port::ArraySlice<DeviceMemory<double> *> &a, int lda,
      const port::ArraySlice<DeviceMemory<double> *> &b, int ldb, double beta,
      const port::ArraySlice<DeviceMemory<double> *> &c, int ldc,
      int batch_count, ScratchAllocator *scratch_allocator) = 0;
  virtual bool DoBlasGemmBatched(
      Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
      uint64 n, uint64 k, std::complex<float> alpha,
      const port::ArraySlice<DeviceMemory<std::complex<float>> *> &a, int lda,
      const port::ArraySlice<DeviceMemory<std::complex<float>> *> &b, int ldb,
      std::complex<float> beta,
      const port::ArraySlice<DeviceMemory<std::complex<float>> *> &c, int ldc,
      int batch_count, ScratchAllocator *scratch_allocator) = 0;
  virtual bool DoBlasGemmBatched(
      Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
      uint64 n, uint64 k, std::complex<double> alpha,
      const port::ArraySlice<DeviceMemory<std::complex<double>> *> &a, int lda,
      const port::ArraySlice<DeviceMemory<std::complex<double>> *> &b, int ldb,
      std::complex<double> beta,
      const port::ArraySlice<DeviceMemory<std::complex<double>> *> &c, int ldc,
      int batch_count, ScratchAllocator *scratch_allocator) = 0;

  // Computes a matrix-matrix product where one input matrix is Hermitian:
  //
  //     c <- alpha * a * b + beta * c,
  // or
  //     c <- alpha * b * a + beta * c,
  //
  // alpha and beta are scalars; a is a Hermitian matrix; b and c are m-by-n
  // matrices.
  virtual bool DoBlasHemm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) = 0;
  virtual bool DoBlasHemm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) = 0;

  // Performs a Hermitian rank-k update.
  //
  //     c <- alpha * a * conj(a') + beta * c,
  // or
  //     c <- alpha * conj(a') * a + beta * c,
  //
  // alpha and beta are scalars; c is a n-by-n Hermitian matrix; a is an n-by-k
  // matrix in the first case and a k-by-n matrix in the second case.
  virtual bool DoBlasHerk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          float alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          float beta, DeviceMemory<std::complex<float>> *c,
                          int ldc) = 0;
  virtual bool DoBlasHerk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          double alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          double beta, DeviceMemory<std::complex<double>> *c,
                          int ldc) = 0;

  // Performs a Hermitian rank-2k update.
  //
  //     c <- alpha * a * conj(b') + conj(alpha) * b * conj(a') + beta * c,
  // or
  //     c <- alpha * conj(b') * a + conj(alpha) * conj(a') * b + beta * c,
  //
  // alpha and beta are scalars; c is a n-by-n Hermitian matrix; a and b are
  // n-by-k matrices in the first case and k-by-n matrices in the second case.
  virtual bool DoBlasHer2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<float> alpha,
                           const DeviceMemory<std::complex<float>> &a, int lda,
                           const DeviceMemory<std::complex<float>> &b, int ldb,
                           float beta, DeviceMemory<std::complex<float>> *c,
                           int ldc) = 0;
  virtual bool DoBlasHer2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<double> alpha,
                           const DeviceMemory<std::complex<double>> &a, int lda,
                           const DeviceMemory<std::complex<double>> &b, int ldb,
                           double beta, DeviceMemory<std::complex<double>> *c,
                           int ldc) = 0;

  // Computes a matrix-matrix product where one input matrix is symmetric.
  //
  //     c <- alpha * a * b + beta * c,
  // or
  //     c <- alpha * b * a + beta * c,
  //
  // alpha and beta are scalars; a is a symmetric matrix; b and c are m-by-n
  // matrices.
  virtual bool DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &b, int ldb, float beta,
                          DeviceMemory<float> *c, int ldc) = 0;
  virtual bool DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &b, int ldb, double beta,
                          DeviceMemory<double> *c, int ldc) = 0;
  virtual bool DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) = 0;
  virtual bool DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) = 0;

  // Performs a symmetric rank-k update.
  //
  //     c <- alpha * a * a' + beta * c,
  // or
  //     c <- alpha * a' * a + beta * c,
  //
  // alpha and beta are scalars; c is a n-by-n symmetric matrix; a is an n-by-k
  // matrix in the first case and a k-by-n matrix in the second case.
  virtual bool DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          float beta, DeviceMemory<float> *c, int ldc) = 0;
  virtual bool DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          double beta, DeviceMemory<double> *c, int ldc) = 0;
  virtual bool DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) = 0;
  virtual bool DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) = 0;

  // Performs a symmetric rank-2k update.
  //
  //     c <- alpha * a * b' + alpha * b * a' + beta * c,
  // or
  //     c <- alpha * b' * a + alpha * a' * b + beta * c,
  //
  // alpha and beta are scalars; c is a n-by-n symmetric matrix; a and b are
  // n-by-k matrices in the first case and k-by-n matrices in the second case.
  virtual bool DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           float alpha, const DeviceMemory<float> &a, int lda,
                           const DeviceMemory<float> &b, int ldb, float beta,
                           DeviceMemory<float> *c, int ldc) = 0;
  virtual bool DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           double alpha, const DeviceMemory<double> &a, int lda,
                           const DeviceMemory<double> &b, int ldb, double beta,
                           DeviceMemory<double> *c, int ldc) = 0;
  virtual bool DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<float> alpha,
                           const DeviceMemory<std::complex<float>> &a, int lda,
                           const DeviceMemory<std::complex<float>> &b, int ldb,
                           std::complex<float> beta,
                           DeviceMemory<std::complex<float>> *c, int ldc) = 0;
  virtual bool DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<double> alpha,
                           const DeviceMemory<std::complex<double>> &a, int lda,
                           const DeviceMemory<std::complex<double>> &b, int ldb,
                           std::complex<double> beta,
                           DeviceMemory<std::complex<double>> *c, int ldc) = 0;

  // Computes a matrix-matrix product where one input matrix is triangular.
  //
  //     b <- alpha * op(a) * b,
  // or
  //     b <- alpha * b * op(a)
  //
  // alpha is a scalar; b is an m-by-n matrix; a is a unit, or non-unit, upper
  // or lower triangular matrix; op(a) is one of op(a) = a, or op(a) = a', or
  // op(a) = conj(a').
  virtual bool DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) = 0;
  virtual bool DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) = 0;
  virtual bool DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) = 0;
  virtual bool DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) = 0;

  // Solves a triangular matrix equation.
  //
  //     op(a) * x = alpha * b,
  // or
  //     x * op(a) = alpha * b
  //
  // alpha is a scalar; x and b are m-by-n matrices; a is a unit, or non-unit,
  // upper or lower triangular matrix; op(a) is one of op(a) = a, or op(a) = a',
  // or op(a) = conj(a').
  virtual bool DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) = 0;
  virtual bool DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) = 0;
  virtual bool DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) = 0;
  virtual bool DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) = 0;

 protected:
  BlasSupport() {}

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(BlasSupport);
};

// Macro used to quickly declare overrides for abstract virtuals in the
// BlasSupport base class.
#define TENSORFLOW_STREAM_EXECUTOR_GPU_BLAS_SUPPORT_OVERRIDES                  \
  bool DoBlasAsum(Stream *stream, uint64 elem_count,                           \
                  const DeviceMemory<float> &x, int incx,                      \
                  DeviceMemory<float> *result) override;                       \
  bool DoBlasAsum(Stream *stream, uint64 elem_count,                           \
                  const DeviceMemory<double> &x, int incx,                     \
                  DeviceMemory<double> *result) override;                      \
  bool DoBlasAsum(Stream *stream, uint64 elem_count,                           \
                  const DeviceMemory<std::complex<float>> &x, int incx,        \
                  DeviceMemory<float> *result) override;                       \
  bool DoBlasAsum(Stream *stream, uint64 elem_count,                           \
                  const DeviceMemory<std::complex<double>> &x, int incx,       \
                  DeviceMemory<double> *result) override;                      \
  bool DoBlasAxpy(Stream *stream, uint64 elem_count, float alpha,              \
                  const DeviceMemory<float> &x, int incx,                      \
                  DeviceMemory<float> *y, int incy) override;                  \
  bool DoBlasAxpy(Stream *stream, uint64 elem_count, double alpha,             \
                  const DeviceMemory<double> &x, int incx,                     \
                  DeviceMemory<double> *y, int incy) override;                 \
  bool DoBlasAxpy(Stream *stream, uint64 elem_count,                           \
                  std::complex<float> alpha,                                   \
                  const DeviceMemory<std::complex<float>> &x, int incx,        \
                  DeviceMemory<std::complex<float>> *y, int incy) override;    \
  bool DoBlasAxpy(Stream *stream, uint64 elem_count,                           \
                  std::complex<double> alpha,                                  \
                  const DeviceMemory<std::complex<double>> &x, int incx,       \
                  DeviceMemory<std::complex<double>> *y, int incy) override;   \
  bool DoBlasCopy(Stream *stream, uint64 elem_count,                           \
                  const DeviceMemory<float> &x, int incx,                      \
                  DeviceMemory<float> *y, int incy) override;                  \
  bool DoBlasCopy(Stream *stream, uint64 elem_count,                           \
                  const DeviceMemory<double> &x, int incx,                     \
                  DeviceMemory<double> *y, int incy) override;                 \
  bool DoBlasCopy(Stream *stream, uint64 elem_count,                           \
                  const DeviceMemory<std::complex<float>> &x, int incx,        \
                  DeviceMemory<std::complex<float>> *y, int incy) override;    \
  bool DoBlasCopy(Stream *stream, uint64 elem_count,                           \
                  const DeviceMemory<std::complex<double>> &x, int incx,       \
                  DeviceMemory<std::complex<double>> *y, int incy) override;   \
  bool DoBlasDot(Stream *stream, uint64 elem_count,                            \
                 const DeviceMemory<float> &x, int incx,                       \
                 const DeviceMemory<float> &y, int incy,                       \
                 DeviceMemory<float> *result) override;                        \
  bool DoBlasDot(Stream *stream, uint64 elem_count,                            \
                 const DeviceMemory<double> &x, int incx,                      \
                 const DeviceMemory<double> &y, int incy,                      \
                 DeviceMemory<double> *result) override;                       \
  bool DoBlasDotc(Stream *stream, uint64 elem_count,                           \
                  const DeviceMemory<std::complex<float>> &x, int incx,        \
                  const DeviceMemory<std::complex<float>> &y, int incy,        \
                  DeviceMemory<std::complex<float>> *result) override;         \
  bool DoBlasDotc(Stream *stream, uint64 elem_count,                           \
                  const DeviceMemory<std::complex<double>> &x, int incx,       \
                  const DeviceMemory<std::complex<double>> &y, int incy,       \
                  DeviceMemory<std::complex<double>> *result) override;        \
  bool DoBlasDotu(Stream *stream, uint64 elem_count,                           \
                  const DeviceMemory<std::complex<float>> &x, int incx,        \
                  const DeviceMemory<std::complex<float>> &y, int incy,        \
                  DeviceMemory<std::complex<float>> *result) override;         \
  bool DoBlasDotu(Stream *stream, uint64 elem_count,                           \
                  const DeviceMemory<std::complex<double>> &x, int incx,       \
                  const DeviceMemory<std::complex<double>> &y, int incy,       \
                  DeviceMemory<std::complex<double>> *result) override;        \
  bool DoBlasNrm2(Stream *stream, uint64 elem_count,                           \
                  const DeviceMemory<float> &x, int incx,                      \
                  DeviceMemory<float> *result) override;                       \
  bool DoBlasNrm2(Stream *stream, uint64 elem_count,                           \
                  const DeviceMemory<double> &x, int incx,                     \
                  DeviceMemory<double> *result) override;                      \
  bool DoBlasNrm2(Stream *stream, uint64 elem_count,                           \
                  const DeviceMemory<std::complex<float>> &x, int incx,        \
                  DeviceMemory<float> *result) override;                       \
  bool DoBlasNrm2(Stream *stream, uint64 elem_count,                           \
                  const DeviceMemory<std::complex<double>> &x, int incx,       \
                  DeviceMemory<double> *result) override;                      \
  bool DoBlasRot(Stream *stream, uint64 elem_count, DeviceMemory<float> *x,    \
                 int incx, DeviceMemory<float> *y, int incy, float c, float s) \
      override;                                                                \
  bool DoBlasRot(Stream *stream, uint64 elem_count, DeviceMemory<double> *x,   \
                 int incx, DeviceMemory<double> *y, int incy, double c,        \
                 double s) override;                                           \
  bool DoBlasRot(Stream *stream, uint64 elem_count,                            \
                 DeviceMemory<std::complex<float>> *x, int incx,               \
                 DeviceMemory<std::complex<float>> *y, int incy, float c,      \
                 float s) override;                                            \
  bool DoBlasRot(Stream *stream, uint64 elem_count,                            \
                 DeviceMemory<std::complex<double>> *x, int incx,              \
                 DeviceMemory<std::complex<double>> *y, int incy, double c,    \
                 double s) override;                                           \
  bool DoBlasRotg(Stream *stream, DeviceMemory<float> *a,                      \
                  DeviceMemory<float> *b, DeviceMemory<float> *c,              \
                  DeviceMemory<float> *s) override;                            \
  bool DoBlasRotg(Stream *stream, DeviceMemory<double> *a,                     \
                  DeviceMemory<double> *b, DeviceMemory<double> *c,            \
                  DeviceMemory<double> *s) override;                           \
  bool DoBlasRotg(Stream *stream, DeviceMemory<std::complex<float>> *a,        \
                  DeviceMemory<std::complex<float>> *b,                        \
                  DeviceMemory<float> *c,                                      \
                  DeviceMemory<std::complex<float>> *s) override;              \
  bool DoBlasRotg(Stream *stream, DeviceMemory<std::complex<double>> *a,       \
                  DeviceMemory<std::complex<double>> *b,                       \
                  DeviceMemory<double> *c,                                     \
                  DeviceMemory<std::complex<double>> *s) override;             \
  bool DoBlasRotm(Stream *stream, uint64 elem_count, DeviceMemory<float> *x,   \
                  int incx, DeviceMemory<float> *y, int incy,                  \
                  const DeviceMemory<float> &param) override;                  \
  bool DoBlasRotm(Stream *stream, uint64 elem_count, DeviceMemory<double> *x,  \
                  int incx, DeviceMemory<double> *y, int incy,                 \
                  const DeviceMemory<double> &param) override;                 \
  bool DoBlasRotmg(Stream *stream, DeviceMemory<float> *d1,                    \
                   DeviceMemory<float> *d2, DeviceMemory<float> *x1,           \
                   const DeviceMemory<float> &y1, DeviceMemory<float> *param)  \
      override;                                                                \
  bool DoBlasRotmg(Stream *stream, DeviceMemory<double> *d1,                   \
                   DeviceMemory<double> *d2, DeviceMemory<double> *x1,         \
                   const DeviceMemory<double> &y1,                             \
                   DeviceMemory<double> *param) override;                      \
  bool DoBlasScal(Stream *stream, uint64 elem_count, float alpha,              \
                  DeviceMemory<float> *x, int incx) override;                  \
  bool DoBlasScal(Stream *stream, uint64 elem_count, double alpha,             \
                  DeviceMemory<double> *x, int incx) override;                 \
  bool DoBlasScal(Stream *stream, uint64 elem_count, float alpha,              \
                  DeviceMemory<std::complex<float>> *x, int incx) override;    \
  bool DoBlasScal(Stream *stream, uint64 elem_count, double alpha,             \
                  DeviceMemory<std::complex<double>> *x, int incx) override;   \
  bool DoBlasScal(Stream *stream, uint64 elem_count,                           \
                  std::complex<float> alpha,                                   \
                  DeviceMemory<std::complex<float>> *x, int incx) override;    \
  bool DoBlasScal(Stream *stream, uint64 elem_count,                           \
                  std::complex<double> alpha,                                  \
                  DeviceMemory<std::complex<double>> *x, int incx) override;   \
  bool DoBlasSwap(Stream *stream, uint64 elem_count, DeviceMemory<float> *x,   \
                  int incx, DeviceMemory<float> *y, int incy) override;        \
  bool DoBlasSwap(Stream *stream, uint64 elem_count, DeviceMemory<double> *x,  \
                  int incx, DeviceMemory<double> *y, int incy) override;       \
  bool DoBlasSwap(Stream *stream, uint64 elem_count,                           \
                  DeviceMemory<std::complex<float>> *x, int incx,              \
                  DeviceMemory<std::complex<float>> *y, int incy) override;    \
  bool DoBlasSwap(Stream *stream, uint64 elem_count,                           \
                  DeviceMemory<std::complex<double>> *x, int incx,             \
                  DeviceMemory<std::complex<double>> *y, int incy) override;   \
  bool DoBlasIamax(Stream *stream, uint64 elem_count,                          \
                   const DeviceMemory<float> &x, int incx,                     \
                   DeviceMemory<int> *result) override;                        \
  bool DoBlasIamax(Stream *stream, uint64 elem_count,                          \
                   const DeviceMemory<double> &x, int incx,                    \
                   DeviceMemory<int> *result) override;                        \
  bool DoBlasIamax(Stream *stream, uint64 elem_count,                          \
                   const DeviceMemory<std::complex<float>> &x, int incx,       \
                   DeviceMemory<int> *result) override;                        \
  bool DoBlasIamax(Stream *stream, uint64 elem_count,                          \
                   const DeviceMemory<std::complex<double>> &x, int incx,      \
                   DeviceMemory<int> *result) override;                        \
  bool DoBlasIamin(Stream *stream, uint64 elem_count,                          \
                   const DeviceMemory<float> &x, int incx,                     \
                   DeviceMemory<int> *result) override;                        \
  bool DoBlasIamin(Stream *stream, uint64 elem_count,                          \
                   const DeviceMemory<double> &x, int incx,                    \
                   DeviceMemory<int> *result) override;                        \
  bool DoBlasIamin(Stream *stream, uint64 elem_count,                          \
                   const DeviceMemory<std::complex<float>> &x, int incx,       \
                   DeviceMemory<int> *result) override;                        \
  bool DoBlasIamin(Stream *stream, uint64 elem_count,                          \
                   const DeviceMemory<std::complex<double>> &x, int incx,      \
                   DeviceMemory<int> *result) override;                        \
  bool DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m, uint64 n,   \
                  uint64 kl, uint64 ku, float alpha,                           \
                  const DeviceMemory<float> &a, int lda,                       \
                  const DeviceMemory<float> &x, int incx, float beta,          \
                  DeviceMemory<float> *y, int incy) override;                  \
  bool DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m, uint64 n,   \
                  uint64 kl, uint64 ku, double alpha,                          \
                  const DeviceMemory<double> &a, int lda,                      \
                  const DeviceMemory<double> &x, int incx, double beta,        \
                  DeviceMemory<double> *y, int incy) override;                 \
  bool DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m, uint64 n,   \
                  uint64 kl, uint64 ku, std::complex<float> alpha,             \
                  const DeviceMemory<std::complex<float>> &a, int lda,         \
                  const DeviceMemory<std::complex<float>> &x, int incx,        \
                  std::complex<float> beta,                                    \
                  DeviceMemory<std::complex<float>> *y, int incy) override;    \
  bool DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m, uint64 n,   \
                  uint64 kl, uint64 ku, std::complex<double> alpha,            \
                  const DeviceMemory<std::complex<double>> &a, int lda,        \
                  const DeviceMemory<std::complex<double>> &x, int incx,       \
                  std::complex<double> beta,                                   \
                  DeviceMemory<std::complex<double>> *y, int incy) override;   \
  bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m, uint64 n,   \
                  float alpha, const DeviceMemory<float> &a, int lda,          \
                  const DeviceMemory<float> &x, int incx, float beta,          \
                  DeviceMemory<float> *y, int incy) override;                  \
  bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m, uint64 n,   \
                  double alpha, const DeviceMemory<double> &a, int lda,        \
                  const DeviceMemory<double> &x, int incx, double beta,        \
                  DeviceMemory<double> *y, int incy) override;                 \
  bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m, uint64 n,   \
                  std::complex<float> alpha,                                   \
                  const DeviceMemory<std::complex<float>> &a, int lda,         \
                  const DeviceMemory<std::complex<float>> &x, int incx,        \
                  std::complex<float> beta,                                    \
                  DeviceMemory<std::complex<float>> *y, int incy) override;    \
  bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m, uint64 n,   \
                  std::complex<double> alpha,                                  \
                  const DeviceMemory<std::complex<double>> &a, int lda,        \
                  const DeviceMemory<std::complex<double>> &x, int incx,       \
                  std::complex<double> beta,                                   \
                  DeviceMemory<std::complex<double>> *y, int incy) override;   \
  bool DoBlasGer(Stream *stream, uint64 m, uint64 n, float alpha,              \
                 const DeviceMemory<float> &x, int incx,                       \
                 const DeviceMemory<float> &y, int incy,                       \
                 DeviceMemory<float> *a, int lda) override;                    \
  bool DoBlasGer(Stream *stream, uint64 m, uint64 n, double alpha,             \
                 const DeviceMemory<double> &x, int incx,                      \
                 const DeviceMemory<double> &y, int incy,                      \
                 DeviceMemory<double> *a, int lda) override;                   \
  bool DoBlasGerc(Stream *stream, uint64 m, uint64 n,                          \
                  std::complex<float> alpha,                                   \
                  const DeviceMemory<std::complex<float>> &x, int incx,        \
                  const DeviceMemory<std::complex<float>> &y, int incy,        \
                  DeviceMemory<std::complex<float>> *a, int lda) override;     \
  bool DoBlasGerc(Stream *stream, uint64 m, uint64 n,                          \
                  std::complex<double> alpha,                                  \
                  const DeviceMemory<std::complex<double>> &x, int incx,       \
                  const DeviceMemory<std::complex<double>> &y, int incy,       \
                  DeviceMemory<std::complex<double>> *a, int lda) override;    \
  bool DoBlasGeru(Stream *stream, uint64 m, uint64 n,                          \
                  std::complex<float> alpha,                                   \
                  const DeviceMemory<std::complex<float>> &x, int incx,        \
                  const DeviceMemory<std::complex<float>> &y, int incy,        \
                  DeviceMemory<std::complex<float>> *a, int lda) override;     \
  bool DoBlasGeru(Stream *stream, uint64 m, uint64 n,                          \
                  std::complex<double> alpha,                                  \
                  const DeviceMemory<std::complex<double>> &x, int incx,       \
                  const DeviceMemory<std::complex<double>> &y, int incy,       \
                  DeviceMemory<std::complex<double>> *a, int lda) override;    \
  bool DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64 n, uint64 k,   \
                  std::complex<float> alpha,                                   \
                  const DeviceMemory<std::complex<float>> &a, int lda,         \
                  const DeviceMemory<std::complex<float>> &x, int incx,        \
                  std::complex<float> beta,                                    \
                  DeviceMemory<std::complex<float>> *y, int incy) override;    \
  bool DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64 n, uint64 k,   \
                  std::complex<double> alpha,                                  \
                  const DeviceMemory<std::complex<double>> &a, int lda,        \
                  const DeviceMemory<std::complex<double>> &x, int incx,       \
                  std::complex<double> beta,                                   \
                  DeviceMemory<std::complex<double>> *y, int incy) override;   \
  bool DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64 n,             \
                  std::complex<float> alpha,                                   \
                  const DeviceMemory<std::complex<float>> &a, int lda,         \
                  const DeviceMemory<std::complex<float>> &x, int incx,        \
                  std::complex<float> beta,                                    \
                  DeviceMemory<std::complex<float>> *y, int incy) override;    \
  bool DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64 n,             \
                  std::complex<double> alpha,                                  \
                  const DeviceMemory<std::complex<double>> &a, int lda,        \
                  const DeviceMemory<std::complex<double>> &x, int incx,       \
                  std::complex<double> beta,                                   \
                  DeviceMemory<std::complex<double>> *y, int incy) override;   \
  bool DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64 n, float alpha, \
                 const DeviceMemory<std::complex<float>> &x, int incx,         \
                 DeviceMemory<std::complex<float>> *a, int lda) override;      \
  bool DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64 n,              \
                 double alpha, const DeviceMemory<std::complex<double>> &x,    \
                 int incx, DeviceMemory<std::complex<double>> *a, int lda)     \
      override;                                                                \
  bool DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64 n,             \
                  std::complex<float> alpha,                                   \
                  const DeviceMemory<std::complex<float>> &x, int incx,        \
                  const DeviceMemory<std::complex<float>> &y, int incy,        \
                  DeviceMemory<std::complex<float>> *a, int lda) override;     \
  bool DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64 n,             \
                  std::complex<double> alpha,                                  \
                  const DeviceMemory<std::complex<double>> &x, int incx,       \
                  const DeviceMemory<std::complex<double>> &y, int incy,       \
                  DeviceMemory<std::complex<double>> *a, int lda) override;    \
  bool DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64 n,             \
                  std::complex<float> alpha,                                   \
                  const DeviceMemory<std::complex<float>> &ap,                 \
                  const DeviceMemory<std::complex<float>> &x, int incx,        \
                  std::complex<float> beta,                                    \
                  DeviceMemory<std::complex<float>> *y, int incy) override;    \
  bool DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64 n,             \
                  std::complex<double> alpha,                                  \
                  const DeviceMemory<std::complex<double>> &ap,                \
                  const DeviceMemory<std::complex<double>> &x, int incx,       \
                  std::complex<double> beta,                                   \
                  DeviceMemory<std::complex<double>> *y, int incy) override;   \
  bool DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64 n, float alpha, \
                 const DeviceMemory<std::complex<float>> &x, int incx,         \
                 DeviceMemory<std::complex<float>> *ap) override;              \
  bool DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64 n,              \
                 double alpha, const DeviceMemory<std::complex<double>> &x,    \
                 int incx, DeviceMemory<std::complex<double>> *ap) override;   \
  bool DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64 n,             \
                  std::complex<float> alpha,                                   \
                  const DeviceMemory<std::complex<float>> &x, int incx,        \
                  const DeviceMemory<std::complex<float>> &y, int incy,        \
                  DeviceMemory<std::complex<float>> *ap) override;             \
  bool DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64 n,             \
                  std::complex<double> alpha,                                  \
                  const DeviceMemory<std::complex<double>> &x, int incx,       \
                  const DeviceMemory<std::complex<double>> &y, int incy,       \
                  DeviceMemory<std::complex<double>> *ap) override;            \
  bool DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64 n, uint64 k,   \
                  float alpha, const DeviceMemory<float> &a, int lda,          \
                  const DeviceMemory<float> &x, int incx, float beta,          \
                  DeviceMemory<float> *y, int incy) override;                  \
  bool DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64 n, uint64 k,   \
                  double alpha, const DeviceMemory<double> &a, int lda,        \
                  const DeviceMemory<double> &x, int incx, double beta,        \
                  DeviceMemory<double> *y, int incy) override;                 \
  bool DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64 n,             \
                  float alpha, const DeviceMemory<float> &ap,                  \
                  const DeviceMemory<float> &x, int incx, float beta,          \
                  DeviceMemory<float> *y, int incy) override;                  \
  bool DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64 n,             \
                  double alpha, const DeviceMemory<double> &ap,                \
                  const DeviceMemory<double> &x, int incx, double beta,        \
                  DeviceMemory<double> *y, int incy) override;                 \
  bool DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64 n, float alpha, \
                 const DeviceMemory<float> &x, int incx,                       \
                 DeviceMemory<float> *ap) override;                            \
  bool DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64 n,              \
                 double alpha, const DeviceMemory<double> &x, int incx,        \
                 DeviceMemory<double> *ap) override;                           \
  bool DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64 n,             \
                  float alpha, const DeviceMemory<float> &x, int incx,         \
                  const DeviceMemory<float> &y, int incy,                      \
                  DeviceMemory<float> *ap) override;                           \
  bool DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64 n,             \
                  double alpha, const DeviceMemory<double> &x, int incx,       \
                  const DeviceMemory<double> &y, int incy,                     \
                  DeviceMemory<double> *ap) override;                          \
  bool DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64 n,             \
                  float alpha, const DeviceMemory<float> &a, int lda,          \
                  const DeviceMemory<float> &x, int incx, float beta,          \
                  DeviceMemory<float> *y, int incy) override;                  \
  bool DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64 n,             \
                  double alpha, const DeviceMemory<double> &a, int lda,        \
                  const DeviceMemory<double> &x, int incx, double beta,        \
                  DeviceMemory<double> *y, int incy) override;                 \
  bool DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64 n, float alpha, \
                 const DeviceMemory<float> &x, int incx,                       \
                 DeviceMemory<float> *a, int lda) override;                    \
  bool DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64 n,              \
                 double alpha, const DeviceMemory<double> &x, int incx,        \
                 DeviceMemory<double> *a, int lda) override;                   \
  bool DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64 n,             \
                  float alpha, const DeviceMemory<float> &x, int incx,         \
                  const DeviceMemory<float> &y, int incy,                      \
                  DeviceMemory<float> *a, int lda) override;                   \
  bool DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64 n,             \
                  double alpha, const DeviceMemory<double> &x, int incx,       \
                  const DeviceMemory<double> &y, int incy,                     \
                  DeviceMemory<double> *a, int lda) override;                  \
  bool DoBlasTbmv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  uint64 k, const DeviceMemory<float> &a, int lda,             \
                  DeviceMemory<float> *x, int incx) override;                  \
  bool DoBlasTbmv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  uint64 k, const DeviceMemory<double> &a, int lda,            \
                  DeviceMemory<double> *x, int incx) override;                 \
  bool DoBlasTbmv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  uint64 k, const DeviceMemory<std::complex<float>> &a,        \
                  int lda, DeviceMemory<std::complex<float>> *x, int incx)     \
      override;                                                                \
  bool DoBlasTbmv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  uint64 k, const DeviceMemory<std::complex<double>> &a,       \
                  int lda, DeviceMemory<std::complex<double>> *x, int incx)    \
      override;                                                                \
  bool DoBlasTbsv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  uint64 k, const DeviceMemory<float> &a, int lda,             \
                  DeviceMemory<float> *x, int incx) override;                  \
  bool DoBlasTbsv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  uint64 k, const DeviceMemory<double> &a, int lda,            \
                  DeviceMemory<double> *x, int incx) override;                 \
  bool DoBlasTbsv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  uint64 k, const DeviceMemory<std::complex<float>> &a,        \
                  int lda, DeviceMemory<std::complex<float>> *x, int incx)     \
      override;                                                                \
  bool DoBlasTbsv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  uint64 k, const DeviceMemory<std::complex<double>> &a,       \
                  int lda, DeviceMemory<std::complex<double>> *x, int incx)    \
      override;                                                                \
  bool DoBlasTpmv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  const DeviceMemory<float> &ap, DeviceMemory<float> *x,       \
                  int incx) override;                                          \
  bool DoBlasTpmv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  const DeviceMemory<double> &ap, DeviceMemory<double> *x,     \
                  int incx) override;                                          \
  bool DoBlasTpmv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  const DeviceMemory<std::complex<float>> &ap,                 \
                  DeviceMemory<std::complex<float>> *x, int incx) override;    \
  bool DoBlasTpmv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  const DeviceMemory<std::complex<double>> &ap,                \
                  DeviceMemory<std::complex<double>> *x, int incx) override;   \
  bool DoBlasTpsv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  const DeviceMemory<float> &ap, DeviceMemory<float> *x,       \
                  int incx) override;                                          \
  bool DoBlasTpsv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  const DeviceMemory<double> &ap, DeviceMemory<double> *x,     \
                  int incx) override;                                          \
  bool DoBlasTpsv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  const DeviceMemory<std::complex<float>> &ap,                 \
                  DeviceMemory<std::complex<float>> *x, int incx) override;    \
  bool DoBlasTpsv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  const DeviceMemory<std::complex<double>> &ap,                \
                  DeviceMemory<std::complex<double>> *x, int incx) override;   \
  bool DoBlasTrmv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  const DeviceMemory<float> &a, int lda,                       \
                  DeviceMemory<float> *x, int incx) override;                  \
  bool DoBlasTrmv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  const DeviceMemory<double> &a, int lda,                      \
                  DeviceMemory<double> *x, int incx) override;                 \
  bool DoBlasTrmv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  const DeviceMemory<std::complex<float>> &a, int lda,         \
                  DeviceMemory<std::complex<float>> *x, int incx) override;    \
  bool DoBlasTrmv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  const DeviceMemory<std::complex<double>> &a, int lda,        \
                  DeviceMemory<std::complex<double>> *x, int incx) override;   \
  bool DoBlasTrsv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  const DeviceMemory<float> &a, int lda,                       \
                  DeviceMemory<float> *x, int incx) override;                  \
  bool DoBlasTrsv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  const DeviceMemory<double> &a, int lda,                      \
                  DeviceMemory<double> *x, int incx) override;                 \
  bool DoBlasTrsv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  const DeviceMemory<std::complex<float>> &a, int lda,         \
                  DeviceMemory<std::complex<float>> *x, int incx) override;    \
  bool DoBlasTrsv(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, blas::Diagonal diag, uint64 n,        \
                  const DeviceMemory<std::complex<double>> &a, int lda,        \
                  DeviceMemory<std::complex<double>> *x, int incx) override;   \
  bool DoBlasGemm(Stream *stream, blas::Transpose transa,                      \
                  blas::Transpose transb, uint64 m, uint64 n, uint64 k,        \
                  float alpha, const DeviceMemory<float> &a, int lda,          \
                  const DeviceMemory<float> &b, int ldb, float beta,           \
                  DeviceMemory<float> *c, int ldc) override;                   \
  bool DoBlasGemm(Stream *stream, blas::Transpose transa,                      \
                  blas::Transpose transb, uint64 m, uint64 n, uint64 k,        \
                  double alpha, const DeviceMemory<double> &a, int lda,        \
                  const DeviceMemory<double> &b, int ldb, double beta,         \
                  DeviceMemory<double> *c, int ldc) override;                  \
  bool DoBlasGemm(Stream *stream, blas::Transpose transa,                      \
                  blas::Transpose transb, uint64 m, uint64 n, uint64 k,        \
                  std::complex<float> alpha,                                   \
                  const DeviceMemory<std::complex<float>> &a, int lda,         \
                  const DeviceMemory<std::complex<float>> &b, int ldb,         \
                  std::complex<float> beta,                                    \
                  DeviceMemory<std::complex<float>> *c, int ldc) override;     \
  bool DoBlasGemm(Stream *stream, blas::Transpose transa,                      \
                  blas::Transpose transb, uint64 m, uint64 n, uint64 k,        \
                  std::complex<double> alpha,                                  \
                  const DeviceMemory<std::complex<double>> &a, int lda,        \
                  const DeviceMemory<std::complex<double>> &b, int ldb,        \
                  std::complex<double> beta,                                   \
                  DeviceMemory<std::complex<double>> *c, int ldc) override;    \
  bool DoBlasGemmBatched(                                                      \
      Stream *stream, blas::Transpose transa, blas::Transpose transb,          \
      uint64 m, uint64 n, uint64 k, float alpha,                               \
      const port::ArraySlice<DeviceMemory<float> *> &a, int lda,               \
      const port::ArraySlice<DeviceMemory<float> *> &b, int ldb, float beta,   \
      const port::ArraySlice<DeviceMemory<float> *> &c, int ldc,               \
      int batch_count, ScratchAllocator *scratch_allocator) override;          \
  bool DoBlasGemmBatched(                                                      \
      Stream *stream, blas::Transpose transa, blas::Transpose transb,          \
      uint64 m, uint64 n, uint64 k, double alpha,                              \
      const port::ArraySlice<DeviceMemory<double> *> &a, int lda,              \
      const port::ArraySlice<DeviceMemory<double> *> &b, int ldb, double beta, \
      const port::ArraySlice<DeviceMemory<double> *> &c, int ldc,              \
      int batch_count, ScratchAllocator *scratch_allocator) override;          \
  bool DoBlasGemmBatched(                                                      \
      Stream *stream, blas::Transpose transa, blas::Transpose transb,          \
      uint64 m, uint64 n, uint64 k, std::complex<float> alpha,                 \
      const port::ArraySlice<DeviceMemory<std::complex<float>> *> &a, int lda, \
      const port::ArraySlice<DeviceMemory<std::complex<float>> *> &b, int ldb, \
      std::complex<float> beta,                                                \
      const port::ArraySlice<DeviceMemory<std::complex<float>> *> &c, int ldc, \
      int batch_count, ScratchAllocator *scratch_allocator) override;          \
  bool DoBlasGemmBatched(                                                      \
      Stream *stream, blas::Transpose transa, blas::Transpose transb,          \
      uint64 m, uint64 n, uint64 k, std::complex<double> alpha,                \
      const port::ArraySlice<DeviceMemory<std::complex<double>> *> &a,         \
      int lda,                                                                 \
      const port::ArraySlice<DeviceMemory<std::complex<double>> *> &b,         \
      int ldb, std::complex<double> beta,                                      \
      const port::ArraySlice<DeviceMemory<std::complex<double>> *> &c,         \
      int ldc, int batch_count, ScratchAllocator *scratch_allocator) override; \
  bool DoBlasHemm(Stream *stream, blas::Side side, blas::UpperLower uplo,      \
                  uint64 m, uint64 n, std::complex<float> alpha,               \
                  const DeviceMemory<std::complex<float>> &a, int lda,         \
                  const DeviceMemory<std::complex<float>> &b, int ldb,         \
                  std::complex<float> beta,                                    \
                  DeviceMemory<std::complex<float>> *c, int ldc) override;     \
  bool DoBlasHemm(Stream *stream, blas::Side side, blas::UpperLower uplo,      \
                  uint64 m, uint64 n, std::complex<double> alpha,              \
                  const DeviceMemory<std::complex<double>> &a, int lda,        \
                  const DeviceMemory<std::complex<double>> &b, int ldb,        \
                  std::complex<double> beta,                                   \
                  DeviceMemory<std::complex<double>> *c, int ldc) override;    \
  bool DoBlasHerk(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, uint64 n, uint64 k, float alpha,      \
                  const DeviceMemory<std::complex<float>> &a, int lda,         \
                  float beta, DeviceMemory<std::complex<float>> *c, int ldc)   \
      override;                                                                \
  bool DoBlasHerk(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, uint64 n, uint64 k, double alpha,     \
                  const DeviceMemory<std::complex<double>> &a, int lda,        \
                  double beta, DeviceMemory<std::complex<double>> *c, int ldc) \
      override;                                                                \
  bool DoBlasHer2k(                                                            \
      Stream *stream, blas::UpperLower uplo, blas::Transpose trans, uint64 n,  \
      uint64 k, std::complex<float> alpha,                                     \
      const DeviceMemory<std::complex<float>> &a, int lda,                     \
      const DeviceMemory<std::complex<float>> &b, int ldb, float beta,         \
      DeviceMemory<std::complex<float>> *c, int ldc) override;                 \
  bool DoBlasHer2k(                                                            \
      Stream *stream, blas::UpperLower uplo, blas::Transpose trans, uint64 n,  \
      uint64 k, std::complex<double> alpha,                                    \
      const DeviceMemory<std::complex<double>> &a, int lda,                    \
      const DeviceMemory<std::complex<double>> &b, int ldb, double beta,       \
      DeviceMemory<std::complex<double>> *c, int ldc) override;                \
  bool DoBlasSymm(Stream *stream, blas::Side side, blas::UpperLower uplo,      \
                  uint64 m, uint64 n, float alpha,                             \
                  const DeviceMemory<float> &a, int lda,                       \
                  const DeviceMemory<float> &b, int ldb, float beta,           \
                  DeviceMemory<float> *c, int ldc) override;                   \
  bool DoBlasSymm(Stream *stream, blas::Side side, blas::UpperLower uplo,      \
                  uint64 m, uint64 n, double alpha,                            \
                  const DeviceMemory<double> &a, int lda,                      \
                  const DeviceMemory<double> &b, int ldb, double beta,         \
                  DeviceMemory<double> *c, int ldc) override;                  \
  bool DoBlasSymm(Stream *stream, blas::Side side, blas::UpperLower uplo,      \
                  uint64 m, uint64 n, std::complex<float> alpha,               \
                  const DeviceMemory<std::complex<float>> &a, int lda,         \
                  const DeviceMemory<std::complex<float>> &b, int ldb,         \
                  std::complex<float> beta,                                    \
                  DeviceMemory<std::complex<float>> *c, int ldc) override;     \
  bool DoBlasSymm(Stream *stream, blas::Side side, blas::UpperLower uplo,      \
                  uint64 m, uint64 n, std::complex<double> alpha,              \
                  const DeviceMemory<std::complex<double>> &a, int lda,        \
                  const DeviceMemory<std::complex<double>> &b, int ldb,        \
                  std::complex<double> beta,                                   \
                  DeviceMemory<std::complex<double>> *c, int ldc) override;    \
  bool DoBlasSyrk(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, uint64 n, uint64 k, float alpha,      \
                  const DeviceMemory<float> &a, int lda, float beta,           \
                  DeviceMemory<float> *c, int ldc) override;                   \
  bool DoBlasSyrk(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, uint64 n, uint64 k, double alpha,     \
                  const DeviceMemory<double> &a, int lda, double beta,         \
                  DeviceMemory<double> *c, int ldc) override;                  \
  bool DoBlasSyrk(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, uint64 n, uint64 k,                   \
                  std::complex<float> alpha,                                   \
                  const DeviceMemory<std::complex<float>> &a, int lda,         \
                  std::complex<float> beta,                                    \
                  DeviceMemory<std::complex<float>> *c, int ldc) override;     \
  bool DoBlasSyrk(Stream *stream, blas::UpperLower uplo,                       \
                  blas::Transpose trans, uint64 n, uint64 k,                   \
                  std::complex<double> alpha,                                  \
                  const DeviceMemory<std::complex<double>> &a, int lda,        \
                  std::complex<double> beta,                                   \
                  DeviceMemory<std::complex<double>> *c, int ldc) override;    \
  bool DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,                      \
                   blas::Transpose trans, uint64 n, uint64 k, float alpha,     \
                   const DeviceMemory<float> &a, int lda,                      \
                   const DeviceMemory<float> &b, int ldb, float beta,          \
                   DeviceMemory<float> *c, int ldc) override;                  \
  bool DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,                      \
                   blas::Transpose trans, uint64 n, uint64 k, double alpha,    \
                   const DeviceMemory<double> &a, int lda,                     \
                   const DeviceMemory<double> &b, int ldb, double beta,        \
                   DeviceMemory<double> *c, int ldc) override;                 \
  bool DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,                      \
                   blas::Transpose trans, uint64 n, uint64 k,                  \
                   std::complex<float> alpha,                                  \
                   const DeviceMemory<std::complex<float>> &a, int lda,        \
                   const DeviceMemory<std::complex<float>> &b, int ldb,        \
                   std::complex<float> beta,                                   \
                   DeviceMemory<std::complex<float>> *c, int ldc) override;    \
  bool DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,                      \
                   blas::Transpose trans, uint64 n, uint64 k,                  \
                   std::complex<double> alpha,                                 \
                   const DeviceMemory<std::complex<double>> &a, int lda,       \
                   const DeviceMemory<std::complex<double>> &b, int ldb,       \
                   std::complex<double> beta,                                  \
                   DeviceMemory<std::complex<double>> *c, int ldc) override;   \
  bool DoBlasTrmm(Stream *stream, blas::Side side, blas::UpperLower uplo,      \
                  blas::Transpose transa, blas::Diagonal diag, uint64 m,       \
                  uint64 n, float alpha, const DeviceMemory<float> &a,         \
                  int lda, DeviceMemory<float> *b, int ldb) override;          \
  bool DoBlasTrmm(Stream *stream, blas::Side side, blas::UpperLower uplo,      \
                  blas::Transpose transa, blas::Diagonal diag, uint64 m,       \
                  uint64 n, double alpha, const DeviceMemory<double> &a,       \
                  int lda, DeviceMemory<double> *b, int ldb) override;         \
  bool DoBlasTrmm(Stream *stream, blas::Side side, blas::UpperLower uplo,      \
                  blas::Transpose transa, blas::Diagonal diag, uint64 m,       \
                  uint64 n, std::complex<float> alpha,                         \
                  const DeviceMemory<std::complex<float>> &a, int lda,         \
                  DeviceMemory<std::complex<float>> *b, int ldb) override;     \
  bool DoBlasTrmm(Stream *stream, blas::Side side, blas::UpperLower uplo,      \
                  blas::Transpose transa, blas::Diagonal diag, uint64 m,       \
                  uint64 n, std::complex<double> alpha,                        \
                  const DeviceMemory<std::complex<double>> &a, int lda,        \
                  DeviceMemory<std::complex<double>> *b, int ldb) override;    \
  bool DoBlasTrsm(Stream *stream, blas::Side side, blas::UpperLower uplo,      \
                  blas::Transpose transa, blas::Diagonal diag, uint64 m,       \
                  uint64 n, float alpha, const DeviceMemory<float> &a,         \
                  int lda, DeviceMemory<float> *b, int ldb) override;          \
  bool DoBlasTrsm(Stream *stream, blas::Side side, blas::UpperLower uplo,      \
                  blas::Transpose transa, blas::Diagonal diag, uint64 m,       \
                  uint64 n, double alpha, const DeviceMemory<double> &a,       \
                  int lda, DeviceMemory<double> *b, int ldb) override;         \
  bool DoBlasTrsm(Stream *stream, blas::Side side, blas::UpperLower uplo,      \
                  blas::Transpose transa, blas::Diagonal diag, uint64 m,       \
                  uint64 n, std::complex<float> alpha,                         \
                  const DeviceMemory<std::complex<float>> &a, int lda,         \
                  DeviceMemory<std::complex<float>> *b, int ldb) override;     \
  bool DoBlasTrsm(Stream *stream, blas::Side side, blas::UpperLower uplo,      \
                  blas::Transpose transa, blas::Diagonal diag, uint64 m,       \
                  uint64 n, std::complex<double> alpha,                        \
                  const DeviceMemory<std::complex<double>> &a, int lda,        \
                  DeviceMemory<std::complex<double>> *b, int ldb) override;

}  // namespace blas
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_BLAS_H_
