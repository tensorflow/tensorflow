/* Copyright 2015 The OpenXLA Authors.

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
// Note that this interface is optionally supported by platforms.

#ifndef XLA_STREAM_EXECUTOR_BLAS_H_
#define XLA_STREAM_EXECUTOR_BLAS_H_

#include <complex>
#include <cstdint>
#include <limits>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/stream_executor/data_type.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/numeric_options.h"
#include "xla/stream_executor/platform/port.h"
#include "tsl/platform/errors.h"
#include "tsl/protobuf/dnn.pb.h"

namespace Eigen {
struct half;
}  // namespace Eigen

namespace stream_executor {

namespace gpu {
struct BlasLt;
struct MatrixDescriptor;
struct OutputMatrixDescriptor;
}  // namespace gpu

class Stream;
class ScratchAllocator;

template <typename ElemT>
class DeviceMemory;

template <typename ElemT>
class HostOrDeviceScalar;

template <typename T>
using DeviceMemorySlice = absl::Span<DeviceMemory<T> *const>;

namespace blas {

// Specifies whether the input matrix will be transposed or
// transposed+conjugated before any BLAS operations.
enum class Transpose { kNoTranspose, kTranspose, kConjugateTranspose };

// Returns a name for t.
std::string TransposeString(Transpose t);

// Specifies whether the upper or lower triangular part of a
// symmetric/Hermitian matrix is used.
enum class UpperLower { kUpper, kLower };

// Returns a name for ul.
std::string UpperLowerString(UpperLower ul);

// Specifies whether a matrix is unit triangular.
enum class Diagonal { kUnit, kNonUnit };

// Returns a name for d.
std::string DiagonalString(Diagonal d);

// Specifies whether a Hermitian matrix appears on the left or right in
// operation.
enum class Side { kLeft, kRight };

// Returns a name for s.
std::string SideString(Side s);

// Type with which intermediate computations of a blas routine are performed.
//
// Some blas calls can perform computations with a type that's different than
// the type of their inputs/outputs.  This lets you e.g. multiply two matrices
// of int8s using float32s to store the matmul's intermediate values.
enum class ComputationType {
  kF16,  // 16-bit floating-point
  kF32,  // 32-bit floating-point
  kF64,  // 64-bit floating-point
  kI32,  // 32-bit integer
  // The below values use float32 for accumulation, but allow the inputs and
  // outputs to be downcast to a lower precision:
  kF16AsF32,   // Allow downcast to F16 precision.
  kBF16AsF32,  // Allow downcast to BF16 precision.
  kTF32AsF32,  // Allow downcast to TF32 precision.
};

// Call context information for GEMM API calls
// This is extra information that can optionally be passed down to the blas
// library, so that it can pick the efficient imlpementation based on context
enum class CallContext {
  kNone = 0,            // No information
  kForward = 1,         // call happens in "forward" pass
  kBackpropInput1 = 2,  // call happens in "backprop" pass for the first input
  kBackpropInput2 = 4,  // call happens in "backprop" pass for the second input
};

// Converts a ComputationType to a string.
std::string ComputationTypeString(ComputationType ty);

std::ostream &operator<<(std::ostream &os, ComputationType ty);

using dnn::DataType;
using dnn::ToDataType;

// Converts a ComputationType to a string.
std::string DataTypeString(DataType ty);

std::ostream &operator<<(std::ostream &os, DataType ty);

// Opaque identifier for an "algorithm" used by a blas routine.  This functions
// as a hint to the blas library.
typedef int64_t AlgorithmType;
constexpr AlgorithmType kDefaultAlgorithm = -1;
constexpr AlgorithmType kDefaultBlasGemm = -2;
constexpr AlgorithmType kDefaultBlasGemv = -3;
constexpr AlgorithmType kNoAlgorithm = -4;
constexpr AlgorithmType kRuntimeAutotuning = -5;

// blas uses -1 to represent the default algorithm. This happens to match up
// with the CUBLAS_GEMM_DFALT constant, so cuda_blas.cc is using static_cast
// to convert from AlgorithmType to cublasGemmAlgo_t, and uses a static_assert
// to ensure that this assumption does not break.
// If another blas implementation uses a different value for the default
// algorithm, then it needs to convert kDefaultGemmAlgo to that value
// (e.g. via a function called ToWhateverGemmAlgo).
constexpr AlgorithmType kDefaultGemmAlgo = -1;

// Describes the result of a performance experiment, usually timing the speed of
// a particular AlgorithmType.
//
// If the call we were benchmarking failed (a common occurrence; not all
// algorithms are valid for all calls), is_valid() will be false.
class ProfileResult {
 public:
  bool is_valid() const { return is_valid_; }
  void set_is_valid(bool val) { is_valid_ = val; }
  bool warmup_run_executed() const { return warmup_run_executed_; }
  void set_warmup_run_executed(bool val) { warmup_run_executed_ = val; }
  AlgorithmType algorithm() const { return algorithm_; }
  void set_algorithm(AlgorithmType val) { algorithm_ = val; }
  float elapsed_time_in_ms() const { return elapsed_time_in_ms_; }
  void set_elapsed_time_in_ms(float val) { elapsed_time_in_ms_ = val; }

 private:
  bool is_valid_ = false, warmup_run_executed_ = false;
  AlgorithmType algorithm_ = kDefaultAlgorithm;
  float elapsed_time_in_ms_ = std::numeric_limits<float>::max();
};

class AlgorithmConfig {
 public:
  AlgorithmConfig() : algorithm_(kDefaultAlgorithm) {}
  explicit AlgorithmConfig(AlgorithmType algorithm) : algorithm_(algorithm) {}
  AlgorithmType algorithm() const { return algorithm_; }
  void set_algorithm(AlgorithmType val) { algorithm_ = val; }
  bool operator==(const AlgorithmConfig &other) const {
    return this->algorithm_ == other.algorithm_;
  }
  bool operator!=(const AlgorithmConfig &other) const {
    return !(*this == other);
  }
  std::string ToString() const;

 private:
  AlgorithmType algorithm_;
};

// Opaque identifier specifying the precision to use in gemm calls.
typedef int64_t ComputePrecision;
constexpr ComputePrecision kDefaultComputePrecision = 0;

namespace detail {

// Helper to return if `T` is the same type as `First` or any or `Rest`.
template <typename T>
constexpr bool is_any_of() {
  return false;
}

template <typename T, typename First, typename... Rest>
constexpr bool is_any_of() {
  return std::is_same_v<T, First> || is_any_of<T, Rest...>();
}

}  // namespace detail

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

  virtual gpu::BlasLt *GetBlasLt() = 0;

  // Performs a BLAS y <- ax+y operation.
  virtual bool DoBlasAxpy(Stream *stream, uint64_t elem_count, float alpha,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) = 0;

  // Copies vector to another vector: y <- x.
  virtual bool DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) = 0;

  // Computes the product of a vector by a scalar: x <- a*x.
  virtual bool DoBlasScal(Stream *stream, uint64_t elem_count, float alpha,
                          DeviceMemory<float> *x, int incx) = 0;
  virtual bool DoBlasScal(Stream *stream, uint64_t elem_count, double alpha,
                          DeviceMemory<double> *x, int incx) = 0;
  virtual bool DoBlasScal(Stream *stream, uint64_t elem_count, float alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) = 0;
  virtual bool DoBlasScal(Stream *stream, uint64_t elem_count, double alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) = 0;
  virtual bool DoBlasScal(Stream *stream, uint64_t elem_count,
                          std::complex<float> alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) = 0;
  virtual bool DoBlasScal(Stream *stream, uint64_t elem_count,
                          std::complex<double> alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) = 0;

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
  virtual bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) = 0;
  virtual bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) = 0;
  virtual bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) = 0;
  virtual bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) = 0;

  // Computes a matrix-vector product using a symmetric band matrix.
  //
  //     y <- alpha * a * x + beta * y,
  //
  // alpha and beta are scalars; a is an n-by-n symmetric band matrix, with k
  // super-diagonals; x and y are n-element vectors.
  virtual bool DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          uint64_t k, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) = 0;

  // Computes a matrix-matrix product with general matrices:
  //
  //     c <- alpha * op(a) * op(b) + beta * c,
  //
  // op(X) is one of op(X) = X, or op(X) = X', or op(X) = conj(X'); alpha and
  // beta are scalars; a, b, and c are matrices; op(a) is an m-by-k matrix;
  // op(b) is a k-by-n matrix; c is an m-by-n matrix.
  //
  // Note: The half interface uses float precision internally; the version
  // that uses half precision internally is not yet supported. There is no
  // batched version of the half-precision interface.
  //
  // Alpha/beta type matches `dtype`, unless `dtype` is `Eigen::half`, in that
  // case the expected alpha/beta type is `float`.
  virtual absl::Status DoBlasGemm(
      Stream *stream, blas::Transpose transa, blas::Transpose transb,
      uint64_t m, uint64 n, uint64_t k, DataType dtype, const void *alpha,
      const DeviceMemoryBase &a, int lda, const DeviceMemoryBase &b, int ldb,
      const void *beta, DeviceMemoryBase *c, int ldc,
      const NumericOptions &numeric_options, blas::CallContext context) = 0;

  // Gets a list of supported algorithms for DoBlasGemmWithAlgorithm.
  virtual bool GetBlasGemmAlgorithms(
      Stream *stream, const gpu::MatrixDescriptor &a,
      const gpu::MatrixDescriptor &b, gpu::OutputMatrixDescriptor *c,
      const void *alpha, const void *beta,
      std::vector<blas::AlgorithmType> *out_algorithms) = 0;

  // Like DoBlasGemm, but accepts an algorithm and an compute type.
  //
  // The compute type lets you say (e.g.) that the inputs and outputs are
  // Eigen::halfs, but you want the internal computations to be done with
  // float32 precision.
  //
  // If output_profile_result is not null, a failure here does not put the
  // stream in a failure state.  Instead, success/failure is indicated by
  // output_profile_result->is_valid().  This lets you use this function for
  // choosing the best algorithm among many (some of which may fail) without
  // creating a new Stream for each attempt.
  virtual absl::Status DoBlasGemmWithAlgorithm(
      Stream *stream, blas::Transpose transa, blas::Transpose transb,
      uint64_t m, uint64_t n, uint64 k, const void *alpha,
      const DeviceMemoryBase &a, DataType type_a, int lda,
      const DeviceMemoryBase &b, DataType type_b, int ldb, const void *beta,
      DeviceMemoryBase *c, DataType type_c, int ldc,
      ComputationType computation_type, AlgorithmType algorithm,
      const NumericOptions &numeric_options,
      ProfileResult *output_profile_result, blas::CallContext context) = 0;
  virtual absl::Status DoBlasGemmStridedBatchedWithAlgorithm(
      Stream *stream, blas::Transpose transa, blas::Transpose transb,
      uint64_t m, uint64_t n, uint64 k, const void *alpha,
      const DeviceMemoryBase &a, DataType type_a, int lda, int64_t stride_a,
      const DeviceMemoryBase &b, DataType type_b, int ldb, int64_t stride_b,
      const void *beta, DeviceMemoryBase *c, DataType type_c, int ldc,
      int64_t stride_c, int batch_count, ComputationType computation_type,
      AlgorithmType algorithm, const NumericOptions &numeric_options,
      ProfileResult *output_profile_result, blas::CallContext context) = 0;

  // Computes a batch of matrix-matrix product with general matrices.
  // This is a batched version of DoBlasGemm.
  // The batched GEMM computes matrix product for each input/output in a, b,
  // and c, which contain batch_count DeviceMemory objects.
  virtual bool DoBlasGemmBatched(Stream *stream, blas::Transpose transa,
                                 blas::Transpose transb, uint64_t m, uint64_t n,
                                 uint64 k, float alpha,
                                 DeviceMemorySlice<Eigen::half> a, int lda,
                                 DeviceMemorySlice<Eigen::half> b, int ldb,
                                 float beta, DeviceMemorySlice<Eigen::half> c,
                                 int ldc, int batch_count,
                                 const NumericOptions &numeric_options,
                                 ScratchAllocator *scratch_allocator,
                                 blas::CallContext context) = 0;
  virtual bool DoBlasGemmBatched(
      Stream *stream, blas::Transpose transa, blas::Transpose transb,
      uint64_t m, uint64_t n, uint64 k, float alpha,
      DeviceMemorySlice<Eigen::bfloat16> a, int lda,
      DeviceMemorySlice<Eigen::bfloat16> b, int ldb, float beta,
      DeviceMemorySlice<Eigen::bfloat16> c, int ldc, int batch_count,
      const NumericOptions &numeric_options,
      ScratchAllocator *scratch_allocator, blas::CallContext context) = 0;
  virtual bool DoBlasGemmBatched(
      Stream *stream, blas::Transpose transa, blas::Transpose transb,
      uint64_t m, uint64_t n, uint64 k, float alpha, DeviceMemorySlice<float> a,
      int lda, DeviceMemorySlice<float> b, int ldb, float beta,
      DeviceMemorySlice<float> c, int ldc, int batch_count,
      const NumericOptions &numeric_options,
      ScratchAllocator *scratch_allocator, blas::CallContext context) = 0;
  virtual bool DoBlasGemmBatched(
      Stream *stream, blas::Transpose transa, blas::Transpose transb,
      uint64_t m, uint64_t n, uint64 k, double alpha,
      DeviceMemorySlice<double> a, int lda, DeviceMemorySlice<double> b,
      int ldb, double beta, DeviceMemorySlice<double> c, int ldc,
      int batch_count, const NumericOptions &numeric_options,
      ScratchAllocator *scratch_allocator, blas::CallContext context) = 0;
  virtual bool DoBlasGemmBatched(
      Stream *stream, blas::Transpose transa, blas::Transpose transb,
      uint64_t m, uint64_t n, uint64 k, std::complex<float> alpha,
      DeviceMemorySlice<std::complex<float>> a, int lda,
      DeviceMemorySlice<std::complex<float>> b, int ldb,
      std::complex<float> beta, DeviceMemorySlice<std::complex<float>> c,
      int ldc, int batch_count, const NumericOptions &numeric_options,
      ScratchAllocator *scratch_allocator, blas::CallContext context) = 0;
  virtual bool DoBlasGemmBatched(
      Stream *stream, blas::Transpose transa, blas::Transpose transb,
      uint64_t m, uint64_t n, uint64 k, std::complex<double> alpha,
      DeviceMemorySlice<std::complex<double>> a, int lda,
      DeviceMemorySlice<std::complex<double>> b, int ldb,
      std::complex<double> beta, DeviceMemorySlice<std::complex<double>> c,
      int ldc, int batch_count, const NumericOptions &numeric_options,
      ScratchAllocator *scratch_allocator, blas::CallContext context) = 0;
  // Batched gemm with strides instead of pointer arrays.
  virtual absl::Status DoBlasGemmStridedBatched(
      Stream *stream, blas::Transpose transa, blas::Transpose transb,
      uint64_t m, uint64_t n, uint64 k, DataType dtype, const void *alpha,
      const DeviceMemoryBase &a, int lda, int64_t stride_a,
      const DeviceMemoryBase &b, int ldb, int64_t stride_b, const void *beta,
      DeviceMemoryBase *c, int ldc, int64_t stride_c, int batch_count,
      const NumericOptions &numeric_options, blas::CallContext context) = 0;

  template <typename InputType, typename OutputType, typename ConstantType>
  absl::Status BlasGemmStridedBatchedWithAlgorithm(
      Stream *stream, blas::Transpose transa, blas::Transpose transb,
      uint64_t m, uint64 n, uint64_t k, ConstantType alpha,
      const DeviceMemory<InputType> &a, int lda, int64_t stride_a,
      const DeviceMemory<InputType> &b, int ldb, int64_t stride_b,
      ConstantType beta, DeviceMemory<OutputType> *c, int ldc, int64_t stride_c,
      int batch_count, blas::ComputationType computation_type,
      blas::AlgorithmType algorithm, const NumericOptions &numeric_options,
      blas::ProfileResult *output_profile_result, blas::CallContext context) {
    TF_RETURN_IF_ERROR(
        CheckTypesForExtendedBlas<InputType, OutputType, ConstantType>(
            computation_type));

    void *alpha_ptr = &alpha;
    void *beta_ptr = &beta;
    float alpha_storage, beta_storage;
    UpcastHalfToFloat<ConstantType>(&alpha_ptr, &beta_ptr, &alpha_storage,
                                    &beta_storage);
    absl::Status status = DoBlasGemmStridedBatchedWithAlgorithm(
        stream, transa, transb, m, n, k, alpha_ptr, a,
        blas::ToDataType<InputType>::value, lda, stride_a, b,
        blas::ToDataType<InputType>::value, ldb, stride_b, beta_ptr, c,
        blas::ToDataType<OutputType>::value, ldc, stride_c, batch_count,
        computation_type, algorithm, numeric_options, output_profile_result,
        context);
    if (output_profile_result) {
      // The error is recorded in the profile.
      return absl::OkStatus();
    }
    return status;
  }

  template <typename InputType, typename OutputType, typename ConstantType>
  absl::Status BlasGemm(Stream *stream, blas::Transpose transa,
                        blas::Transpose transb, uint64_t m, uint64 n, uint64 k,
                        ConstantType alpha, const DeviceMemory<InputType> &a,
                        int lda, const DeviceMemory<InputType> &b, int ldb,
                        ConstantType beta, DeviceMemory<OutputType> *c, int ldc,
                        const NumericOptions &numeric_options,
                        blas::CallContext context) {
    static_assert(
        detail::is_any_of<InputType, int8_t, Eigen::half, Eigen::bfloat16,
                          float, double, std::complex<float>,
                          std::complex<double>>(),
        "Input can be int8_t, half, bf16, float, double, std::complex<float> "
        "or "
        "std::complex<double>");
    static_assert(!std::is_same_v<InputType, Eigen::half> ||
                      detail::is_any_of<ConstantType, float, Eigen::half>(),
                  "If input is Eigen::half, constant has to be either "
                  "Eigen::half or float");
    static_assert(detail::is_any_of<InputType, int8_t, Eigen::half,
                                    Eigen::bfloat16, ConstantType>(),
                  "If input is not int8_t, Eigen::half, constant and input "
                  "types have to match");
    void *alpha_ptr = &alpha;
    void *beta_ptr = &beta;
    float alpha_storage, beta_storage;
    UpcastHalfToFloat<ConstantType>(&alpha_ptr, &beta_ptr, &alpha_storage,
                                    &beta_storage);

    return DoBlasGemm(stream, transa, transb, m, n, k,
                      blas::ToDataType<InputType>::value, alpha_ptr, a, lda, b,
                      ldb, beta_ptr, c, ldc, numeric_options, context);
  }

  template <typename InputType, typename OutputType>
  absl::Status BlasGemm(Stream *stream, blas::Transpose transa,
                        blas::Transpose transb, uint64_t m, uint64 n, uint64 k,
                        const DeviceMemory<InputType> &a, int lda,
                        const DeviceMemory<InputType> &b, int ldb,
                        DeviceMemory<OutputType> *c, int ldc,
                        const NumericOptions &numeric_options,
                        blas::CallContext context) {
    InputType alpha{1.0};
    InputType beta{0.0};
    return BlasGemm(stream, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                    beta, c, ldc, numeric_options, context);
  }

  template <typename InputType, typename OutputType, typename ConstantType>
  absl::Status BlasGemmWithAlgorithm(
      Stream *stream, blas::Transpose transa, blas::Transpose transb,
      uint64_t m, uint64 n, uint64_t k, ConstantType alpha,
      const DeviceMemory<InputType> &a, int lda,
      const DeviceMemory<InputType> &b, int ldb, ConstantType beta,
      DeviceMemory<OutputType> *c, int ldc,
      blas::ComputationType computation_type, blas::AlgorithmType algorithm,
      const NumericOptions &numeric_options,
      blas::ProfileResult *output_profile_result, blas::CallContext context) {
    TF_RETURN_IF_ERROR(
        CheckTypesForExtendedBlas<InputType, OutputType, ConstantType>(
            computation_type));

    void *alpha_ptr = &alpha;
    void *beta_ptr = &beta;
    float alpha_storage, beta_storage;
    UpcastHalfToFloat<ConstantType>(&alpha_ptr, &beta_ptr, &alpha_storage,
                                    &beta_storage);

    absl::Status st = DoBlasGemmWithAlgorithm(
        stream, transa, transb, m, n, k, alpha_ptr, a,
        blas::ToDataType<InputType>::value, lda, b,
        blas::ToDataType<InputType>::value, ldb, beta_ptr, c,
        blas::ToDataType<OutputType>::value, ldc, computation_type, algorithm,
        numeric_options, output_profile_result, context);

    if (output_profile_result) {
      // The error is recorded in the profile.
      return absl::OkStatus();
    }
    return st;
  }

  template <typename InputType, typename OutputType>
  absl::Status BlasGemmWithAlgorithm(
      Stream *stream, blas::Transpose transa, blas::Transpose transb,
      uint64_t m, uint64 n, uint64_t k, const DeviceMemory<InputType> &a,
      int lda, const DeviceMemory<InputType> &b, int ldb,
      DeviceMemory<OutputType> *c, int ldc,
      blas::ComputationType computation_type, blas::AlgorithmType algorithm,
      blas::ProfileResult *output_profile_result, blas::CallContext context) {
    OutputType alpha{1};
    OutputType beta{0};

    return BlasGemmWithAlgorithm(stream, transa, transb, m, n, k, alpha, a, lda,
                                 b, ldb, beta, c, ldc, computation_type,
                                 algorithm, NumericOptions{},
                                 output_profile_result, context);
  }

  template <typename InputType, typename OutputType, typename ConstantType>
  absl::Status BlasGemmStridedBatched(
      Stream *stream, blas::Transpose transa, blas::Transpose transb,
      uint64_t m, uint64 n, uint64_t k, ConstantType alpha,
      const DeviceMemory<InputType> &a, int lda, int64_t stride_a,
      const DeviceMemory<InputType> &b, int ldb, int64_t stride_b,
      ConstantType beta, DeviceMemory<OutputType> *c, int ldc, int64_t stride_c,
      int batch_count, const NumericOptions &numeric_options,
      blas::CallContext context) {
    static_assert(
        detail::is_any_of<InputType, int8_t, float, Eigen::half,
                          Eigen::bfloat16, double, std::complex<float>,
                          std::complex<double>>(),
        "Unsupported input type");
    static_assert(std::is_same_v<ConstantType, InputType> ||
                      (detail::is_any_of<InputType, int8_t, Eigen::half,
                                         Eigen::bfloat16>() &&
                       std::is_same_v<ConstantType, float>),
                  "Mismatched input and alpha/beta types");

    void *alpha_ptr = &alpha;
    void *beta_ptr = &beta;
    float alpha_storage, beta_storage;
    UpcastHalfToFloat<ConstantType>(&alpha_ptr, &beta_ptr, &alpha_storage,
                                    &beta_storage);

    return DoBlasGemmStridedBatched(
        stream, transa, transb, m, n, k, blas::ToDataType<InputType>::value,
        alpha_ptr, a, lda, stride_a, b, ldb, stride_b, beta_ptr, c, ldc,
        stride_c, batch_count, numeric_options, context);
  }

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
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) = 0;
  virtual bool DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) = 0;
  virtual bool DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) = 0;
  virtual bool DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) = 0;

  // Same as DoBlasTrsm, but operates over a list of a's and b's.  The lists
  // `as` and `bs` must have the same length.
  virtual bool DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 float alpha, const DeviceMemory<float *> &as,
                                 int lda, DeviceMemory<float *> *bs, int ldb,
                                 int batch_count) = 0;
  virtual bool DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 double alpha, const DeviceMemory<double *> &as,
                                 int lda, DeviceMemory<double *> *bs, int ldb,
                                 int batch_count) = 0;
  virtual bool DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 std::complex<float> alpha,
                                 const DeviceMemory<std::complex<float> *> &as,
                                 int lda,
                                 DeviceMemory<std::complex<float> *> *bs,
                                 int ldb, int batch_count) = 0;
  virtual bool DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 std::complex<double> alpha,
                                 const DeviceMemory<std::complex<double> *> &as,
                                 int lda,
                                 DeviceMemory<std::complex<double> *> *bs,
                                 int ldb, int batch_count) = 0;

  // TODO(ezhulenev): We should never pass ScratchAllocator to any of the APIs
  // in this file, because it makes them incompatible with command buffers (CUDA
  // graphs). We should pass workspace memory explicitly to all APIs. However
  // this is a giant change, so currently we work around it by setting a thread
  // local workspace and rely on `ScopedBlasWorkspace` RAII helper to reset it.
  //
  // APIs that get ScratchAllocator ignore this workspace, and continue
  // allocating scratch memory on demand.
  class ScopedWorkspace {
   public:
    ScopedWorkspace(BlasSupport *blas, DeviceMemoryBase *workspace);
    ~ScopedWorkspace();

   private:
    BlasSupport *blas_;
  };

  virtual absl::Status GetVersion(std::string *version) = 0;

 protected:
  DeviceMemoryBase *GetWorkspace();

  BlasSupport() {}

 private:
  // Workspace memory pointer is thread local, once it is set all Blas
  // operations issued from a caller thread might use it if it has large enough
  // size. It's a user responsibility to make sure that workspace will outlive
  // all issued BLAS operations.
  //
  // TODO(ezhulenev): This is a giant footgun! We have to remove it and use
  // explicit workspace memory argument for all BLAS operations.
  void SetWorkspace(DeviceMemoryBase *workspace);

  // Resets user-defined workspace memory, so that Blas operations can use their
  // own memory pool for allocating workspace.
  void ResetWorkspace();

  // Checks whether types match before a call to extended BLAS version.
  template <typename ABType, typename CType, typename ScaleType>
  absl::Status CheckTypesForExtendedBlas(
      blas::ComputationType computation_type) {
    static_assert(
        detail::is_any_of<ABType, Eigen::half, Eigen::bfloat16, float, double,
                          int8_t, std::complex<float>, std::complex<double>>(),
        "The only buffer types supported are: Eigen::half, float, "
        "double, int8, std::complex<float> and std::complex<double>");
    static_assert(
        std::is_same_v<ScaleType, CType> ||
            (std::is_same_v<ScaleType, float> &&
             detail::is_any_of<CType, Eigen::half, Eigen::bfloat16>()),
        "Mismatched alpha/beta and output types");

    bool valid_computation_type = [computation_type] {
      switch (computation_type) {
        case blas::ComputationType::kF16:
          return std::is_same_v<CType, Eigen::half>;
        case blas::ComputationType::kF32:
          return detail::is_any_of<CType, Eigen::half, Eigen::bfloat16, float,
                                   std::complex<float>>();
        case blas::ComputationType::kF64:
          return detail::is_any_of<CType, double, std::complex<double>>();
        case blas::ComputationType::kI32:
          return std::is_same_v<CType, int32_t>;
        case blas::ComputationType::kF16AsF32:   // fall-through
        case blas::ComputationType::kBF16AsF32:  // fall-through
        case blas::ComputationType::kTF32AsF32:
          return detail::is_any_of<CType, float, std::complex<float>>();
      }
    }();

    if (!valid_computation_type) {
      return absl::InternalError(absl::StrCat(
          "Invalid computation type ",
          blas::ComputationTypeString(computation_type), " for output type: ",
          blas::DataTypeString(blas::ToDataType<CType>::value)));
    }
    return absl::OkStatus();
  }

  // Non-extended BLAS interface requires alpha/beta to be floats when input
  // type is Eigen::half. However, for consistency purposes it is convenient
  // for the interface to accept Eigen::half.
  template <typename T>
  void UpcastHalfToFloat(void **alpha_ptr, void **beta_ptr,
                         float *alpha_storage, float *beta_storage) {
    if (std::is_same<T, Eigen::half>::value) {
      *alpha_storage =
          static_cast<float>(*reinterpret_cast<Eigen::half *>(*alpha_ptr));
      *beta_storage =
          static_cast<float>(*reinterpret_cast<Eigen::half *>(*beta_ptr));
      *alpha_ptr = alpha_storage;
      *beta_ptr = beta_storage;
    } else if (std::is_same<T, Eigen::bfloat16>::value) {
      *alpha_storage =
          static_cast<float>(*reinterpret_cast<Eigen::bfloat16 *>(*alpha_ptr));
      *beta_storage =
          static_cast<float>(*reinterpret_cast<Eigen::bfloat16 *>(*beta_ptr));
      *alpha_ptr = alpha_storage;
      *beta_ptr = beta_storage;
    }
  }

  BlasSupport(const BlasSupport &) = delete;
  void operator=(const BlasSupport &) = delete;
};

// Macro used to quickly declare overrides for abstract virtuals in the
// BlasSupport base class.
#define TENSORFLOW_STREAM_EXECUTOR_GPU_BLAS_SUPPORT_OVERRIDES                  \
  bool DoBlasAxpy(Stream *stream, uint64_t elem_count, float alpha,            \
                  const DeviceMemory<float> &x, int incx,                      \
                  DeviceMemory<float> *y, int incy) override;                  \
  bool DoBlasCopy(Stream *stream, uint64_t elem_count,                         \
                  const DeviceMemory<float> &x, int incx,                      \
                  DeviceMemory<float> *y, int incy) override;                  \
  bool DoBlasScal(Stream *stream, uint64_t elem_count, float alpha,            \
                  DeviceMemory<float> *x, int incx) override;                  \
  bool DoBlasScal(Stream *stream, uint64_t elem_count, double alpha,           \
                  DeviceMemory<double> *x, int incx) override;                 \
  bool DoBlasScal(Stream *stream, uint64_t elem_count, float alpha,            \
                  DeviceMemory<std::complex<float>> *x, int incx) override;    \
  bool DoBlasScal(Stream *stream, uint64_t elem_count, double alpha,           \
                  DeviceMemory<std::complex<double>> *x, int incx) override;   \
  bool DoBlasScal(Stream *stream, uint64_t elem_count,                         \
                  std::complex<float> alpha,                                   \
                  DeviceMemory<std::complex<float>> *x, int incx) override;    \
  bool DoBlasScal(Stream *stream, uint64_t elem_count,                         \
                  std::complex<double> alpha,                                  \
                  DeviceMemory<std::complex<double>> *x, int incx) override;   \
  bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m, uint64 n, \
                  float alpha, const DeviceMemory<float> &a, int lda,          \
                  const DeviceMemory<float> &x, int incx, float beta,          \
                  DeviceMemory<float> *y, int incy) override;                  \
  bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m, uint64 n, \
                  double alpha, const DeviceMemory<double> &a, int lda,        \
                  const DeviceMemory<double> &x, int incx, double beta,        \
                  DeviceMemory<double> *y, int incy) override;                 \
  bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m, uint64 n, \
                  std::complex<float> alpha,                                   \
                  const DeviceMemory<std::complex<float>> &a, int lda,         \
                  const DeviceMemory<std::complex<float>> &x, int incx,        \
                  std::complex<float> beta,                                    \
                  DeviceMemory<std::complex<float>> *y, int incy) override;    \
  bool DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m, uint64 n, \
                  std::complex<double> alpha,                                  \
                  const DeviceMemory<std::complex<double>> &a, int lda,        \
                  const DeviceMemory<std::complex<double>> &x, int incx,       \
                  std::complex<double> beta,                                   \
                  DeviceMemory<std::complex<double>> *y, int incy) override;   \
  bool DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64_t n, uint64 k, \
                  float alpha, const DeviceMemory<float> &a, int lda,          \
                  const DeviceMemory<float> &x, int incx, float beta,          \
                  DeviceMemory<float> *y, int incy) override;                  \
  absl::Status DoBlasGemm(                                                     \
      Stream *stream, blas::Transpose transa, blas::Transpose transb,          \
      uint64_t m, uint64 n, uint64 k, blas::DataType dtype, const void *alpha, \
      const DeviceMemoryBase &a, int lda, const DeviceMemoryBase &b, int ldb,  \
      const void *beta, DeviceMemoryBase *c, int ldc,                          \
      const NumericOptions &numeric_options, blas::CallContext context)        \
      override;                                                                \
  bool GetBlasGemmAlgorithms(                                                  \
      Stream *stream, const gpu::MatrixDescriptor &a,                          \
      const gpu::MatrixDescriptor &b, gpu::OutputMatrixDescriptor *c,          \
      const void *alpha, const void *beta,                                     \
      std::vector<blas::AlgorithmType> *out_algorithms) override;              \
  absl::Status DoBlasGemmWithAlgorithm(                                        \
      Stream *stream, blas::Transpose transa, blas::Transpose transb,          \
      uint64_t m, uint64 n, uint64 k, const void *alpha,                       \
      const DeviceMemoryBase &a, blas::DataType type_a, int lda,               \
      const DeviceMemoryBase &b, blas::DataType type_b, int ldb,               \
      const void *beta, DeviceMemoryBase *c, blas::DataType type_c, int ldc,   \
      blas::ComputationType computation_type, blas::AlgorithmType algorithm,   \
      const NumericOptions &numeric_options,                                   \
      blas::ProfileResult *output_profile_result, blas::CallContext context)   \
      override;                                                                \
  bool DoBlasGemmBatched(                                                      \
      Stream *stream, blas::Transpose transa, blas::Transpose transb,          \
      uint64_t m, uint64 n, uint64 k, float alpha,                             \
      DeviceMemorySlice<Eigen::half> a, int lda,                               \
      DeviceMemorySlice<Eigen::half> b, int ldb, float beta,                   \
      DeviceMemorySlice<Eigen::half> c, int ldc, int batch_count,              \
      const NumericOptions &numeric_options,                                   \
      ScratchAllocator *scratch_allocator, blas::CallContext context)          \
      override;                                                                \
  bool DoBlasGemmBatched(                                                      \
      Stream *stream, blas::Transpose transa, blas::Transpose transb,          \
      uint64_t m, uint64 n, uint64 k, float alpha,                             \
      DeviceMemorySlice<Eigen::bfloat16> a, int lda,                           \
      DeviceMemorySlice<Eigen::bfloat16> b, int ldb, float beta,               \
      DeviceMemorySlice<Eigen::bfloat16> c, int ldc, int batch_count,          \
      const NumericOptions &numeric_options,                                   \
      ScratchAllocator *scratch_allocator, blas::CallContext context)          \
      override;                                                                \
  bool DoBlasGemmBatched(                                                      \
      Stream *stream, blas::Transpose transa, blas::Transpose transb,          \
      uint64_t m, uint64 n, uint64 k, float alpha, DeviceMemorySlice<float> a, \
      int lda, DeviceMemorySlice<float> b, int ldb, float beta,                \
      DeviceMemorySlice<float> c, int ldc, int batch_count,                    \
      const NumericOptions &numeric_options,                                   \
      ScratchAllocator *scratch_allocator, blas::CallContext context)          \
      override;                                                                \
  bool DoBlasGemmBatched(                                                      \
      Stream *stream, blas::Transpose transa, blas::Transpose transb,          \
      uint64_t m, uint64 n, uint64 k, double alpha,                            \
      DeviceMemorySlice<double> a, int lda, DeviceMemorySlice<double> b,       \
      int ldb, double beta, DeviceMemorySlice<double> c, int ldc,              \
      int batch_count, const NumericOptions &numeric_options,                  \
      ScratchAllocator *scratch_allocator, blas::CallContext context)          \
      override;                                                                \
  bool DoBlasGemmBatched(                                                      \
      Stream *stream, blas::Transpose transa, blas::Transpose transb,          \
      uint64_t m, uint64 n, uint64 k, std::complex<float> alpha,               \
      DeviceMemorySlice<std::complex<float>> a, int lda,                       \
      DeviceMemorySlice<std::complex<float>> b, int ldb,                       \
      std::complex<float> beta, DeviceMemorySlice<std::complex<float>> c,      \
      int ldc, int batch_count, const NumericOptions &numeric_options,         \
      ScratchAllocator *scratch_allocator, blas::CallContext context)          \
      override;                                                                \
  bool DoBlasGemmBatched(                                                      \
      Stream *stream, blas::Transpose transa, blas::Transpose transb,          \
      uint64_t m, uint64 n, uint64 k, std::complex<double> alpha,              \
      DeviceMemorySlice<std::complex<double>> a, int lda,                      \
      DeviceMemorySlice<std::complex<double>> b, int ldb,                      \
      std::complex<double> beta, DeviceMemorySlice<std::complex<double>> c,    \
      int ldc, int batch_count, const NumericOptions &numeric_options,         \
      ScratchAllocator *scratch_allocator, blas::CallContext context)          \
      override;                                                                \
  absl::Status DoBlasGemmStridedBatched(                                       \
      Stream *stream, blas::Transpose transa, blas::Transpose transb,          \
      uint64_t m, uint64 n, uint64 k, blas::DataType dtype, const void *alpha, \
      const DeviceMemoryBase &a, int lda, int64_t stride_a,                    \
      const DeviceMemoryBase &b, int ldb, int64_t stride_b, const void *beta,  \
      DeviceMemoryBase *c, int ldc, int64_t stride_c, int batch_count,         \
      const NumericOptions &numeric_options, blas::CallContext context)        \
      override;                                                                \
  absl::Status DoBlasGemmStridedBatchedWithAlgorithm(                          \
      Stream *stream, blas::Transpose transa, blas::Transpose transb,          \
      uint64_t m, uint64 n, uint64 k, const void *alpha,                       \
      const DeviceMemoryBase &a, blas::DataType type_a, int lda,               \
      int64_t stride_a, const DeviceMemoryBase &b, blas::DataType type_b,      \
      int ldb, int64_t stride_b, const void *beta, DeviceMemoryBase *c,        \
      blas::DataType type_c, int ldc, int64_t stride_c, int batch_count,       \
      blas::ComputationType computation_type, blas::AlgorithmType algorithm,   \
      const NumericOptions &numeric_options,                                   \
      blas::ProfileResult *output_profile_result, blas::CallContext context)   \
      override;                                                                \
  bool DoBlasTrsm(Stream *stream, blas::Side side, blas::UpperLower uplo,      \
                  blas::Transpose transa, blas::Diagonal diag, uint64_t m,     \
                  uint64_t n, float alpha, const DeviceMemory<float> &a,       \
                  int lda, DeviceMemory<float> *b, int ldb) override;          \
  bool DoBlasTrsm(Stream *stream, blas::Side side, blas::UpperLower uplo,      \
                  blas::Transpose transa, blas::Diagonal diag, uint64_t m,     \
                  uint64_t n, double alpha, const DeviceMemory<double> &a,     \
                  int lda, DeviceMemory<double> *b, int ldb) override;         \
  bool DoBlasTrsm(Stream *stream, blas::Side side, blas::UpperLower uplo,      \
                  blas::Transpose transa, blas::Diagonal diag, uint64_t m,     \
                  uint64_t n, std::complex<float> alpha,                       \
                  const DeviceMemory<std::complex<float>> &a, int lda,         \
                  DeviceMemory<std::complex<float>> *b, int ldb) override;     \
  bool DoBlasTrsm(Stream *stream, blas::Side side, blas::UpperLower uplo,      \
                  blas::Transpose transa, blas::Diagonal diag, uint64_t m,     \
                  uint64_t n, std::complex<double> alpha,                      \
                  const DeviceMemory<std::complex<double>> &a, int lda,        \
                  DeviceMemory<std::complex<double>> *b, int ldb) override;    \
  bool DoBlasTrsmBatched(                                                      \
      Stream *stream, blas::Side side, blas::UpperLower uplo,                  \
      blas::Transpose transa, blas::Diagonal diag, uint64_t m, uint64 n,       \
      float alpha, const DeviceMemory<float *> &as, int lda,                   \
      DeviceMemory<float *> *bs, int ldb, int batch_count) override;           \
  bool DoBlasTrsmBatched(                                                      \
      Stream *stream, blas::Side side, blas::UpperLower uplo,                  \
      blas::Transpose transa, blas::Diagonal diag, uint64_t m, uint64 n,       \
      double alpha, const DeviceMemory<double *> &as, int lda,                 \
      DeviceMemory<double *> *bs, int ldb, int batch_count) override;          \
  bool DoBlasTrsmBatched(Stream *stream, blas::Side side,                      \
                         blas::UpperLower uplo, blas::Transpose transa,        \
                         blas::Diagonal diag, uint64_t m, uint64 n,            \
                         std::complex<float> alpha,                            \
                         const DeviceMemory<std::complex<float> *> &as,        \
                         int lda, DeviceMemory<std::complex<float> *> *bs,     \
                         int ldb, int batch_count) override;                   \
  bool DoBlasTrsmBatched(Stream *stream, blas::Side side,                      \
                         blas::UpperLower uplo, blas::Transpose transa,        \
                         blas::Diagonal diag, uint64_t m, uint64 n,            \
                         std::complex<double> alpha,                           \
                         const DeviceMemory<std::complex<double> *> &as,       \
                         int lda, DeviceMemory<std::complex<double> *> *bs,    \
                         int ldb, int batch_count) override;                   \
  absl::Status GetVersion(std::string *version) override;

}  // namespace blas
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_BLAS_H_
