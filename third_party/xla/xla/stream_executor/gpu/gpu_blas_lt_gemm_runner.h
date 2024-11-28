/* Copyright 2023 The OpenXLA Authors.
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

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_BLAS_LT_GEMM_RUNNER_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_BLAS_LT_GEMM_RUNNER_H_

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/scratch_allocator.h"

namespace xla {
namespace gpu {
class AutotuneConfig;

}
}  // namespace xla

namespace stream_executor {

namespace gpu {

struct StridedGemmConfig {
  int64_t m, n, k, batch_count;
  blas::Transpose trans_a, trans_b;
  xla::complex128 alpha;
  double beta;
  blas::DataType type_a, type_b, type_c;
  int64_t lda, ldb, ldc;
  int64_t stride_a, stride_b, stride_c;
  blas::ComputationType compute_type;
};

namespace detail {

inline auto AsTuple(const GroupedGemmConfig& p) {
  // NOTE: alpha, beta and data pointers are not included in cache !!
  return std::make_tuple(p.m, p.n, p.k, p.batch_count, p.trans_a, p.trans_b,
                         p.type_a, p.type_b, p.type_c, p.type_d, p.lda, p.ldb,
                         p.ldc, p.ldd, p.compute_type);
}

inline auto AsTuple(const StridedGemmConfig& p) {
  return std::make_tuple(p.m, p.n, p.k, p.batch_count, p.trans_a, p.trans_b,
                         p.alpha.real(), p.alpha.imag(), p.beta, p.type_a,
                         p.type_b, p.type_c, p.lda, p.ldb, p.ldc, p.stride_a,
                         p.stride_b, p.stride_c);
}

}  // namespace detail

bool operator==(const GroupedGemmConfig& rhs, const GroupedGemmConfig& lhs);
bool operator==(const StridedGemmConfig& rhs, const StridedGemmConfig& lhs);

template <typename H>
H AbslHashValue(H h, const GroupedGemmConfig& params) {
  return H::combine(std::move(h), detail::AsTuple(params));
}

template <typename H>
H AbslHashValue(H h, const StridedGemmConfig& params) {
  return H::combine(std::move(h), detail::AsTuple(params));
}

struct BlasLtGemmRunner {
  static BlasLtGemmRunner& i(const Stream* stream);
  const static size_t MaxSolutions = 128;

  template <class Scalar>
  xla::complex128 Convert(Scalar x) {
    if constexpr (std::is_same<Scalar, xla::complex64>::value ||
                  std::is_same<Scalar, xla::complex128>::value) {
      return static_cast<xla::complex128>(x);
    } else {
      return static_cast<double>(x);
    }
  }

  template <class Scalar, class TypeA, class TypeB, class TypeC>
  absl::Status Run(Stream& stream, blas::Transpose trans_a,
                   blas::Transpose trans_b, int64_t m, int64_t n, int64_t k,
                   Scalar alpha, const DeviceMemory<TypeA>& a, int64_t lda,
                   const DeviceMemory<TypeB>& b, int64_t ldb, Scalar beta,
                   DeviceMemory<TypeC>* c, int64_t ldc,
                   ScratchAllocator* allocator) {
    auto type_a = dnn::ToDataType<TypeA>::value,
         type_b = dnn::ToDataType<TypeB>::value,
         type_c = dnn::ToDataType<TypeC>::value;
    return RunStridedBatchedImpl(
        stream, trans_a, trans_b, m, n, k, Convert(alpha), type_a, a, lda, 0,
        type_b, b, ldb, 0,
        Convert(beta).real(),  // only real betas are supported!!
        type_c, c, ldc, 0, 1, allocator);
  }

  template <class Scalar, class TypeA, class TypeB, class TypeC>
  absl::Status Run(Stream& stream, blas::Transpose trans_a,
                   blas::Transpose trans_b, int64_t m, int64_t n, int64_t k,
                   Scalar alpha, const TypeA* a, int64_t lda, const TypeB* b,
                   int64_t ldb, Scalar beta, TypeC* c, int64_t ldc,
                   ScratchAllocator* allocator) {
    auto type_a = dnn::ToDataType<TypeA>::value,
         type_b = dnn::ToDataType<TypeB>::value,
         type_c = dnn::ToDataType<TypeC>::value;

    DeviceMemoryBase mem_c{c};
    return RunStridedBatchedImpl(
        stream, trans_a, trans_b, m, n, k, Convert(alpha), type_a,
        DeviceMemoryBase{const_cast<TypeA*>(a)}, lda, 0, type_b,
        DeviceMemoryBase{const_cast<TypeB*>(b)}, ldb, 0,
        Convert(beta).real(),  // only real betas are supported!!
        type_c, &mem_c, ldc, 0, 1, allocator);
  }

  template <class Scalar, class TypeA, class TypeB, class TypeC>
  absl::Status RunStridedBatched(Stream& stream, blas::Transpose trans_a,
                                 blas::Transpose trans_b, int64_t m, int64_t n,
                                 int64_t k, Scalar alpha, const TypeA* a,
                                 int64_t lda, int64_t stride_a, const TypeB* b,
                                 int64_t ldb, int64_t stride_b, Scalar beta,
                                 TypeC* c, int64_t ldc, int64_t stride_c,
                                 int64_t batch_count,
                                 ScratchAllocator* allocator) {
    auto type_a = dnn::ToDataType<TypeA>::value,
         type_b = dnn::ToDataType<TypeB>::value,
         type_c = dnn::ToDataType<TypeC>::value;
    DeviceMemoryBase mem_c{c};
    return RunStridedBatchedImpl(
        stream, trans_a, trans_b, m, n, k, Convert(alpha), type_a,
        DeviceMemoryBase{const_cast<TypeA*>(a)}, lda, stride_a, type_b,
        DeviceMemoryBase{const_cast<TypeB*>(a)}, ldb, stride_b,
        Convert(beta).real(),  // only real betas are supported!!
        type_c, &mem_c, ldc, stride_c, batch_count, allocator);
  }

  template <class Scalar, class TypeA, class TypeB, class TypeC>
  absl::Status RunStridedBatched(Stream& stream, blas::Transpose trans_a,
                                 blas::Transpose trans_b, int64_t m, int64_t n,
                                 int64_t k, Scalar alpha,
                                 const DeviceMemory<TypeA>& a, int64_t lda,
                                 int64_t stride_a, const DeviceMemory<TypeB>& b,
                                 int64_t ldb, int64_t stride_b, Scalar beta,
                                 DeviceMemory<TypeC>* c, int64_t ldc,
                                 int64_t stride_c, int64_t batch_count,
                                 ScratchAllocator* allocator) {
    auto type_a = dnn::ToDataType<TypeA>::value,
         type_b = dnn::ToDataType<TypeB>::value,
         type_c = dnn::ToDataType<TypeC>::value;
    return RunStridedBatchedImpl(
        stream, trans_a, trans_b, m, n, k, Convert(alpha), type_a, a, lda,
        stride_a, type_b, b, ldb, stride_b,
        Convert(beta).real(),  // only real betas are supported!!
        type_c, c, ldc, stride_c, batch_count, allocator);
  }

  template <class Scalar, class T>
  absl::Status RunBatched(Stream& stream, blas::Transpose trans_a,
                          blas::Transpose trans_b, int64_t m, int64_t n,
                          int64_t k, Scalar alpha,
                          const DeviceMemorySlice<T>& a, int64_t lda,
                          const DeviceMemorySlice<T>& b, int64_t ldb,
                          Scalar beta, const DeviceMemorySlice<T>& c,
                          int64_t ldc, int64_t batch_count,
                          ScratchAllocator* allocator) {
    // NOTE: Scalar types shall be verified for correctness vs T!!
    auto type = dnn::ToDataType<T>::value;

    auto res = ContiguousStrides(a, b, c, batch_count);
    if (res.ok()) {
      auto strides = std::move(res.value());
      return RunStridedBatchedImpl(
          stream, trans_a, trans_b, m, n, k, Convert(alpha), type, *a[0], lda,
          strides[0] / sizeof(T), type, *b[0], ldb, strides[1] / sizeof(T),
          Convert(beta).real(),  // only real betas are supported!!
          type, c[0], ldc, strides[2] / sizeof(T), batch_count, allocator);
    }

    return absl::InternalError("RunBatched: port::ArraySlice NYI!");
  }

  template <class Scalar, class T>
  absl::Status RunBatched(Stream& stream, blas::Transpose trans_a,
                          blas::Transpose trans_b, int64_t m, int64_t n,
                          int64_t k, Scalar alpha, const T** a, int lda,
                          const T** b, int ldb, Scalar beta, T** c, int64_t ldc,
                          int64_t batch_count, ScratchAllocator* allocator) {
    auto type = dnn::ToDataType<T>::value;
    return RunBatchedImpl(stream, trans_a, trans_b, m, n, k, &alpha, type,
                          reinterpret_cast<const void**>(a), lda, type,
                          reinterpret_cast<const void**>(b), ldb, &beta, type,
                          reinterpret_cast<void**>(c), ldc, batch_count,
                          allocator);
  }

  ~BlasLtGemmRunner();
  BlasLtGemmRunner& operator=(BlasLtGemmRunner&& rhs) noexcept = default;
  BlasLtGemmRunner(BlasLtGemmRunner&& rhs) noexcept = default;

 private:
  explicit BlasLtGemmRunner(StreamExecutor* parent);

  template <class TuneFunc>
  absl::StatusOr<gpu::BlasLt::MatmulAlgorithm> Autotune(
      const std::vector<gpu::BlasLt::MatmulAlgorithm>& algorithms,
      TuneFunc&& benchmark_func);

  absl::Status RunBatchedImpl(Stream& stream, blas::Transpose trans_a,
                              blas::Transpose trans_b, int64_t m, int64_t n,
                              int64_t k, const void* alpha,
                              blas::DataType type_a, const void** a,
                              int64_t lda, blas::DataType type_b,
                              const void** b, int64_t ldb, const void* beta,
                              blas::DataType type_c, void** c, int64_t ldc,
                              int64_t batch_count, ScratchAllocator* allocator);

  absl::Status RunStridedBatchedImpl(
      Stream& stream, blas::Transpose trans_a, blas::Transpose trans_b,
      int64_t m, int64_t n, int64_t k, xla::complex128 alpha,
      blas::DataType type_a, const DeviceMemoryBase& a, int64_t lda,
      int64_t stride_a, blas::DataType type_b, const DeviceMemoryBase& b,
      int64_t ldb, int64_t stride_b, double beta, blas::DataType type_c,
      DeviceMemoryBase* c, int64_t ldc, int64_t stride_c, int64_t batch_count,
      ScratchAllocator* allocator);

  template <typename T>
  using ArraySlice = absl::Span<const T>;

  template <typename T>
  absl::StatusOr<std::array<int64_t, 3>> ContiguousStrides(
      const ArraySlice<DeviceMemory<T>*>& a,
      const ArraySlice<DeviceMemory<T>*>& b,
      const ArraySlice<DeviceMemory<T>*>& c, int64_t batch_count) {
    int64_t bsa = 0, bsb = 0, bsc = 0;
    using CT = const uint8_t;
    for (int64_t i = 0; i < batch_count - 1; i++) {
      int64_t da = (CT*)a[i + 1]->opaque() - (CT*)a[i]->opaque(),
              db = (CT*)b[i + 1]->opaque() - (CT*)b[i]->opaque(),
              dc = (CT*)c[i + 1]->opaque() - (CT*)c[i]->opaque();
      if (i == 0) {
        bsa = da, bsb = db, bsc = dc;
      } else if (!(bsa == da && bsb == db && bsc == dc)) {  // strides mismatch
        return absl::InternalError("Strides are not consistent!");
      }
    }
    return std::array<int64_t, 3>{bsa, bsb, bsc};
  }

  std::unique_ptr<absl::Mutex> mutex_;

  absl::flat_hash_map<StridedGemmConfig, std::unique_ptr<BlasLt::MatmulPlan>>
      strided_gemm_map_;
};

}  // namespace gpu

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_BLAS_LT_GEMM_RUNNER_H_
