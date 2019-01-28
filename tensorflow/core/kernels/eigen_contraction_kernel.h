/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_EIGEN_CONTRACTION_KERNEL_H_
#define TENSORFLOW_CORE_KERNELS_EIGEN_CONTRACTION_KERNEL_H_

// Depending on a build configuration this header provides custom kernel for
// Eigen tensor contractions (small matrix multiplication kernel used to
// multiple together blocks of the original tensors).
//
// 1) --define tensorflow_mkldnn_contraction_kernel=1
//    Use Mkldnn single threaded sgemm. The mkldnn kernels are generated at
//    runtime and use avx/avx2/fma/avx512 based on cpu status registers
//    (https://en.wikipedia.org/wiki/CPUID).
//
// If you use `tensor.contract(other_tensor)` in your code, you must include
// this header to get the benefit of custom contraction kernel:
//
//   #if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
//   #include "tensorflow/core/kernels/eigen_contraction_kernel.h"
//   #endif

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if defined(TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL)
#include "mkldnn.h"
#endif

namespace Eigen {
namespace internal {

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
// Returns `true` iff we can use custom contraction kernels. This is a runtime
// check, that uses environment variables.
bool UseCustomContractionKernels();
#endif  // TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL

// Enabled by build option: "--define tensorflow_mkldnn_contraction_kernel=1"
#if defined(TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL)

template <typename Scalar, typename IndexType, typename DataMapper,
          int StorageOrder>
struct mkldnn_gemm_pack;

// mkl_gemm_pack for ColMajor storage order.
template <typename Scalar, typename IndexType, typename DataMapper>
struct mkldnn_gemm_pack<Scalar, IndexType, DataMapper,
                        /*StorageOrder*/ ColMajor> {
  typedef typename internal::packet_traits<Scalar>::type Packet;
  typedef typename DataMapper::LinearMapper LinearMapper;

  enum { PacketSize = internal::packet_traits<Scalar>::size };

  EIGEN_DONT_INLINE
  void operator()(Scalar* block, const DataMapper& data_mapper, IndexType rows,
                  IndexType cols) {
    const IndexType unrolled_rows =
        (rows / (4 * PacketSize)) * (4 * PacketSize);
    const IndexType vectorized_rows = (rows / PacketSize) * PacketSize;

    for (IndexType col = 0; col < cols; ++col) {
      LinearMapper lm = data_mapper.getLinearMapper(0, col);

      // Give compiler a strong possibility to unroll the loop.
      for (IndexType i = 0; i < unrolled_rows; i += 4 * PacketSize) {
        for (IndexType j = 0; j < 4; ++j) {
          const Packet p = lm.template loadPacket<Packet>(i + j * PacketSize);
          internal::pstoreu(block + j * PacketSize, p);
        }
        block += 4 * PacketSize;
      }

      // Process remaining rows with packets.
      for (IndexType i = unrolled_rows; i < vectorized_rows; i += PacketSize) {
        const Packet p = lm.template loadPacket<Packet>(i);
        internal::pstoreu(block, p);
        block += PacketSize;
      }

      // Finalize with coefficients.
      for (IndexType i = vectorized_rows; i < rows; ++i) {
        *block = lm(i);
        ++block;
      }
    }
  }
};

template <typename Scalar, typename IndexType, typename OutputMapper,
          bool ConjugateLhs = false, bool ConjugateRhs = false>
struct mkldnn_gemm_kernel;

// mkldnn_gemm_kernel for floats defined as a thin layer on top of mkldnn_sgemm.
template <typename IndexType, typename OutputMapper, bool ConjugateLhs,
          bool ConjugateRhs>
struct mkldnn_gemm_kernel</*Scalar*/ float, IndexType, OutputMapper,
                          ConjugateLhs, ConjugateRhs> {
  EIGEN_DONT_INLINE
  void operator()(const OutputMapper& output, const float* blockA,
                  const float* blockB, const IndexType rows,
                  const IndexType depth, const IndexType cols, float alpha) {
    static const int max_index = (std::numeric_limits<int>::max)();

    eigen_assert(max_index >= rows);
    eigen_assert(max_index >= cols);
    eigen_assert(max_index >= depth);
    eigen_assert(max_index >= output.stride());

    const int m = static_cast<int>(rows);
    const int n = static_cast<int>(cols);
    const int k = static_cast<int>(depth);

    const char transposeA = ConjugateLhs ? 'Y' : 'N';
    const char transposeB = ConjugateRhs ? 'Y' : 'N';

    const int ldA = ConjugateLhs ? k : m;
    const int ldB = ConjugateRhs ? n : k;
    const int ldC = static_cast<int>(output.stride());

    const float beta = 1.0;

    mkldnn_status_t st = mkldnn_sgemm(&transposeA, &transposeB, &m, &n, &k,
                                      &alpha, blockA, &ldA, blockB, &ldB, &beta,
                                      const_cast<float*>(output.data()), &ldC);
    eigen_assert(st == 0);

    // eigen_assert is a no-op in optimized mode so we add these to avoid
    // compiler's unused-variable errors.
    EIGEN_UNUSED_VARIABLE(max_index);
    EIGEN_UNUSED_VARIABLE(st);
  }
};

// For mkldnn_sgemm having the right dimensions (especially for small matrices)
// is more important than fitting all the working set in L1/L2 caches.
// TODO(ezhulenev): Do better heuristics.
template <typename StorageIndex, int sharding_type>
class TensorContractionBlocking<float, float, float, StorageIndex,
                                sharding_type> {
  // For now mkldnn has only mkldnn_sgemm (gemm for floats).
  using Scalar = float;

  // Adjust the block sizes to work well with mkldnn kernels.

  // Multiply default choice of block size along M and N dimensions.
  // TODO(ezhulenev): Explore if this can work in general (kScaleM=2.0 worked
  // well in some of models).
  static constexpr float kScaleM = 1.5;
  static constexpr float kScaleN = 1.0;

  // Mkldnn Avx/Avx2/Avx512 unroll factors are: 8/16/48.
  static const StorageIndex kUnrollM = 48;

  // Mkldnn Avx/Avx2/Avx512 unroll factors are: 6/6/8.
  static const StorageIndex kUnrollN = 24;

 public:
  TensorContractionBlocking(StorageIndex k, StorageIndex m, StorageIndex n,
                            StorageIndex num_threads = 1)
      : kc_(k), mc_(m), nc_(n) {
    // 1. Compute block sizes using default Eigen heuristics.
    if (sharding_type == ShardByCol) {
      computeProductBlockingSizes<Scalar, Scalar, 1>(kc_, mc_, nc_,
                                                     num_threads);
    } else {
      computeProductBlockingSizes<Scalar, Scalar, 1>(kc_, nc_, mc_,
                                                     num_threads);
    }

    // If dimensions do not pass basic sanity checks return immediately.
    if (kc_ <= 0 || mc_ <= 0 || nc_ <= 0) return;

    // If we are using default Eigen gebp kernel there is no need to adjust the
    // block sizes for MKL-DNN.
    if (!UseCustomContractionKernels()) return;

    // 2. And refine them to work well with mkldnn sgemm.
    mc_ = (std::min)(
        m, Eigen::divup(static_cast<StorageIndex>(mc_ * kScaleM), kUnrollM) *
               kUnrollM);
    nc_ = (std::min)(
        n, Eigen::divup(static_cast<StorageIndex>(nc_ * kScaleN), kUnrollN) *
               kUnrollN);

    // We split Kth dimensions in roughly equal slices.
    StorageIndex target_k_slices =
        (std::max)(StorageIndex(1), Eigen::divup(k, kc_));
    StorageIndex packet_size = internal::packet_traits<Scalar>::size;
    if (packet_size < 8) packet_size = 8;
    StorageIndex target_bk =
        Eigen::divup(k / target_k_slices, packet_size) * packet_size;
    kc_ = (std::min)(k, target_bk);
  }

  EIGEN_ALWAYS_INLINE StorageIndex kc() const { return kc_; }
  EIGEN_ALWAYS_INLINE StorageIndex mc() const { return mc_; }
  EIGEN_ALWAYS_INLINE StorageIndex nc() const { return nc_; }

 private:
  StorageIndex kc_;
  StorageIndex mc_;
  StorageIndex nc_;
};

template <typename StorageIndex, typename OutputMapper, typename LhsMapper,
          typename RhsMapper>
struct TensorContractionKernel<float, float, float, StorageIndex, OutputMapper,
                               LhsMapper, RhsMapper> {
  // For now mkldnn has only mkldnn_sgemm (gemm for floats).
  using Scalar = float;
  using Traits = typename internal::gebp_traits<Scalar, Scalar>;

  using LhsPacker = mkldnn_gemm_pack<Scalar, StorageIndex,
                                     typename LhsMapper::SubMapper, ColMajor>;
  using RhsPacker = mkldnn_gemm_pack<Scalar, StorageIndex,
                                     typename RhsMapper::SubMapper, ColMajor>;
  using GemmKernel = mkldnn_gemm_kernel<Scalar, StorageIndex, OutputMapper>;

  // Fallback on default Eigen pack and GEBP kernel if custom contraction
  // kernels disabled at runtime.
  using EigenLhsPacker =
      gemm_pack_lhs<Scalar, StorageIndex, typename LhsMapper::SubMapper,
                    Traits::mr, Traits::LhsProgress,
                    typename Traits::LhsPacket4Packing, ColMajor>;
  using EigenRhsPacker =
      gemm_pack_rhs<Scalar, StorageIndex, typename RhsMapper::SubMapper,
                    Traits::nr, ColMajor>;
  using GebpKernel =
      gebp_kernel<Scalar, Scalar, StorageIndex, OutputMapper, Traits::mr,
                  Traits::nr,
                  /*ConjugateLhs*/ false, /*ConjugateRhs*/ false>;

  EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE static void packLhs(
      Scalar* lhsBlock, const typename LhsMapper::SubMapper& data_mapper,
      const StorageIndex depth, const StorageIndex rows) {
    if (UseCustomContractionKernels()) {
      LhsPacker()(lhsBlock, data_mapper, rows, depth);
    } else {
      EigenLhsPacker()(lhsBlock, data_mapper, depth, rows, /*stride*/ 0,
                       /*offset*/ 0);
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE static void packRhs(
      Scalar* rhsBlock, const typename RhsMapper::SubMapper& data_mapper,
      const StorageIndex depth, const StorageIndex cols) {
    if (UseCustomContractionKernels()) {
      RhsPacker()(rhsBlock, data_mapper, depth, cols);
    } else {
      EigenRhsPacker()(rhsBlock, data_mapper, depth, cols);
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE static void invoke(
      const OutputMapper& output_mapper, const Scalar* lhsBlock,
      const Scalar* rhsBlock, const StorageIndex rows, const StorageIndex depth,
      const StorageIndex cols, const Scalar alpha) {
    if (UseCustomContractionKernels()) {
      GemmKernel()(output_mapper, lhsBlock, rhsBlock, rows, depth, cols, alpha);
    } else {
      GebpKernel()(output_mapper, lhsBlock, rhsBlock, rows, depth, cols, alpha,
                   /*strideA*/ -1, /*strideB*/ -1,
                   /*offsetA*/ 0, /*offsetB*/ 0);
    }
  }
};

#endif  // defined(TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL)

}  // namespace internal
}  // namespace Eigen

#endif  // TENSORFLOW_CORE_KERNELS_EIGEN_CONTRACTION_KERNEL_H_
