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

#ifndef TENSORFLOW_CORE_KERNELS_EIGEN_MKLDNN_H_
#define TENSORFLOW_CORE_KERNELS_EIGEN_MKLDNN_H_

// Support for Mkldnn sgemm kernel in Eigen/Tensor contractions:
//
// 1. Prepare packed Lhs/Rhs blocks from tensor expressions using
//    DataMapper (see TensorContractionInputMapper).
// 2. Invoke gemm kernel with packed blocks (replacement for default
//    gebp_kernel).

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "third_party/intel_mkl_dnn/include/mkldnn.h"

namespace Eigen {
namespace internal {

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
  void operator()(Scalar *block, const DataMapper &data_mapper, IndexType rows,
                  IndexType cols) {
    const IndexType unrolled_rows =
        (rows / (4 * PacketSize)) * (4 * PacketSize);
    const IndexType vectorized_rows = (rows / PacketSize) * PacketSize;

    for (IndexType col = 0; col < cols; ++col) {
      LinearMapper lm = data_mapper.getLinearMapper(0, col);

      // Give compiler a strong possibility to unroll the loop.
      for (IndexType i = 0; i < unrolled_rows; i += 4 * PacketSize) {
        for (IndexType j = 0; j < 4; ++j) {
          const Packet p = lm.loadPacket(i + j * PacketSize);
          internal::pstoreu(block + j * PacketSize, p);
        }
        block += 4 * PacketSize;
      }

      // Process remaining rows with packets.
      for (IndexType i = unrolled_rows; i < vectorized_rows; i += PacketSize) {
        const Packet p = lm.loadPacket(i);
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
  void operator()(const OutputMapper &output, const float *blockA,
                  const float *blockB, const IndexType rows,
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
                                      const_cast<float *>(output.data()), &ldC);
    eigen_assert(st == 0);
  }
};

}  // namespace internal
}  // namespace Eigen

#endif  // TENSORFLOW_CORE_KERNELS_EIGEN_MKLDNN_H_
