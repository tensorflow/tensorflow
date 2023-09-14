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

#ifndef TENSORFLOW_TSL_FRAMEWORK_CONTRACTION_EIGEN_CONTRACTION_KERNEL_H_
#define TENSORFLOW_TSL_FRAMEWORK_CONTRACTION_EIGEN_CONTRACTION_KERNEL_H_

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
//   #include
//   "third_party/tensorflow/tsl/framework/contraction/eigen_contraction_kernel.h"
//   #endif

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tsl/framework/fixedpoint/FixedPoint.h"

#if defined(TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL)
#include "dnnl.h"
#endif

#include "tsl/platform/dynamic_annotations.h"

namespace Eigen {
namespace internal {

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
// Returns `true` iff we can use custom contraction kernels. This is a runtime
// check, that uses environment variables.
EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE bool UseCustomContractionKernels();

// Pack a 2D block of a Tensor expression into contiguous block of memory with
// col-major storage order. We do not have access to the underlying Tensor
// expression, we only have a DataMapper (TensorContractionInputMapper for
// tensor contractions, or blas_data_mapper for plain tensors), that provides a
// two-dimensional view into the Tensor expression.
//
// Default Eigen gemm_pack_rhs and gemm_pack_lhs pack blocks of tensor
// expressions into the packed format described in "Anatomy of High-Performance
// Matrix Multiplication" paper (1). Eigen::internal::gebp_kernel relies on this
// packing format for efficient micro-panel multiplication.
//
// This simple packing can be used with any '?gemm' function from BLAS
// libraries, that work with col-major matrices.
//
// (1) http://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf
//
// IMPORTANT: `gemm_pack_colmajor_block` always packs the block in column major
// order, DataMapperStorageOrder specifies the storage order of the underlying
// Tensor expression.
template <typename Scalar, typename IndexType, typename DataMapper,
          int DataMapperStorageOrder>
struct gemm_pack_colmajor_block;

// gemm_pack_colmajor_block for ColMajor storage order.
template <typename Scalar, typename IndexType, typename DataMapper>
struct gemm_pack_colmajor_block<Scalar, IndexType, DataMapper,
                                /*DataMapperStorageOrder*/ ColMajor> {
  typedef typename internal::packet_traits<Scalar>::type Packet;
  typedef typename DataMapper::LinearMapper LinearMapper;

  enum { PacketSize = internal::packet_traits<Scalar>::size };

  EIGEN_DONT_INLINE
  void operator()(Scalar* block, const DataMapper& data_mapper, IndexType rows,
                  IndexType cols) {
    const IndexType unrolled_rows = rows - 4 * PacketSize;
    const IndexType vectorized_rows = rows - PacketSize;

    for (IndexType col = 0; col < cols; ++col) {
      LinearMapper lm = data_mapper.getLinearMapper(0, col);

      IndexType row = 0;
      // Give compiler a strong possibility to unroll the loop.
      for (; row <= unrolled_rows; row += 4 * PacketSize) {
        for (IndexType j = 0; j < 4; ++j) {
          const Packet p = lm.template loadPacket<Packet>(row + j * PacketSize);
          internal::pstoreu(block + j * PacketSize, p);
        }
        block += 4 * PacketSize;
      }
      // Process remaining rows with packets.
      for (; row <= vectorized_rows; row += PacketSize) {
        const Packet p = lm.template loadPacket<Packet>(row);
        internal::pstoreu(block, p);
        block += PacketSize;
      }
      // Finalize with coefficients.
      for (; row < rows; ++row) {
        *block = lm(row);
        ++block;
      }
    }
  }
};

#endif  // TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL

// Enabled by build option: "--define tensorflow_mkldnn_contraction_kernel=1"
#if defined(TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL)

template <typename Scalar, typename IndexType, typename OutputMapper,
          bool ConjugateLhs = false, bool ConjugateRhs = false>
struct dnnl_gemm_kernel;

// dnnl_gemm_kernel for floats defined as a thin layer on top of mkldnn_sgemm.
template <typename IndexType, typename OutputMapper, bool ConjugateLhs,
          bool ConjugateRhs>
struct dnnl_gemm_kernel</*Scalar*/ float, IndexType, OutputMapper, ConjugateLhs,
                        ConjugateRhs> {
  static_assert(!ConjugateLhs, "DNNL kernel doesn't support ConjugateLhs");
  static_assert(!ConjugateRhs, "DNNL kernel doesn't support ConjugateRhs");

  static constexpr int kComputeStrideFromBlockDimensions = -1;

  using LhsScalar = float;
  using RhsScalar = float;
  using ResScalar = float;

  EIGEN_DONT_INLINE
  void operator()(const OutputMapper& output, const LhsScalar* blockA,
                  const RhsScalar* blockB, const IndexType rows,
                  const IndexType depth, const IndexType cols, float alpha,
                  float beta, int ldA = kComputeStrideFromBlockDimensions,
                  int ldB = kComputeStrideFromBlockDimensions,
                  char transposeA = 'N', char transposeB = 'N') {
    static const int max_index = (std::numeric_limits<int>::max)();

    eigen_assert(max_index >= rows);
    eigen_assert(max_index >= cols);
    eigen_assert(max_index >= depth);
    eigen_assert(max_index >= output.stride());

    const int m = static_cast<int>(rows);
    const int n = static_cast<int>(cols);
    const int k = static_cast<int>(depth);

    ldA = ldA == kComputeStrideFromBlockDimensions ? m : ldA;
    ldB = ldB == kComputeStrideFromBlockDimensions ? k : ldB;
    const int ldC = static_cast<int>(output.stride());

    // DNNL takes row-major matrices. Our packed column-major matrices can be
    // viewed as a transposed row-major matrix, i.e.,
    //   C_colmajor = C_rowmajor^T = (A_rowmajor * B_rowmajor)^T
    //                             = B_rowmajor^T * A_rowmajor^T
    //                             = B_colmajor * A_colmajor
    // So we can just swap the input matrices A and B for DNNL.
    // TODO(penporn): Switch to row-major packing instead.
    dnnl_status_t st =
        dnnl_sgemm(transposeB, transposeA, n, m, k, alpha, blockB, ldB, blockA,
                   ldA, beta, const_cast<ResScalar*>(output.data()), ldC);
    eigen_assert(st == 0);

#if DYNAMIC_ANNOTATIONS_ENABLED == 1 || defined(MEMORY_SANITIZER)
    for (IndexType col = 0; col < cols; ++col) {
      ResScalar* row_base = &output(0, col);
      EIGEN_UNUSED_VARIABLE(row_base);  // Suppress unused variable error.
      TF_ANNOTATE_MEMORY_IS_INITIALIZED(row_base, sizeof(ResScalar) * rows);
    }
#endif

    // eigen_assert is a no-op in optimized mode so we add these to avoid
    // compiler's unused-variable errors.
    EIGEN_UNUSED_VARIABLE(max_index);
    EIGEN_UNUSED_VARIABLE(st);
  }
};

template <typename IndexType, typename OutputMapper, bool ConjugateLhs = false,
          bool ConjugateRhs = false>
struct mkldnn_gemm_s8u8s32_kernel {
  static_assert(!ConjugateLhs, "DNNL kernel doesn't support ConjugateLhs");
  static_assert(!ConjugateRhs, "DNNL kernel doesn't support ConjugateRhs");

  static constexpr int kComputeStrideFromBlockDimensions = -1;

  using LhsScalar = Eigen::QInt8;
  using RhsScalar = Eigen::QUInt8;
  using ResScalar = Eigen::QInt32;

  EIGEN_DONT_INLINE
  void operator()(const OutputMapper& output, const LhsScalar* blockA,
                  const RhsScalar* blockB, const IndexType rows,
                  const IndexType depth, const IndexType cols, float alpha,
                  float beta, int ldA = kComputeStrideFromBlockDimensions,
                  int ldB = kComputeStrideFromBlockDimensions,
                  char transposeA = 'N', char transposeB = 'N') {
    static const int max_index = (std::numeric_limits<int>::max)();

    eigen_assert(max_index >= rows);
    eigen_assert(max_index >= cols);
    eigen_assert(max_index >= depth);
    eigen_assert(max_index >= output.stride());

    const int m = static_cast<int>(rows);
    const int n = static_cast<int>(cols);
    const int k = static_cast<int>(depth);

    ldA = ldA == kComputeStrideFromBlockDimensions ? m : ldA;
    ldB = ldB == kComputeStrideFromBlockDimensions ? k : ldB;
    const int ldC = static_cast<int>(output.stride());

    // Currently we support only symmetric quantization with zero point at 0.
    const int8_t ao = 0;
    const int8_t bo = 0;

    // Don't add any offset to the result C.
    const char offsetc = 'F';
    const int32_t co = 0;

    const auto* A = reinterpret_cast<const int8_t*>(blockA);
    const auto* B = reinterpret_cast<const uint8_t*>(blockB);
    auto* C = reinterpret_cast<int32_t*>(const_cast<ResScalar*>(output.data()));

    // DNNL takes row-major matrices. Our packed column-major matrices can be
    // viewed as a transposed row-major matrix, i.e., C_colmajor = C_rowmajor^T.
    // C_colmajor = C_rowmajor^T = (A_rowmajor * B_rowmajor)^T
    //                           = B_rowmajor^T * A_rowmajor^T
    //                           = B_colmajor * A_colmajor
    // So we can just swap the input matrices A and B for DNNL.
    // TODO(penporn): Switch to row-major packing instead.
    dnnl_status_t st = dnnl_gemm_u8s8s32(transposeB, transposeA, offsetc,  //
                                         n, m, k,                          //
                                         alpha,                            //
                                         B, ldB, bo,                       //
                                         A, ldA, ao,                       //
                                         beta,                             //
                                         C, ldC, &co);
    eigen_assert(st == 0);

#if DYNAMIC_ANNOTATIONS_ENABLED == 1 || defined(MEMORY_SANITIZER)
    for (IndexType col = 0; col < cols; ++col) {
      ResScalar* row_base = &output(0, col);
      EIGEN_UNUSED_VARIABLE(row_base);  // Suppress unused variable error.
      TF_ANNOTATE_MEMORY_IS_INITIALIZED(row_base, sizeof(ResScalar) * rows);
    }
#endif

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
  static constexpr StorageIndex kUnrollM = 48;

  // Mkldnn Avx/Avx2/Avx512 unroll factors are: 6/6/8.
  static constexpr StorageIndex kUnrollN = 24;

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
    // block sizes for DNNL.
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

template <typename StorageIndex, int sharding_type>
class TensorContractionBlocking<Eigen::QInt32, Eigen::QInt8, Eigen::QUInt8,
                                StorageIndex, sharding_type> {
  // TODO(ezhulenev): Define proper gebp_traits in Eigen for quantized types?

  // Default Eigen block heuristics for `QInt8xQUInt8 -> QInt32` are wrong.
  // Mostly because gebp_traits are not correctly defined. But we know that we
  // are going to use s8u8s32_gemm from DNNL, so we use float heuristics, and
  // adjust them to work well with DNNL.
  using LhsScalar = Eigen::QInt8;
  using RhsScalar = Eigen::QUInt8;
  using ResScalar = Eigen::QInt32;

  // Multiply default choice of block size along M, N and K dimensions.
  static constexpr float kScaleM = 1.5;
  static constexpr float kScaleN = 1.5;
  static constexpr float kScaleK = 1.5;

 public:
  TensorContractionBlocking(StorageIndex k, StorageIndex m, StorageIndex n,
                            StorageIndex num_threads = 1)
      : kc_(k), mc_(m), nc_(n) {
    // Each dimension is a multiple of 32 (fits into _m256i).
    mc_ = (std::min)(m, static_cast<StorageIndex>(192));
    nc_ = (std::min)(n, static_cast<StorageIndex>(288));
    kc_ = (std::min)(k, static_cast<StorageIndex>(320));
  }

  EIGEN_ALWAYS_INLINE StorageIndex kc() const { return kc_; }
  EIGEN_ALWAYS_INLINE StorageIndex mc() const { return mc_; }
  EIGEN_ALWAYS_INLINE StorageIndex nc() const { return nc_; }

 private:
  StorageIndex kc_;
  StorageIndex mc_;
  StorageIndex nc_;
};

// If the Lhs or Rhs Tensor expressions are already evaluated and have access to
// raw data, we can skip packing step and setup pointers and a stride to the
// underlying memory buffer and pass them directly to Gemm.
template <typename Scalar, typename StorageIndex>
struct ColMajorBlock {
  bool is_direct_access;

  // Valid iff `is_direct_access == false`
  Scalar* packed_data;

  // Valid iff `is_direct_access == true`
  Scalar* raw_data;
  StorageIndex stride;
  char transpose;
};

template <typename DataMapper>
struct DirectColMajorAccess {
  enum { value = false };

  template <typename Scalar, typename StorageIndex>
  static bool block(const typename DataMapper::SubMapper& data_mapper,
                    const StorageIndex rows, const StorageIndex cols,
                    const StorageIndex num_kernels,
                    ColMajorBlock<Scalar, StorageIndex>* block) {
    eigen_assert(false && "Not implemented");
    return false;
  }
};

// If we have an access to raw memory of the contraction input, we can safely
// skip packing if:
//   (1) Packing is a no-op.
//   (2) Packed block will be used just once.
//
// If a packed block is used many times, it's more efficient to pack it into
// contiguous block of memory to reduce pressure on TLB.
//
// TODO(ezhulenev): Add support for more tensor expressions that matters.
#define REGISTER_DIRECT_COL_MAJOR_ACCESS(TENSOR_EXPR)                          \
  template <typename Scalar, typename StorageIndex, int Side, typename Device, \
            typename nocontract_t, typename contract_t, int packet_size,       \
            int Alignment>                                                     \
  struct DirectColMajorAccess<TensorContractionInputMapper<                    \
      Scalar, StorageIndex, Side, TensorEvaluator<TENSOR_EXPR, Device>,        \
      nocontract_t, contract_t, packet_size, /*inner_dim_contiguous=*/true,    \
      /*inner_dim_reordered=*/false, Alignment>> {                             \
    enum { value = true };                                                     \
                                                                               \
    using DataMapper = TensorContractionInputMapper<                           \
        Scalar, StorageIndex, Side, TensorEvaluator<TENSOR_EXPR, Device>,      \
        nocontract_t, contract_t, packet_size, /*inner_dim_contiguous=*/true,  \
        /*inner_dim_reordered=*/false, Alignment>;                             \
                                                                               \
    static bool block(const typename DataMapper::SubMapper& data_mapper,       \
                      const StorageIndex rows, const StorageIndex cols,        \
                      const StorageIndex num_kernels,                          \
                      ColMajorBlock<Scalar, StorageIndex>* block) {            \
      static_assert(DataMapper::DirectOffsets == true,                         \
                    "DataMapper must support direct offsets");                 \
                                                                               \
      const StorageIndex vert_offset = data_mapper.vert_offset();              \
      const StorageIndex horiz_offset = data_mapper.horiz_offset();            \
      const StorageIndex stride =                                              \
          Side == Lhs ? data_mapper.base_mapper().stride()                     \
                      : data_mapper.base_mapper().nocontract_strides()[0];     \
      const Scalar* data = data_mapper.base_mapper().tensor().data();          \
      data = Side == Lhs ? data : data + vert_offset + horiz_offset * stride;  \
                                                                               \
      const bool is_no_op_packing = stride == rows;                            \
      const StorageIndex addressable_mem = (stride * cols * sizeof(Scalar));   \
      const bool use_direct_access =                                           \
          is_no_op_packing || num_kernels == 1 /* used once */ ||              \
          ((num_kernels == 2) &&                                               \
           (addressable_mem < (256 << 10) /* 256 kb */));                      \
                                                                               \
      if (use_direct_access) {                                                 \
        block->is_direct_access = true;                                        \
        block->raw_data = const_cast<Scalar*>(data);                           \
        block->stride = stride;                                                \
        block->transpose = 'N';                                                \
        return true;                                                           \
      }                                                                        \
      return false;                                                            \
    }                                                                          \
  }

#define SIMPLE_TENSOR const Tensor<Scalar, 2, Eigen::ColMajor, StorageIndex>

#define TENSOR_MAP_ROWMAJOR                                               \
  const TensorMap<Tensor<const Scalar, 2, Eigen::RowMajor, StorageIndex>, \
                  Eigen::Aligned>

#define TENSOR_MAP_COLMAJOR                                               \
  const TensorMap<Tensor<const Scalar, 2, Eigen::ColMajor, StorageIndex>, \
                  Eigen::Aligned>

#define TENSOR_MAP_CONST_ROWMAJOR                                   \
  const TensorMap<Tensor<Scalar, 2, Eigen::RowMajor, StorageIndex>, \
                  Eigen::Aligned>

#define TENSOR_MAP_CONST_COLMAJOR                                   \
  const TensorMap<Tensor<Scalar, 2, Eigen::ColMajor, StorageIndex>, \
                  Eigen::Aligned>

// This is reshaped convolution filter from `eigen_spatial_convolutions.h`.
#define TENSOR_RESHAPE                                                        \
  const TensorReshapingOp<                                                    \
      const Eigen::DSizes<StorageIndex, 2>,                                   \
      const TensorMap<Tensor<const Scalar, 4, Eigen::RowMajor, StorageIndex>, \
                      Eigen::Aligned>>

REGISTER_DIRECT_COL_MAJOR_ACCESS(SIMPLE_TENSOR);
REGISTER_DIRECT_COL_MAJOR_ACCESS(TENSOR_MAP_ROWMAJOR);
REGISTER_DIRECT_COL_MAJOR_ACCESS(TENSOR_MAP_COLMAJOR);
REGISTER_DIRECT_COL_MAJOR_ACCESS(TENSOR_MAP_CONST_ROWMAJOR);
REGISTER_DIRECT_COL_MAJOR_ACCESS(TENSOR_MAP_CONST_COLMAJOR);
REGISTER_DIRECT_COL_MAJOR_ACCESS(TENSOR_RESHAPE);

#undef SIMPLE_TENSOR
#undef TENSOR_MAP_ROWMAJOR
#undef TENSOR_MAP_COLMAJOR
#undef TENSOR_MAP_CONST_ROWMAJOR
#undef TENSOR_MAP_CONST_COLMAJOR
#undef TENSOR_RESHAPE
#undef REGISTER_DIRECT_COL_MAJOR_ACCESS

template <typename ResScalar, typename LhsScalar, typename RhsScalar,
          typename StorageIndex, typename OutputMapper>
struct GemmKernelProvider {
  enum { Defined = 0 };
  using GemmKernel = void;
};

template <typename StorageIndex, typename OutputMapper>
struct GemmKernelProvider<float, float, float, StorageIndex, OutputMapper> {
  enum { Defined = 1 };
  using GemmKernel = dnnl_gemm_kernel<float, StorageIndex, OutputMapper>;
};

template <typename StorageIndex, typename OutputMapper>
struct GemmKernelProvider<Eigen::QInt32, Eigen::QInt8, Eigen::QUInt8,
                          StorageIndex, OutputMapper> {
  enum { Defined = 1 };
  using GemmKernel = mkldnn_gemm_s8u8s32_kernel<StorageIndex, OutputMapper>;
};

// NOTE: 'std::enable_if' doesn't work for template specializations. See
// "default template argument in a class template partial specialization".

// Tensor contraction kernel that can fallback on Eigen gebp_kernel at runtime.
#define REGISTER_TENSOR_CONTRACTION_KERNEL_WITH_FALLBACK(                      \
    RES_SCALAR, LHS_SCALAR, RHS_SCALAR)                                        \
                                                                               \
  template <typename StorageIndex, typename OutputMapper, typename LhsMapper,  \
            typename RhsMapper>                                                \
  struct TensorContractionKernel<RES_SCALAR, LHS_SCALAR, RHS_SCALAR,           \
                                 StorageIndex, OutputMapper, LhsMapper,        \
                                 RhsMapper> {                                  \
    TensorContractionKernel(StorageIndex m, StorageIndex k, StorageIndex n,    \
                            StorageIndex bm, StorageIndex bk, StorageIndex bn) \
        : m(m), k(k), n(n), bm(bm), bk(bk), bn(bn) {}                          \
                                                                               \
    enum { HasBeta = true };                                                   \
                                                                               \
    using ResScalar = RES_SCALAR;                                              \
    using LhsScalar = LHS_SCALAR;                                              \
    using RhsScalar = RHS_SCALAR;                                              \
                                                                               \
    using Traits = typename internal::gebp_traits<LhsScalar, RhsScalar>;       \
                                                                               \
    using LhsBlock = ColMajorBlock<LhsScalar, StorageIndex>;                   \
    using RhsBlock = ColMajorBlock<RhsScalar, StorageIndex>;                   \
                                                                               \
    using DirectLhsAccess = DirectColMajorAccess<LhsMapper>;                   \
    using DirectRhsAccess = DirectColMajorAccess<RhsMapper>;                   \
                                                                               \
    /* Packed Lhs/Rhs block memory allocator.*/                                \
    typedef TensorContractionBlockMemAllocator<LhsScalar, RhsScalar>           \
        BlockMemAllocator;                                                     \
    typedef typename BlockMemAllocator::BlockMemHandle BlockMemHandle;         \
                                                                               \
    using LhsPacker =                                                          \
        gemm_pack_colmajor_block<LhsScalar, StorageIndex,                      \
                                 typename LhsMapper::SubMapper, ColMajor>;     \
    using RhsPacker =                                                          \
        gemm_pack_colmajor_block<RhsScalar, StorageIndex,                      \
                                 typename RhsMapper::SubMapper, ColMajor>;     \
                                                                               \
    using GemmKernelProviderType =                                             \
        GemmKernelProvider<ResScalar, LhsScalar, RhsScalar, StorageIndex,      \
                           OutputMapper>;                                      \
    static_assert(                                                             \
        GemmKernelProviderType::Defined,                                       \
        "Custom GEMM kernel is not registered for given scalar types");        \
    using GemmKernel = typename GemmKernelProviderType::GemmKernel;            \
                                                                               \
    /* Fallback on default Eigen pack and GEBP kernel if custom contraction */ \
    /* kernels disabled at runtime.                                         */ \
    using EigenLhsPacker =                                                     \
        gemm_pack_lhs<LhsScalar, StorageIndex, typename LhsMapper::SubMapper,  \
                      Traits::mr, Traits::LhsProgress,                         \
                      typename Traits::LhsPacket4Packing, ColMajor>;           \
    using EigenRhsPacker =                                                     \
        gemm_pack_rhs<RhsScalar, StorageIndex, typename RhsMapper::SubMapper,  \
                      Traits::nr, ColMajor>;                                   \
    using GebpKernel =                                                         \
        gebp_kernel<LhsScalar, RhsScalar, StorageIndex, OutputMapper,          \
                    Traits::mr, Traits::nr, /*ConjugateLhs*/ false,            \
                    /*ConjugateRhs*/ false>;                                   \
                                                                               \
    template <typename Device>                                                 \
    EIGEN_DEVICE_FUNC BlockMemHandle allocate(Device& d, LhsBlock* lhs_block,  \
                                              RhsBlock* rhs_block) {           \
      return BlockMemAllocator::allocate(                                      \
          d, bm, bk, bn, &lhs_block->packed_data, &rhs_block->packed_data);    \
    }                                                                          \
                                                                               \
    template <typename Device>                                                 \
    EIGEN_DEVICE_FUNC BlockMemHandle                                           \
    allocateSlices(Device& d, const int num_lhs, const int num_rhs,            \
                   const int num_slices, std::vector<LhsBlock>* lhs_blocks,    \
                   std::vector<RhsBlock>* rhs_blocks) {                        \
      eigen_assert(num_slices > 0);                                            \
      std::vector<std::vector<LhsScalar*>> lhs_mem(num_slices);                \
      std::vector<std::vector<RhsScalar*>> rhs_mem(num_slices);                \
                                                                               \
      BlockMemHandle block_mem = BlockMemAllocator::allocateSlices(            \
          d, bm, bk, bn, num_lhs, num_rhs, num_slices, lhs_mem.data(),         \
          rhs_mem.data());                                                     \
                                                                               \
      for (Index x = 0; x < num_slices; x++) {                                 \
        if (num_lhs > 0) lhs_blocks[x].resize(num_lhs);                        \
        for (Index m = 0; m < num_lhs; m++) {                                  \
          lhs_blocks[x][m].packed_data = lhs_mem[x][m];                        \
        }                                                                      \
        if (num_rhs > 0) rhs_blocks[x].resize(num_rhs);                        \
        for (Index n = 0; n < num_rhs; n++) {                                  \
          rhs_blocks[x][n].packed_data = rhs_mem[x][n];                        \
        }                                                                      \
      }                                                                        \
                                                                               \
      return block_mem;                                                        \
    }                                                                          \
                                                                               \
    template <typename Device>                                                 \
    EIGEN_DEVICE_FUNC static void deallocate(Device& d,                        \
                                             BlockMemHandle handle) {          \
      BlockMemAllocator::deallocate(d, handle);                                \
    }                                                                          \
                                                                               \
    EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE void packLhs(                          \
        LhsBlock* lhsBlock, const typename LhsMapper::SubMapper& data_mapper,  \
        const StorageIndex depth, const StorageIndex rows) {                   \
      if (UseCustomContractionKernels()) {                                     \
        const bool is_direct_access =                                          \
            DirectLhsAccess::value &&                                          \
            DirectLhsAccess::block(data_mapper, rows, depth,                   \
                                   bn > 0 ? divup(n, bn) : 0, lhsBlock);       \
                                                                               \
        if (!is_direct_access) {                                               \
          lhsBlock->is_direct_access = false;                                  \
          LhsPacker()(lhsBlock->packed_data, data_mapper, rows, depth);        \
        }                                                                      \
      } else {                                                                 \
        lhsBlock->is_direct_access = false;                                    \
        EigenLhsPacker()(lhsBlock->packed_data, data_mapper, depth, rows,      \
                         /*stride*/ 0, /*offset*/ 0);                          \
      }                                                                        \
    }                                                                          \
                                                                               \
    EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE void packRhs(                          \
        RhsBlock* rhsBlock, const typename RhsMapper::SubMapper& data_mapper,  \
        const StorageIndex depth, const StorageIndex cols) {                   \
      if (UseCustomContractionKernels()) {                                     \
        const bool is_direct_access =                                          \
            DirectRhsAccess::value &&                                          \
            DirectRhsAccess::block(data_mapper, depth, cols,                   \
                                   bm > 0 ? divup(m, bm) : 0, rhsBlock);       \
                                                                               \
        if (!is_direct_access) {                                               \
          rhsBlock->is_direct_access = false;                                  \
          RhsPacker()(rhsBlock->packed_data, data_mapper, depth, cols);        \
        }                                                                      \
      } else {                                                                 \
        rhsBlock->is_direct_access = false;                                    \
        EigenRhsPacker()(rhsBlock->packed_data, data_mapper, depth, cols);     \
      }                                                                        \
    }                                                                          \
                                                                               \
    EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE void invoke(                           \
        const OutputMapper& output_mapper, const LhsBlock& lhsBlock,           \
        const RhsBlock& rhsBlock, const StorageIndex rows,                     \
        const StorageIndex depth, const StorageIndex cols, const float alpha,  \
        const float beta) {                                                    \
      if (UseCustomContractionKernels()) {                                     \
        if ((DirectLhsAccess::value && lhsBlock.is_direct_access) &&           \
            (DirectRhsAccess::value && rhsBlock.is_direct_access)) {           \
          GemmKernel()(output_mapper, lhsBlock.raw_data, rhsBlock.raw_data,    \
                       rows, depth, cols, alpha, beta,                         \
                       /*ldA=*/lhsBlock.stride, /*ldB=*/rhsBlock.stride,       \
                       /*transposeA=*/lhsBlock.transpose,                      \
                       /*transposeB=*/rhsBlock.transpose);                     \
                                                                               \
        } else if (DirectLhsAccess::value && lhsBlock.is_direct_access) {      \
          GemmKernel()(output_mapper, lhsBlock.raw_data, rhsBlock.packed_data, \
                       rows, depth, cols, alpha, beta,                         \
                       /*ldA=*/lhsBlock.stride,                                \
                       /*ldB=*/GemmKernel::kComputeStrideFromBlockDimensions,  \
                       /*transposeA=*/lhsBlock.transpose, /*transposeB=*/'N'); \
                                                                               \
        } else if (DirectRhsAccess::value && rhsBlock.is_direct_access) {      \
          GemmKernel()(output_mapper, lhsBlock.packed_data, rhsBlock.raw_data, \
                       rows, depth, cols, alpha, beta,                         \
                       /*ldA=*/GemmKernel::kComputeStrideFromBlockDimensions,  \
                       /*ldB=*/rhsBlock.stride, /*transposeA=*/'N',            \
                       /*transposeB=*/rhsBlock.transpose);                     \
                                                                               \
        } else {                                                               \
          GemmKernel()(output_mapper, lhsBlock.packed_data,                    \
                       rhsBlock.packed_data, rows, depth, cols, alpha, beta);  \
        }                                                                      \
      } else {                                                                 \
        /* Gebp kernel does not support beta, so we have to clear memory in */ \
        /* the output mapper manually.                                      */ \
        /* WARNING(ezhulenev): This is optimized into a memset in a loop,   */ \
        /* could be much slower for small matrices. Currently this code     */ \
        /* path used only for testing, and performance does not matter.     */ \
        if (beta == 0.0) {                                                     \
          for (StorageIndex col = 0; col < cols; ++col) {                      \
            ResScalar* output_base = &output_mapper(0, col);                   \
            typedef Array<ResScalar, Dynamic, 1> OutputRow;                    \
            typedef Map<OutputRow, 0, InnerStride<1>> OutputRowMap;            \
            OutputRowMap(output_base, rows).setZero();                         \
          }                                                                    \
        }                                                                      \
                                                                               \
        GebpKernel()(                                                          \
            output_mapper, lhsBlock.packed_data, rhsBlock.packed_data, rows,   \
            depth, cols, alpha,                                                \
            /*strideA*/ GemmKernel::kComputeStrideFromBlockDimensions,         \
            /*strideB*/ GemmKernel::kComputeStrideFromBlockDimensions,         \
            /*offsetA*/ 0, /*offsetB*/ 0);                                     \
      }                                                                        \
    }                                                                          \
                                                                               \
   private:                                                                    \
    /* These are dimensions of the original Tensors, and selected block     */ \
    /* sizes. The actual block sizes passed to all function above might be  */ \
    /* smaller because of the partial blocks at the end.                    */ \
    const StorageIndex m;                                                      \
    const StorageIndex k;                                                      \
    const StorageIndex n;                                                      \
    const StorageIndex bm;                                                     \
    const StorageIndex bk;                                                     \
    const StorageIndex bn;                                                     \
  }

// Tensor contraction kernel that do not fallback on Eigen. Currently not all
// data types are supported by Eigen data packing and default gebp_kernel.
#define REGISTER_TENSOR_CONTRACTION_KERNEL_NO_FALLBACK(RES_SCALAR, LHS_SCALAR, \
                                                       RHS_SCALAR)             \
                                                                               \
  template <typename StorageIndex, typename OutputMapper, typename LhsMapper,  \
            typename RhsMapper>                                                \
  struct TensorContractionKernel<RES_SCALAR, LHS_SCALAR, RHS_SCALAR,           \
                                 StorageIndex, OutputMapper, LhsMapper,        \
                                 RhsMapper> {                                  \
    TensorContractionKernel(StorageIndex m, StorageIndex k, StorageIndex n,    \
                            StorageIndex bm, StorageIndex bk, StorageIndex bn) \
        : m(m), k(k), n(n), bm(bm), bk(bk), bn(bn) {}                          \
                                                                               \
    enum { HasBeta = true };                                                   \
                                                                               \
    using ResScalar = RES_SCALAR;                                              \
    using LhsScalar = LHS_SCALAR;                                              \
    using RhsScalar = RHS_SCALAR;                                              \
                                                                               \
    using Traits = typename internal::gebp_traits<LhsScalar, RhsScalar>;       \
                                                                               \
    using LhsBlock = ColMajorBlock<LhsScalar, StorageIndex>;                   \
    using RhsBlock = ColMajorBlock<RhsScalar, StorageIndex>;                   \
                                                                               \
    using DirectLhsAccess = DirectColMajorAccess<LhsMapper>;                   \
    using DirectRhsAccess = DirectColMajorAccess<RhsMapper>;                   \
                                                                               \
    /* Packed Lhs/Rhs block memory allocator.*/                                \
    typedef TensorContractionBlockMemAllocator<LhsScalar, RhsScalar>           \
        BlockMemAllocator;                                                     \
    typedef typename BlockMemAllocator::BlockMemHandle BlockMemHandle;         \
                                                                               \
    using LhsPacker =                                                          \
        gemm_pack_colmajor_block<LhsScalar, StorageIndex,                      \
                                 typename LhsMapper::SubMapper, ColMajor>;     \
    using RhsPacker =                                                          \
        gemm_pack_colmajor_block<RhsScalar, StorageIndex,                      \
                                 typename RhsMapper::SubMapper, ColMajor>;     \
                                                                               \
    using GemmKernelProviderType =                                             \
        GemmKernelProvider<ResScalar, LhsScalar, RhsScalar, StorageIndex,      \
                           OutputMapper>;                                      \
    static_assert(                                                             \
        GemmKernelProviderType::Defined,                                       \
        "Custom GEMM kernel is not registered for given scalar types");        \
    using GemmKernel = typename GemmKernelProviderType::GemmKernel;            \
                                                                               \
    template <typename Device>                                                 \
    EIGEN_DEVICE_FUNC BlockMemHandle allocate(Device& d, LhsBlock* lhs_block,  \
                                              RhsBlock* rhs_block) {           \
      return BlockMemAllocator::allocate(                                      \
          d, bm, bk, bn, &lhs_block->packed_data, &rhs_block->packed_data);    \
    }                                                                          \
                                                                               \
    template <typename Device>                                                 \
    EIGEN_DEVICE_FUNC BlockMemHandle                                           \
    allocateSlices(Device& d, const int num_lhs, const int num_rhs,            \
                   const int num_slices, std::vector<LhsBlock>* lhs_blocks,    \
                   std::vector<RhsBlock>* rhs_blocks) {                        \
      eigen_assert(num_slices > 0);                                            \
      std::vector<std::vector<LhsScalar*>> lhs_mem(num_slices);                \
      std::vector<std::vector<RhsScalar*>> rhs_mem(num_slices);                \
                                                                               \
      BlockMemHandle block_mem = BlockMemAllocator::allocateSlices(            \
          d, bm, bk, bn, num_lhs, num_rhs, num_slices, lhs_mem.data(),         \
          rhs_mem.data());                                                     \
                                                                               \
      for (Index x = 0; x < num_slices; x++) {                                 \
        if (num_lhs > 0) lhs_blocks[x].resize(num_lhs);                        \
        for (Index m = 0; m < num_lhs; m++) {                                  \
          lhs_blocks[x][m].packed_data = lhs_mem[x][m];                        \
        }                                                                      \
        if (num_rhs > 0) rhs_blocks[x].resize(num_rhs);                        \
        for (Index n = 0; n < num_rhs; n++) {                                  \
          rhs_blocks[x][n].packed_data = rhs_mem[x][n];                        \
        }                                                                      \
      }                                                                        \
                                                                               \
      return block_mem;                                                        \
    }                                                                          \
                                                                               \
    template <typename Device>                                                 \
    EIGEN_DEVICE_FUNC static void deallocate(Device& d,                        \
                                             BlockMemHandle handle) {          \
      BlockMemAllocator::deallocate(d, handle);                                \
    }                                                                          \
                                                                               \
    EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE void packLhs(                          \
        LhsBlock* lhsBlock, const typename LhsMapper::SubMapper& data_mapper,  \
        const StorageIndex depth, const StorageIndex rows) {                   \
      const bool is_direct_access =                                            \
          DirectLhsAccess::value &&                                            \
          DirectLhsAccess::block(data_mapper, rows, depth,                     \
                                 bn > 0 ? divup(n, bn) : 0, lhsBlock);         \
                                                                               \
      if (!is_direct_access) {                                                 \
        lhsBlock->is_direct_access = false;                                    \
        LhsPacker()(lhsBlock->packed_data, data_mapper, rows, depth);          \
      }                                                                        \
    }                                                                          \
                                                                               \
    EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE void packRhs(                          \
        RhsBlock* rhsBlock, const typename RhsMapper::SubMapper& data_mapper,  \
        const StorageIndex depth, const StorageIndex cols) {                   \
      const bool is_direct_access =                                            \
          DirectRhsAccess::value &&                                            \
          DirectRhsAccess::block(data_mapper, depth, cols,                     \
                                 bm > 0 ? divup(m, bm) : 0, rhsBlock);         \
                                                                               \
      if (!is_direct_access) {                                                 \
        rhsBlock->is_direct_access = false;                                    \
        RhsPacker()(rhsBlock->packed_data, data_mapper, depth, cols);          \
      }                                                                        \
    }                                                                          \
                                                                               \
    EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE void invoke(                           \
        const OutputMapper& output_mapper, const LhsBlock& lhsBlock,           \
        const RhsBlock& rhsBlock, const StorageIndex rows,                     \
        const StorageIndex depth, const StorageIndex cols, const float alpha,  \
        const float beta) {                                                    \
      if ((DirectLhsAccess::value && lhsBlock.is_direct_access) &&             \
          (DirectRhsAccess::value && rhsBlock.is_direct_access)) {             \
        GemmKernel()(output_mapper, lhsBlock.raw_data, rhsBlock.raw_data,      \
                     rows, depth, cols, alpha, beta, /*ldA=*/lhsBlock.stride,  \
                     /*ldB=*/rhsBlock.stride,                                  \
                     /*transposeA=*/lhsBlock.transpose,                        \
                     /*transposeB=*/rhsBlock.transpose);                       \
                                                                               \
      } else if (DirectLhsAccess::value && lhsBlock.is_direct_access) {        \
        GemmKernel()(output_mapper, lhsBlock.raw_data, rhsBlock.packed_data,   \
                     rows, depth, cols, alpha, beta, /*ldA=*/lhsBlock.stride,  \
                     /*ldB=*/GemmKernel::kComputeStrideFromBlockDimensions,    \
                     /*transposeA=*/lhsBlock.transpose, /*transposeB=*/'N');   \
                                                                               \
      } else if (DirectRhsAccess::value && rhsBlock.is_direct_access) {        \
        GemmKernel()(output_mapper, lhsBlock.packed_data, rhsBlock.raw_data,   \
                     rows, depth, cols, alpha, beta,                           \
                     /*ldA=*/GemmKernel::kComputeStrideFromBlockDimensions,    \
                     /*ldB=*/rhsBlock.stride, /*transposeA=*/'N',              \
                     /*transposeB=*/rhsBlock.transpose);                       \
                                                                               \
      } else {                                                                 \
        GemmKernel()(output_mapper, lhsBlock.packed_data,                      \
                     rhsBlock.packed_data, rows, depth, cols, alpha, beta);    \
      }                                                                        \
    }                                                                          \
                                                                               \
   private:                                                                    \
    /* These are dimensions of the original Tensors, and selected block     */ \
    /* sizes. The actual block sizes passed to all function above might be  */ \
    /* smaller because of the partial blocks at the end.                    */ \
    const StorageIndex m;                                                      \
    const StorageIndex k;                                                      \
    const StorageIndex n;                                                      \
    const StorageIndex bm;                                                     \
    const StorageIndex bk;                                                     \
    const StorageIndex bn;                                                     \
  }

REGISTER_TENSOR_CONTRACTION_KERNEL_WITH_FALLBACK(float, float, float);
REGISTER_TENSOR_CONTRACTION_KERNEL_NO_FALLBACK(Eigen::QInt32, Eigen::QInt8,
                                               Eigen::QUInt8);

#undef REGISTER_TENSOR_CONTRACTION_KERNEL

#endif  // defined(TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL)

}  // namespace internal
}  // namespace Eigen

#endif  // TENSORFLOW_TSL_FRAMEWORK_CONTRACTION_EIGEN_CONTRACTION_KERNEL_H_
