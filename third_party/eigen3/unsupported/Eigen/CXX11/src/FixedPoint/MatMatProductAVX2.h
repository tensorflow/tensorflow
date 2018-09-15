// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
// Copyright (C) 2015 Matthew Sarett <msarett@google.com>
// Copyright (C) 2016 Nishant Patil <nishantpatil@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef CXX11_SRC_FIXEDPOINT_MATMATPRODUCTAVX2_H_
#define CXX11_SRC_FIXEDPOINT_MATMATPRODUCTAVX2_H_

namespace Eigen {
namespace internal {

// AVX2 optimized implementation of Mat-Mat product.
// LHS is encoded using signed 16-bit integers.
// RHS is encoded using signed 16-bit integers.
#ifdef EIGEN_USE_OPTIMIZED_INT16_INT16_MAT_MAT_PRODUCT

// Define quantized traits
template <bool _ConjLhs, bool _ConjRhs>
class gebp_traits<QInt16, QInt16, _ConjLhs, _ConjRhs> {
 public:
  typedef QInt16 LhsScalar;
  typedef QInt16 RhsScalar;
  typedef QInt32 ResScalar;

  enum {
    // Define register blocking scheme.
    nr = 16,
    mr = 16,
    kr = 4,
    // Ignore progress tracking per loop iteration.
    LhsProgress = -1,
    RhsProgress = -1
  };
};

// Specialized blocking for quantized implementations.
// Used by TensorContractionThreadPool, inputs must have dimensions that are
// multiples of 32.
template <typename Index, int ShardingType>
class TensorContractionBlocking<QInt16, QInt16, Index, ShardingType> {
 public:
  TensorContractionBlocking(Index k, Index m, Index n, Index num_threads = 1)
      : kc_(((k + 15) / 16) * 16),
        mc_(((m + 15) / 16) * 16),
        nc_(((n + 15) / 16) * 16) {
    eigen_assert(mc_ % 16 == 0);
    eigen_assert(kc_ % 16 == 0);
    if (!k || !m || !n) {
      return;
    }

    if (ShardingType == ShardByCol) {
      eigen_assert(nc_ % 16 == 0);
      nc_ = (((nc_ / num_threads) + 15) / 16) * 16;
    } else {
      eigen_assert(nc_ % 16 == 0);
      mc_ = (((mc_ / num_threads) + 15) / 16) * 16;
    }
  }

  EIGEN_ALWAYS_INLINE Index kc() const { return kc_; }
  EIGEN_ALWAYS_INLINE Index mc() const { return mc_; }
  EIGEN_ALWAYS_INLINE Index nc() const { return nc_; }

 private:
  Index kc_;
  Index mc_;
  Index nc_;
};

// Specialized blocking for quantized implementations.
// Used by TensorContraction and GeneralMatrixMatrix, inputs are padded to
// multiples of 32.
template <int MaxRows, int MaxCols, int MaxDepth, int KcFactor>
class gemm_blocking_space<ColMajor, QInt16, QInt16, MaxRows, MaxCols, MaxDepth,
                          KcFactor, false>
    : public level3_blocking<QInt16, QInt16> {
  DenseIndex m_sizeA;
  DenseIndex m_sizeB;

 public:
  gemm_blocking_space(DenseIndex rows, DenseIndex cols, DenseIndex depth,
                      DenseIndex /*num_threads*/, bool /*l3_blocking*/) {
    this->m_mc = ((rows + 15) / 16) * 16;
    this->m_nc = ((cols + 15) / 16) * 16;
    this->m_kc = ((depth + 15) / 16) * 16;
    m_sizeA = this->m_mc * this->m_kc;
    m_sizeB = this->m_kc * this->m_nc;
  }
  void allocateA() {
    if (this->m_blockA == 0) this->m_blockA = aligned_new<QInt16>(m_sizeA);
  }
  void allocateB() {
    if (this->m_blockB == 0) this->m_blockB = aligned_new<QInt16>(m_sizeB);
  }
  void allocateAll() {
    allocateA();
    allocateB();
  }
  ~gemm_blocking_space() {
    aligned_delete(this->m_blockA, m_sizeA);
    aligned_delete(this->m_blockB, m_sizeB);
  }
};

// Below are the fully optimized versions that are correct only for sizes that
// are multiple of 16.  It is about a 10% performance benefit to keep these
// implementations separate.

// Arrange a block of the left input matrix in contiguous memory.
//
// Given column major input (A0 beside A1 in memory):
// A0 B0 C0 D0 E0 F0 G0 H0 ...
// A1 B1 C1 D1 E1 F1 G1 H1 ...
// A2 B2 C2 D2 E2 F2 G2 H2 ...
// A3 B3 C3 D3 E3 F3 G3 H3 ...
// A4 B4 C4 D4 E4 F4 G4 H4 ...
// A5 B5 C5 D5 E5 F5 G5 H5 ...
// A6 B6 C6 D6 E6 F6 G6 H6 ...
// A7 B7 C7 D7 E7 F7 G7 H7 ...
// A8 ...
// ...
//
// Packing with m = 8 yields row major output (A0 beside B0 in memory):
// A0 B0
// A1 B1
// A2 B2
// A3 B3
// A4 B4
// A5 B5
// A6 B6
// A7 B7
// ...
//
// The purpose is to collect m rows of size k.  Two elements of the same
// row are arranged contiguously because madd performs an adjacent addition
// in the kernel.

template <typename Index, typename DataMapper, int Pack1, int Pack2,
          bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<QInt16, Index, DataMapper, Pack1, Pack2, ColMajor,
                     Conjugate, PanelMode> {
  EIGEN_DONT_INLINE void operator()(QInt16* blockA, const DataMapper& lhs,
                                    Index depth, Index rows, Index stride = 0,
                                    Index offset = 0);
};

template <typename Index, typename DataMapper, int Pack1, int Pack2,
          bool Conjugate, bool PanelMode>
EIGEN_DONT_INLINE void gemm_pack_lhs<QInt16, Index, DataMapper, Pack1, Pack2,
                                     ColMajor, Conjugate, PanelMode>::
operator()(QInt16* blockA, const DataMapper& lhs, Index depth, Index rows,
           Index stride, Index offset) {
  eigen_assert(stride == 0);
  eigen_assert(offset == 0);

  // Use alternate function for weird sizes
  if (rows % 16 != 0 || depth % 16 != 0) {
    assert(false &&
           "only depths and rows that are a multiple of 16 are currently "
           "supported");
    // gemm_pack_lhs_any<QInt16, Index, DataMapper, Pack1, Pack2, ColMajor,
    // Conjugate, PanelMode> lhs_pack;
    // return lhs_pack(blockA, lhs, depth, rows, stride, offset);
  }

  // Get vector pointer
  __m256i* blockA_256 = reinterpret_cast<__m256i*>(blockA);

  // Pack rows in sets of 16
  for (Index m = 0; m < rows; m += 16) {
    // Pack depth in sets of 4
    for (Index k = 0; k < depth; k += 4) {
      // Load vectors
      __m256i L_A = lhs.loadPacket(m, k);
      __m256i L_B = lhs.loadPacket(m, k + 1);
      __m256i L_C = lhs.loadPacket(m, k + 2);
      __m256i L_D = lhs.loadPacket(m, k + 3);

      // Rearrange the inputs as required by the kernel
      __m256i L_AB0_AB7 = _mm256_unpacklo_epi16(L_A, L_B);
      __m256i L_AB8_AB15 = _mm256_unpackhi_epi16(L_A, L_B);
      __m256i L_CD0_CD7 = _mm256_unpacklo_epi16(L_C, L_D);
      __m256i L_CD8_CD15 = _mm256_unpackhi_epi16(L_C, L_D);

      __m256i L_AD0 = _mm256_permute2x128_si256(L_AB0_AB7, L_AB8_AB15, 0x20);
      _mm256_store_si256(blockA_256++, L_AD0);
      __m256i L_AD8 = _mm256_permute2x128_si256(L_CD0_CD7, L_CD8_CD15, 0x20);
      _mm256_store_si256(blockA_256++, L_AD8);
      __m256i L_AD16 = _mm256_permute2x128_si256(L_AB0_AB7, L_AB8_AB15, 0x31);
      _mm256_store_si256(blockA_256++, L_AD16);
      __m256i L_AD24 = _mm256_permute2x128_si256(L_CD0_CD7, L_CD8_CD15, 0x31);
      _mm256_store_si256(blockA_256++, L_AD24);
    }
  }
}

// Arrange a block of the right input matrix in contiguous memory.
//
// Given column major input (A0 beside A1 in memory):
// A0 B0 C0 D0 E0 F0 G0 H0 ...
// A1 B1 C1 D1 E1 F1 G1 H1 ...
// A2 B2 C2 D2 E2 F2 G2 H2 ...
// A3 B3 C3 D3 E3 F3 G3 H3 ...
// A4 B4 C4 D4 E4 F4 G4 H4 ...
// A5 B5 C5 D5 E5 F5 G5 H5 ...
// A6 B6 C6 D6 E6 F6 G6 H6 ...
// A7 B7 C7 D7 E7 F7 G7 H7 ...
// A8 ...
// ...
// Packing yields row major output (A0 beside A1 in memory):
// A0 A1 A2 A3 A4 A5 A6 A7
// B0 B1 B2 B3 B4 B5 B6 B7
// ...
//
// At least two elements of the same col are arranged contiguously because
// maddubs and madd both perform an adjacent addition in the kernel.  We can
// save work by leaving 4 adjacent elements because kr = 4.
// The purpose is to collect n cols of size k.  Two elements of the same
// col are arranged contiguously because madd performs an adjacent addition
// in the kernel.
template <typename Index, typename DataMapper, int nr, bool Conjugate,
          bool PanelMode>
struct gemm_pack_rhs<QInt16, Index, DataMapper, nr, ColMajor, Conjugate,
                     PanelMode> {
  EIGEN_DONT_INLINE void operator()(QInt16* blockB, const DataMapper& rhs,
                                    Index depth, Index cols, Index stride = 0,
                                    Index offset = 0);
};

template <typename Index, typename DataMapper, int nr, bool Conjugate,
          bool PanelMode>
EIGEN_DONT_INLINE void
gemm_pack_rhs<QInt16, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode>::
operator()(QInt16* blockB, const DataMapper& rhs, Index depth, Index cols,
           Index stride, Index offset) {
  eigen_assert(stride == 0);
  eigen_assert(offset == 0);

  // Use alternate function for weird sizes
  if (cols % 16 != 0 || depth % 16 != 0) {
    assert(false &&
           "only depths and cols that are a multiple of 16 are currently "
           "supported");
    // gemm_pack_rhs_any<QInt16, Index, DataMapper, nr, ColMajor, Conjugate,
    // PanelMode> rhs_pack;
    // return rhs_pack(blockB, rhs, depth, cols, stride, offset);
  }

  // Get vector pointer
  __m256i* blockB_256 = reinterpret_cast<__m256i*>(blockB);

  // Perform a step of the packing for 4 columns
  __m256i R_AB_L, R_AB_H, R_CD_L, R_CD_H, R_AD_0, R_AD_4, R_AD_8, R_AD_12;
#define PACK_STEP                                            \
  R_AB_L = _mm256_unpacklo_epi64(R_A, R_B);                  \
  R_CD_L = _mm256_unpacklo_epi64(R_C, R_D);                  \
  R_AB_H = _mm256_unpackhi_epi64(R_A, R_B);                  \
  R_CD_H = _mm256_unpackhi_epi64(R_C, R_D);                  \
  R_AD_0 = _mm256_permute2x128_si256(R_AB_L, R_CD_L, 0x20);  \
  R_AD_8 = _mm256_permute2x128_si256(R_AB_L, R_CD_L, 0x31);  \
  R_AD_4 = _mm256_permute2x128_si256(R_AB_H, R_CD_H, 0x20);  \
  R_AD_12 = _mm256_permute2x128_si256(R_AB_H, R_CD_H, 0x31); \
  _mm256_store_si256(blockB_256, R_AD_0);                    \
  _mm256_store_si256(blockB_256 + 4, R_AD_4);                \
  _mm256_store_si256(blockB_256 + 8, R_AD_8);                \
  _mm256_store_si256(blockB_256 + 12, R_AD_12);              \
  blockB_256++;

  // Pack cols in sets of 16
  for (Index n = 0; n < cols; n += 16) {
    // Pack depth in sets of 16
    for (Index k = 0; k < depth; k += 16) {
      __m256i R_A = rhs.loadPacket(k, n);
      __m256i R_B = rhs.loadPacket(k, n + 1);
      __m256i R_C = rhs.loadPacket(k, n + 2);
      __m256i R_D = rhs.loadPacket(k, n + 3);
      PACK_STEP;

      R_A = rhs.loadPacket(k, n + 4);
      R_B = rhs.loadPacket(k, n + 5);
      R_C = rhs.loadPacket(k, n + 6);
      R_D = rhs.loadPacket(k, n + 7);
      PACK_STEP;

      R_A = rhs.loadPacket(k, n + 8);
      R_B = rhs.loadPacket(k, n + 9);
      R_C = rhs.loadPacket(k, n + 10);
      R_D = rhs.loadPacket(k, n + 11);
      PACK_STEP;

      R_A = rhs.loadPacket(k, n + 12);
      R_B = rhs.loadPacket(k, n + 13);
      R_C = rhs.loadPacket(k, n + 14);
      R_D = rhs.loadPacket(k, n + 15);
      PACK_STEP;

      blockB_256 += 12;
    }
  }
#undef PACK_STEP
}

// Perform the actual multiplication on packed inputs
template <typename Index, typename DataMapper, int mr, int nr,
          bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<QInt16, QInt16, Index, DataMapper, mr, nr, ConjugateLhs,
                   ConjugateRhs> {
  typedef typename DataMapper::LinearMapper LinearMapper;

  EIGEN_DONT_INLINE
  void operator()(const DataMapper& res, const QInt16* blockA,
                  const QInt16* blockB, Index rows, Index depth, Index cols,
                  QInt32 alpha, Index strideA = -1, Index strideB = -1,
                  Index offsetA = 0, Index offsetB = 0);
};

template <typename Index, typename DataMapper, int mr, int nr,
          bool ConjugateLhs, bool ConjugateRhs>
EIGEN_DONT_INLINE void gebp_kernel<QInt16, QInt16, Index, DataMapper, mr, nr,
                                   ConjugateLhs, ConjugateRhs>::
operator()(const DataMapper& res, const QInt16* blockA, const QInt16* blockB,
           Index rows, Index depth, Index cols, QInt32 alpha, Index strideA,
           Index strideB, Index offsetA, Index offsetB) {
  EIGEN_STATIC_ASSERT(!ConjugateLhs, YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT(!ConjugateRhs, YOU_MADE_A_PROGRAMMING_MISTAKE);
  eigen_assert(alpha.value == 1);
  eigen_assert(strideA == -1);
  eigen_assert(strideB == -1);
  eigen_assert(offsetA == 0);
  eigen_assert(offsetB == 0);
  eigen_assert(rows > 0);
  eigen_assert(cols > 0);
  eigen_assert(depth > 0);
  eigen_assert(blockA);
  eigen_assert(blockB);

  // Use alternate function for weird sizes
  if (rows % 16 != 0 || cols % 16 != 0 || depth % 16 != 0) {
    assert(false &&
           "only depths, cols and rows that are a multiple of 16 are currently "
           "supported");
    // gebp_kernel_any<QInt16, QInt16, Index, DataMapper, mr, nr, ConjugateLhs,
    // ConjugateRhs> gebp;
    // return gebp(res, blockA, blockB, rows, depth, cols, alpha, strideA,
    // strideB, offsetA, offsetB);
  }

  // Create result block
  QInt32* blockO = aligned_new<QInt32>(16 * 16);
  memset(blockO, 0, 16 * 16 * sizeof(QInt32));

  // Get vectorized pointers
  __m256i* blockO_256 = reinterpret_cast<__m256i*>(blockO);
  const __m256i* blockA_256 = reinterpret_cast<const __m256i*>(blockA);
  const __m256i* blockB_256 = reinterpret_cast<const __m256i*>(blockB);

  // Loop over blocks of 16 columns
  for (Index n = 0; n < cols; n += 16) {
    // Reset index into blockA
    Index indexL = 0;
    // Loop over blocks of 16 rows
    for (Index m = 0; m < rows; m += 16) {
      // Reset index into blockB
      Index indexR = n / 16 * depth;
      // Loop over blocks of 4 on depth
      for (Index k = 0; k < depth; k += 4) {
        // Load inputs
        __m256i L_AD0 = blockA_256[indexL++];
        __m256i L_AD8 = blockA_256[indexL++];
        __m256i L_EH0 = blockA_256[indexL++];
        __m256i L_EH8 = blockA_256[indexL++];

        __m256i R_AH0 = blockB_256[indexR++];
        __m256i R_AH4 = blockB_256[indexR++];
        __m256i R_AH8 = blockB_256[indexR++];
        __m256i R_AH12 = blockB_256[indexR++];

        // Declare variables used in COMPUTE_STEP
        __m256i P_32_A, P_32_B, P_32;

#define COMPUTE_STEP(R_INPUT_A, R_INPUT_B, OFFSET)                         \
  P_32_A = _mm256_madd_epi16(R_INPUT_A, L_AD0);                            \
  P_32_B = _mm256_madd_epi16(R_INPUT_B, L_AD8);                            \
  P_32 = _mm256_add_epi32(P_32_A, P_32_B);                                 \
  _mm256_store_si256(                                                      \
      blockO_256 + 2 * OFFSET,                                             \
      _mm256_add_epi32(_mm256_load_si256(blockO_256 + 2 * OFFSET), P_32)); \
                                                                           \
  P_32_A = _mm256_madd_epi16(R_INPUT_A, L_EH0);                            \
  P_32_B = _mm256_madd_epi16(R_INPUT_B, L_EH8);                            \
  P_32 = _mm256_add_epi32(P_32_A, P_32_B);                                 \
  _mm256_store_si256(                                                      \
      blockO_256 + 2 * OFFSET + 1,                                         \
      _mm256_add_epi32(_mm256_load_si256(blockO_256 + 2 * OFFSET + 1), P_32));

        // Permute and shuffle to copy a single value across the entire vector
        // Then compute the multiplication
        // Replicate lower 128-bits of R_AH0 across both lanes
        __m256i R_AH0_ = _mm256_permute2x128_si256(R_AH0, R_AH0, 0x00);
        // Copy first two elements of R_AH0 across entire vector
        __m256i R_AD0 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        // Copy second two elements of R_AH0 across entire vector
        __m256i R_EH0 = _mm256_shuffle_epi32(R_AH0_, 0x55);

        COMPUTE_STEP(R_AD0, R_EH0, 0);
        __m256i R_AD1 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        __m256i R_EH1 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD1, R_EH1, 1);

        // Replicate upper 128-bits of R_AH0 across both lanes
        R_AH0_ = _mm256_permute2x128_si256(R_AH0, R_AH0, 0x11);
        __m256i R_AD2 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        __m256i R_EH2 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD2, R_EH2, 2);
        __m256i R_AD3 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        __m256i R_EH3 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD3, R_EH3, 3);

        R_AH0_ = _mm256_permute2x128_si256(R_AH4, R_AH4, 0x00);
        R_AD0 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH0 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD0, R_EH0, 4);
        R_AD1 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH1 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD1, R_EH1, 5);
        R_AH0_ = _mm256_permute2x128_si256(R_AH4, R_AH4, 0x11);
        R_AD2 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH2 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD2, R_EH2, 6);
        R_AD3 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH3 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD3, R_EH3, 7);

        R_AH0_ = _mm256_permute2x128_si256(R_AH8, R_AH8, 0x00);
        R_AD0 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH0 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD0, R_EH0, 8);
        R_AD1 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH1 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD1, R_EH1, 9);
        R_AH0_ = _mm256_permute2x128_si256(R_AH8, R_AH8, 0x11);
        R_AD2 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH2 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD2, R_EH2, 10);
        R_AD3 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH3 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD3, R_EH3, 11);

        R_AH0_ = _mm256_permute2x128_si256(R_AH12, R_AH12, 0x00);
        R_AD0 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH0 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD0, R_EH0, 12);
        R_AD1 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH1 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD1, R_EH1, 13);
        R_AH0_ = _mm256_permute2x128_si256(R_AH12, R_AH12, 0x11);
        R_AD2 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH2 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD2, R_EH2, 14);
        R_AD3 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH3 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD3, R_EH3, 15);

#undef COMPUTE_STEP
      }

      // Transfer the results to the result matrix
      Index i = 0;
      for (Index j = n; j < n + 16; j++) {
        LinearMapper r0 = res.getLinearMapper(m, j);
        LinearMapper r1 = res.getLinearMapper(m + 8, j);

        r0.storePacket(0, _mm256_add_epi32(blockO_256[i++], r0.loadPacket(0)));
        r1.storePacket(0, _mm256_add_epi32(blockO_256[i++], r1.loadPacket(0)));
      }

      // Zero the result block so it can be reused
      memset(blockO, 0, 16 * 16 * sizeof(QInt32));
    }
  }
  aligned_delete(blockO, 16 * 16);
}

#endif

// AVX2 optimized implementation of Mat-Mat product.
// LHS is encoded using signed 8-bit integers.
// RHS is encoded using unsigned 8-bit integers.
#ifdef EIGEN_USE_OPTIMIZED_INT8_UINT8_MAT_MAT_PRODUCT

// Define quantized traits
template<bool _ConjLhs, bool _ConjRhs>
class gebp_traits<QInt8, QUInt8, _ConjLhs, _ConjRhs>
{
public:
  typedef QInt8 LhsScalar;
  typedef QUInt8 RhsScalar;
  typedef QInt32 ResScalar;

  enum {
    // Define register blocking scheme.
    nr = 32,
    mr = 32,
    kr = 8,
    // Ignore progress tracking per loop iteration.
    LhsProgress = -1,
    RhsProgress = -1
  };
};

// Specialized blocking for quantized implementations.
// Used by TensorContractionThreadPool, inputs must have dimensions that are
// multiples of 32.
template<typename Index,
         typename LeftTensor,
         typename left_nocontract_t, typename left_contract_t,
         bool left_inner_dim_contiguous, bool left_inner_dim_reordered, int LeftAlignment,
         typename RightTensor,
         typename right_nocontract_t, typename right_contract_t,
         bool right_inner_dim_contiguous, bool right_inner_dim_reordered, int RightAlignment, int ShardingType>
class TensorContractionBlocking<TensorContractionInputMapper<QInt8, Index, Lhs, LeftTensor, left_nocontract_t, left_contract_t, 32, left_inner_dim_contiguous, left_inner_dim_reordered, LeftAlignment>, TensorContractionInputMapper<QUInt8, Index, Rhs, RightTensor, right_nocontract_t, right_contract_t, 32, right_inner_dim_contiguous, right_inner_dim_reordered, RightAlignment>, Index, ShardingType> {
 public:

  typedef QInt8  LhsScalar;
  typedef QUInt8 RhsScalar;

  TensorContractionBlocking(Index k, Index m, Index n, Index num_threads = 1) :
      kc_(k), mc_(m), nc_(n)
  {
    eigen_assert(m % 32 == 0);
    eigen_assert(k % 32 == 0);
    if (!k || !m || !n) {
      return;
    }

    if (ShardingType == ShardByCol) {
      eigen_assert(n % 32 == 0);
      nc_ = (((n / num_threads) + 31) / 32) * 32;
    }
    else {
      eigen_assert(n % 32 == 0 || n == 1);
      // Special case to avoid breaking the unimplemented matrix-vector case
      if (n == 1) {
        nc_ = 32;
      }
      mc_ = (((m / num_threads) + 31) / 32) * 32;
    }
  }

  EIGEN_ALWAYS_INLINE Index kc() const { return kc_; }
  EIGEN_ALWAYS_INLINE Index mc() const { return mc_; }
  EIGEN_ALWAYS_INLINE Index nc() const { return nc_; }

 private:
  Index kc_;
  Index mc_;
  Index nc_;
};

// Specialized blocking for quantized implementations.
// Used by TensorContraction and GeneralMatrixMatrix, inputs are padded to
// multiples of 32.
template <int MaxRows, int MaxCols, int MaxDepth, int KcFactor>
class gemm_blocking_space<ColMajor, QInt8, QInt8, MaxRows, MaxCols, MaxDepth,
                          KcFactor, false>
    : public level3_blocking<QInt8, QInt8> {
  DenseIndex m_sizeA;
  DenseIndex m_sizeB;

 public:
  gemm_blocking_space(DenseIndex rows, DenseIndex cols, DenseIndex depth,
                      DenseIndex /*num_threads*/, bool /*l3_blocking*/) {
    this->m_mc = ((rows + 31) / 32) * 32;
    this->m_nc = ((cols + 31) / 32) * 32;
    this->m_kc = ((depth + 31) / 32) * 32;
    m_sizeA = this->m_mc * this->m_kc;
    m_sizeB = this->m_kc * this->m_nc;
  }
  void allocateA() {
    if (this->m_blockA == 0) this->m_blockA = aligned_new<QInt8>(m_sizeA);
  }
  void allocateB() {
    if (this->m_blockB == 0) this->m_blockB = aligned_new<QInt8>(m_sizeB);
  }
  void allocateAll() {
    allocateA();
    allocateB();
  }
  ~gemm_blocking_space() {
    aligned_delete(this->m_blockA, m_sizeA);
    aligned_delete(this->m_blockB, m_sizeB);
  }
};


template <int MaxRows, int MaxCols, int MaxDepth, int KcFactor>
class gemm_blocking_space<ColMajor, QInt8, QUInt8, MaxRows, MaxCols, MaxDepth,
                          KcFactor, false>
    : public level3_blocking<QInt8, QUInt8> {
  DenseIndex m_sizeA;
  DenseIndex m_sizeB;

 public:
  gemm_blocking_space(DenseIndex rows, DenseIndex cols, DenseIndex depth,
                      DenseIndex /*num_threads*/, bool /*l3_blocking*/) {
    this->m_mc = ((rows + 31) / 32) * 32;
    this->m_nc = ((cols + 31) / 32) * 32;
    this->m_kc = ((depth + 31) / 32) * 32;
    m_sizeA = this->m_mc * this->m_kc;
    m_sizeB = this->m_kc * this->m_nc;
  }
  void allocateA() {
    if (this->m_blockA == 0) this->m_blockA = aligned_new<QInt8>(m_sizeA);
  }
  void allocateB() {
    if (this->m_blockB == 0) this->m_blockB = aligned_new<QUInt8>(m_sizeB);
  }
  void allocateAll() {
    allocateA();
    allocateB();
  }
  ~gemm_blocking_space() {
    aligned_delete(this->m_blockA, m_sizeA);
    aligned_delete(this->m_blockB, m_sizeB);
  }
};

// Alternate templates for any input sizes
template<typename Scalar, typename Index, typename DataMapper, int Pack1, int Pack2, int StorageOrder, bool Conjugate = false, bool PanelMode = false>
struct gemm_pack_lhs_any;
template <typename Index, typename DataMapper, int Pack1, int Pack2, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs_any<QInt8, Index, DataMapper, Pack1, Pack2, ColMajor, Conjugate, PanelMode> {
  EIGEN_DONT_INLINE void operator()
      (QInt8* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride = 0, Index offset = 0);
};

template<typename Scalar, typename Index, typename DataMapper, int nr, int StorageOrder, bool Conjugate = false, bool PanelMode=false>
struct gemm_pack_rhs_any;
template <typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs_any<QUInt8, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode> {
  EIGEN_DONT_INLINE void operator()
      (QUInt8* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride = 0, Index offset = 0);
};

template<typename LhsScalar, typename RhsScalar, typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs=false, bool ConjugateRhs=false>
struct gebp_kernel_any;
template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel_any<QInt8, QUInt8, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
{
  typedef typename DataMapper::LinearMapper LinearMapper;

  EIGEN_DONT_INLINE
  void operator()(const DataMapper& res, const QInt8* blockA, const QUInt8* blockB,
                  Index rows, Index depth, Index cols, QInt32 alpha,
                  Index strideA=-1, Index strideB=-1, Index offsetA=0, Index offsetB=0);
};

// Alternate implementations for any input sizes
template <typename Index, typename DataMapper, int Pack1, int Pack2, bool Conjugate, bool PanelMode>
EIGEN_DONT_INLINE void gemm_pack_lhs_any<QInt8, Index, DataMapper, Pack1, Pack2, ColMajor, Conjugate, PanelMode>::
operator()(QInt8* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset) {
  eigen_assert(stride == 0);
  eigen_assert(offset == 0);

  // Get vector pointer
  __m256i* blockA_256 = reinterpret_cast<__m256i*>(blockA);

  // Get even multiples of the dimensions
  Index rows_32 = (rows / 32) * 32;
  Index depth_8 = (depth / 8) * 8;

  // Get padding for when depth is not a multiple of 32
  int padding = 0;
  if (depth % 32 != 0) {
    int depth_32 = (depth / 32) * 32;
    int extra_depth = depth - depth_32;
    int extra_depth_8 = ((extra_depth + 7) / 8) * 8;
    padding = 32 - extra_depth_8;
  }

  // Pack rows in sets of 32
  for (Index m = 0; m < rows_32; m += 32) {
    // Pack depth in sets of 8
    for (Index k = 0; k < depth_8; k += 8) {
      // Load vectors
      __m256i L_A = lhs.loadPacket(m, k);
      __m256i L_B = lhs.loadPacket(m, k + 1);

      // Interleave 8-bit elements
      __m256i L_AB0_AB16 = _mm256_unpacklo_epi8(L_A, L_B);
      __m256i L_AB8_AB24 = _mm256_unpackhi_epi8(L_A, L_B);

      __m256i L_C = lhs.loadPacket(m, k + 2);
      __m256i L_D = lhs.loadPacket(m, k + 3);
      __m256i L_CD0_CD16 = _mm256_unpacklo_epi8(L_C, L_D);
      __m256i L_CD8_CD24 = _mm256_unpackhi_epi8(L_C, L_D);

      // Interleave 16-bit elements
      __m256i L_AD0_AD16 = _mm256_unpacklo_epi16(L_AB0_AB16, L_CD0_CD16);
      __m256i L_AD4_AD20 = _mm256_unpackhi_epi16(L_AB0_AB16, L_CD0_CD16);

      // Use permute before we store to cross 128-bit lanes
      __m256i L_AD0 = _mm256_permute2x128_si256(L_AD0_AD16, L_AD4_AD20, 0x20);
      _mm256_store_si256(blockA_256++, L_AD0);

      // Complete packing for 32 x 8 block
      __m256i L_AD16 = _mm256_permute2x128_si256(L_AD0_AD16, L_AD4_AD20, 0x31);
      __m256i L_AD8_AD24 = _mm256_unpacklo_epi16(L_AB8_AB24, L_CD8_CD24);
      __m256i L_AD12_AD28 = _mm256_unpackhi_epi16(L_AB8_AB24, L_CD8_CD24);
      __m256i L_AD8 = _mm256_permute2x128_si256(L_AD8_AD24, L_AD12_AD28, 0x20);
      _mm256_store_si256(blockA_256++, L_AD8);
      _mm256_store_si256(blockA_256++, L_AD16);
      __m256i L_AD24 = _mm256_permute2x128_si256(L_AD8_AD24, L_AD12_AD28, 0x31);
      _mm256_store_si256(blockA_256++, L_AD24);
      __m256i L_E = lhs.loadPacket(m, k + 4);
      __m256i L_F = lhs.loadPacket(m, k + 5);
      __m256i L_EF0_EF16 = _mm256_unpacklo_epi8(L_E, L_F);
      __m256i L_EF8_EF24 = _mm256_unpackhi_epi8(L_E, L_F);
      __m256i L_G = lhs.loadPacket(m, k + 6);
      __m256i L_H = lhs.loadPacket(m, k + 7);
      __m256i L_GH0_GH16 = _mm256_unpacklo_epi8(L_G, L_H);
      __m256i L_GH8_GH24 = _mm256_unpackhi_epi8(L_G, L_H);
      __m256i L_EH0_EH16 = _mm256_unpacklo_epi16(L_EF0_EF16, L_GH0_GH16);
      __m256i L_EH4_EH20 = _mm256_unpackhi_epi16(L_EF0_EF16, L_GH0_GH16);
      __m256i L_EH0 = _mm256_permute2x128_si256(L_EH0_EH16, L_EH4_EH20, 0x20);
      _mm256_store_si256(blockA_256++, L_EH0);
      __m256i L_EH16 = _mm256_permute2x128_si256(L_EH0_EH16, L_EH4_EH20, 0x31);
      __m256i L_EH8_EH24 = _mm256_unpacklo_epi16(L_EF8_EF24, L_GH8_GH24);
      __m256i L_EH12_EH28 = _mm256_unpackhi_epi16(L_EF8_EF24, L_GH8_GH24);
      __m256i L_EH8 = _mm256_permute2x128_si256(L_EH8_EH24, L_EH12_EH28, 0x20);
      _mm256_store_si256(blockA_256++, L_EH8);
      _mm256_store_si256(blockA_256++, L_EH16);
      __m256i L_EH24 = _mm256_permute2x128_si256(L_EH8_EH24, L_EH12_EH28, 0x31);
      _mm256_store_si256(blockA_256++, L_EH24);
    }

    // Finish the k dimension, padding with zeros
    if (depth_8 < depth) {
      __m256i L_A, L_B, L_C, L_D, L_E, L_F, L_G, L_H;
      switch (depth - depth_8) {
      case 1:
        L_A = lhs.loadPacket(m, depth_8);
        L_B = _mm256_setzero_si256();
        L_C = _mm256_setzero_si256();
        L_D = _mm256_setzero_si256();
        L_E = _mm256_setzero_si256();
        L_F = _mm256_setzero_si256();
        L_G = _mm256_setzero_si256();
        L_H = _mm256_setzero_si256();
        break;
      case 2:
        L_A = lhs.loadPacket(m, depth_8);
        L_B = lhs.loadPacket(m, depth_8 + 1);
        L_C = _mm256_setzero_si256();
        L_D = _mm256_setzero_si256();
        L_E = _mm256_setzero_si256();
        L_F = _mm256_setzero_si256();
        L_G = _mm256_setzero_si256();
        L_H = _mm256_setzero_si256();
        break;
      case 3:
        L_A = lhs.loadPacket(m, depth_8);
        L_B = lhs.loadPacket(m, depth_8 + 1);
        L_C = lhs.loadPacket(m, depth_8 + 2);
        L_D = _mm256_setzero_si256();
        L_E = _mm256_setzero_si256();
        L_F = _mm256_setzero_si256();
        L_G = _mm256_setzero_si256();
        L_H = _mm256_setzero_si256();
        break;
      case 4:
        L_A = lhs.loadPacket(m, depth_8);
        L_B = lhs.loadPacket(m, depth_8 + 1);
        L_C = lhs.loadPacket(m, depth_8 + 2);
        L_D = lhs.loadPacket(m, depth_8 + 3);
        L_E = _mm256_setzero_si256();
        L_F = _mm256_setzero_si256();
        L_G = _mm256_setzero_si256();
        L_H = _mm256_setzero_si256();
        break;
      case 5:
        L_A = lhs.loadPacket(m, depth_8);
        L_B = lhs.loadPacket(m, depth_8 + 1);
        L_C = lhs.loadPacket(m, depth_8 + 2);
        L_D = lhs.loadPacket(m, depth_8 + 3);
        L_E = lhs.loadPacket(m, depth_8 + 4);
        L_F = _mm256_setzero_si256();
        L_G = _mm256_setzero_si256();
        L_H = _mm256_setzero_si256();
        break;
      case 6:
        L_A = lhs.loadPacket(m, depth_8);
        L_B = lhs.loadPacket(m, depth_8 + 1);
        L_C = lhs.loadPacket(m, depth_8 + 2);
        L_D = lhs.loadPacket(m, depth_8 + 3);
        L_E = lhs.loadPacket(m, depth_8 + 4);
        L_F = lhs.loadPacket(m, depth_8 + 5);
        L_G = _mm256_setzero_si256();
        L_H = _mm256_setzero_si256();
        break;
      case 7:
        L_A = lhs.loadPacket(m, depth_8);
        L_B = lhs.loadPacket(m, depth_8 + 1);
        L_C = lhs.loadPacket(m, depth_8 + 2);
        L_D = lhs.loadPacket(m, depth_8 + 3);
        L_E = lhs.loadPacket(m, depth_8 + 4);
        L_F = lhs.loadPacket(m, depth_8 + 5);
        L_G = lhs.loadPacket(m, depth_8 + 6);
        L_H = _mm256_setzero_si256();
        break;
      }

      // Interleave 8-bit elements
      __m256i L_AB0_AB16 = _mm256_unpacklo_epi8(L_A, L_B);
      __m256i L_AB8_AB24 = _mm256_unpackhi_epi8(L_A, L_B);

      __m256i L_CD0_CD16 = _mm256_unpacklo_epi8(L_C, L_D);
      __m256i L_CD8_CD24 = _mm256_unpackhi_epi8(L_C, L_D);

      // Interleave 16-bit elements
      __m256i L_AD0_AD16 = _mm256_unpacklo_epi16(L_AB0_AB16, L_CD0_CD16);
      __m256i L_AD4_AD20 = _mm256_unpackhi_epi16(L_AB0_AB16, L_CD0_CD16);

      // Use permute before we store to cross 128-bit lanes
      __m256i L_AD0 = _mm256_permute2x128_si256(L_AD0_AD16, L_AD4_AD20, 0x20);
      _mm256_store_si256(blockA_256++, L_AD0);

      // Complete packing
      __m256i L_AD16 = _mm256_permute2x128_si256(L_AD0_AD16, L_AD4_AD20, 0x31);
      __m256i L_AD8_AD24 = _mm256_unpacklo_epi16(L_AB8_AB24, L_CD8_CD24);
      __m256i L_AD12_AD28 = _mm256_unpackhi_epi16(L_AB8_AB24, L_CD8_CD24);
      __m256i L_AD8 = _mm256_permute2x128_si256(L_AD8_AD24, L_AD12_AD28, 0x20);
      _mm256_store_si256(blockA_256++, L_AD8);
      _mm256_store_si256(blockA_256++, L_AD16);
      __m256i L_AD24 = _mm256_permute2x128_si256(L_AD8_AD24, L_AD12_AD28, 0x31);
      _mm256_store_si256(blockA_256++, L_AD24);
      __m256i L_EF0_EF16 = _mm256_unpacklo_epi8(L_E, L_F);
      __m256i L_EF8_EF24 = _mm256_unpackhi_epi8(L_E, L_F);
      __m256i L_GH0_GH16 = _mm256_unpacklo_epi8(L_G, L_H);
      __m256i L_GH8_GH24 = _mm256_unpackhi_epi8(L_G, L_H);
      __m256i L_EH0_EH16 = _mm256_unpacklo_epi16(L_EF0_EF16, L_GH0_GH16);
      __m256i L_EH4_EH20 = _mm256_unpackhi_epi16(L_EF0_EF16, L_GH0_GH16);
      __m256i L_EH0 = _mm256_permute2x128_si256(L_EH0_EH16, L_EH4_EH20, 0x20);
      _mm256_store_si256(blockA_256++, L_EH0);
      __m256i L_EH16 = _mm256_permute2x128_si256(L_EH0_EH16, L_EH4_EH20, 0x31);
      __m256i L_EH8_EH24 = _mm256_unpacklo_epi16(L_EF8_EF24, L_GH8_GH24);
      __m256i L_EH12_EH28 = _mm256_unpackhi_epi16(L_EF8_EF24, L_GH8_GH24);
      __m256i L_EH8 = _mm256_permute2x128_si256(L_EH8_EH24, L_EH12_EH28, 0x20);
      _mm256_store_si256(blockA_256++, L_EH8);
      _mm256_store_si256(blockA_256++, L_EH16);
      __m256i L_EH24 = _mm256_permute2x128_si256(L_EH8_EH24, L_EH12_EH28, 0x31);
      _mm256_store_si256(blockA_256++, L_EH24);
    }
    blockA_256 += padding;
  }

  // Finish the m dimension, padding with zeros
  if (rows_32 < rows) {
    // Pack depth in sets of 8
    for (Index k = 0; k < depth_8; k += 8) {
      // Load vectors
      __m256i L_A = _mm256_setzero_si256();
      __m256i L_B = _mm256_setzero_si256();
      __m256i L_C = _mm256_setzero_si256();
      __m256i L_D = _mm256_setzero_si256();
      __m256i L_E = _mm256_setzero_si256();
      __m256i L_F = _mm256_setzero_si256();
      __m256i L_G = _mm256_setzero_si256();
      __m256i L_H = _mm256_setzero_si256();
      for (Index m = 0; m < rows - rows_32; m++) {
        QInt8* ptr = (QInt8*) &L_A;
        ptr[m] = lhs(rows_32 + m, k);
        ptr = (QInt8*) &L_B;
        ptr[m] = lhs(rows_32 + m, k + 1);
        ptr = (QInt8*) &L_C;
        ptr[m] = lhs(rows_32 + m, k + 2);
        ptr = (QInt8*) &L_D;
        ptr[m] = lhs(rows_32 + m, k + 3);
        ptr = (QInt8*) &L_E;
        ptr[m] = lhs(rows_32 + m, k + 4);
        ptr = (QInt8*) &L_F;
        ptr[m] = lhs(rows_32 + m, k + 5);
        ptr = (QInt8*) &L_G;
        ptr[m] = lhs(rows_32 + m, k + 6);
        ptr = (QInt8*) &L_H;
        ptr[m] = lhs(rows_32 + m, k + 7);
      }

      // Interleave 8-bit elements
      __m256i L_AB0_AB16 = _mm256_unpacklo_epi8(L_A, L_B);
      __m256i L_AB8_AB24 = _mm256_unpackhi_epi8(L_A, L_B);
      __m256i L_CD0_CD16 = _mm256_unpacklo_epi8(L_C, L_D);
      __m256i L_CD8_CD24 = _mm256_unpackhi_epi8(L_C, L_D);

      // Interleave 16-bit elements
      __m256i L_AD0_AD16 = _mm256_unpacklo_epi16(L_AB0_AB16, L_CD0_CD16);
      __m256i L_AD4_AD20 = _mm256_unpackhi_epi16(L_AB0_AB16, L_CD0_CD16);

      // Use permute before we store to cross 128-bit lanes
      __m256i L_AD0 = _mm256_permute2x128_si256(L_AD0_AD16, L_AD4_AD20, 0x20);
      _mm256_store_si256(blockA_256++, L_AD0);

      // Complete packing for 32 x 8 block
      __m256i L_AD16 = _mm256_permute2x128_si256(L_AD0_AD16, L_AD4_AD20, 0x31);
      __m256i L_AD8_AD24 = _mm256_unpacklo_epi16(L_AB8_AB24, L_CD8_CD24);
      __m256i L_AD12_AD28 = _mm256_unpackhi_epi16(L_AB8_AB24, L_CD8_CD24);
      __m256i L_AD8 = _mm256_permute2x128_si256(L_AD8_AD24, L_AD12_AD28, 0x20);
      _mm256_store_si256(blockA_256++, L_AD8);
      _mm256_store_si256(blockA_256++, L_AD16);
      __m256i L_AD24 = _mm256_permute2x128_si256(L_AD8_AD24, L_AD12_AD28, 0x31);
      _mm256_store_si256(blockA_256++, L_AD24);
      __m256i L_EF0_EF16 = _mm256_unpacklo_epi8(L_E, L_F);
      __m256i L_EF8_EF24 = _mm256_unpackhi_epi8(L_E, L_F);
      __m256i L_GH0_GH16 = _mm256_unpacklo_epi8(L_G, L_H);
      __m256i L_GH8_GH24 = _mm256_unpackhi_epi8(L_G, L_H);
      __m256i L_EH0_EH16 = _mm256_unpacklo_epi16(L_EF0_EF16, L_GH0_GH16);
      __m256i L_EH4_EH20 = _mm256_unpackhi_epi16(L_EF0_EF16, L_GH0_GH16);
      __m256i L_EH0 = _mm256_permute2x128_si256(L_EH0_EH16, L_EH4_EH20, 0x20);
      _mm256_store_si256(blockA_256++, L_EH0);
      __m256i L_EH16 = _mm256_permute2x128_si256(L_EH0_EH16, L_EH4_EH20, 0x31);
      __m256i L_EH8_EH24 = _mm256_unpacklo_epi16(L_EF8_EF24, L_GH8_GH24);
      __m256i L_EH12_EH28 = _mm256_unpackhi_epi16(L_EF8_EF24, L_GH8_GH24);
      __m256i L_EH8 = _mm256_permute2x128_si256(L_EH8_EH24, L_EH12_EH28, 0x20);
      _mm256_store_si256(blockA_256++, L_EH8);
      _mm256_store_si256(blockA_256++, L_EH16);
      __m256i L_EH24 = _mm256_permute2x128_si256(L_EH8_EH24, L_EH12_EH28, 0x31);
      _mm256_store_si256(blockA_256++, L_EH24);
    }

    // Finish the k dimension, padding with zeros
    if (depth_8 < depth) {
      __m256i L_A, L_B, L_C, L_D, L_E, L_F, L_G, L_H;
      QInt8* ptr;
      switch (depth - depth_8) {
      case 1:
        L_A = _mm256_setzero_si256();
        L_B = _mm256_setzero_si256();
        L_C = _mm256_setzero_si256();
        L_D = _mm256_setzero_si256();
        L_E = _mm256_setzero_si256();
        L_F = _mm256_setzero_si256();
        L_G = _mm256_setzero_si256();
        L_H = _mm256_setzero_si256();
        for (Index m = 0; m < rows - rows_32; m++) {
          QInt8* ptr = (QInt8*) &L_A;
          ptr[m] = lhs(rows_32 + m, depth_8);
        }
        break;
      case 2:
        L_A = _mm256_setzero_si256();
        L_B = _mm256_setzero_si256();
        L_C = _mm256_setzero_si256();
        L_D = _mm256_setzero_si256();
        L_E = _mm256_setzero_si256();
        L_F = _mm256_setzero_si256();
        L_G = _mm256_setzero_si256();
        L_H = _mm256_setzero_si256();
        for (Index m = 0; m < rows - rows_32; m++) {
          ptr = (QInt8*) &L_A;
          ptr[m] = lhs(rows_32 + m, depth_8);
          ptr = (QInt8*) &L_B;
          ptr[m] = lhs(rows_32 + m, depth_8 + 1);
        }
        break;
      case 3:
        L_A = _mm256_setzero_si256();
        L_B = _mm256_setzero_si256();
        L_C = _mm256_setzero_si256();
        L_D = _mm256_setzero_si256();
        L_E = _mm256_setzero_si256();
        L_F = _mm256_setzero_si256();
        L_G = _mm256_setzero_si256();
        L_H = _mm256_setzero_si256();
        for (Index m = 0; m < rows - rows_32; m++) {
          ptr = (QInt8*) &L_A;
          ptr[m] = lhs(rows_32 + m, depth_8);
          ptr = (QInt8*) &L_B;
          ptr[m] = lhs(rows_32 + m, depth_8 + 1);
          ptr = (QInt8*) &L_C;
          ptr[m] = lhs(rows_32 + m, depth_8 + 2);
        }
        break;
      case 4:
        L_A = _mm256_setzero_si256();
        L_B = _mm256_setzero_si256();
        L_C = _mm256_setzero_si256();
        L_D = _mm256_setzero_si256();
        L_E = _mm256_setzero_si256();
        L_F = _mm256_setzero_si256();
        L_G = _mm256_setzero_si256();
        L_H = _mm256_setzero_si256();
        for (Index m = 0; m < rows - rows_32; m++) {
          ptr = (QInt8*) &L_A;
          ptr[m] = lhs(rows_32 + m, depth_8);
          ptr = (QInt8*) &L_B;
          ptr[m] = lhs(rows_32 + m, depth_8 + 1);
          ptr = (QInt8*) &L_C;
          ptr[m] = lhs(rows_32 + m, depth_8 + 2);
          ptr = (QInt8*) &L_D;
          ptr[m] = lhs(rows_32 + m, depth_8 + 3);
        }
        break;
      case 5:
        L_A = _mm256_setzero_si256();
        L_B = _mm256_setzero_si256();
        L_C = _mm256_setzero_si256();
        L_D = _mm256_setzero_si256();
        L_E = _mm256_setzero_si256();
        L_F = _mm256_setzero_si256();
        L_G = _mm256_setzero_si256();
        L_H = _mm256_setzero_si256();
        for (Index m = 0; m < rows - rows_32; m++) {
          ptr = (QInt8*) &L_A;
          ptr[m] = lhs(rows_32 + m, depth_8);
          ptr = (QInt8*) &L_B;
          ptr[m] = lhs(rows_32 + m, depth_8 + 1);
          ptr = (QInt8*) &L_C;
          ptr[m] = lhs(rows_32 + m, depth_8 + 2);
          ptr = (QInt8*) &L_D;
          ptr[m] = lhs(rows_32 + m, depth_8 + 3);
          ptr = (QInt8*) &L_E;
          ptr[m] = lhs(rows_32 + m, depth_8 + 4);
        }
        break;
      case 6:
        L_A = _mm256_setzero_si256();
        L_B = _mm256_setzero_si256();
        L_C = _mm256_setzero_si256();
        L_D = _mm256_setzero_si256();
        L_E = _mm256_setzero_si256();
        L_F = _mm256_setzero_si256();
        L_G = _mm256_setzero_si256();
        L_H = _mm256_setzero_si256();
        for (Index m = 0; m < rows - rows_32; m++) {
          ptr = (QInt8*) &L_A;
          ptr[m] = lhs(rows_32 + m, depth_8);
          ptr = (QInt8*) &L_B;
          ptr[m] = lhs(rows_32 + m, depth_8 + 1);
          ptr = (QInt8*) &L_C;
          ptr[m] = lhs(rows_32 + m, depth_8 + 2);
          ptr = (QInt8*) &L_D;
          ptr[m] = lhs(rows_32 + m, depth_8 + 3);
          ptr = (QInt8*) &L_E;
          ptr[m] = lhs(rows_32 + m, depth_8 + 4);
          ptr = (QInt8*) &L_F;
          ptr[m] = lhs(rows_32 + m, depth_8 + 5);
        }
        break;
      case 7:
        L_A = _mm256_setzero_si256();
        L_B = _mm256_setzero_si256();
        L_C = _mm256_setzero_si256();
        L_D = _mm256_setzero_si256();
        L_E = _mm256_setzero_si256();
        L_F = _mm256_setzero_si256();
        L_G = _mm256_setzero_si256();
        L_H = _mm256_setzero_si256();
        for (Index m = 0; m < rows - rows_32; m++) {
          ptr = (QInt8*) &L_A;
          ptr[m] = lhs(rows_32 + m, depth_8);
          ptr = (QInt8*) &L_B;
          ptr[m] = lhs(rows_32 + m, depth_8 + 1);
          ptr = (QInt8*) &L_C;
          ptr[m] = lhs(rows_32 + m, depth_8 + 2);
          ptr = (QInt8*) &L_D;
          ptr[m] = lhs(rows_32 + m, depth_8 + 3);
          ptr = (QInt8*) &L_E;
          ptr[m] = lhs(rows_32 + m, depth_8 + 4);
          ptr = (QInt8*) &L_F;
          ptr[m] = lhs(rows_32 + m, depth_8 + 5);
          ptr = (QInt8*) &L_G;
          ptr[m] = lhs(rows_32 + m, depth_8 + 6);
        }
        break;
      }

      // Interleave 8-bit elements
      __m256i L_AB0_AB16 = _mm256_unpacklo_epi8(L_A, L_B);
      __m256i L_AB8_AB24 = _mm256_unpackhi_epi8(L_A, L_B);
      __m256i L_CD0_CD16 = _mm256_unpacklo_epi8(L_C, L_D);
      __m256i L_CD8_CD24 = _mm256_unpackhi_epi8(L_C, L_D);

      // Interleave 16-bit elements
      __m256i L_AD0_AD16 = _mm256_unpacklo_epi16(L_AB0_AB16, L_CD0_CD16);
      __m256i L_AD4_AD20 = _mm256_unpackhi_epi16(L_AB0_AB16, L_CD0_CD16);

      // Use permute before we store to cross 128-bit lanes
      __m256i L_AD0 = _mm256_permute2x128_si256(L_AD0_AD16, L_AD4_AD20, 0x20);
      _mm256_store_si256(blockA_256++, L_AD0);

      // Complete packing
      __m256i L_AD16 = _mm256_permute2x128_si256(L_AD0_AD16, L_AD4_AD20, 0x31);
      __m256i L_AD8_AD24 = _mm256_unpacklo_epi16(L_AB8_AB24, L_CD8_CD24);
      __m256i L_AD12_AD28 = _mm256_unpackhi_epi16(L_AB8_AB24, L_CD8_CD24);
      __m256i L_AD8 = _mm256_permute2x128_si256(L_AD8_AD24, L_AD12_AD28, 0x20);
      _mm256_store_si256(blockA_256++, L_AD8);
      _mm256_store_si256(blockA_256++, L_AD16);
      __m256i L_AD24 = _mm256_permute2x128_si256(L_AD8_AD24, L_AD12_AD28, 0x31);
      _mm256_store_si256(blockA_256++, L_AD24);
      __m256i L_EF0_EF16 = _mm256_unpacklo_epi8(L_E, L_F);
      __m256i L_EF8_EF24 = _mm256_unpackhi_epi8(L_E, L_F);
      __m256i L_GH0_GH16 = _mm256_unpacklo_epi8(L_G, L_H);
      __m256i L_GH8_GH24 = _mm256_unpackhi_epi8(L_G, L_H);
      __m256i L_EH0_EH16 = _mm256_unpacklo_epi16(L_EF0_EF16, L_GH0_GH16);
      __m256i L_EH4_EH20 = _mm256_unpackhi_epi16(L_EF0_EF16, L_GH0_GH16);
      __m256i L_EH0 = _mm256_permute2x128_si256(L_EH0_EH16, L_EH4_EH20, 0x20);
      _mm256_store_si256(blockA_256++, L_EH0);
      __m256i L_EH16 = _mm256_permute2x128_si256(L_EH0_EH16, L_EH4_EH20, 0x31);
      __m256i L_EH8_EH24 = _mm256_unpacklo_epi16(L_EF8_EF24, L_GH8_GH24);
      __m256i L_EH12_EH28 = _mm256_unpackhi_epi16(L_EF8_EF24, L_GH8_GH24);
      __m256i L_EH8 = _mm256_permute2x128_si256(L_EH8_EH24, L_EH12_EH28, 0x20);
      _mm256_store_si256(blockA_256++, L_EH8);
      _mm256_store_si256(blockA_256++, L_EH16);
      __m256i L_EH24 = _mm256_permute2x128_si256(L_EH8_EH24, L_EH12_EH28, 0x31);
      _mm256_store_si256(blockA_256++, L_EH24);
    }
  }
}

template <typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
EIGEN_DONT_INLINE void gemm_pack_rhs_any<QUInt8, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode>::
operator()(QUInt8* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset) {
  eigen_assert(stride == 0);
  eigen_assert(offset == 0);

  // Get vector pointer
  __m256i* blockB_256 = reinterpret_cast<__m256i*>(blockB);

  // Get even multiples of the dimensions
  Index cols_32 = (cols / 32) * 32;
  Index depth_32 = (depth / 32) * 32;

  // Perform a step of the packing for 4 columns
  __m256i R_AB_L, R_AB_H, R_CD_L, R_CD_H, R_AD_0, R_AD_8, R_AD_16, R_AD_24;
#define PACK_STEP                                            \
  R_AB_L = _mm256_unpacklo_epi64(R_A, R_B);                  \
  R_CD_L = _mm256_unpacklo_epi64(R_C, R_D);                  \
  R_AB_H = _mm256_unpackhi_epi64(R_A, R_B);                  \
  R_CD_H = _mm256_unpackhi_epi64(R_C, R_D);                  \
  R_AD_0 = _mm256_permute2x128_si256(R_AB_L, R_CD_L, 0x20);  \
  R_AD_16 = _mm256_permute2x128_si256(R_AB_L, R_CD_L, 0x31); \
  R_AD_8 = _mm256_permute2x128_si256(R_AB_H, R_CD_H, 0x20);  \
  R_AD_24 = _mm256_permute2x128_si256(R_AB_H, R_CD_H, 0x31); \
  _mm256_store_si256(blockB_256, R_AD_0);                    \
  _mm256_store_si256(blockB_256 + 8, R_AD_8);                \
  _mm256_store_si256(blockB_256 + 16, R_AD_16);              \
  _mm256_store_si256(blockB_256 + 24, R_AD_24);              \
  blockB_256++;

  // Pack cols in sets of 32
  for (Index n = 0; n < cols_32; n += 32) {
    // Pack depth in sets of 32
    for (Index k = 0; k < depth_32; k += 32) {
      __m256i R_A = rhs.loadPacket(k, n);
      __m256i R_B = rhs.loadPacket(k, n + 1);
      __m256i R_C = rhs.loadPacket(k, n + 2);
      __m256i R_D = rhs.loadPacket(k, n + 3);
      PACK_STEP;

      R_A = rhs.loadPacket(k, n + 4);
      R_B = rhs.loadPacket(k, n + 5);
      R_C = rhs.loadPacket(k, n + 6);
      R_D = rhs.loadPacket(k, n + 7);
      PACK_STEP;

      R_A = rhs.loadPacket(k, n + 8);
      R_B = rhs.loadPacket(k, n + 9);
      R_C = rhs.loadPacket(k, n + 10);
      R_D = rhs.loadPacket(k, n + 11);
      PACK_STEP;

      R_A = rhs.loadPacket(k, n + 12);
      R_B = rhs.loadPacket(k, n + 13);
      R_C = rhs.loadPacket(k, n + 14);
      R_D = rhs.loadPacket(k, n + 15);
      PACK_STEP;

      R_A = rhs.loadPacket(k, n + 16);
      R_B = rhs.loadPacket(k, n + 17);
      R_C = rhs.loadPacket(k, n + 18);
      R_D = rhs.loadPacket(k, n + 19);
      PACK_STEP;

      R_A = rhs.loadPacket(k, n + 20);
      R_B = rhs.loadPacket(k, n + 21);
      R_C = rhs.loadPacket(k, n + 22);
      R_D = rhs.loadPacket(k, n + 23);
      PACK_STEP;

      R_A = rhs.loadPacket(k, n + 24);
      R_B = rhs.loadPacket(k, n + 25);
      R_C = rhs.loadPacket(k, n + 26);
      R_D = rhs.loadPacket(k, n + 27);
      PACK_STEP;

      R_A = rhs.loadPacket(k, n + 28);
      R_B = rhs.loadPacket(k, n + 29);
      R_C = rhs.loadPacket(k, n + 30);
      R_D = rhs.loadPacket(k, n + 31);
      PACK_STEP;

      blockB_256 += 24;
    }

    if (depth_32 < depth) {
      QUInt8* ptr;
      __m256i R_A = _mm256_setzero_si256();
      __m256i R_B = _mm256_setzero_si256();
      __m256i R_C = _mm256_setzero_si256();
      __m256i R_D = _mm256_setzero_si256();
      for (Index k = depth_32; k < depth; k++) {
        ptr = (QUInt8*) &R_A;
        ptr[k - depth_32] = rhs(k, n);
        ptr = (QUInt8*) &R_B;
        ptr[k - depth_32] = rhs(k, n + 1);
        ptr = (QUInt8*) &R_C;
        ptr[k - depth_32] = rhs(k, n + 2);
        ptr = (QUInt8*) &R_D;
        ptr[k - depth_32] = rhs(k, n + 3);
      }
      PACK_STEP;

      R_A = _mm256_setzero_si256();
      R_B = _mm256_setzero_si256();
      R_C = _mm256_setzero_si256();
      R_D = _mm256_setzero_si256();
      for (Index k = depth_32; k < depth; k++) {
        ptr = (QUInt8*) &R_A;
        ptr[k - depth_32] = rhs(k, n + 4);
        ptr = (QUInt8*) &R_B;
        ptr[k - depth_32] = rhs(k, n + 5);
        ptr = (QUInt8*) &R_C;
        ptr[k - depth_32] = rhs(k, n + 6);
        ptr = (QUInt8*) &R_D;
        ptr[k - depth_32] = rhs(k, n + 7);
      }
      PACK_STEP;

      R_A = _mm256_setzero_si256();
      R_B = _mm256_setzero_si256();
      R_C = _mm256_setzero_si256();
      R_D = _mm256_setzero_si256();
      for (Index k = depth_32; k < depth; k++) {
        ptr = (QUInt8*) &R_A;
        ptr[k - depth_32] = rhs(k, n + 8);
        ptr = (QUInt8*) &R_B;
        ptr[k - depth_32] = rhs(k, n + 9);
        ptr = (QUInt8*) &R_C;
        ptr[k - depth_32] = rhs(k, n + 10);
        ptr = (QUInt8*) &R_D;
        ptr[k - depth_32] = rhs(k, n + 11);
      }
      PACK_STEP;

      R_A = _mm256_setzero_si256();
      R_B = _mm256_setzero_si256();
      R_C = _mm256_setzero_si256();
      R_D = _mm256_setzero_si256();
      for (Index k = depth_32; k < depth; k++) {
        ptr = (QUInt8*) &R_A;
        ptr[k - depth_32] = rhs(k, n + 12);
        ptr = (QUInt8*) &R_B;
        ptr[k - depth_32] = rhs(k, n + 13);
        ptr = (QUInt8*) &R_C;
        ptr[k - depth_32] = rhs(k, n + 14);
        ptr = (QUInt8*) &R_D;
        ptr[k - depth_32] = rhs(k, n + 15);
      }
      PACK_STEP;

      R_A = _mm256_setzero_si256();
      R_B = _mm256_setzero_si256();
      R_C = _mm256_setzero_si256();
      R_D = _mm256_setzero_si256();
      for (Index k = depth_32; k < depth; k++) {
        ptr = (QUInt8*) &R_A;
        ptr[k - depth_32] = rhs(k, n + 16);
        ptr = (QUInt8*) &R_B;
        ptr[k - depth_32] = rhs(k, n + 17);
        ptr = (QUInt8*) &R_C;
        ptr[k - depth_32] = rhs(k, n + 18);
        ptr = (QUInt8*) &R_D;
        ptr[k - depth_32] = rhs(k, n + 19);
      }
      PACK_STEP;

      R_A = _mm256_setzero_si256();
      R_B = _mm256_setzero_si256();
      R_C = _mm256_setzero_si256();
      R_D = _mm256_setzero_si256();
      for (Index k = depth_32; k < depth; k++) {
        ptr = (QUInt8*) &R_A;
        ptr[k - depth_32] = rhs(k, n + 20);
        ptr = (QUInt8*) &R_B;
        ptr[k - depth_32] = rhs(k, n + 21);
        ptr = (QUInt8*) &R_C;
        ptr[k - depth_32] = rhs(k, n + 22);
        ptr = (QUInt8*) &R_D;
        ptr[k - depth_32] = rhs(k, n + 23);
      }
      PACK_STEP;

      R_A = _mm256_setzero_si256();
      R_B = _mm256_setzero_si256();
      R_C = _mm256_setzero_si256();
      R_D = _mm256_setzero_si256();
      for (Index k = depth_32; k < depth; k++) {
        ptr = (QUInt8*) &R_A;
        ptr[k - depth_32] = rhs(k, n + 24);
        ptr = (QUInt8*) &R_B;
        ptr[k - depth_32] = rhs(k, n + 25);
        ptr = (QUInt8*) &R_C;
        ptr[k - depth_32] = rhs(k, n + 26);
        ptr = (QUInt8*) &R_D;
        ptr[k - depth_32] = rhs(k, n + 27);
      }
      PACK_STEP;

      R_A = _mm256_setzero_si256();
      R_B = _mm256_setzero_si256();
      R_C = _mm256_setzero_si256();
      R_D = _mm256_setzero_si256();
      for (Index k = depth_32; k < depth; k++) {
        ptr = (QUInt8*) &R_A;
        ptr[k - depth_32] = rhs(k, n + 28);
        ptr = (QUInt8*) &R_B;
        ptr[k - depth_32] = rhs(k, n + 29);
        ptr = (QUInt8*) &R_C;
        ptr[k - depth_32] = rhs(k, n + 30);
        ptr = (QUInt8*) &R_D;
        ptr[k - depth_32] = rhs(k, n + 31);
      }
      PACK_STEP;
      blockB_256 += 24;
    }
  }

  // Finish packing cols
  if (cols_32 < cols) {
    // Pack depth in sets of 32
    for (Index k = 0; k < depth_32; k += 32) {
      __m256i R_A, R_B, R_C, R_D;
      Index n;
      for (n = cols_32; n < cols; n += 4) {
        switch (cols - n) {
        case 1:
          R_A = rhs.loadPacket(k, n);
          R_B = _mm256_setzero_si256();
          R_C = _mm256_setzero_si256();
          R_D = _mm256_setzero_si256();
          PACK_STEP;
          break;
        case 2:
          R_A = rhs.loadPacket(k, n);
          R_B = rhs.loadPacket(k, n + 1);
          R_C = _mm256_setzero_si256();
          R_D = _mm256_setzero_si256();
          PACK_STEP;
          break;
        case 3:
          R_A = rhs.loadPacket(k, n);
          R_B = rhs.loadPacket(k, n + 1);
          R_C = rhs.loadPacket(k, n + 2);
          R_D = _mm256_setzero_si256();
          PACK_STEP;
          break;
        default:
          R_A = rhs.loadPacket(k, n);
          R_B = rhs.loadPacket(k, n + 1);
          R_C = rhs.loadPacket(k, n + 2);
          R_D = rhs.loadPacket(k, n + 3);
          PACK_STEP;
          break;
        }
      }

      // Increment the block pointer.
      // We must pad if cols is not a multiple of 32.
      blockB_256 += 32 - (n - cols_32) / 4;
    }

    if (depth_32 < depth) {
      for (Index n = cols_32; n < cols; n += 4) {
        QUInt8* ptr;
        __m256i R_A = _mm256_setzero_si256();
        __m256i R_B = _mm256_setzero_si256();
        __m256i R_C = _mm256_setzero_si256();
        __m256i R_D = _mm256_setzero_si256();
        switch (cols - n) {
        case 1:
          for (Index k = depth_32; k < depth; k++) {
            ptr = (QUInt8*) &R_A;
            ptr[k - depth_32] = rhs(k, n);
          }
          PACK_STEP;
          break;
        case 2:
          for (Index k = depth_32; k < depth; k++) {
            ptr = (QUInt8*) &R_A;
            ptr[k - depth_32] = rhs(k, n);
            ptr = (QUInt8*) &R_B;
            ptr[k - depth_32] = rhs(k, n + 1);
          }
          PACK_STEP;
          break;
        case 3:
          for (Index k = depth_32; k < depth; k++) {
            ptr = (QUInt8*) &R_A;
            ptr[k - depth_32] = rhs(k, n);
            ptr = (QUInt8*) &R_B;
            ptr[k - depth_32] = rhs(k, n + 1);
            ptr = (QUInt8*) &R_C;
            ptr[k - depth_32] = rhs(k, n + 2);
          }
          PACK_STEP;
          break;
        default:
          for (Index k = depth_32; k < depth; k++) {
            ptr = (QUInt8*) &R_A;
            ptr[k - depth_32] = rhs(k, n);
            ptr = (QUInt8*) &R_B;
            ptr[k - depth_32] = rhs(k, n + 1);
            ptr = (QUInt8*) &R_C;
            ptr[k - depth_32] = rhs(k, n + 2);
            ptr = (QUInt8*) &R_D;
            ptr[k - depth_32] = rhs(k, n + 3);
          }
          PACK_STEP;
          break;
        }
      }
    }
  }
#undef PACK_STEP
}

template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
EIGEN_DONT_INLINE
void gebp_kernel_any<QInt8, QUInt8, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
::operator()(const DataMapper& res, const QInt8* blockA, const QUInt8* blockB,
             Index rows, Index depth, Index cols, QInt32 alpha,
             Index strideA, Index strideB, Index offsetA, Index offsetB)
{
  EIGEN_STATIC_ASSERT(!ConjugateLhs, YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT(!ConjugateRhs, YOU_MADE_A_PROGRAMMING_MISTAKE);
  eigen_assert(alpha.value == 1);
  eigen_assert(strideA == -1);
  eigen_assert(strideB == -1);
  eigen_assert(offsetA == 0);
  eigen_assert(offsetB == 0);
  eigen_assert(rows > 0);
  eigen_assert(cols > 0);
  eigen_assert(depth > 0);
  eigen_assert(blockA);
  eigen_assert(blockB);

  Index rows_32 = ((rows + 31) / 32) * 32;
  Index cols_32 = ((cols + 31) / 32) * 32;
  Index depth_32 = ((depth + 31) / 32) * 32;

  // Create result block
  ei_declare_aligned_stack_constructed_variable(QInt32, blockO, 32 * 32, 0);
  memset(blockO, 0, 32 * 32 * sizeof(QInt32));

  // Get vectorized pointers
  __m256i* blockO_256 = reinterpret_cast<__m256i*>(blockO);
  const __m256i* blockA_256 = reinterpret_cast<const __m256i*>(blockA);
  const __m256i* blockB_256 = reinterpret_cast<const __m256i*>(blockB);

  // Loop over blocks of 32 columns
  for (Index n = 0; n < cols_32; n += 32) {
    // Reset index into blockA
    Index indexL = 0;
    // Loop over blocks of 32 rows
    for (Index m = 0; m < rows_32; m += 32) {
      // Reset index into blockB
      Index indexR = n / 32 * depth_32;
      // Loop over blocks of 8 on depth
      for (Index k = 0; k < depth_32; k += 8) {
        // Load inputs
        __m256i L_AD0 = blockA_256[indexL++];
        __m256i L_AD8 = blockA_256[indexL++];
        __m256i L_AD16 = blockA_256[indexL++];
        __m256i L_AD24 = blockA_256[indexL++];
        __m256i L_EH0 = blockA_256[indexL++];
        __m256i L_EH8 = blockA_256[indexL++];
        __m256i L_EH16 = blockA_256[indexL++];
        __m256i L_EH24 = blockA_256[indexL++];
        __m256i R_AH0 = blockB_256[indexR++];
        __m256i R_AH4 = blockB_256[indexR++];
        __m256i R_AH8 = blockB_256[indexR++];
        __m256i R_AH12 = blockB_256[indexR++];
        __m256i R_AH16 = blockB_256[indexR++];
        __m256i R_AH20 = blockB_256[indexR++];
        __m256i R_AH24 = blockB_256[indexR++];
        __m256i R_AH28 = blockB_256[indexR++];

        // This constant is used with madd to convert 16 bit to 32 bit
        const __m256i ONE = _mm256_set1_epi32(0x00010001);

        // Declare variables used in COMPUTE_STEP
        __m256i P_16_A, P_16_B, P_32_A, P_32_B, P_32;

#define COMPUTE_STEP(R_INPUT_A, R_INPUT_B, OFFSET)                             \
  P_16_A = _mm256_maddubs_epi16(R_INPUT_A, L_AD0);                             \
  P_32_A = _mm256_madd_epi16(P_16_A, ONE);                                     \
  P_16_B = _mm256_maddubs_epi16(R_INPUT_B, L_EH0);                             \
  P_32_B = _mm256_madd_epi16(P_16_B, ONE);                                     \
  P_32 = _mm256_add_epi32(P_32_A, P_32_B);                                     \
  _mm256_store_si256(                                                          \
      blockO_256 + 4 * OFFSET,                                                 \
      _mm256_add_epi32(_mm256_load_si256(blockO_256 + 4 * OFFSET), P_32));     \
                                                                               \
  P_16_A = _mm256_maddubs_epi16(R_INPUT_A, L_AD8);                             \
  P_32_A = _mm256_madd_epi16(P_16_A, ONE);                                     \
  P_16_B = _mm256_maddubs_epi16(R_INPUT_B, L_EH8);                             \
  P_32_B = _mm256_madd_epi16(P_16_B, ONE);                                     \
  P_32 = _mm256_add_epi32(P_32_A, P_32_B);                                     \
  _mm256_store_si256(                                                          \
      blockO_256 + 4 * OFFSET + 1,                                             \
      _mm256_add_epi32(_mm256_load_si256(blockO_256 + 4 * OFFSET + 1), P_32)); \
                                                                               \
  P_16_A = _mm256_maddubs_epi16(R_INPUT_A, L_AD16);                            \
  P_32_A = _mm256_madd_epi16(P_16_A, ONE);                                     \
  P_16_B = _mm256_maddubs_epi16(R_INPUT_B, L_EH16);                            \
  P_32_B = _mm256_madd_epi16(P_16_B, ONE);                                     \
  P_32 = _mm256_add_epi32(P_32_A, P_32_B);                                     \
  _mm256_store_si256(                                                          \
      blockO_256 + 4 * OFFSET + 2,                                             \
      _mm256_add_epi32(_mm256_load_si256(blockO_256 + 4 * OFFSET + 2), P_32)); \
                                                                               \
  P_16_A = _mm256_maddubs_epi16(R_INPUT_A, L_AD24);                            \
  P_32_A = _mm256_madd_epi16(P_16_A, ONE);                                     \
  P_16_B = _mm256_maddubs_epi16(R_INPUT_B, L_EH24);                            \
  P_32_B = _mm256_madd_epi16(P_16_B, ONE);                                     \
  P_32 = _mm256_add_epi32(P_32_A, P_32_B);                                     \
  _mm256_store_si256(                                                          \
      blockO_256 + 4 * OFFSET + 3,                                             \
      _mm256_add_epi32(_mm256_load_si256(blockO_256 + 4 * OFFSET + 3), P_32));

        // Permute and shuffle to copy a single value across the entire vector
        // Then compute the multiplication
        __m256i R_AH0_ = _mm256_permute2x128_si256(R_AH0, R_AH0, 0x00);
        __m256i R_AD0 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        __m256i R_EH0 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD0, R_EH0, 0);
        __m256i R_AD1 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        __m256i R_EH1 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD1, R_EH1, 1);
        R_AH0_ = _mm256_permute2x128_si256(R_AH0, R_AH0, 0x11);
        __m256i R_AD2 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        __m256i R_EH2 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD2, R_EH2, 2);
        __m256i R_AD3 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        __m256i R_EH3 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD3, R_EH3, 3);

        R_AH0_ = _mm256_permute2x128_si256(R_AH4, R_AH4, 0x00);
        R_AD0 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH0 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD0, R_EH0, 4);
        R_AD1 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH1 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD1, R_EH1, 5);
        R_AH0_ = _mm256_permute2x128_si256(R_AH4, R_AH4, 0x11);
        R_AD2 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH2 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD2, R_EH2, 6);
        R_AD3 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH3 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD3, R_EH3, 7);

        R_AH0_ = _mm256_permute2x128_si256(R_AH8, R_AH8, 0x00);
        R_AD0 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH0 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD0, R_EH0, 8);
        R_AD1 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH1 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD1, R_EH1, 9);
        R_AH0_ = _mm256_permute2x128_si256(R_AH8, R_AH8, 0x11);
        R_AD2 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH2 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD2, R_EH2, 10);
        R_AD3 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH3 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD3, R_EH3, 11);

        R_AH0_ = _mm256_permute2x128_si256(R_AH12, R_AH12, 0x00);
        R_AD0 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH0 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD0, R_EH0, 12);
        R_AD1 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH1 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD1, R_EH1, 13);
        R_AH0_ = _mm256_permute2x128_si256(R_AH12, R_AH12, 0x11);
        R_AD2 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH2 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD2, R_EH2, 14);
        R_AD3 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH3 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD3, R_EH3, 15);

        R_AH0_ = _mm256_permute2x128_si256(R_AH16, R_AH16, 0x00);
        R_AD0 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH0 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD0, R_EH0, 16);
        R_AD1 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH1 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD1, R_EH1, 17);
        R_AH0_ = _mm256_permute2x128_si256(R_AH16, R_AH16, 0x11);
        R_AD2 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH2 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD2, R_EH2, 18);
        R_AD3 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH3 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD3, R_EH3, 19);

        R_AH0_ = _mm256_permute2x128_si256(R_AH20, R_AH20, 0x00);
        R_AD0 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH0 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD0, R_EH0, 20);
        R_AD1 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH1 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD1, R_EH1, 21);
        R_AH0_ = _mm256_permute2x128_si256(R_AH20, R_AH20, 0x11);
        R_AD2 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH2 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD2, R_EH2, 22);
        R_AD3 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH3 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD3, R_EH3, 23);

        R_AH0_ = _mm256_permute2x128_si256(R_AH24, R_AH24, 0x00);
        R_AD0 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH0 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD0, R_EH0, 24);
        R_AD1 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH1 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD1, R_EH1, 25);
        R_AH0_ = _mm256_permute2x128_si256(R_AH24, R_AH24, 0x11);
        R_AD2 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH2 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD2, R_EH2, 26);
        R_AD3 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH3 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD3, R_EH3, 27);

        R_AH0_ = _mm256_permute2x128_si256(R_AH28, R_AH28, 0x00);
        R_AD0 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH0 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD0, R_EH0, 28);
        R_AD1 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH1 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD1, R_EH1, 29);
        R_AH0_ = _mm256_permute2x128_si256(R_AH28, R_AH28, 0x11);
        R_AD2 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH2 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD2, R_EH2, 30);
        R_AD3 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH3 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD3, R_EH3, 31);

#undef COMPUTE_STEP
      }

      // Transfer the results to the result matrix.
      if (m + 32 <= rows && n + 32 <= cols) {
        Index i = 0;
        for (Index j = n; j < n + 32; j++) {
          LinearMapper r0 = res.getLinearMapper(m, j);
          LinearMapper r1 = res.getLinearMapper(m + 8, j);
          LinearMapper r2 = res.getLinearMapper(m + 16, j);
          LinearMapper r3 = res.getLinearMapper(m + 24, j);
          r0.storePacket(
              0, _mm256_add_epi32(blockO_256[i++], r0.loadPacket(0)));
          r1.storePacket(
              0, _mm256_add_epi32(blockO_256[i++], r1.loadPacket(0)));
          r2.storePacket(
              0, _mm256_add_epi32(blockO_256[i++], r2.loadPacket(0)));
          r3.storePacket(
              0, _mm256_add_epi32(blockO_256[i++], r3.loadPacket(0)));
        }
      }
      else {
        for (Index j = n; j < cols; j++) {
          for (Index i = m; i < rows; i++) {
            res(i, j) = blockO[(j - n) * 32 + (i - m)];
          }
        }
      }

      // Zero the result block so it can be reused
      memset(blockO, 0, 32 * 32 * sizeof(QInt32));
    }
  }
}

// Below are the fully optimized versions that are correct only for sizes that
// are multiple of 32.  It is about a 10% performance benefit to keep these
// implementations separate.

// Arrange a block of the left input matrix in contiguous memory.
//
// Given column major input (A0 beside A1 in memory):
// A0 B0 C0 D0 E0 F0 G0 H0 ...
// A1 B1 C1 D1 E1 F1 G1 H1 ...
// A2 B2 C2 D2 E2 F2 G2 H2 ...
// A3 B3 C3 D3 E3 F3 G3 H3 ...
// A4 B4 C4 D4 E4 F4 G4 H4 ...
// A5 B5 C5 D5 E5 F5 G5 H5 ...
// A6 B6 C6 D6 E6 F6 G6 H6 ...
// A7 B7 C7 D7 E7 F7 G7 H7 ...
// A8 ...
// ...
//
// Packing yields output (A0 beside B0 in memory):
// A0 B0 C0 D0
// A1 B1 C1 D1
// A2 B2 C2 D2
// A3 B3 C3 D3
// A4 B4 C4 D4
// A5 B5 C5 D5
// A6 B6 C6 D6
// A7 B7 C7 D7
// ...
// A31 B31 C31 D31
// E0 F0 G0 H0
// E1 F1 G1 H1
// E2 F2 G2 H2
// E3 F3 G3 H3
// E4 F4 G4 H4
// E5 F5 G5 H5
// E6 F6 G6 H6
// E7 F7 G7 H7
// ...
//
// Four elements of the same row are arranged contiguously because maddubs and
// madd both perform an adjacent addition in the kernel.
template <typename Index, typename DataMapper, int Pack1, int Pack2,
          bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<QInt8, Index, DataMapper, Pack1, Pack2, ColMajor,
                     Conjugate, PanelMode> {
  EIGEN_DONT_INLINE void operator()(QInt8* blockA, const DataMapper& lhs,
                                    Index depth, Index rows, Index stride = 0,
                                    Index offset = 0);
};

template <typename Index, typename DataMapper, int Pack1, int Pack2,
          bool Conjugate, bool PanelMode>
EIGEN_DONT_INLINE void gemm_pack_lhs<QInt8, Index, DataMapper, Pack1, Pack2,
                                     ColMajor, Conjugate, PanelMode>::
operator()(QInt8* blockA, const DataMapper& lhs, Index depth, Index rows,
           Index stride, Index offset) {
  eigen_assert(stride == 0);
  eigen_assert(offset == 0);

  // Use alternate function for weird sizes
  if (rows % 32 != 0 || depth % 32 != 0) {
    gemm_pack_lhs_any<QInt8, Index, DataMapper, Pack1, Pack2, ColMajor, Conjugate, PanelMode> lhs_pack;
    return lhs_pack(blockA, lhs, depth, rows, stride, offset);
  }

  // Get vector pointer
  __m256i* blockA_256 = reinterpret_cast<__m256i*>(blockA);

  // Pack rows in sets of 32
  for (Index m = 0; m < rows; m += 32) {
    // Pack depth in sets of 8
    for (Index k = 0; k < depth; k += 8) {
      // Load vectors
      __m256i L_A = lhs.loadPacket(m, k);
      __m256i L_B = lhs.loadPacket(m, k + 1);

      // Interleave 8-bit elements
      __m256i L_AB0_AB16 = _mm256_unpacklo_epi8(L_A, L_B);
      __m256i L_AB8_AB24 = _mm256_unpackhi_epi8(L_A, L_B);

      __m256i L_C = lhs.loadPacket(m, k + 2);
      __m256i L_D = lhs.loadPacket(m, k + 3);
      __m256i L_CD0_CD16 = _mm256_unpacklo_epi8(L_C, L_D);
      __m256i L_CD8_CD24 = _mm256_unpackhi_epi8(L_C, L_D);

      // Interleave 16-bit elements
      __m256i L_AD0_AD16 = _mm256_unpacklo_epi16(L_AB0_AB16, L_CD0_CD16);
      __m256i L_AD4_AD20 = _mm256_unpackhi_epi16(L_AB0_AB16, L_CD0_CD16);

      // Use permute before we store to cross 128-bit lanes
      __m256i L_AD0 = _mm256_permute2x128_si256(L_AD0_AD16, L_AD4_AD20, 0x20);
      _mm256_store_si256(blockA_256++, L_AD0);

      // Complete packing for 32 x 8 block
      __m256i L_AD16 = _mm256_permute2x128_si256(L_AD0_AD16, L_AD4_AD20, 0x31);
      __m256i L_AD8_AD24 = _mm256_unpacklo_epi16(L_AB8_AB24, L_CD8_CD24);
      __m256i L_AD12_AD28 = _mm256_unpackhi_epi16(L_AB8_AB24, L_CD8_CD24);
      __m256i L_AD8 = _mm256_permute2x128_si256(L_AD8_AD24, L_AD12_AD28, 0x20);
      _mm256_store_si256(blockA_256++, L_AD8);
      _mm256_store_si256(blockA_256++, L_AD16);
      __m256i L_AD24 = _mm256_permute2x128_si256(L_AD8_AD24, L_AD12_AD28, 0x31);
      _mm256_store_si256(blockA_256++, L_AD24);
      __m256i L_E = lhs.loadPacket(m, k + 4);
      __m256i L_F = lhs.loadPacket(m, k + 5);
      __m256i L_EF0_EF16 = _mm256_unpacklo_epi8(L_E, L_F);
      __m256i L_EF8_EF24 = _mm256_unpackhi_epi8(L_E, L_F);
      __m256i L_G = lhs.loadPacket(m, k + 6);
      __m256i L_H = lhs.loadPacket(m, k + 7);
      __m256i L_GH0_GH16 = _mm256_unpacklo_epi8(L_G, L_H);
      __m256i L_GH8_GH24 = _mm256_unpackhi_epi8(L_G, L_H);
      __m256i L_EH0_EH16 = _mm256_unpacklo_epi16(L_EF0_EF16, L_GH0_GH16);
      __m256i L_EH4_EH20 = _mm256_unpackhi_epi16(L_EF0_EF16, L_GH0_GH16);
      __m256i L_EH0 = _mm256_permute2x128_si256(L_EH0_EH16, L_EH4_EH20, 0x20);
      _mm256_store_si256(blockA_256++, L_EH0);
      __m256i L_EH16 = _mm256_permute2x128_si256(L_EH0_EH16, L_EH4_EH20, 0x31);
      __m256i L_EH8_EH24 = _mm256_unpacklo_epi16(L_EF8_EF24, L_GH8_GH24);
      __m256i L_EH12_EH28 = _mm256_unpackhi_epi16(L_EF8_EF24, L_GH8_GH24);
      __m256i L_EH8 = _mm256_permute2x128_si256(L_EH8_EH24, L_EH12_EH28, 0x20);
      _mm256_store_si256(blockA_256++, L_EH8);
      _mm256_store_si256(blockA_256++, L_EH16);
      __m256i L_EH24 = _mm256_permute2x128_si256(L_EH8_EH24, L_EH12_EH28, 0x31);
      _mm256_store_si256(blockA_256++, L_EH24);
    }
  }
}

// Arrange a block of the right input matrix in contiguous memory.
//
// Given column major input (A0 beside A1 in memory):
// A0 B0 C0 D0 E0 F0 G0 H0 ...
// A1 B1 C1 D1 E1 F1 G1 H1 ...
// A2 B2 C2 D2 E2 F2 G2 H2 ...
// A3 B3 C3 D3 E3 F3 G3 H3 ...
// A4 B4 C4 D4 E4 F4 G4 H4 ...
// A5 B5 C5 D5 E5 F5 G5 H5 ...
// A6 B6 C6 D6 E6 F6 G6 H6 ...
// A7 B7 C7 D7 E7 F7 G7 H7 ...
// A8 ...
// ...
//
// Packing yields row major output (A0 beside A1 in memory):
// A0 A1 A2 A3 A4 A5 A6 A7
// B0 B1 B2 B3 B4 B5 B6 B7
// ...
//
// At least four elements of the same col are arranged contiguously because
// maddubs and madd both perform an adjacent addition in the kernel.  We can
// save work by leaving 8 adjacent elements because kr = 8.
template <typename Index, typename DataMapper, int nr, bool Conjugate,
          bool PanelMode>
struct gemm_pack_rhs<QUInt8, Index, DataMapper, nr, ColMajor, Conjugate,
                     PanelMode> {
  EIGEN_DONT_INLINE void operator()(QUInt8* blockB, const DataMapper& rhs,
                                    Index depth, Index cols, Index stride = 0,
                                    Index offset = 0);
};

template <typename Index, typename DataMapper, int nr, bool Conjugate,
          bool PanelMode>
EIGEN_DONT_INLINE void gemm_pack_rhs<QUInt8, Index, DataMapper, nr, ColMajor,
                                     Conjugate, PanelMode>::
operator()(QUInt8* blockB, const DataMapper& rhs, Index depth, Index cols,
           Index stride, Index offset) {
  eigen_assert(stride == 0);
  eigen_assert(offset == 0);

  // Use alternate function for weird sizes
  if (cols % 32 != 0 || depth % 32 != 0) {
    gemm_pack_rhs_any<QUInt8, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode> rhs_pack;
    return rhs_pack(blockB, rhs, depth, cols, stride, offset);
  }

  // Get vector pointer
  __m256i* blockB_256 = reinterpret_cast<__m256i*>(blockB);

  // Perform a step of the packing for 4 columns
  __m256i R_AB_L, R_AB_H, R_CD_L, R_CD_H, R_AD_0, R_AD_8, R_AD_16, R_AD_24;
#define PACK_STEP                                            \
  R_AB_L = _mm256_unpacklo_epi64(R_A, R_B);                  \
  R_CD_L = _mm256_unpacklo_epi64(R_C, R_D);                  \
  R_AB_H = _mm256_unpackhi_epi64(R_A, R_B);                  \
  R_CD_H = _mm256_unpackhi_epi64(R_C, R_D);                  \
  R_AD_0 = _mm256_permute2x128_si256(R_AB_L, R_CD_L, 0x20);  \
  R_AD_16 = _mm256_permute2x128_si256(R_AB_L, R_CD_L, 0x31); \
  R_AD_8 = _mm256_permute2x128_si256(R_AB_H, R_CD_H, 0x20);  \
  R_AD_24 = _mm256_permute2x128_si256(R_AB_H, R_CD_H, 0x31); \
  _mm256_store_si256(blockB_256, R_AD_0);                    \
  _mm256_store_si256(blockB_256 + 8, R_AD_8);                \
  _mm256_store_si256(blockB_256 + 16, R_AD_16);              \
  _mm256_store_si256(blockB_256 + 24, R_AD_24);              \
  blockB_256++;

  // Pack cols in sets of 32
  for (Index n = 0; n < cols; n += 32) {
    // Pack depth in sets of 32
    for (Index k = 0; k < depth; k += 32) {
      __m256i R_A = rhs.loadPacket(k, n);
      __m256i R_B = rhs.loadPacket(k, n + 1);
      __m256i R_C = rhs.loadPacket(k, n + 2);
      __m256i R_D = rhs.loadPacket(k, n + 3);
      PACK_STEP;

      R_A = rhs.loadPacket(k, n + 4);
      R_B = rhs.loadPacket(k, n + 5);
      R_C = rhs.loadPacket(k, n + 6);
      R_D = rhs.loadPacket(k, n + 7);
      PACK_STEP;

      R_A = rhs.loadPacket(k, n + 8);
      R_B = rhs.loadPacket(k, n + 9);
      R_C = rhs.loadPacket(k, n + 10);
      R_D = rhs.loadPacket(k, n + 11);
      PACK_STEP;

      R_A = rhs.loadPacket(k, n + 12);
      R_B = rhs.loadPacket(k, n + 13);
      R_C = rhs.loadPacket(k, n + 14);
      R_D = rhs.loadPacket(k, n + 15);
      PACK_STEP;

      R_A = rhs.loadPacket(k, n + 16);
      R_B = rhs.loadPacket(k, n + 17);
      R_C = rhs.loadPacket(k, n + 18);
      R_D = rhs.loadPacket(k, n + 19);
      PACK_STEP;

      R_A = rhs.loadPacket(k, n + 20);
      R_B = rhs.loadPacket(k, n + 21);
      R_C = rhs.loadPacket(k, n + 22);
      R_D = rhs.loadPacket(k, n + 23);
      PACK_STEP;

      R_A = rhs.loadPacket(k, n + 24);
      R_B = rhs.loadPacket(k, n + 25);
      R_C = rhs.loadPacket(k, n + 26);
      R_D = rhs.loadPacket(k, n + 27);
      PACK_STEP;

      R_A = rhs.loadPacket(k, n + 28);
      R_B = rhs.loadPacket(k, n + 29);
      R_C = rhs.loadPacket(k, n + 30);
      R_D = rhs.loadPacket(k, n + 31);
      PACK_STEP;

      blockB_256 += 24;
    }
  }
#undef PACK_STEP
}

// Perform the actual multiplication on packed inputs
template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<QInt8, QUInt8, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
{
  typedef typename DataMapper::LinearMapper LinearMapper;

  EIGEN_DONT_INLINE
  void operator()(const DataMapper& res, const QInt8* blockA, const QUInt8* blockB,
                  Index rows, Index depth, Index cols, QInt32 alpha,
                  Index strideA=-1, Index strideB=-1, Index offsetA=0, Index offsetB=0);
};

template<typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
EIGEN_DONT_INLINE
void gebp_kernel<QInt8, QUInt8, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
::operator()(const DataMapper& res, const QInt8* blockA, const QUInt8* blockB,
             Index rows, Index depth, Index cols, QInt32 alpha,
             Index strideA, Index strideB, Index offsetA, Index offsetB)
{
  EIGEN_STATIC_ASSERT(!ConjugateLhs, YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT(!ConjugateRhs, YOU_MADE_A_PROGRAMMING_MISTAKE);
  eigen_assert(alpha.value == 1);
  eigen_assert(strideA == -1);
  eigen_assert(strideB == -1);
  eigen_assert(offsetA == 0);
  eigen_assert(offsetB == 0);
  eigen_assert(rows > 0);
  eigen_assert(cols > 0);
  eigen_assert(depth > 0);
  eigen_assert(blockA);
  eigen_assert(blockB);

  // Use alternate function for weird sizes
  if (rows % 32 != 0 || cols % 32 != 0 || depth % 32 != 0) {
    gebp_kernel_any<QInt8, QUInt8, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs> gebp;
    return gebp(res, blockA, blockB, rows, depth, cols, alpha, strideA, strideB, offsetA, offsetB);
  }

  // Create result block
  QInt32* blockO = aligned_new<QInt32>(32 * 32);
  // Allocating the result block is about 5-10% faster than declaring stack
  // space.  It is unclear why this is the case.
  // ei_declare_aligned_stack_constructed_variable(QInt32, blockO, 32 * 32, 0);
  memset(blockO, 0, 32 * 32 * sizeof(QInt32));

  // Get vectorized pointers
  __m256i* blockO_256 = reinterpret_cast<__m256i*>(blockO);
  const __m256i* blockA_256 = reinterpret_cast<const __m256i*>(blockA);
  const __m256i* blockB_256 = reinterpret_cast<const __m256i*>(blockB);

  // Loop over blocks of 32 columns
  for (Index n = 0; n < cols; n += 32) {
    // Reset index into blockA
    Index indexL = 0;
    // Loop over blocks of 32 rows
    for (Index m = 0; m < rows; m += 32) {
      // Reset index into blockB
      Index indexR = n / 32 * depth;
      // Loop over blocks of 8 on depth
      for (Index k = 0; k < depth; k += 8) {
        // Load inputs
        __m256i L_AD0 = blockA_256[indexL++];
        __m256i L_AD8 = blockA_256[indexL++];
        __m256i L_AD16 = blockA_256[indexL++];
        __m256i L_AD24 = blockA_256[indexL++];
        __m256i L_EH0 = blockA_256[indexL++];
        __m256i L_EH8 = blockA_256[indexL++];
        __m256i L_EH16 = blockA_256[indexL++];
        __m256i L_EH24 = blockA_256[indexL++];
        __m256i R_AH0 = blockB_256[indexR++];
        __m256i R_AH4 = blockB_256[indexR++];
        __m256i R_AH8 = blockB_256[indexR++];
        __m256i R_AH12 = blockB_256[indexR++];
        __m256i R_AH16 = blockB_256[indexR++];
        __m256i R_AH20 = blockB_256[indexR++];
        __m256i R_AH24 = blockB_256[indexR++];
        __m256i R_AH28 = blockB_256[indexR++];

        // This constant is used with madd to convert 16 bit to 32 bit
        const __m256i ONE = _mm256_set1_epi32(0x00010001);

        // Declare variables used in COMPUTE_STEP
        __m256i P_16_A, P_16_B, P_32_A, P_32_B, P_32;

#define COMPUTE_STEP(R_INPUT_A, R_INPUT_B, OFFSET)                             \
  P_16_A = _mm256_maddubs_epi16(R_INPUT_A, L_AD0);                             \
  P_32_A = _mm256_madd_epi16(P_16_A, ONE);                                     \
  P_16_B = _mm256_maddubs_epi16(R_INPUT_B, L_EH0);                             \
  P_32_B = _mm256_madd_epi16(P_16_B, ONE);                                     \
  P_32 = _mm256_add_epi32(P_32_A, P_32_B);                                     \
  _mm256_store_si256(                                                          \
      blockO_256 + 4 * OFFSET,                                                 \
      _mm256_add_epi32(_mm256_load_si256(blockO_256 + 4 * OFFSET), P_32));     \
                                                                               \
  P_16_A = _mm256_maddubs_epi16(R_INPUT_A, L_AD8);                             \
  P_32_A = _mm256_madd_epi16(P_16_A, ONE);                                     \
  P_16_B = _mm256_maddubs_epi16(R_INPUT_B, L_EH8);                             \
  P_32_B = _mm256_madd_epi16(P_16_B, ONE);                                     \
  P_32 = _mm256_add_epi32(P_32_A, P_32_B);                                     \
  _mm256_store_si256(                                                          \
      blockO_256 + 4 * OFFSET + 1,                                             \
      _mm256_add_epi32(_mm256_load_si256(blockO_256 + 4 * OFFSET + 1), P_32)); \
                                                                               \
  P_16_A = _mm256_maddubs_epi16(R_INPUT_A, L_AD16);                            \
  P_32_A = _mm256_madd_epi16(P_16_A, ONE);                                     \
  P_16_B = _mm256_maddubs_epi16(R_INPUT_B, L_EH16);                            \
  P_32_B = _mm256_madd_epi16(P_16_B, ONE);                                     \
  P_32 = _mm256_add_epi32(P_32_A, P_32_B);                                     \
  _mm256_store_si256(                                                          \
      blockO_256 + 4 * OFFSET + 2,                                             \
      _mm256_add_epi32(_mm256_load_si256(blockO_256 + 4 * OFFSET + 2), P_32)); \
                                                                               \
  P_16_A = _mm256_maddubs_epi16(R_INPUT_A, L_AD24);                            \
  P_32_A = _mm256_madd_epi16(P_16_A, ONE);                                     \
  P_16_B = _mm256_maddubs_epi16(R_INPUT_B, L_EH24);                            \
  P_32_B = _mm256_madd_epi16(P_16_B, ONE);                                     \
  P_32 = _mm256_add_epi32(P_32_A, P_32_B);                                     \
  _mm256_store_si256(                                                          \
      blockO_256 + 4 * OFFSET + 3,                                             \
      _mm256_add_epi32(_mm256_load_si256(blockO_256 + 4 * OFFSET + 3), P_32));

        // Permute and shuffle to copy a single value across the entire vector
        // Then compute the multiplication
        __m256i R_AH0_ = _mm256_permute2x128_si256(R_AH0, R_AH0, 0x00);
        __m256i R_AD0 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        __m256i R_EH0 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD0, R_EH0, 0);
        __m256i R_AD1 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        __m256i R_EH1 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD1, R_EH1, 1);
        R_AH0_ = _mm256_permute2x128_si256(R_AH0, R_AH0, 0x11);
        __m256i R_AD2 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        __m256i R_EH2 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD2, R_EH2, 2);
        __m256i R_AD3 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        __m256i R_EH3 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD3, R_EH3, 3);

        R_AH0_ = _mm256_permute2x128_si256(R_AH4, R_AH4, 0x00);
        R_AD0 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH0 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD0, R_EH0, 4);
        R_AD1 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH1 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD1, R_EH1, 5);
        R_AH0_ = _mm256_permute2x128_si256(R_AH4, R_AH4, 0x11);
        R_AD2 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH2 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD2, R_EH2, 6);
        R_AD3 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH3 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD3, R_EH3, 7);

        R_AH0_ = _mm256_permute2x128_si256(R_AH8, R_AH8, 0x00);
        R_AD0 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH0 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD0, R_EH0, 8);
        R_AD1 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH1 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD1, R_EH1, 9);
        R_AH0_ = _mm256_permute2x128_si256(R_AH8, R_AH8, 0x11);
        R_AD2 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH2 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD2, R_EH2, 10);
        R_AD3 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH3 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD3, R_EH3, 11);

        R_AH0_ = _mm256_permute2x128_si256(R_AH12, R_AH12, 0x00);
        R_AD0 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH0 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD0, R_EH0, 12);
        R_AD1 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH1 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD1, R_EH1, 13);
        R_AH0_ = _mm256_permute2x128_si256(R_AH12, R_AH12, 0x11);
        R_AD2 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH2 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD2, R_EH2, 14);
        R_AD3 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH3 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD3, R_EH3, 15);

        R_AH0_ = _mm256_permute2x128_si256(R_AH16, R_AH16, 0x00);
        R_AD0 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH0 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD0, R_EH0, 16);
        R_AD1 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH1 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD1, R_EH1, 17);
        R_AH0_ = _mm256_permute2x128_si256(R_AH16, R_AH16, 0x11);
        R_AD2 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH2 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD2, R_EH2, 18);
        R_AD3 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH3 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD3, R_EH3, 19);

        R_AH0_ = _mm256_permute2x128_si256(R_AH20, R_AH20, 0x00);
        R_AD0 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH0 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD0, R_EH0, 20);
        R_AD1 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH1 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD1, R_EH1, 21);
        R_AH0_ = _mm256_permute2x128_si256(R_AH20, R_AH20, 0x11);
        R_AD2 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH2 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD2, R_EH2, 22);
        R_AD3 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH3 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD3, R_EH3, 23);

        R_AH0_ = _mm256_permute2x128_si256(R_AH24, R_AH24, 0x00);
        R_AD0 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH0 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD0, R_EH0, 24);
        R_AD1 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH1 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD1, R_EH1, 25);
        R_AH0_ = _mm256_permute2x128_si256(R_AH24, R_AH24, 0x11);
        R_AD2 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH2 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD2, R_EH2, 26);
        R_AD3 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH3 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD3, R_EH3, 27);

        R_AH0_ = _mm256_permute2x128_si256(R_AH28, R_AH28, 0x00);
        R_AD0 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH0 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD0, R_EH0, 28);
        R_AD1 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH1 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD1, R_EH1, 29);
        R_AH0_ = _mm256_permute2x128_si256(R_AH28, R_AH28, 0x11);
        R_AD2 = _mm256_shuffle_epi32(R_AH0_, 0x00);
        R_EH2 = _mm256_shuffle_epi32(R_AH0_, 0x55);
        COMPUTE_STEP(R_AD2, R_EH2, 30);
        R_AD3 = _mm256_shuffle_epi32(R_AH0_, 0xAA);
        R_EH3 = _mm256_shuffle_epi32(R_AH0_, 0xFF);
        COMPUTE_STEP(R_AD3, R_EH3, 31);

#undef COMPUTE_STEP
      }

      // Transfer the results to the result matrix
      Index i = 0;
      for (Index j = n; j < n + 32; j++) {
        LinearMapper r0 = res.getLinearMapper(m, j);
        LinearMapper r1 = res.getLinearMapper(m + 8, j);
        LinearMapper r2 = res.getLinearMapper(m + 16, j);
        LinearMapper r3 = res.getLinearMapper(m + 24, j);
        r0.storePacket(
            0, _mm256_add_epi32(blockO_256[i++], r0.loadPacket(0)));
        r1.storePacket(
            0, _mm256_add_epi32(blockO_256[i++], r1.loadPacket(0)));
        r2.storePacket(
            0, _mm256_add_epi32(blockO_256[i++], r2.loadPacket(0)));
        r3.storePacket(
            0, _mm256_add_epi32(blockO_256[i++], r3.loadPacket(0)));
      }

      // Zero the result block so it can be reused
      memset(blockO, 0, 32 * 32 * sizeof(QInt32));
    }
  }
  aligned_delete(blockO, 32 * 32);
}

#endif  // EIGEN_USE_OPTIMIZED_INT8_UINT8_MAT_MAT_PRODUCT

}  // namespace internal
}  // namespace Eigen

#endif  // CXX11_SRC_FIXEDPOINT_MATMATPRODUCTAVX2_H_
