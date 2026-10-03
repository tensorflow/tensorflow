/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/math_ops.cc.

#ifndef TENSORFLOW_CORE_KERNELS_SPARSE_MATMUL_OP_COMMON_H_
#define TENSORFLOW_CORE_KERNELS_SPARSE_MATMUL_OP_COMMON_H_

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/sparse_matmul_op.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <vector>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "xla/tsl/framework/contraction/eigen_contraction_kernel.h"
#endif

#define ALWAYS_INLINE EIGEN_ALWAYS_INLINE

namespace tensorflow {
namespace {

template <typename T>
using BasicMatrix = Eigen::Tensor<T, 2, Eigen::RowMajor>;

template <typename T>
using BasicMatrixMap =
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned>;

using Matrix = BasicMatrix<float>;
using MatrixMap = BasicMatrixMap<float>;
using CPUDevice = Eigen::ThreadPoolDevice;
using DSizes = Eigen::DSizes<Eigen::DenseIndex, 2>;

// Two commonly used static dsizes. We use Eigen::type2index to allow as much
// compile time optimization as possible.
inline Eigen::IndexList<Eigen::type2index<0>, Eigen::type2index<0>>
dsizes_00() {
  return Eigen::IndexList<Eigen::type2index<0>, Eigen::type2index<0>>();
}
inline Eigen::IndexList<Eigen::type2index<1>, Eigen::type2index<0>>
dsizes_10() {
  return Eigen::IndexList<Eigen::type2index<1>, Eigen::type2index<0>>();
}

// Blocksizes
// TODO(agarwal): compute these sizes based on cache sizes.
const int K = 64;
const int M = 64;
const int N = 128;

// This stores a sparse representation of a slice of a matrix with size
// (num_rows, num_cols). The slice is represented as a series of blocks of size
// (num_rows, b), where b = block_size for all but the last block, which may
// have fewer columns.
//
// num_rows and block_size are assumed to be <= 256. This allows storing
// different indices as uint8.
//
// For each block, we store all the non zero entries in data/data3 vector and
// the corresponding coordinates of the element in index/index3 vectors. index3
// vector stores index of 3 elements in the same row so that these elements can
// share the same row coordinate. Each entry in Index3 corresponds to 3 entries
// in data3.
//
// Note that all the data/indices of all the blocks are stored in the same
// vectors respectively. To identify block boundaries, we store the block
// offsets using index3_offset/index_offset. If there are n blocks in the slice,
// index3_offset and index_offset have n entries. The indices for the ith block
// are the values in the following range:
// [index3[index3_offset[i-1]], index3[index3_offset[i]]). Similarly for
// index_offset.
template <typename T>
struct SparseSlice {
  using ConstMatrixMap = BasicMatrixMap<const T>;

 public:
  // Indices of three elements on the same row.
  struct Index3 {
    Index3(uint8_t m, uint8_t k1, uint8_t k2, uint8_t k3)
        : m(m), k1(k1), k2(k2), k3(k3) {}

    uint8_t m;  // row
    // columns
    uint8_t k1;
    uint8_t k2;
    uint8_t k3;
  };

  // Index of one element.
  struct Index {
    Index(uint8_t m, uint8_t k) : m(m), k(k) {}

    uint8_t m;
    uint8_t k;
  };

  SparseSlice(int nrows, int ncols, int bsize)
      : num_rows(nrows), num_cols(ncols), block_size(bsize) {
    DCHECK_LE(nrows, 256);
    DCHECK_LE(block_size, 256);
  }

  // Initializes the slice with data starting at mat(0, col_offset) and with
  // size (num_rows, num_cols).
  // If Transpose is true, implicitly transposes mat.
  template <bool Transpose = false>
  void Initialize(const ConstMatrixMap& mat, int col_offset);

  void Clear();

  // See comments above.
  std::vector<int> index3_offset;
  std::vector<Index3> index3;
  std::vector<T> data3;

  // See comments above. Similar to "index3" except that each element in "index"
  // corresponds to one element in data.
  std::vector<int> index_offset;
  std::vector<Index> index;
  std::vector<T> data;

  // Number of rows and columns for the slice.
  const int num_rows;
  const int num_cols;

  // Block size used to initialize from a matrix.
  const int block_size;
};

template <typename T>
bool IsZero(T v);

template <>
ALWAYS_INLINE bool IsZero(bfloat16 v) {
  return !static_cast<bool>(v);
}

template <>
ALWAYS_INLINE bool IsZero(float v) {
  return v == 0.0f;
}

// Note: this is intended to be used as a value type with all inline methods so
// that the compiler can optimize.
template <typename T>
class StridedIterator {
 public:
  StridedIterator(int stride, const T* start, const T* end)
      : stride_(stride), k_(0), curr_(start), end_(end) {}

  ALWAYS_INLINE bool Done() const { return curr_ >= end_; }

  // Requires `!Done()`.
  ALWAYS_INLINE T Value() const { return *curr_; }

  ALWAYS_INLINE uint8_t K() const { return k_; }

  ALWAYS_INLINE void Next() {
    curr_ += stride_;
    ++k_;
  }

  ALWAYS_INLINE void EatZeros() {
    while (curr_ < end_ && IsZero<T>(*curr_)) {
      Next();
    }
  }

 private:
  const int stride_;
  uint8_t k_;
  const T* curr_;
  const T* const end_;
};

template <typename T>
template <bool Transpose>
void SparseSlice<T>::Initialize(
    const typename SparseSlice<T>::ConstMatrixMap& mat, int col_offset) {
  const int mat_rows = Transpose ? mat.dimension(1) : mat.dimension(0);
  const int mat_cols = Transpose ? mat.dimension(0) : mat.dimension(1);
  DCHECK_LE(num_rows, mat_rows);
  DCHECK_LE(num_cols + col_offset, mat_cols);

  int num_blocks = (num_cols + block_size - 1) / block_size;
  int mat_size = num_rows * num_cols;

  index3_offset.reserve(num_blocks);
  data3.reserve(mat_size);
  index3.reserve(mat_size / 3);

  index_offset.reserve(num_blocks);
  data.reserve(num_blocks * num_rows * 2);
  index.reserve(num_blocks * num_rows * 2);

  const int stride = Transpose ? mat.dimension(1) : 1;

  for (int i = 0; i < num_blocks; ++i) {
    int num_block_cols = std::min(block_size, num_cols - block_size * i);
    for (int row = 0; row < num_rows; ++row) {
      const uint8_t m = static_cast<uint8_t>(row);
      // Safety note: The following code has a race, since it checks whether
      // *curr is nonzero and then reads it again on use.  However, the result
      // of the race is only that some of the "nonzeros" in the resulting sparse
      // representation may actually be zero, which is harmless.
      const auto* start =
          Transpose ? &mat(col_offset, row) : &mat(row, col_offset);
      const auto* end = start + stride * num_block_cols;
      StridedIterator<T> iter(stride, start, end);
      while (true) {
        iter.EatZeros();
        if (iter.Done()) break;
        const uint8_t k1 = iter.K();
        const T value1 = iter.Value();
        iter.Next();

        iter.EatZeros();
        if (iter.Done()) {
          data.push_back(value1);
          index.emplace_back(m, k1);
          break;
        }
        const uint8_t k2 = iter.K();
        const T value2 = iter.Value();
        iter.Next();

        iter.EatZeros();
        if (iter.Done()) {
          data.push_back(value2);
          index.emplace_back(m, k2);
          data.push_back(value1);
          index.emplace_back(m, k1);
          break;
        }
        const uint8_t k3 = iter.K();
        data3.push_back(value1);
        data3.push_back(value2);
        data3.push_back(iter.Value());
        iter.Next();
        ;
        index3.emplace_back(m, k1, k2, k3);
      }
    }
    col_offset += block_size;
    index3_offset.push_back(index3.size());
    index_offset.push_back(index.size());
  }
  DCHECK_EQ(index3_offset.size(), num_blocks);
  DCHECK_EQ(index_offset.size(), num_blocks);
  DCHECK_EQ(3 * index3.size(), data3.size());
  DCHECK_EQ(index.size(), data.size());
}

template <typename T>
void SparseSlice<T>::Clear() {
  index3_offset.clear();
  index3.clear();
  data3.clear();
  index_offset.clear();
  index.clear();
  data.clear();
}

ALWAYS_INLINE float ConvertBfloat16ToFloat(const bfloat16* src) {
  float out = 0;
  auto tmp = reinterpret_cast<bfloat16*>(&out);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  tmp[0] = *src;
#else
  tmp[1] = *src;
#endif
  return out;
}

ALWAYS_INLINE void ScalarMulAdd(const float a, const float** inp, float** out) {
  **out += a * **inp;
  ++*inp;
  ++*out;
}

ALWAYS_INLINE void ScalarMulAdd(const float a, const bfloat16** inp,
                                float** out) {
  float inp_f = ConvertBfloat16ToFloat(*inp);
  **out += a * inp_f;
  ++*inp;
  ++*out;
}

ALWAYS_INLINE void ScalarMulAdd3Way(const float a1, const float a2,
                                    const float a3, const bfloat16** inp1,
                                    const bfloat16** inp2,
                                    const bfloat16** inp3, float** out) {
  float inp1_f = ConvertBfloat16ToFloat(*inp1);
  float inp2_f = ConvertBfloat16ToFloat(*inp2);
  float inp3_f = ConvertBfloat16ToFloat(*inp3);
  **out += a1 * inp1_f + a2 * inp2_f + a3 * inp3_f;
  ++*out;
  ++*inp1;
  ++*inp2;
  ++*inp3;
}

ALWAYS_INLINE void ScalarMulAdd3Way(const float a1, const float a2,
                                    const float a3, const float** inp1,
                                    const float** inp2, const float** inp3,
                                    float** out) {
  **out += a1 * **inp1 + a2 * **inp2 + a3 * **inp3;
  ++*out;
  ++*inp1;
  ++*inp2;
  ++*inp3;
}

template <typename TL_, typename TR_>
struct TypePair {
  using TL = TL_;
  using TR = TR_;
};

}  // namespace
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SPARSE_MATMUL_OP_COMMON_H_

#undef HWY_TARGET_INCLUDE
#if TSL_IS_IN_OSS
#define HWY_TARGET_INCLUDE "tensorflow/core/kernels/sparse_matmul_op.cc"
#else
#define HWY_TARGET_INCLUDE \
  "third_party/tensorflow/core/kernels/sparse_matmul_op.cc"
#endif
#include "hwy/foreach_target.h"  // from @highway  // IWYU pragma: keep
#include "hwy/highway.h"  // from @highway

HWY_BEFORE_NAMESPACE();
namespace tensorflow {
namespace {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Use up to 512-bit float vectors (16 lanes on AVX-512 / Genoa, 8 on AVX2,
// 4 on SSE4/NEON, 1 on HWY_SCALAR).
using D = hn::CappedTag<float, 16>;
using Packet = hn::Vec<D>;
constexpr int kNumOperands = hn::MaxLanes(D());

ALWAYS_INLINE void LoadSingleScalar(const bfloat16** data, Packet* l) {
  const D d;
  *l = hn::Set(d, ConvertBfloat16ToFloat(*data));
  ++*data;
}

ALWAYS_INLINE void LoadTwoScalars(const bfloat16** data, Packet* l1,
                                  Packet* l2) {
  LoadSingleScalar(data, l1);
  LoadSingleScalar(data, l2);
}

ALWAYS_INLINE void LoadFourScalars(const bfloat16** data, Packet* l1,
                                   Packet* l2, Packet* l3, Packet* l4) {
  LoadTwoScalars(data, l1, l2);
  LoadTwoScalars(data, l3, l4);
}

ALWAYS_INLINE void LoadSingleScalar(const float** data, Packet* l) {
  const D d;
  *l = hn::Set(d, **data);
  ++(*data);
}

ALWAYS_INLINE void LoadTwoScalars(const float** data, Packet* l1, Packet* l2) {
  LoadSingleScalar(data, l1);
  LoadSingleScalar(data, l2);
}

ALWAYS_INLINE void LoadFourScalars(const float** data, Packet* l1, Packet* l2,
                                   Packet* l3, Packet* l4) {
  LoadTwoScalars(data, l1, l2);
  LoadTwoScalars(data, l3, l4);
}

template <typename T>
ALWAYS_INLINE void LoadThreeScalars(const T** data, Packet* l1, Packet* l2,
                                    Packet* l3) {
  LoadTwoScalars(data, l1, l2);
  LoadSingleScalar(data, l3);
}

template <typename T>
ALWAYS_INLINE void LoadSixScalars(const T** data, Packet* l1, Packet* l2,
                                  Packet* l3, Packet* l4, Packet* l5,
                                  Packet* l6) {
  LoadFourScalars(data, l1, l2, l3, l4);
  LoadTwoScalars(data, l5, l6);
}

ALWAYS_INLINE void ExpandBfloat16(const bfloat16* inp, Packet* b_0,
                                  Packet* b_1) {
  const D d;
#if HWY_TARGET != HWY_SCALAR && !HWY_IS_BIG_ENDIAN
  if (kNumOperands >= 4) {
    const hn::Repartition<uint16_t, D> d16;
    const auto zero = hn::Zero(d16);
    const auto b = hn::Load(d16, reinterpret_cast<const uint16_t*>(inp));
    *b_0 = hn::BitCast(d, hn::InterleaveLower(d16, zero, b));
    *b_1 = hn::BitCast(d, hn::InterleaveUpper(d16, zero, b));
    return;
  }
#endif
  const hn::Rebind<hwy::bfloat16_t, D> d_bf16;
  const auto* bf_inp = reinterpret_cast<const hwy::bfloat16_t*>(inp);
  *b_0 = hn::PromoteTo(d, hn::LoadU(d_bf16, bf_inp));
  *b_1 = hn::PromoteTo(d, hn::LoadU(d_bf16, bf_inp + kNumOperands));
}

// Vectorized version of ScalarMulAdd.
ALWAYS_INLINE void MulAdd(const Packet a, const bfloat16** binp, float** out) {
  const D d;
  Packet b_0, b_1;
  ExpandBfloat16(*binp, &b_0, &b_1);
  *binp += 2 * kNumOperands;
  Packet c1 = hn::Load(d, *out);
  Packet c2 = hn::Load(d, *out + kNumOperands);
  c1 = hn::MulAdd(a, b_0, c1);
  c2 = hn::MulAdd(a, b_1, c2);
  hn::Store(c1, d, *out);
  hn::Store(c2, d, *out + kNumOperands);
  *out += 2 * kNumOperands;
}

// Vectorized version of ScalarMulAdd3Way.
ALWAYS_INLINE void MulAdd3Way(const Packet a1, const Packet a2, const Packet a3,
                              const bfloat16** binp1, const bfloat16** binp2,
                              const bfloat16** binp3, float** out) {
  const D d;
  Packet c1 = hn::Load(d, *out);
  Packet c2 = hn::Load(d, *out + kNumOperands);
  Packet b1_0, b1_1, b2_0, b2_1, b3_0, b3_1;
  ExpandBfloat16(*binp1, &b1_0, &b1_1);
  *binp1 += 2 * kNumOperands;
  ExpandBfloat16(*binp2, &b2_0, &b2_1);
  *binp2 += 2 * kNumOperands;
  ExpandBfloat16(*binp3, &b3_0, &b3_1);
  *binp3 += 2 * kNumOperands;
  c1 = hn::MulAdd(a1, b1_0, c1);
  c2 = hn::MulAdd(a1, b1_1, c2);
  c1 = hn::MulAdd(a2, b2_0, c1);
  c2 = hn::MulAdd(a2, b2_1, c2);
  c1 = hn::MulAdd(a3, b3_0, c1);
  c2 = hn::MulAdd(a3, b3_1, c2);
  hn::Store(c1, d, *out);
  hn::Store(c2, d, *out + kNumOperands);
  *out += 2 * kNumOperands;
}

// Unroll MulAdd3Way for two iterations
ALWAYS_INLINE void TwoMulAdd3Way(const Packet a1, const Packet a2,
                                 const Packet a3, const bfloat16** binp1,
                                 const bfloat16** binp2, const bfloat16** binp3,
                                 float** out) {
  const D d;
  Packet c1 = hn::Load(d, *out);
  Packet c2 = hn::Load(d, *out + kNumOperands);
  Packet b1_0, b1_1, b2_0, b2_1, b3_0, b3_1;
  ExpandBfloat16(*binp1, &b1_0, &b1_1);
  ExpandBfloat16(*binp2, &b2_0, &b2_1);
  ExpandBfloat16(*binp3, &b3_0, &b3_1);

  Packet c3 = hn::Load(d, *out + 2 * kNumOperands);
  Packet c4 = hn::Load(d, *out + 3 * kNumOperands);
  Packet b4_0, b4_1, b5_0, b5_1, b6_0, b6_1;
  ExpandBfloat16(*binp1 + 2 * kNumOperands, &b4_0, &b4_1);
  ExpandBfloat16(*binp2 + 2 * kNumOperands, &b5_0, &b5_1);
  ExpandBfloat16(*binp3 + 2 * kNumOperands, &b6_0, &b6_1);

  c1 = hn::MulAdd(a1, b1_0, c1);
  c2 = hn::MulAdd(a1, b1_1, c2);
  c3 = hn::MulAdd(a1, b4_0, c3);
  c4 = hn::MulAdd(a1, b4_1, c4);
  c1 = hn::MulAdd(a2, b2_0, c1);
  c2 = hn::MulAdd(a2, b2_1, c2);
  c3 = hn::MulAdd(a2, b5_0, c3);
  c4 = hn::MulAdd(a2, b5_1, c4);
  c1 = hn::MulAdd(a3, b3_0, c1);
  c2 = hn::MulAdd(a3, b3_1, c2);
  c3 = hn::MulAdd(a3, b6_0, c3);
  c4 = hn::MulAdd(a3, b6_1, c4);
  hn::Store(c1, d, *out);
  hn::Store(c2, d, *out + kNumOperands);
  hn::Store(c3, d, *out + 2 * kNumOperands);
  hn::Store(c4, d, *out + 3 * kNumOperands);
  *out += 4 * kNumOperands;
  *binp1 += 4 * kNumOperands;
  *binp2 += 4 * kNumOperands;
  *binp3 += 4 * kNumOperands;
}

// Apply MulAdd3Way on 128 operands.
ALWAYS_INLINE void MulAdd3Way128(const Packet a1, const Packet a2,
                                 const Packet a3, const bfloat16** inp1,
                                 const bfloat16** inp2, const bfloat16** inp3,
                                 float** out) {
  for (int k = 0; k < 128 / (8 * kNumOperands); ++k) {
    TwoMulAdd3Way(a1, a2, a3, inp1, inp2, inp3, out);
    TwoMulAdd3Way(a1, a2, a3, inp1, inp2, inp3, out);
  }
}

// Vectorized version of ScalarMulAdd
ALWAYS_INLINE void MulAdd(const Packet a, const float** inp, float** out) {
  const D d;
  const Packet b = hn::Load(d, *inp);
  *inp += kNumOperands;
  Packet c = hn::Load(d, *out);
  c = hn::MulAdd(a, b, c);
  hn::Store(c, d, *out);
  *out += kNumOperands;
}

// Vectorized version of ScalarMulAdd3Way
ALWAYS_INLINE void MulAdd3Way(const Packet a1, const Packet a2, const Packet a3,
                              const float** inp1, const float** inp2,
                              const float** inp3, float** out) {
  const D d;
  Packet c = hn::Load(d, *out);
  const Packet b1 = hn::Load(d, *inp1);
  *inp1 += kNumOperands;
  const Packet b2 = hn::Load(d, *inp2);
  *inp2 += kNumOperands;
  const Packet b3 = hn::Load(d, *inp3);
  *inp3 += kNumOperands;
  c = hn::MulAdd(a1, b1, c);
  c = hn::MulAdd(a2, b2, c);
  c = hn::MulAdd(a3, b3, c);
  hn::Store(c, d, *out);
  *out += kNumOperands;
}

// Unroll MulAdd3Way for two iterations
ALWAYS_INLINE void TwoMulAdd3Way(const Packet a1, const Packet a2,
                                 const Packet a3, const float** inp1,
                                 const float** inp2, const float** inp3,
                                 float** out) {
  const D d;
  Packet c1 = hn::Load(d, *out);
  const Packet b1 = hn::Load(d, *inp1);
  const Packet b2 = hn::Load(d, *inp2);
  const Packet b3 = hn::Load(d, *inp3);

  Packet c2 = hn::Load(d, *out + kNumOperands);
  const Packet b4 = hn::Load(d, *inp1 + kNumOperands);
  const Packet b5 = hn::Load(d, *inp2 + kNumOperands);
  const Packet b6 = hn::Load(d, *inp3 + kNumOperands);

  c1 = hn::MulAdd(a1, b1, c1);
  c2 = hn::MulAdd(a1, b4, c2);
  c1 = hn::MulAdd(a2, b2, c1);
  c2 = hn::MulAdd(a2, b5, c2);
  c1 = hn::MulAdd(a3, b3, c1);
  c2 = hn::MulAdd(a3, b6, c2);
  hn::Store(c1, d, *out);
  hn::Store(c2, d, *out + kNumOperands);
  *out += 2 * kNumOperands;
  *inp1 += 2 * kNumOperands;
  *inp2 += 2 * kNumOperands;
  *inp3 += 2 * kNumOperands;
}

// Unroll MulAdd3Way for four iterations
ALWAYS_INLINE void FourMulAdd3Way(const Packet a1, const Packet a2,
                                  const Packet a3, const float** inp1,
                                  const float** inp2, const float** inp3,
                                  float** out) {
  const D d;
  Packet c1 = hn::Load(d, *out);
  Packet c2 = hn::Load(d, *out + kNumOperands);
  Packet c3 = hn::Load(d, *out + 2 * kNumOperands);
  Packet c4 = hn::Load(d, *out + 3 * kNumOperands);

  const Packet b1_0 = hn::Load(d, *inp1);
  const Packet b1_1 = hn::Load(d, *inp1 + kNumOperands);
  const Packet b2_0 = hn::Load(d, *inp2);
  const Packet b2_1 = hn::Load(d, *inp2 + kNumOperands);
  const Packet b3_0 = hn::Load(d, *inp3);
  const Packet b3_1 = hn::Load(d, *inp3 + kNumOperands);

  const Packet b4_0 = hn::Load(d, *inp1 + 2 * kNumOperands);
  const Packet b4_1 = hn::Load(d, *inp1 + 3 * kNumOperands);
  const Packet b5_0 = hn::Load(d, *inp2 + 2 * kNumOperands);
  const Packet b5_1 = hn::Load(d, *inp2 + 3 * kNumOperands);
  const Packet b6_0 = hn::Load(d, *inp3 + 2 * kNumOperands);
  const Packet b6_1 = hn::Load(d, *inp3 + 3 * kNumOperands);

  c1 = hn::MulAdd(a1, b1_0, c1);
  c2 = hn::MulAdd(a1, b1_1, c2);
  c3 = hn::MulAdd(a1, b4_0, c3);
  c4 = hn::MulAdd(a1, b4_1, c4);
  c1 = hn::MulAdd(a2, b2_0, c1);
  c2 = hn::MulAdd(a2, b2_1, c2);
  c3 = hn::MulAdd(a2, b5_0, c3);
  c4 = hn::MulAdd(a2, b5_1, c4);
  c1 = hn::MulAdd(a3, b3_0, c1);
  c2 = hn::MulAdd(a3, b3_1, c2);
  c3 = hn::MulAdd(a3, b6_0, c3);
  c4 = hn::MulAdd(a3, b6_1, c4);
  hn::Store(c1, d, *out);
  hn::Store(c2, d, *out + kNumOperands);
  hn::Store(c3, d, *out + 2 * kNumOperands);
  hn::Store(c4, d, *out + 3 * kNumOperands);
  *out += 4 * kNumOperands;
  *inp1 += 4 * kNumOperands;
  *inp2 += 4 * kNumOperands;
  *inp3 += 4 * kNumOperands;
}

// Apply MulAdd3Way on 128 operands.
ALWAYS_INLINE void MulAdd3Way128(const Packet a1, const Packet a2,
                                 const Packet a3, const float** inp1,
                                 const float** inp2, const float** inp3,
                                 float** out) {
  if (kNumOperands == 16) {
    FourMulAdd3Way(a1, a2, a3, inp1, inp2, inp3, out);
    FourMulAdd3Way(a1, a2, a3, inp1, inp2, inp3, out);
  } else if (kNumOperands == 8) {
    FourMulAdd3Way(a1, a2, a3, inp1, inp2, inp3, out);
    FourMulAdd3Way(a1, a2, a3, inp1, inp2, inp3, out);
    FourMulAdd3Way(a1, a2, a3, inp1, inp2, inp3, out);
    FourMulAdd3Way(a1, a2, a3, inp1, inp2, inp3, out);
  } else {
    DCHECK_LE(4 * kNumOperands, 128);
    for (int i = 0; i < 128 / (4 * kNumOperands); ++i) {
      MulAdd3Way(a1, a2, a3, inp1, inp2, inp3, out);
      MulAdd3Way(a1, a2, a3, inp1, inp2, inp3, out);
      MulAdd3Way(a1, a2, a3, inp1, inp2, inp3, out);
      MulAdd3Way(a1, a2, a3, inp1, inp2, inp3, out);
    }
  }
}

// Computes product of "left_slices" with "num_cols" columns of "right", and
// stores the output in *"output".
// Note that left_slices is a list of SparseSlices, which are conceptually
// assumed to be concatenated along the column dimension. Also each SparseSlice
// is encoded as a list of blocks with upto N columns. See SparseSlice for more
// details.
template <typename TL, typename TR, int Cols>
inline void GEPP(
    const std::vector<SparseSlice<TL>*>& left_slices,
    const Eigen::TensorMap<Eigen::Tensor<const TR, 2, Eigen::RowMajor>,
                           Eigen::Aligned>& right,
    const int num_cols, MatrixMap* output) {
  const int cols = (Cols == -1) ? num_cols : Cols;
  DCHECK_EQ(num_cols, cols);
  const int right_num_cols = right.dimension(1);
  const int output_num_cols = output->dimension(1);
  static constexpr int kNumOperandsR =
      kNumOperands * sizeof(float) / sizeof(TR);
  const int cols_mod = cols % kNumOperandsR;
  int k_offset = 0;
  // Pre-compute pointers for output matrix.
  float* out_ptrs[M];
  float* const out_start = &(*output)(0, 0);
  for (int j = 0; j < M; ++j) {
    out_ptrs[j] = out_start + output_num_cols * j;
  }
  for (const auto* left_slice : left_slices) {
    const auto& left = *left_slice;
    const auto* data3 = (!left.data3.empty()) ? &left.data3[0] : nullptr;
    const auto* data = (!left.data.empty()) ? &left.data[0] : nullptr;
    const int num_blocks = left.index3_offset.size();
    int begin3 = 0;
    int begin = 0;
    for (int i = 0; i < num_blocks; ++i) {
      // Pre-compute pointers for right matrix
      const TR* right_ptrs[K];
      const auto* const right_start = &right(k_offset, 0);
      DCHECK_LT(k_offset, right.dimension(0));
      for (int j = 0; j < K; ++j) {
        right_ptrs[j] = right_start + right_num_cols * j;
      }

      const int end3 = left.index3_offset[i];
      int j = begin3;
      // Loop unrolled for 2 iterations.
      for (; j + 1 < end3; j += 2) {
        Packet l1, l2, l3, nl1, nl2, nl3;
        LoadSixScalars(&data3, &l1, &l2, &l3, &nl1, &nl2, &nl3);
        const auto& index = left.index3[j];
        const auto& nindex = left.index3[j + 1];
        float* out = out_ptrs[index.m];
        float* nout = out_ptrs[nindex.m];
        const auto* r1 = right_ptrs[index.k1];
        const auto* r2 = right_ptrs[index.k2];
        const auto* r3 = right_ptrs[index.k3];

        const auto* nr1 = right_ptrs[nindex.k1];
        const auto* nr2 = right_ptrs[nindex.k2];
        const auto* nr3 = right_ptrs[nindex.k3];
        if (cols == 128) {
          MulAdd3Way128(l1, l2, l3, &r1, &r2, &r3, &out);
          MulAdd3Way128(nl1, nl2, nl3, &nr1, &nr2, &nr3, &nout);
        } else {
          for (int n = 0; n < cols / kNumOperandsR; ++n) {
            MulAdd3Way(l1, l2, l3, &r1, &r2, &r3, &out);
            MulAdd3Way(nl1, nl2, nl3, &nr1, &nr2, &nr3, &nout);
          }

          const float sl1 = hn::GetLane(l1);
          const float sl2 = hn::GetLane(l2);
          const float sl3 = hn::GetLane(l3);
          const float nsl1 = hn::GetLane(nl1);
          const float nsl2 = hn::GetLane(nl2);
          const float nsl3 = hn::GetLane(nl3);
          for (int k = 0; k < cols_mod; ++k) {
            ScalarMulAdd3Way(sl1, sl2, sl3, &r1, &r2, &r3, &out);
            ScalarMulAdd3Way(nsl1, nsl2, nsl3, &nr1, &nr2, &nr3, &nout);
          }
        }
      }
      if (j < end3) {
        Packet l1, l2, l3;
        LoadThreeScalars(&data3, &l1, &l2, &l3);

        const auto& index = left.index3[j];
        float* out = out_ptrs[index.m];
        const auto* r1 = right_ptrs[index.k1];
        const auto* r2 = right_ptrs[index.k2];
        const auto* r3 = right_ptrs[index.k3];
        if (cols == 128) {
          MulAdd3Way128(l1, l2, l3, &r1, &r2, &r3, &out);
        } else {
          for (int n = 0; n < cols / kNumOperandsR; ++n) {
            MulAdd3Way(l1, l2, l3, &r1, &r2, &r3, &out);
          }
          const float sl1 = hn::GetLane(l1);
          const float sl2 = hn::GetLane(l2);
          const float sl3 = hn::GetLane(l3);
          for (int k = 0; k < cols_mod; ++k) {
            ScalarMulAdd3Way(sl1, sl2, sl3, &r1, &r2, &r3, &out);
          }
        }
      }
      begin3 = end3;
      int end = left.index_offset[i];
      // Loop unrolled for 4 iterations.
      j = begin;
      for (; j + 3 < end; j += 4) {
        Packet l, nl, n2l, n3l;
        LoadFourScalars(&data, &l, &nl, &n2l, &n3l);

        const auto& index = left.index[j];
        const auto& nindex = left.index[j + 1];
        const auto& n2index = left.index[j + 2];
        const auto& n3index = left.index[j + 3];
        const auto* r = right_ptrs[index.k];
        const auto* nr = right_ptrs[nindex.k];
        const auto* n2r = right_ptrs[n2index.k];
        const auto* n3r = right_ptrs[n3index.k];
        float* out = out_ptrs[index.m];
        float* nout = out_ptrs[nindex.m];
        float* n2out = out_ptrs[n2index.m];
        float* n3out = out_ptrs[n3index.m];

        for (int n = 0; n < cols / kNumOperandsR; ++n) {
          MulAdd(l, &r, &out);
          MulAdd(nl, &nr, &nout);
          MulAdd(n2l, &n2r, &n2out);
          MulAdd(n3l, &n3r, &n3out);
        }

        const float sl1 = hn::GetLane(l);
        const float sl2 = hn::GetLane(nl);
        const float sl3 = hn::GetLane(n2l);
        const float sl4 = hn::GetLane(n3l);
        for (int k = 0; k < cols_mod; ++k) {
          ScalarMulAdd(sl1, &r, &out);
          ScalarMulAdd(sl2, &nr, &nout);
          ScalarMulAdd(sl3, &n2r, &n2out);
          ScalarMulAdd(sl4, &n3r, &n3out);
        }
      }
      while (j < end) {
        Packet l;
        LoadSingleScalar(&data, &l);
        const auto& index = left.index[j];
        const auto* r = right_ptrs[index.k];
        float* out = out_ptrs[index.m];
        for (int n = 0; n < cols / kNumOperandsR; ++n) {
          MulAdd(l, &r, &out);
        }
        const float sl = hn::GetLane(l);
        for (int k = 0; k < cols_mod; ++k) {
          ScalarMulAdd(sl, &r, &out);
        }
        j++;
      }
      k_offset += left.block_size;
      begin = end;
    }
  }
}

template <int NUM_ELEM = -1>
ALWAYS_INLINE void CopyAndMayBeInterleaveBfloat16(void* bdst, const void* bsrc,
                                                  int num_elements) {
#if HWY_TARGET != HWY_SCALAR && !HWY_IS_BIG_ENDIAN
  if (kNumOperands >= 8) {
    static constexpr int kStep =
        kNumOperands * sizeof(float) / sizeof(bfloat16);
    const int num = (NUM_ELEM == -1) ? num_elements : NUM_ELEM;
    DCHECK_EQ(num, num_elements);
    const hn::Repartition<uint64_t, D> d64;
    const uint64_t* src = reinterpret_cast<const uint64_t*>(bsrc);
    uint64_t* dst = reinterpret_cast<uint64_t*>(bdst);
    for (int index = 0; index + kStep <= num; index += kStep) {
      auto in = hn::LoadU(d64, src);
      if (kNumOperands == 16) {
        alignas(64) static constexpr uint64_t kIdx[8] = {0, 4, 1, 5,
                                                         2, 6, 3, 7};
        in = hn::TableLookupLanes(in, hn::SetTableIndices(d64, kIdx));
      } else if (kNumOperands == 8) {
        alignas(32) static constexpr uint64_t kIdx[4] = {0, 2, 1, 3};
        in = hn::TableLookupLanes(in, hn::SetTableIndices(d64, kIdx));
      }
      hn::Store(in, d64, dst);
      src += kNumOperands / 2;
      dst += kNumOperands / 2;
    }
    if (num % kStep != 0) {
      memcpy(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src),
             (num % kStep) * sizeof(bfloat16));
    }
    return;
  }
#endif
  memcpy(bdst, bsrc, num_elements * sizeof(bfloat16));
}

template <typename T>
ALWAYS_INLINE void CopyAndMayBeInterleave(void* dst, const void* src,
                                          int num_elements) {
  if (std::is_same<T, float>::value || kNumOperands < 8) {
    memcpy(dst, src, num_elements * sizeof(T));
  } else if (std::is_same<T, bfloat16>::value) {
    if (num_elements == N) {
      CopyAndMayBeInterleaveBfloat16<N>(dst, src, num_elements);
    } else {
      CopyAndMayBeInterleaveBfloat16<-1>(dst, src, num_elements);
    }
  } else {
    LOG(FATAL) << "Unsupported type";
  }
}

template <typename TR>
void ShuffleMatrixWorkImpl(const BasicMatrixMap<const TR>& mat,
                           int slice_row_start, int slice_num_rows,
                           int slice_col_start, int slice_num_cols, const int N,
                           int s, int e, BasicMatrixMap<TR>* buffer) {
  const int row_start = s % slice_num_rows + slice_row_start;
  const int col_start = s / slice_num_rows * N + slice_col_start;
  auto* out_start = &(*buffer)(s, 0);
  const auto* input_start = &mat(row_start, col_start);
  const auto* input_end = &mat(slice_row_start + slice_num_rows - 1,
                               slice_col_start + slice_num_cols - 1);
  const int mat_num_cols = mat.dimension(1);
  const int row_slice_size = slice_num_rows * mat_num_cols;

  const int aligned_end = slice_num_cols / N * slice_num_rows;
  const int e1 = std::min(e, aligned_end);
  while (s < e1) {
    CopyAndMayBeInterleave<TR>(out_start, input_start, N);
    out_start += N;
    input_start += mat_num_cols;
    if (input_start > input_end) {
      input_start = input_start - row_slice_size + N;
    }
    ++s;
  }
  int s1 = std::max(s, aligned_end);
  const int copy_num_cols = slice_num_cols % N;
  while (s1 < e) {
    CopyAndMayBeInterleave<TR>(out_start, input_start, copy_num_cols);
    out_start += N;
    input_start += mat_num_cols;
    ++s1;
  }
}

template <typename Pair>
void ComputeOutputBlockImpl(
    const std::vector<SparseSlice<typename Pair::TL>*>& left,
    const BasicMatrixMap<const typename Pair::TR>& right, int num_cols,
    int output_row_offset, int output_col_offset, bool assign,
    bool transpose_output, MatrixMap* output) {
  using TL = typename Pair::TL;
  using TR = typename Pair::TR;
  const auto perm = dsizes_10();
  int num_rows = left[0]->num_rows;
  const int rhs_num_cols = right.dimension(1);
  DCHECK_LE(num_cols, rhs_num_cols);
  float* out_data =
      static_cast<float*>(Eigen::internal::handmade_aligned_malloc(
          sizeof(float) * num_rows * rhs_num_cols, 64));
  std::unique_ptr<float, void (*)(void*)> out_heap(
      out_data, Eigen::internal::handmade_aligned_free);
  memset(out_data, 0, sizeof(float) * num_rows * rhs_num_cols);
  MatrixMap out(out_data, num_rows, rhs_num_cols);
  if (num_cols == N) {
    GEPP<TL, TR, N>(left, right, num_cols, &out);
  } else {
    GEPP<TL, TR, -1>(left, right, num_cols, &out);
  }
  if (!assign) {
    const D d;
    if (transpose_output) {
      for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
          (*output)(output_col_offset + j, output_row_offset + i) += out(i, j);
        }
      }
    } else {
      for (int i = 0; i < num_rows; ++i) {
        float* dst = &(*output)(output_row_offset + i, output_col_offset);
        const float* src = &out(i, 0);
        int j = 0;
        for (; j + kNumOperands <= num_cols; j += kNumOperands) {
          hn::StoreU(hn::Add(hn::LoadU(d, dst + j), hn::Load(d, src + j)), d,
                     dst + j);
        }
        for (; j < num_cols; ++j) {
          dst[j] += src[j];
        }
      }
    }
  } else {
    std::unique_ptr<Matrix> out_tr;
    if (transpose_output) {
      out_tr.reset(new Matrix(rhs_num_cols, num_rows));
      *out_tr = out.shuffle(perm);
      std::swap(output_row_offset, output_col_offset);
      std::swap(num_rows, num_cols);
    }
    for (int i = 0; i < num_rows; ++i) {
      const float* src = transpose_output ? &(*out_tr)(i, 0) : &out(i, 0);
      memcpy(&(*output)(output_row_offset + i, output_col_offset), src,
             num_cols * sizeof(float));
    }
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace
}  // namespace tensorflow
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace tensorflow {

template <typename TL, typename TR>
class SparseMatMul {
  using MatrixL = BasicMatrix<TL>;
  using MatrixR = BasicMatrix<TR>;
  using ConstMatrixMapL = BasicMatrixMap<const TL>;
  using ConstMatrixMapR = BasicMatrixMap<const TR>;
  using MatrixMapR = BasicMatrixMap<TR>;

 public:
  // Perform matrix multiplication of "left" and "right", and store the result
  // in *"output".
 public:
  static inline void Compute(const ConstMatrixMapL& left,
                             const ConstMatrixMapR& right, bool transpose_left,
                             const DeviceBase::CpuWorkerThreads* thread_pool,
                             bool transpose_output, MatrixMap* output);

 private:
  // Computes multiplication of left and num_cols columns of right, and stores
  // the output block in *"output" at offsets "output_row_offset" and
  // "output_col_offset". If assign is true, assigns the value to that block,
  // else adds the values to the existing values.
  static inline void ComputeOutputBlock(
      const std::vector<SparseSlice<TL>*>& left, const ConstMatrixMapR& right,
      int num_cols, int output_row_offset, int output_col_offset, bool assign,
      bool transpose_output, MatrixMap* output);

  // Encodes "mat" using a sparse representation and stores that in
  // "mat_slices". "mat" is broken into a grid with sizes "slice_num_rows" and
  // "slice_num_cols", each grid element is converted into a SparseSlice and
  // stored in mat_slices. "slice_block_size" is used to perform further column
  // blocking of each slice.
  static inline std::unique_ptr<BlockingCounter> CreateSparseSlices(
      const ConstMatrixMapL& mat, bool transpose, int slice_num_rows,
      int slice_block_size, int slice_num_cols,
      std::vector<std::vector<SparseSlice<TL>*>>* mat_slices,
      const DeviceBase::CpuWorkerThreads* thread_pool);

  // This function chops "mat" along column dimension into pieces with at most N
  // columns, and concatenates the pieces one after the other in "buffer". It
  // returns the list of the pieces in "slices". It returns a BlockingCounter
  // which should be used to wait for the shuffle operations to complete.
  static inline std::unique_ptr<BlockingCounter> CreateDenseSlices(
      const ConstMatrixMapR& mat, int row_start, int num_rows, int col_start,
      int num_cols, const DeviceBase::CpuWorkerThreads* thread_pool,
      MatrixMapR* buffer, std::vector<ConstMatrixMapR*>* slices);

  // Helper function for CreateDenseSlices to move the data around. It returns a
  // BlockingCounter which should be used to wait for the shuffle operations to
  // complete.
  static inline BlockingCounter* ShuffleMatrix(
      const ConstMatrixMapR& mat, int slice_row_start, int slice_num_rows,
      int slice_col_start, int slice_num_cols, const int N,
      const DeviceBase::CpuWorkerThreads* thread_pool, MatrixMapR* buffer);

  // Helper function for CreateDenseSlices to create slices.
  static inline void SliceMatrix(const MatrixMapR& mat, const int num_rows,
                                 const int num_slices,
                                 std::vector<ConstMatrixMapR*>* slices);

  // Heuristics to compute various block sizes.
  // KR, NR: block sizes for "right". We run blocking iterations that operate on
  // matrices with at most this size.
  // KL: grid size along the column dimension used while encoding left.
  // IB, JB: number of left and right slices to multiply together. This is used
  // for ordering different ComputeBlockOutput operations inside each blocking
  // iteration so as to potentially reduce the working set size.
  static inline void ComputeBlockSizes(const ConstMatrixMapL& left,
                                       const ConstMatrixMapR& right,
                                       bool transpose_left, int num_threads,
                                       int* KR, int* NR, int* KL, int* JB,
                                       int* IB);

  SparseMatMul(const SparseMatMul&) = delete;
  void operator=(const SparseMatMul&) = delete;
};

template <typename TL, typename TR,
          template <typename TL2, typename TR2> class DoMatMul>
class SparseMatMulOp : public OpKernel {
  using MatrixR = BasicMatrix<TR>;
  using ConstMatrixMapR = BasicMatrixMap<const TR>;

 public:
  explicit SparseMatMulOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("a_is_sparse", &a_is_sparse_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("b_is_sparse", &b_is_sparse_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a.shape()),
                absl::InvalidArgumentError("a is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b.shape()),
                absl::InvalidArgumentError("b is not a matrix"));

    const int m = transpose_a_ ? a.dim_size(1) : a.dim_size(0);
    const int k = transpose_a_ ? a.dim_size(0) : a.dim_size(1);
    const int n = transpose_b_ ? b.dim_size(0) : b.dim_size(1);
    const int k2 = transpose_b_ ? b.dim_size(1) : b.dim_size(0);

    OP_REQUIRES(ctx, k == k2,
                absl::InvalidArgumentError(absl::StrCat(
                    "Matrix size incompatible: a: ", a.shape().DebugString(),
                    ", b: ", b.shape().DebugString())));
    OP_REQUIRES(
        ctx, m >= 0 && n >= 0 && k >= 0,
        absl::InvalidArgumentError(absl::StrCat(
            "Matrix dimensions cannot be negative: a: ",
            a.shape().DebugString(), ", b: ", b.shape().DebugString())));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({m, n}), &output));

    // Return early if at least one of the output dimension size is 0.
    if (m == 0 || n == 0) {
      return;
    }

    if (k == 0) {
      // If the inner dimension k in the matrix multiplication is zero, we fill
      // the output with zeros.
      functor::SetZeroFunctor<CPUDevice, float> f;
      f(ctx->eigen_device<CPUDevice>(), output->flat<float>());
      return;
    }

    auto out = output->matrix<float>();

    std::unique_ptr<Tensor> a_float;
    std::unique_ptr<Tensor> b_float;
    if (!a_is_sparse_ && !b_is_sparse_) {
      auto left = &a;
      auto right = &b;
      // TODO(agarwal): multi-thread the conversions from bfloat16 to float.
      if (std::is_same<TL, bfloat16>::value) {
        a_float.reset(new Tensor(DT_FLOAT, a.shape()));
        BFloat16ToFloat(a.flat<bfloat16>().data(),
                        a_float->flat<float>().data(), a.NumElements());
        left = a_float.get();
      }
      if (std::is_same<TR, bfloat16>::value) {
        b_float.reset(new Tensor(DT_FLOAT, b.shape()));
        BFloat16ToFloat(b.flat<bfloat16>().data(),
                        b_float->flat<float>().data(), b.NumElements());
        right = b_float.get();
      }
      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0].first = transpose_a_ ? 0 : 1;
      dim_pair[0].second = transpose_b_ ? 1 : 0;

      out.device(ctx->template eigen_device<CPUDevice>()) =
          left->matrix<float>().contract(right->matrix<float>(), dim_pair);
      return;
    }

    auto left = &a;
    auto right = &b;
    bool transpose_output = false;
    bool transpose_a = transpose_a_;
    bool transpose_b = transpose_b_;
    if (!a_is_sparse_) {
      // Swap the order of multiplications using the identity:
      // A * B = (B' *  A')'.
      std::swap(left, right);
      std::swap(transpose_a, transpose_b);
      transpose_a = !transpose_a;
      transpose_b = !transpose_b;
      transpose_output = !transpose_output;
    }

    std::unique_ptr<Tensor> right_tr;
    if (transpose_b) {
      // TODO(agarwal): avoid transposing the matrix here and directly handle
      // transpose in CreateDenseSlices.
      OP_REQUIRES(
          ctx, right->dim_size(0) != 0,
          absl::InvalidArgumentError("b has an entry 0 in it's shape."));
      OP_REQUIRES(
          ctx, right->dim_size(1) != 0,
          absl::InvalidArgumentError("b has an entry 0 in it's shape."));
      right_tr.reset(
          new Tensor(right->dtype(),
                     TensorShape({right->dim_size(1), right->dim_size(0)})));

      const auto perm = dsizes_10();
      if (transpose_output) {
        right_tr->matrix<TL>().device(ctx->template eigen_device<CPUDevice>()) =
            right->matrix<TL>().shuffle(perm);
      } else {
        right_tr->matrix<TR>().device(ctx->template eigen_device<CPUDevice>()) =
            right->matrix<TR>().shuffle(perm);
      }
      right = right_tr.get();
    }

    if (transpose_output) {
      DoMatMul<TR, TL>::Compute(left->matrix<TR>(), right->matrix<TL>(),
                                transpose_a,
                                ctx->device()->tensorflow_cpu_worker_threads(),
                                transpose_output, &out);
    } else {
      DoMatMul<TL, TR>::Compute(left->matrix<TL>(), right->matrix<TR>(),
                                transpose_a,
                                ctx->device()->tensorflow_cpu_worker_threads(),
                                transpose_output, &out);
    }
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
  bool a_is_sparse_;
  bool b_is_sparse_;

  SparseMatMulOp(const SparseMatMulOp&) = delete;
  void operator=(const SparseMatMulOp&) = delete;
};

template <typename TL, typename TR>
inline void SparseMatMul<TL, TR>::ComputeOutputBlock(
    const std::vector<SparseSlice<TL>*>& left,
    const typename SparseMatMul<TL, TR>::ConstMatrixMapR& right, int num_cols,
    int output_row_offset, int output_col_offset, bool assign,
    bool transpose_output, MatrixMap* output) {
  using Pair = TypePair<TL, TR>;
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(ComputeOutputBlockImpl<Pair>)(
      left, right, num_cols, output_row_offset, output_col_offset, assign,
      transpose_output, output);
}

template <typename TL, typename TR>
inline std::unique_ptr<BlockingCounter>
SparseMatMul<TL, TR>::CreateSparseSlices(
    const typename SparseMatMul<TL, TR>::ConstMatrixMapL& mat, bool transpose,
    int slice_num_rows, int slice_block_size, int slice_num_cols,
    std::vector<std::vector<SparseSlice<TL>*>>* mat_slices,
    const DeviceBase::CpuWorkerThreads* thread_pool) {
  const int mat_num_rows = transpose ? mat.dimension(1) : mat.dimension(0);
  const int mat_num_cols = transpose ? mat.dimension(0) : mat.dimension(1);
  const int num_slices_dim0 =
      std::max(1, (mat_num_rows + slice_num_rows - 1) / slice_num_rows);
  const int num_slices_dim1 =
      std::max(1, (mat_num_cols + slice_num_cols - 1) / slice_num_cols);
  mat_slices->resize(num_slices_dim0);
  BlockingCounter* counter =
      new BlockingCounter(num_slices_dim0 * num_slices_dim1);
  auto work = [counter, transpose](SparseSlice<TL>* sparse_slice,
                                   SparseMatMul<TL, TR>::ConstMatrixMapL* slice,
                                   int col_offset) {
    if (transpose) {
      sparse_slice->template Initialize<true>(*slice, col_offset);
    } else {
      sparse_slice->template Initialize<false>(*slice, col_offset);
    }
    delete slice;
    counter->DecrementCount();
  };
  for (int i = 0; i < num_slices_dim0; ++i) {
    (*mat_slices)[i].resize(num_slices_dim1);
    int num_rows =
        std::min<int>(slice_num_rows, mat_num_rows - i * slice_num_rows);
    for (int j = 0; j < num_slices_dim1; ++j) {
      int num_cols =
          std::min<int>(slice_num_cols, mat_num_cols - j * slice_num_cols);
      SparseMatMul<TL, TR>::ConstMatrixMapL* slice = nullptr;
      if (transpose) {
        slice = new SparseMatMul<TL, TR>::ConstMatrixMapL(
            &mat(0, i * slice_num_rows), mat.dimensions());
      } else {
        DSizes d(num_rows, mat_num_cols);
        slice = new SparseMatMul<TL, TR>::ConstMatrixMapL(
            &mat(i * slice_num_rows, 0), d);
      }
      auto* sparse_slice =
          new SparseSlice<TL>(num_rows, num_cols, slice_block_size);
      (*mat_slices)[i][j] = sparse_slice;
      thread_pool->workers->Schedule(
          [=]() { work(sparse_slice, slice, slice_num_cols * j); });
    }
  }
  return std::unique_ptr<BlockingCounter>(counter);
}

template <typename TL, typename TR>
inline BlockingCounter* SparseMatMul<TL, TR>::ShuffleMatrix(
    const typename SparseMatMul<TL, TR>::ConstMatrixMapR& mat,
    int slice_row_start, int slice_num_rows, int slice_col_start,
    int slice_num_cols, const int N,
    const DeviceBase::CpuWorkerThreads* thread_pool, MatrixMapR* buffer) {
  DCHECK_EQ(N % 2, 0);
  // Note(nikhilsarda): This heuristic is optimal in benchmarks as of
  // Jan 21, 2020.
  int num_threads = std::min(thread_pool->num_threads, 8);
  BlockingCounter* counter = new BlockingCounter(num_threads);
  DCHECK_EQ(N, buffer->dimension(1));
  auto shuffle_work = [&mat, slice_row_start, slice_num_rows, slice_col_start,
                       slice_num_cols, N, buffer, counter](int s, int e) {
    HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(ShuffleMatrixWorkImpl<TR>)(
        mat, slice_row_start, slice_num_rows, slice_col_start, slice_num_cols,
        N, s, e, buffer);
    if (counter) counter->DecrementCount();
  };

  int start = 0;
  int end = 0;
  int num_out_rows = (slice_num_cols + N - 1) / N * slice_num_rows;
  DCHECK_LE(num_out_rows, buffer->dimension(0));
  for (int i = std::max(1, num_threads); i > 0; --i) {
    end = start + num_out_rows / i;
    thread_pool->workers->Schedule([=]() { shuffle_work(start, end); });
    num_out_rows -= (end - start);
    start = end;
  }
  return counter;
}

template <typename TL, typename TR>
inline void SparseMatMul<TL, TR>::SliceMatrix(
    const MatrixMapR& mat, const int num_rows, const int num_slices,
    std::vector<typename SparseMatMul<TL, TR>::ConstMatrixMapR*>* slices) {
  slices->resize(num_slices);
  DSizes d(num_rows, mat.dimension(1));
  DCHECK_LE(num_rows * num_slices, mat.dimension(0));
  for (int i = 0; i < num_slices; ++i) {
    (*slices)[i] = new ConstMatrixMapR(&mat(i * num_rows, 0), d);
  }
}

template <typename TL, typename TR>
inline std::unique_ptr<BlockingCounter> SparseMatMul<TL, TR>::CreateDenseSlices(
    const typename SparseMatMul<TL, TR>::ConstMatrixMapR& mat, int row_start,
    int num_rows, int col_start, int num_cols,
    const DeviceBase::CpuWorkerThreads* thread_pool, MatrixMapR* buffer,
    std::vector<typename SparseMatMul<TL, TR>::ConstMatrixMapR*>* slices) {
  std::unique_ptr<BlockingCounter> shuffle_counter(ShuffleMatrix(
      mat, row_start, num_rows, col_start, num_cols, N, thread_pool, buffer));
  const int num_slices = (num_cols + N - 1) / N;
  SliceMatrix(*buffer, num_rows, num_slices, slices);
  return shuffle_counter;
}

template <typename TL, typename TR>
inline void SparseMatMul<TL, TR>::ComputeBlockSizes(
    const typename SparseMatMul<TL, TR>::ConstMatrixMapL& left,
    const typename SparseMatMul<TL, TR>::ConstMatrixMapR& right,
    bool transpose_left, int num_threads, int* KR, int* NR, int* KL, int* JB,
    int* IB) {
  // Heuristics for calculating block sizes
  // Assume two hyperthreads per core.
  const int est_num_cores = std::max(1, (num_threads + 1) / 2);
  // Use block of rhs with at most 128K floats per core.
  const int mem = est_num_cores * 128 * 1024;
  *KR = std::min(static_cast<int>(right.dimension(0)), mem / 256);
  *NR = right.dimension(1);
  if (*KR * *NR > mem) {
    // 4096 may be enough to amortize the cost of writes.
    *KR = std::min<int>(*KR, 4096);
  }
  // Use sizes that are multiples of K and 256.
  *KR = std::max(1, *KR / K) * K;
  *NR = std::max(1, *NR / 256) * 256;
  if (*KR * *NR > mem) {
    *NR = mem / *KR;
  }
  *NR = std::max(1, *NR / 256) * 256;

  const int left_dim0 = transpose_left ? left.dimension(1) : left.dimension(0);
  const int left_dim1 = transpose_left ? left.dimension(0) : left.dimension(1);
  for (*KL = 1024; *KL > K; *KL /= 2) {
    if (*KR % *KL == 0 &&
        std::max<int>(1, left_dim0 / 64) * (left_dim1 / *KL) > est_num_cores) {
      break;
    }
  }
  DCHECK_EQ(*KL % K, 0);
  DCHECK_GE(*KR, *KL);
  if (*KR < right.dimension(0)) {
    CHECK_EQ(*KR % *KL, 0);
  }

  *JB = std::max(1, static_cast<int>(sqrt(num_threads) / 2.0));
  *IB = 8 * *JB;
  DCHECK_EQ(N * sizeof(float) % 64, size_t{0});
}

// Here is an overview of the SparseMatMul code. Note that we assume that the
// left matrix is sparse.
//
// The matrix "left" is divided into a grid with blocksize of (M, KL). Each
// block is encoded as a SparseSlice. These grid elements are stored as
// std::vector<std::vector<SparseSlice>>. Each element of the outer vector
// represents M rows of the left matrix. Lets call these elements l_i and lets
// call each element of the inner vector L_mk.
//
// The matrix "right" is divided into a grid with block size KR * NR.  Lets
// denote the blocks on the right as R_kn. Note that we ensure that KL divides
// KR so that for each element R_kn, we don't need to multiply it with any
// partial L_mk blocks.
//
// We then multiply each right side block R_kn with the full "left" matrix and
// update the output. These iterations are run sequentially since R_kn are
// packed into the same underlying temporary buffer.
//
// In each iteration we do the following:
// 1. Create slices r_j of R_kn: We split R_kn into vertical blocks with N
//    (=128) columns and then concatenating these slices into a buffer. This is
//    done so that each slice r_j of R_kn is stored contiguously in memory. Note
//    that if R_kj has dimensions (KR, NR), we create NR / N slices, and the
//    buffer has dimensions (KR * NR / N, N) (assuming N divides NR).
// 2. For each (l_i, r_j), we compute the inner product using the GEPP function
//    and update the output block o_ij. These calls are further blocked to
//    reduce the working set size. In each iteration we take IB elements from
//    {l_i} and JB elements from {r_j} and compute the IB * JB inner products.
template <typename TL, typename TR>
inline void SparseMatMul<TL, TR>::Compute(
    const typename SparseMatMul<TL, TR>::ConstMatrixMapL& left,
    const typename SparseMatMul<TL, TR>::ConstMatrixMapR& right,
    bool transpose_left, const DeviceBase::CpuWorkerThreads* thread_pool,
    bool transpose_output, MatrixMap* output) {
  const int num_threads = thread_pool->num_threads;
  int KR, NR, KL, JB, IB;
  ComputeBlockSizes(left, right, transpose_left, num_threads, &KR, &NR, &KL,
                    &JB, &IB);
  // Slice the left matrix
  std::vector<std::vector<SparseSlice<TL>*>> left_slices;
  std::unique_ptr<BlockingCounter> sparse_slice_counter =
      CreateSparseSlices(ConstMatrixMapL(left.data(), left.dimensions()),
                         transpose_left, M, K, KL, &left_slices, thread_pool);
  const int num_left_slices = left_slices.size();

  const int right_dim0 = right.dimension(0);
  const int right_dim1 = right.dimension(1);
  // Allocate 64-byte aligned buffer for storing slices of right matrix.
  // Note buffer needs enough space to hold at most a KR * NR matrix since that
  // is the block size per iteration.
  const int buffer_num_rows =
      std::min(KR, right_dim0) * ((std::min(NR, right_dim1) + N - 1) / N);
  std::unique_ptr<TR, void (*)(void*)> buffer_data(
      static_cast<TR*>(Eigen::internal::handmade_aligned_malloc(
          sizeof(TR) * buffer_num_rows * N, 64)),
      Eigen::internal::handmade_aligned_free);
  MatrixMapR buffer(buffer_data.get(), buffer_num_rows, N);
  std::vector<ConstMatrixMapR*> right_slices;

  std::vector<SparseSlice<TL>*> block_left_slices;
  std::vector<std::function<void(void)>> tasks;
  // Number of blocks based on block sizes of KR * NR.
  const int num_k_blocks = (right_dim0 + KR - 1) / KR;
  const int num_n_blocks = (right_dim1 + NR - 1) / NR;
  std::unique_ptr<BlockingCounter> dense_slice_counter;

  for (int nb = 0; nb < num_n_blocks; ++nb) {
    const int right_num_cols =
        std::min(NR, static_cast<int>(right_dim1 - NR * nb));
    for (int kb = 0; kb < num_k_blocks; ++kb) {
      const int right_num_rows =
          std::min(KR, static_cast<int>(right_dim0 - KR * kb));
      dense_slice_counter = CreateDenseSlices(
          right, kb * KR, right_num_rows, nb * NR, right_num_cols, thread_pool,
          &buffer, &right_slices);
      const int num_right_slices = right_slices.size();
      tasks.reserve(num_left_slices * num_right_slices);
      for (int j_outer = 0; j_outer < num_right_slices; j_outer += JB) {
        for (int i_outer = 0; i_outer < num_left_slices; i_outer += IB) {
          for (int j_inner = j_outer;
               j_inner < std::min(num_right_slices, j_outer + JB); ++j_inner) {
            const int num_cols = std::min(N, right_num_cols - N * j_inner);
            for (int i_inner = i_outer;
                 i_inner < std::min(num_left_slices, i_outer + IB); ++i_inner) {
              block_left_slices.clear();
              int begin = kb * KR / KL;
              int end = std::min<int>((kb + 1) * KR / KL,
                                      (right.dimension(0) + KL - 1) / KL);
              DCHECK_LT(begin, end);
              block_left_slices.insert(block_left_slices.begin(),
                                       left_slices[i_inner].begin() + begin,
                                       left_slices[i_inner].begin() + end);
              tasks.push_back(std::bind(
                  &ComputeOutputBlock, block_left_slices,
                  std::ref(*right_slices[j_inner]), num_cols, M * i_inner,
                  N * j_inner + nb * NR, kb == 0, transpose_output, output));
            }
          }
        }
      }
      if (sparse_slice_counter) {
        sparse_slice_counter->Wait();
        sparse_slice_counter.reset(nullptr);
      }
      if (dense_slice_counter) {
        dense_slice_counter->Wait();
        dense_slice_counter.reset(nullptr);
      }
      BlockingCounter bc(tasks.size());
      for (const auto& t : tasks) {
        thread_pool->workers->Schedule([&bc, &t]() {
          t();
          bc.DecrementCount();
        });
      }
      bc.Wait();
      tasks.clear();
      for (auto& temp : right_slices) {
        delete temp;
      }
      right_slices.clear();
    }
  }
  for (auto& left_slice : left_slices) {
    for (auto& temp : left_slice) {
      delete temp;
    }
    left_slice.clear();
  }
}

#define REGISTER_SPARSE_MATMUL(TA, TB)                   \
  REGISTER_KERNEL_BUILDER(Name("SparseMatMul")           \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<TA>("Ta")  \
                              .TypeConstraint<TB>("Tb"), \
                          SparseMatMulOp<TA, TB, SparseMatMul>);

REGISTER_SPARSE_MATMUL(float, float);
REGISTER_SPARSE_MATMUL(bfloat16, bfloat16);
REGISTER_SPARSE_MATMUL(float, bfloat16);
REGISTER_SPARSE_MATMUL(bfloat16, float);

#undef REGISTER_SPARSE_MATMUL

}  // end namespace tensorflow
#endif  // HWY_ONCE
