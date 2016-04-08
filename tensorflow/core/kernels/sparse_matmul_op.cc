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

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include <vector>
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

typedef Eigen::Tensor<float, 2, Eigen::RowMajor> Matrix;
typedef Eigen::DSizes<Eigen::DenseIndex, 2> DSizes;
typedef Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>,
                         Eigen::Aligned> MatrixMap;
typedef Eigen::TensorMap<Eigen::Tensor<const float, 2, Eigen::RowMajor>,
                         Eigen::Aligned> ConstMatrixMap;
typedef Eigen::ThreadPoolDevice CPUDevice;

// Blocksizes
// TODO(agarwal): compute these sizes based on cache sizes.
static const int K = 64;
static const int M = 64;
static const int N = 128;

// This stores a sparse representation of a slice of a matrix with size
// (num_rows, num_cols). The slice is represented as a series of blocks of size
// (num_rows, b), where b = block_size for all but the last block, which may
// have
// fewer columns.
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
struct SparseSlice {
 public:
  // Indices of three elements on the same row.
  struct Index3 {
    uint8 m;  // row
    // columns
    uint8 k1;
    uint8 k2;
    uint8 k3;
  };

  // Index of one element.
  struct Index {
    uint8 m;
    uint8 k;
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
  std::vector<float> data3;

  // See comments above. Similar to "index3" except that each element in "index"
  // corresponds to one element in data.
  std::vector<int> index_offset;
  std::vector<Index> index;
  std::vector<float> data;

  // Number of rows and columns for the slice.
  const int num_rows;
  const int num_cols;

  // Block size used to initialize from a matrix.
  const int block_size;
};

template <bool Transpose>
void SparseSlice::Initialize(const ConstMatrixMap& mat, int col_offset) {
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

  Index3 idx3;
  Index idx;
  int data3_size = 0;
  for (int i = 0; i < num_blocks; ++i) {
    int num_block_cols = std::min(block_size, num_cols - block_size * i);
    for (int row = 0; row < num_rows; ++row) {
      idx3.m = static_cast<uint8>(row);
      // Safety note: The following code has a race, since it checks whether
      // *curr is nonzero and then reads it again on use.  However, the result
      // of the race is only that some of the "nonzeros" in the resulting sparse
      // representation may actually be zero, which is harmless.
      const float* start =
          Transpose ? &mat(col_offset, row) : &mat(row, col_offset);
      const float* curr = start;
      const int stride = Transpose ? mat.dimension(1) : 1;
      const float* end = start + stride * num_block_cols;
      uint8 k = 0;
#define NEXT_ELEM \
  curr += stride; \
  ++k;
      while (true) {
        while (curr < end && (*curr == 0)) {
          NEXT_ELEM;
        }
        if (curr >= end) break;
        idx3.k1 = k;
        data3.push_back(*curr);
        NEXT_ELEM;

        while (curr < end && (*curr == 0)) {
          NEXT_ELEM;
        }
        if (curr >= end) break;
        idx3.k2 = k;
        data3.push_back(*curr);
        NEXT_ELEM;

        while (curr < end && (*curr == 0)) {
          NEXT_ELEM;
        }
        if (curr >= end) break;
        idx3.k3 = k;
        data3.push_back(*curr);
        NEXT_ELEM;
        index3.push_back(idx3);
#undef NEXT_ELEM
      }
      int num_inserted_mod = data3.size() % 3;
      // Move some elements to index and data if needed.
      data3_size = data3.size() - num_inserted_mod;
      idx.m = idx3.m;
      switch (num_inserted_mod) {
        case 2:
          idx.k = idx3.k2;
          data.push_back(data3[data3_size + 1]);
          index.push_back(idx);
          TF_FALLTHROUGH_INTENDED;
        case 1:
          idx.k = idx3.k1;
          data.push_back(data3[data3_size]);
          index.push_back(idx);
          data3.resize(data3_size);
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

void SparseSlice::Clear() {
  index3_offset.clear();
  index3.clear();
  data3.clear();
  index_offset.clear();
  index.clear();
  data.clear();
}

#define SCALAR_MULADD(a, inp, out) *out++ += *a * *inp++;

#define SCALAR_MULADD3WAY(a1, a2, a3, inp1, inp2, inp3, out) \
  *out++ += *a1 * *inp1++ + *a2 * *inp2++ + *a3 * *inp3++;

typedef Eigen::internal::packet_traits<float>::type Packet;
static const int kNumOperands = (sizeof(Packet) / sizeof(float));
#define LOAD(x) Eigen::internal::pload<Packet>(x);
#define STORE(x, y) Eigen::internal::pstore<float>(x, y);
#define LOAD_SCALAR(x, y) const auto y = Eigen::internal::pload1<Packet>(x);
#define FMA(a, b, c, d) d = Eigen::internal::pmadd<Packet>(a, b, c);

// Vectorized version of SCALAR_MULADD.
#define MULADD(a, inp, out)   \
  do {                        \
    const auto b = LOAD(inp); \
    inp += kNumOperands;      \
    auto c = LOAD(out);       \
    FMA(a, b, c, c);          \
    STORE(out, c);            \
    out += kNumOperands;      \
  } while (false)

// Vectorized version of SCALAR_MULADD3WAY.
#define MULADD3WAY(a1, a2, a3, inp1, inp2, inp3, out) \
  do {                                                \
    auto c = LOAD(out);                               \
    const auto b1 = LOAD(inp1);                       \
    inp1 += kNumOperands;                             \
    const auto b2 = LOAD(inp2);                       \
    inp2 += kNumOperands;                             \
    const auto b3 = LOAD(inp3);                       \
    inp3 += kNumOperands;                             \
    FMA(a1, b1, c, c);                                \
    FMA(a2, b2, c, c);                                \
    FMA(a3, b3, c, c);                                \
    STORE(out, c);                                    \
    out += kNumOperands;                              \
  } while (false)

#ifdef EIGEN_VECTORIZE_AVX2
// Unroll MULADD3WAY for two iterations
#define MULADD3WAY_16(a1, a2, a3, inp1, inp2, inp3, out) \
  do {                                                   \
    auto c1 = LOAD(out);                                 \
    const auto b1 = LOAD(inp1);                          \
    const auto b2 = LOAD(inp2);                          \
    const auto b3 = LOAD(inp3);                          \
                                                         \
    auto c2 = LOAD(out + kNumOperands);                  \
    const auto b4 = LOAD(inp1 + kNumOperands);           \
    const auto b5 = LOAD(inp2 + kNumOperands);           \
    const auto b6 = LOAD(inp3 + kNumOperands);           \
                                                         \
    FMA(a1, b1, c1, c1);                                 \
    FMA(a1, b4, c2, c2);                                 \
    FMA(a2, b2, c1, c1);                                 \
    FMA(a2, b5, c2, c2);                                 \
    FMA(a3, b3, c1, c1);                                 \
    FMA(a3, b6, c2, c2);                                 \
    STORE(out, c1);                                      \
    STORE(out + kNumOperands, c2);                       \
    out += 2 * kNumOperands;                             \
    inp1 += 2 * kNumOperands;                            \
    inp2 += 2 * kNumOperands;                            \
    inp3 += 2 * kNumOperands;                            \
  } while (false)
// Further unroll MULADD3WAY.
#define MULADD3WAY_32(a1, a2, a3, inp1, inp2, inp3, out) \
  MULADD3WAY_16(a1, a2, a3, inp1, inp2, inp3, out);      \
  MULADD3WAY_16(a1, a2, a3, inp1, inp2, inp3, out);
#define MULADD3WAY_128(a1, a2, a3, inp1, inp2, inp3, out) \
  MULADD3WAY_32(a1, a2, a3, inp1, inp2, inp3, out);       \
  MULADD3WAY_32(a1, a2, a3, inp1, inp2, inp3, out);       \
  MULADD3WAY_32(a1, a2, a3, inp1, inp2, inp3, out);       \
  MULADD3WAY_32(a1, a2, a3, inp1, inp2, inp3, out);
#else
#define MULADD3WAY_128(a1, a2, a3, inp1, inp2, inp3, out)    \
  for (int __i = 0; __i < 128 / (4 * kNumOperands); ++__i) { \
    MULADD3WAY(a1, a2, a3, inp1, inp2, inp3, out);           \
    MULADD3WAY(a1, a2, a3, inp1, inp2, inp3, out);           \
    MULADD3WAY(a1, a2, a3, inp1, inp2, inp3, out);           \
    MULADD3WAY(a1, a2, a3, inp1, inp2, inp3, out);           \
  }
#endif

// Computes product of "left_slices" with "num_cols" columns of "right", and
// stores the output in *"output".
// Note that left_slices is a list of SparseSlices, which are conceptually
// assumed to be concatenated along the column dimension. Also each SparseSlice
// is encoded as a list of blocks with upto N columns. See SparseSlice for more
// details.
template <int Cols>
inline void GEPP(const std::vector<SparseSlice*>& left_slices,
                 const ConstMatrixMap& right, const int num_cols,
                 Matrix* output) {
  const int cols = (Cols == -1) ? num_cols : Cols;
  DCHECK_EQ(num_cols, cols);
  const int right_num_cols = right.dimension(1);
  const int output_num_cols = output->dimension(1);
  const int cols_mod = cols % kNumOperands;
  int k_offset = 0;
  // Pre-compute pointers for output matrix.
  float* out_ptrs[M];
  float* const out_start = &(*output)(0, 0);
  for (int j = 0; j < M; ++j) {
    out_ptrs[j] = out_start + output_num_cols * j;
  }
  for (const auto* left_slice : left_slices) {
    const auto& left = *left_slice;
    const float* data3 = (left.data3.size() > 0) ? &left.data3[0] : nullptr;
    const float* data = (left.data.size() > 0) ? &left.data[0] : nullptr;
    const int num_blocks = left.index3_offset.size();
    int begin3 = 0;
    int begin = 0;
    for (int i = 0; i < num_blocks; ++i) {
      // Pre-compute pointers for right matrix
      const float* right_ptrs[K];
      const float* const right_start = &right(k_offset, 0);
      DCHECK_LT(k_offset, right.dimension(0));
      for (int j = 0; j < K; ++j) {
        right_ptrs[j] = right_start + right_num_cols * j;
      }

      const int end3 = left.index3_offset[i];
      int j = begin3;
      // Loop unrolled for 2 iterations.
      for (; j + 1 < end3; j += 2) {
        const float* sl1 = data3++;
        LOAD_SCALAR(sl1, l1);
        const float* sl2 = data3++;
        LOAD_SCALAR(sl2, l2);
        const float* sl3 = data3++;
        LOAD_SCALAR(sl3, l3);
        const float* nsl1 = data3++;
        LOAD_SCALAR(nsl1, nl1);
        const float* nsl2 = data3++;
        LOAD_SCALAR(nsl2, nl2);
        const float* nsl3 = data3++;
        LOAD_SCALAR(nsl3, nl3);
        const SparseSlice::Index3& index = left.index3[j];
        const SparseSlice::Index3& nindex = left.index3[j + 1];
        float* out = out_ptrs[index.m];
        float* nout = out_ptrs[nindex.m];
        const float* r1 = right_ptrs[index.k1];
        const float* r2 = right_ptrs[index.k2];
        const float* r3 = right_ptrs[index.k3];
        const float* nr1 = right_ptrs[nindex.k1];
        const float* nr2 = right_ptrs[nindex.k2];
        const float* nr3 = right_ptrs[nindex.k3];
        if (cols == 128) {
          MULADD3WAY_128(l1, l2, l3, r1, r2, r3, out);
          MULADD3WAY_128(nl1, nl2, nl3, nr1, nr2, nr3, nout);
        } else {
          for (int n = 0; n < cols / kNumOperands; ++n) {
            MULADD3WAY(l1, l2, l3, r1, r2, r3, out);
            MULADD3WAY(nl1, nl2, nl3, nr1, nr2, nr3, nout);
          }
          for (int k = 0; k < cols_mod; ++k) {
            SCALAR_MULADD3WAY(sl1, sl2, sl3, r1, r2, r3, out);
            SCALAR_MULADD3WAY(nsl1, nsl2, nsl3, nr1, nr2, nr3, nout);
          }
        }
      }
      if (j < end3) {
        const float* sl1 = data3++;
        LOAD_SCALAR(sl1, l1);
        const float* sl2 = data3++;
        LOAD_SCALAR(sl2, l2);
        const float* sl3 = data3++;
        LOAD_SCALAR(sl3, l3);
        const SparseSlice::Index3& index = left.index3[j];
        float* out = out_ptrs[index.m];
        const float* r1 = right_ptrs[index.k1];
        const float* r2 = right_ptrs[index.k2];
        const float* r3 = right_ptrs[index.k3];
        if (cols == 128) {
          MULADD3WAY_128(l1, l2, l3, r1, r2, r3, out);
        } else {
          for (int n = 0; n < cols / kNumOperands; ++n) {
            MULADD3WAY(l1, l2, l3, r1, r2, r3, out);
          }
          for (int k = 0; k < cols_mod; ++k) {
            SCALAR_MULADD3WAY(sl1, sl2, sl3, r1, r2, r3, out);
          }
        }
      }
      begin3 = end3;
      int end = left.index_offset[i];
      for (int j = begin; j < end; ++j) {
        const float* sl = data++;
        LOAD_SCALAR(sl, l);
        const SparseSlice::Index& index = left.index[j];
        const float* r = right_ptrs[index.k];
        float* out = out_ptrs[index.m];
        for (int n = 0; n < cols / kNumOperands; ++n) {
          MULADD(l, r, out);
        }
        for (int k = 0; k < cols_mod; ++k) {
          SCALAR_MULADD(sl, r, out);
        }
      }
      k_offset += left.block_size;
      begin = end;
    }
  }
}

#undef SCALAR_MULADD
#undef SCALAR_MULADD3WAY
#undef LOAD
#undef STORE
#undef LOAD_SCALAR
#undef FMA
#undef MULADD
#undef MULADD3WAY
#undef MULADD3WAY_16
#undef MULADD3WAY_32
#undef MULADD3WAY_128

}  // namespace

class SparseMatMulOp : public OpKernel {
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
                errors::InvalidArgument("a is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b.shape()),
                errors::InvalidArgument("b is not a matrix"));

    auto left = a.matrix<float>();
    auto right = b.matrix<float>();
    const int m = transpose_a_ ? left.dimension(1) : left.dimension(0);
    const int k = transpose_a_ ? left.dimension(0) : left.dimension(1);
    const int n = transpose_b_ ? right.dimension(0) : right.dimension(1);
    const int k2 = transpose_b_ ? right.dimension(1) : right.dimension(0);

    OP_REQUIRES(ctx, k == k2,
                errors::InvalidArgument("Matrix size incompatible: a: ",
                                        a.shape().DebugString(), ", b: ",
                                        b.shape().DebugString()));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({m, n}), &output));
    auto out = output->matrix<float>();

    if (!a_is_sparse_ && !b_is_sparse_) {
      // Fallback to Eigen contract.
      // Note that we currently don't optimize the case where only right is
      // sparse. That can generally be handled by transposing the order of the
      // matmul.
      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0].first = transpose_a_ ? 0 : 1;
      dim_pair[0].second = transpose_b_ ? 1 : 0;
      out.device(ctx->template eigen_device<CPUDevice>()) =
          left.contract(right, dim_pair);
      return;
    }
    auto left_mat = &left;
    auto right_mat = &right;
    bool transpose_output = false;
    bool transpose_a = transpose_a_;
    bool transpose_b = transpose_b_;
    if (!a_is_sparse_) {
      // Swap the order of multiplications using the identity:
      // A * B = (B' *  A')'.
      std::swap(left_mat, right_mat);
      std::swap(transpose_a, transpose_b);
      transpose_a = !transpose_a;
      transpose_b = !transpose_b;
      transpose_output = !transpose_output;
    }
    std::unique_ptr<Matrix> right_tr_mat;
    std::unique_ptr<TTypes<float>::ConstMatrix> right_tr_map;
    if (transpose_b) {
      // TODO(agarwal): avoid transposing the matrix here and directly handle
      // transpose in CreateDenseSlices.
      right_tr_mat.reset(
          new Matrix(right_mat->dimension(1), right_mat->dimension(0)));
      Eigen::array<int, 2> perm({1, 0});
      right_tr_mat->device(ctx->template eigen_device<CPUDevice>()) =
          right_mat->shuffle(perm);
      right_tr_map.reset(new TTypes<float>::ConstMatrix(
          right_tr_mat->data(), right_tr_mat->dimensions()));
      right_mat = right_tr_map.get();
    }

    SparseMatMul(*left_mat, *right_mat, transpose_a,
                 ctx->device()->tensorflow_cpu_worker_threads(),
                 transpose_output, &out);
  }

 private:
  // Perform matrix multiplication of "left" and "right", and store the result
  // in *"output".
  static inline void SparseMatMul(
      const ConstMatrixMap& left, const ConstMatrixMap& right,
      bool transpose_left, const DeviceBase::CpuWorkerThreads* thread_pool,
      bool transpose_output, MatrixMap* output);

  // Computes multiplication of left and num_cols columns of right, and stores
  // the output block in *"output" at offsets "output_row_offset" and
  // "output_col_offset". If assign is true, assigns the value to that block,
  // else adds the values to the existing values.
  static inline void ComputeOutputBlock(const std::vector<SparseSlice*>& left,
                                        const ConstMatrixMap& right,
                                        int num_cols, int output_row_offset,
                                        int output_col_offset, bool assign,
                                        bool transpose_output,
                                        MatrixMap* output);

  // Encodes "mat" using a sparse representation and stores that in
  // "mat_slices". "mat" is broken into a grid with sizes "slice_num_rows" and
  // "slice_num_cols", each grid element is converted into a SparseSlice and
  // stored in mat_slices. "slice_block_size" is used to perform further column
  // blocking of each slice.
  static inline BlockingCounter* CreateSparseSlices(
      const ConstMatrixMap& mat, bool transpose, int slice_num_rows,
      int slice_block_size, int slice_num_cols,
      std::vector<std::vector<SparseSlice*>>* mat_slices,
      const DeviceBase::CpuWorkerThreads* thread_pool);

  // This function chops "mat" along column dimension into pieces with at most N
  // columns, and concatenates the pieces one after the other in "buffer". It
  // returns the list of the pieces in "slices". It returns a BlockingCounter
  // which should be used to wait for the shuffle operations to complete.
  static inline BlockingCounter* CreateDenseSlices(
      const ConstMatrixMap& mat, int row_start, int num_rows, int col_start,
      int num_cols, const DeviceBase::CpuWorkerThreads* thread_pool,
      Matrix* buffer, std::vector<ConstMatrixMap*>* slices);

  // Helper function for CreateDenseSlices to move the data around. It returns a
  // BlockingCounter which should be used to wait for the shuffle operations to
  // complete.
  static inline BlockingCounter* ShuffleMatrix(
      const ConstMatrixMap& mat, int slice_row_start, int slice_num_rows,
      int slice_col_start, int slice_num_cols, const int N,
      const DeviceBase::CpuWorkerThreads* thread_pool, Matrix* buffer);

  // Helper function for CreateDenseSlices to create slices.
  static inline void SliceMatrix(const Matrix& mat, const int num_rows,
                                 const int num_slices,
                                 std::vector<ConstMatrixMap*>* slices);

  // Heuristics to compute various block sizes.
  // KR, NR: block sizes for "right". We run blocking iterations that operate on
  // matrices with at most this size.
  // KL: grid size along the column dimension used while encoding left.
  // IB, JB: number of left and right slices to multiply together. This is used
  // for ordering different ComputeBlockOutput operations inside each blocking
  // iteration so as to potentially reduce the working set size.
  static inline void ComputeBlockSizes(const ConstMatrixMap& left,
                                       const ConstMatrixMap& right,
                                       bool transpose_left, int num_threads,
                                       int* KR, int* NR, int* KL, int* JB,
                                       int* IB);

  bool transpose_a_;
  bool transpose_b_;
  bool a_is_sparse_;
  bool b_is_sparse_;
  TF_DISALLOW_COPY_AND_ASSIGN(SparseMatMulOp);
};

inline void SparseMatMulOp::ComputeOutputBlock(
    const std::vector<SparseSlice*>& left, const ConstMatrixMap& right,
    int num_cols, int output_row_offset, int output_col_offset, bool assign,
    bool transpose_output, MatrixMap* output) {
  static const Eigen::array<int, 2> perm({1, 0});
  int num_rows = left[0]->num_rows;
  const int rhs_num_cols = right.dimension(1);
  DCHECK_LE(num_cols, rhs_num_cols);
  Matrix out(num_rows, rhs_num_cols);
  out.setZero();
  if (num_cols == N) {
    GEPP<N>(left, right, num_cols, &out);
  } else {
    GEPP<-1>(left, right, num_cols, &out);
  }
  if (!assign) {
    const Eigen::array<int, 2> begin = {output_row_offset, output_col_offset};
    const Eigen::array<int, 2> sizes = {num_rows, num_cols};
    if (transpose_output) {
      if (num_cols == rhs_num_cols) {
        output->shuffle(perm).slice(begin, sizes) += out;
      } else {
        static const Eigen::array<int, 2> zero = {0, 0};
        output->shuffle(perm).slice(begin, sizes) += out.slice(zero, sizes);
      }
    } else {
      if (num_cols == rhs_num_cols) {
        output->slice(begin, sizes) += out;
      } else {
        static const Eigen::array<int, 2> zero = {0, 0};
        output->slice(begin, sizes) += out.slice(zero, sizes);
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
    const Matrix& final_out = transpose_output ? *out_tr : out;
    for (int i = 0; i < num_rows; ++i) {
      memcpy(&(*output)(output_row_offset + i, output_col_offset),
             &final_out(i, 0), num_cols * sizeof(float));
    }
  }
}

inline BlockingCounter* SparseMatMulOp::CreateSparseSlices(
    const ConstMatrixMap& mat, bool transpose, int slice_num_rows,
    int slice_block_size, int slice_num_cols,
    std::vector<std::vector<SparseSlice*>>* mat_slices,
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
  auto work = [counter, transpose](SparseSlice* sparse_slice,
                                   ConstMatrixMap* slice, int col_offset) {
    if (transpose) {
      sparse_slice->Initialize<true>(*slice, col_offset);
    } else {
      sparse_slice->Initialize<false>(*slice, col_offset);
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
      ConstMatrixMap* slice = nullptr;
      if (transpose) {
        slice =
            new ConstMatrixMap(&mat(0, i * slice_num_rows), mat.dimensions());
      } else {
        DSizes d(num_rows, mat_num_cols);
        slice = new ConstMatrixMap(&mat(i * slice_num_rows, 0), d);
      }
      SparseSlice* sparse_slice =
          new SparseSlice(num_rows, num_cols, slice_block_size);
      (*mat_slices)[i][j] = sparse_slice;
      thread_pool->workers->Schedule(
          std::bind(work, sparse_slice, slice, slice_num_cols * j));
    }
  }
  return counter;
}

inline BlockingCounter* SparseMatMulOp::ShuffleMatrix(
    const ConstMatrixMap& mat, int slice_row_start, int slice_num_rows,
    int slice_col_start, int slice_num_cols, const int N,
    const DeviceBase::CpuWorkerThreads* thread_pool, Matrix* buffer) {
  int num_threads = std::min(thread_pool->num_threads, 16);
  BlockingCounter* counter = new BlockingCounter(num_threads);
  DCHECK_EQ(N, buffer->dimension(1));
  auto shuffle_work = [&mat, slice_row_start, slice_num_rows, slice_col_start,
                       slice_num_cols, N, buffer, counter](int s, int e) {
    const int row_start = s % slice_num_rows + slice_row_start;
    const int col_start = s / slice_num_rows * N + slice_col_start;
    float* out_start = &(*buffer)(s, 0);
    const float* input_start = &mat(row_start, col_start);
    const float* input_end = &mat(slice_row_start + slice_num_rows - 1,
                                  slice_col_start + slice_num_cols - 1);
    const int mat_num_cols = mat.dimension(1);
    const int row_slice_size = slice_num_rows * mat_num_cols;

    const int aligned_end = slice_num_cols / N * slice_num_rows;
    const int e1 = std::min(e, aligned_end);
    while (s < e1) {
      memcpy(out_start, input_start, N * sizeof(float));
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
      memcpy(out_start, input_start, copy_num_cols * sizeof(float));
      out_start += N;
      input_start += mat_num_cols;
      ++s1;
    }
    if (counter) counter->DecrementCount();
  };

  int start = 0;
  int end = 0;
  int num_out_rows = (slice_num_cols + N - 1) / N * slice_num_rows;
  DCHECK_LE(num_out_rows, buffer->dimension(0));
  for (int i = std::max(1, num_threads); i > 0; --i) {
    end = start + num_out_rows / i;
    thread_pool->workers->Schedule(std::bind(shuffle_work, start, end));
    num_out_rows -= (end - start);
    start = end;
  }
  return counter;
}

inline void SparseMatMulOp::SliceMatrix(const Matrix& mat, const int num_rows,
                                        const int num_slices,
                                        std::vector<ConstMatrixMap*>* slices) {
  slices->resize(num_slices);
  DSizes d(num_rows, mat.dimension(1));
  DCHECK_LE(num_rows * num_slices, mat.dimension(0));
  for (int i = 0; i < num_slices; ++i) {
    (*slices)[i] = new ConstMatrixMap(&mat(i * num_rows, 0), d);
  }
}

inline BlockingCounter* SparseMatMulOp::CreateDenseSlices(
    const ConstMatrixMap& mat, int row_start, int num_rows, int col_start,
    int num_cols, const DeviceBase::CpuWorkerThreads* thread_pool,
    Matrix* buffer, std::vector<ConstMatrixMap*>* slices) {
  BlockingCounter* shuffle_counter = ShuffleMatrix(
      mat, row_start, num_rows, col_start, num_cols, N, thread_pool, buffer);
  const int num_slices = (num_cols + N - 1) / N;
  SliceMatrix(*buffer, num_rows, num_slices, slices);
  return shuffle_counter;
}

inline void SparseMatMulOp::ComputeBlockSizes(const ConstMatrixMap& left,
                                              const ConstMatrixMap& right,
                                              bool transpose_left,
                                              int num_threads, int* KR, int* NR,
                                              int* KL, int* JB, int* IB) {
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

// Here is a an overview of the SparseMatMul code. Note that we assume that the
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
inline void SparseMatMulOp::SparseMatMul(
    const ConstMatrixMap& left, const ConstMatrixMap& right,
    bool transpose_left, const DeviceBase::CpuWorkerThreads* thread_pool,
    bool transpose_output, MatrixMap* output) {
  const int num_threads = thread_pool->num_threads;
  int KR, NR, KL, JB, IB;
  ComputeBlockSizes(left, right, transpose_left, num_threads, &KR, &NR, &KL,
                    &JB, &IB);

  // Slice the left matrix
  std::vector<std::vector<SparseSlice*>> left_slices;
  std::unique_ptr<BlockingCounter> sparse_slice_counter;
  sparse_slice_counter.reset(
      CreateSparseSlices(ConstMatrixMap(left.data(), left.dimensions()),
                         transpose_left, M, K, KL, &left_slices, thread_pool));
  const int num_left_slices = left_slices.size();

  const int right_dim0 = right.dimension(0);
  const int right_dim1 = right.dimension(1);
  // Allocate buffer for storing slices of right matrix.
  // Note buffer needs enough space to hold at most a KR * NR matrix since that
  // is the block size per iteration.
  const int buffer_num_rows =
      std::min(KR, right_dim0) * (std::min(NR, right_dim1) + N - 1) / N;
  Matrix buffer(buffer_num_rows, N);
  std::vector<ConstMatrixMap*> right_slices;

  std::vector<SparseSlice*> block_left_slices;
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
      dense_slice_counter.reset(CreateDenseSlices(
          right, kb * KR, right_num_rows, nb * NR, right_num_cols, thread_pool,
          &buffer, &right_slices));
      const int num_right_slices = right_slices.size();
      tasks.reserve(num_left_slices * num_right_slices);
      for (int j_outer = 0; j_outer < num_right_slices; j_outer += JB) {
        for (int i_outer = 0; i_outer < num_left_slices; i_outer += IB) {
          for (int j_inner = j_outer;
               j_inner < std::min(num_right_slices, j_outer + JB); ++j_inner) {
            const int num_cols = std::min(N, right_num_cols - N * j_inner);
            for (int i_inner = i_outer;
                 i_inner < std::min(num_left_slices, i_outer + IB); ++i_inner) {
              // Figure out which left slices to use.
              block_left_slices.clear();
              int begin = kb * KR / KL;
              int end = std::min<int>((kb + 1) * KR / KL,
                                      (right.dimension(0) + KL - 1) / KL);
              DCHECK_LT(begin, end);
              block_left_slices.insert(block_left_slices.begin(),
                                       left_slices[i_inner].begin() + begin,
                                       left_slices[i_inner].begin() + end);
              tasks.push_back(std::bind(
                  &SparseMatMulOp::ComputeOutputBlock, block_left_slices,
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
      gtl::STLDeleteElements(&right_slices);
      right_slices.clear();
    }
  }
  for (auto& left_slice : left_slices) {
    gtl::STLDeleteElements(&left_slice);
  }
}

REGISTER_KERNEL_BUILDER(Name("SparseMatMul").Device(DEVICE_CPU),
                        SparseMatMulOp);

}  // end namespace tensorflow
