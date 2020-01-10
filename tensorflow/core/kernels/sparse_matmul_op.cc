// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/port.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename T>
void PrefetchBlockNTA(const T& tensor, int si, int ei, int sj, int ej) {
  for (int i = si; i < ei; ++i) {
    for (int j = sj; j < ej; j = j + 16) {
      port::prefetch<port::PREFETCH_HINT_NTA>(&tensor(i, j));
    }
  }
}

template <typename T>
void PrefetchBlockT1(const T& tensor, int si, int ei, int sj, int ej) {
  for (int i = si; i < ei; ++i) {
    for (int j = sj; j < ej; j = j + 16) {
      port::prefetch<port::PREFETCH_HINT_T1>(&tensor(i, j));
    }
  }
}

struct Block {
  Block(int sm, int em, int sk, int ek, int sn, int en)
      : startm(sm), endm(em), startk(sk), endk(ek), startn(sn), endn(en) {}

  int startm;
  int endm;
  int startk;
  int endk;
  int startn;
  int endn;
};

bool NextBlock(const int Bm, const int Bk, const int Bn, const int m_start,
               const int m, const int k, const int n, const Block& b,
               Block* next) {
  *next = b;
  if (b.endk < k) {
    next->startk = b.endk;
    next->endk = std::min(b.endk + Bk, k);
  } else {
    next->startk = 0;
    next->endk = std::min(Bk, k);
    if (b.endm < m) {
      next->startm = b.endm;
      next->endm = std::min(b.endm + Bm, m);
    } else {
      next->startm = m_start;
      next->endm = std::min(m_start + Bm, m);
      next->startn = b.endn;
      next->endn = std::min(b.endn + Bn, n);
    }
  }
  return next->startn == next->endn;
}

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
    auto right_mat = b.matrix<float>();
    const int m = transpose_a_ ? left.dimension(1) : left.dimension(0);
    const int k = transpose_a_ ? left.dimension(0) : left.dimension(1);
    const int n =
        transpose_b_ ? right_mat.dimension(0) : right_mat.dimension(1);
    const int k2 =
        transpose_b_ ? right_mat.dimension(1) : right_mat.dimension(0);

    OP_REQUIRES(ctx, k == k2,
                errors::InvalidArgument("Matrix size incompatible: a: ",
                                        a.shape().DebugString(), ", b: ",
                                        b.shape().DebugString()));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({m, n}), &output));
    auto out = output->matrix<float>();

    if (!a_is_sparse_) {
      // Fallback to Eigen contract.
      // Note that we currently don't optimize the case where only right is
      // sparse. That can generally be handled by tranposing the order of the
      // matmul.
      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0].first = transpose_a_ ? 0 : 1;
      dim_pair[0].second = transpose_b_ ? 1 : 0;
      out.device(ctx->template eigen_device<CPUDevice>()) =
          left.contract(right_mat, dim_pair);
      return;
    }
    typedef Eigen::Tensor<float, 2, Eigen::RowMajor> Matrix;
    std::unique_ptr<Matrix> right_tr_mat;
    std::unique_ptr<TTypes<float>::ConstMatrix> right_tr_map;
    if (transpose_b_) {
      right_tr_mat.reset(new Matrix(k, n));
      Eigen::array<int, 2> perm({1, 0});
      right_tr_mat->device(ctx->template eigen_device<CPUDevice>()) =
          right_mat.shuffle(perm);
      right_tr_map.reset(new TTypes<float>::ConstMatrix(
          right_tr_mat->data(), right_tr_mat->dimensions()));
    }
    TTypes<float>::ConstMatrix& right =
        transpose_b_ ? *right_tr_map : right_mat;

    const bool transpose_a = transpose_a_;

    typedef Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
                             Eigen::Unaligned> TensorMap;
    typedef Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>,
                             Eigen::Unaligned> ConstTensorMap;
    typedef Eigen::DSizes<Eigen::DenseIndex, 1> DSizes;
    const int Bm = 16;
    const int Bk = 16;
    const int Bn = 1024;

    auto work_shard = [m, n, k, transpose_a, Bm, Bk, Bn, &left, &right, &out](
        int64 start64, int64 end64) {
      const int start = static_cast<int>(start64);
      const int end = static_cast<int>(end64);
      Block curr(start, std::min(start + Bm, end), 0, std::min(Bk, k), 0,
                 std::min(Bn, n));
      Block next(curr);
      bool done = false;
      for (int i = start; i < end; ++i) {
        out.chip<0>(i).setZero();
      }
      while (true) {
        done = NextBlock(Bm, Bk, Bn, start, end, k, n, curr, &next);

        PrefetchBlockT1(right, curr.startk, curr.endk, curr.startn, curr.endn);

        // Process current block
        for (int i = curr.startm; i < curr.endm; ++i) {
          PrefetchBlockNTA(left, i, i + 1, curr.startk, curr.endk);
          PrefetchBlockNTA(out, i, i + 1, curr.startn, curr.endn);
          DSizes out_slice_shape(curr.endn - curr.startn);
          TensorMap out_i(&out(i, curr.startn), out_slice_shape);
          for (int j = curr.startk; j < curr.endk; ++j) {
            const float l = transpose_a ? left(j, i) : left(i, j);
            if (l == 0) continue;
            ConstTensorMap right_j(&right(j, curr.startn), out_slice_shape);
            out_i += right_j * l;
          }
        }
        if (done) break;
        curr = next;
      }
    };
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, m, 2 * k * n,
          work_shard);
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
  bool a_is_sparse_;
  bool b_is_sparse_;
  TF_DISALLOW_COPY_AND_ASSIGN(SparseMatMulOp);
};

REGISTER_KERNEL_BUILDER(Name("SparseMatMul").Device(DEVICE_CPU),
                        SparseMatMulOp);

}  // end namespace tensorflow
