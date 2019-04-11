/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/contrib/rnn/kernels/lstm_ops.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/eigen_activations.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

namespace {

struct FloatToHalf {
  __host__ __device__ EIGEN_STRONG_INLINE Eigen::half operator()(
      const float& x) const {
    return Eigen::half_impl::float_to_half_rtne(x);
  }
};

template <typename U, typename T>
__host__ __device__ EIGEN_STRONG_INLINE
    typename std::enable_if<!std::is_same<T, U>::value, U>::type
    strict_cast(T t);

template <typename U, typename T>
__host__ __device__ EIGEN_STRONG_INLINE
    typename std::enable_if<std::is_same<T, U>::value, U>::type
    strict_cast(T t) {
  return t;
}

template <>
__host__ __device__ EIGEN_STRONG_INLINE Eigen::half
strict_cast<Eigen::half, float>(float t) {
  return FloatToHalf()(t);
}

}  // namespace

template <typename T>
struct TensorZero<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat t) {
    t.device(d) = t.constant(strict_cast<T>(0.f));
  }
};

template <typename T>
struct TensorUnalignedZero<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::UnalignedFlat t) {
    t.device(d) = t.constant(strict_cast<T>(0.f));
  }
};

namespace {

// Adds bias, applies non-linearities and gates.
//
// Launch with a 2D setup such that there is one thread per (example,
// activation) with 'x' governing example index and 'y' governing activation.
//
// Launch with blocks of (batch x 32)
//
// TODO(b/67600500): Try making 'use_peephole' a template parameter.
template <typename T, bool use_peephole>
__global__ void lstm_gates(const T* icfo, const T* b, const T* cs_prev,
                           const T* wci, const T* wcf, const T* wco, T* o, T* h,
                           T* ci, T* cs, T* co, T* i, T* f,
                           const float forget_bias, const float cell_clip,
                           const int batch_size, const int cell_size) {
  const int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int act_id = blockIdx.y * blockDim.y + threadIdx.y;

  T forget_bias_t = strict_cast<T>(forget_bias);
  T cell_clip_t = strict_cast<T>(cell_clip);

  if (batch_id >= batch_size || act_id >= cell_size) return;

  // The following code assumes the input arrays are of the following
  // shapes and interpretations.
  //
  // 1) 'icfo' is a matrix such that,
  //
  //   cell_size  cell_size  cell_size  cell_size
  //  +----------+----------+----------+----------+
  //  |          |          |          |          |
  //  |    i     |    c     |    f     |    o     |  batch_size
  //  |          |          |          |          |
  //  +----------+----------+----------+----------+
  //
  // 'gid' is the index assigned to this thread for 'icfo' in the 'i' submatrix.
  //
  // 2) 'b' is a vector such that,
  //
  //   cell_size  cell_size  cell_size  cell_size
  //  +----------+----------+----------+----------+
  //  |    i     |    c     |    f     |    o     |  1
  //  +----------+----------+----------+----------+
  //
  // 'act_id' is the index assigned to this thread for 'b' in the 'i' subvector.
  //
  // 3) 'wc{i,f,o}' are vectors such that,
  //
  //   cell_size
  //  +----------+
  //  |    i     |  1
  //  +----------+
  //
  //  'act_id' is the index to this thread.
  //
  // 4) All other matrices have the form,
  //
  //   cell_size
  //  +----------+
  //  |          |
  //  |    i     |  batch_size
  //  |          |
  //  +----------+
  //
  // 'cid' is the index assigned to this thread.
  //
  const int gid = batch_id * cell_size * 4 + act_id;
  const int cid = batch_id * cell_size + act_id;
  Eigen::internal::scalar_logistic_op<T> sigmoid_op;
  Eigen::internal::scalar_tanh_op<T> tanh_op;
  Eigen::scalar_clip_op<T> clip_op;

  T i_local;
  if (use_peephole) {
    i_local = sigmoid_op(icfo[0 * cell_size + gid] + b[0 * cell_size + act_id] +
                         cs_prev[cid] * wci[act_id]);
  } else {
    i_local = sigmoid_op(icfo[0 * cell_size + gid] + b[0 * cell_size + act_id]);
  }
  i[cid] = i_local;

  const T ci_local =
      tanh_op(icfo[1 * cell_size + gid] + b[1 * cell_size + act_id]);
  ci[cid] = ci_local;

  T f_local;
  if (use_peephole) {
    f_local = sigmoid_op(icfo[2 * cell_size + gid] + b[2 * cell_size + act_id] +
                         forget_bias_t + cs_prev[cid] * wcf[act_id]);
  } else {
    f_local = sigmoid_op(icfo[2 * cell_size + gid] + b[2 * cell_size + act_id] +
                         forget_bias_t);
  }
  f[cid] = f_local;

  T cs_local = i_local * ci_local + f_local * cs_prev[cid];
  if (cell_clip > 0.0f) {
    cs_local = clip_op(cs_local, cell_clip_t);
  }
  cs[cid] = cs_local;

  const T co_local = tanh_op(cs_local);
  co[cid] = co_local;

  T o_local;
  if (use_peephole) {
    o_local = sigmoid_op(icfo[3 * cell_size + gid] + b[3 * cell_size + act_id] +
                         cs_local * wco[act_id]);
  } else {
    o_local = sigmoid_op(icfo[3 * cell_size + gid] + b[3 * cell_size + act_id]);
  }
  o[cid] = o_local;

  h[cid] = o_local * co_local;
}

// Concatenate 'x' and 'h' and copy their contents into 'xh'.
template <typename T>
__global__ void concat_xh(T* xh, const T* x, const T* h_prev,
                          const int batch_size, const int cell_size,
                          const int input_size) {
  // Assumes 'x', 'h', and 'xh' are of the following shape,
  //
  //   input_size  cell_size
  //  +----------+----------+
  //  |          |          |
  //  |    x     |    h     |  batch_size
  //  |          |          |
  //  +----------+----------+
  //
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const int width = input_size + cell_size;

  if (gid >= width * batch_size) return;

  const int output_row = gid / width;
  const int output_col = gid % width;

  if (output_col < input_size) {  // x
    xh[gid] = x[output_row * input_size + output_col];
  } else {  // h
    xh[gid] = h_prev[output_row * cell_size + output_col - input_size];
  }
}

template <typename T>
void LSTMBlockCellFpropWithCUDA(
    OpKernelContext* ctx, const GPUDevice& d, const float forget_bias,
    const float cell_clip, bool use_peephole, typename TTypes<T>::ConstMatrix x,
    typename TTypes<T>::ConstMatrix cs_prev,
    typename TTypes<T>::ConstMatrix h_prev, typename TTypes<T>::ConstMatrix w,
    typename TTypes<T>::ConstVec wci, typename TTypes<T>::ConstVec wcf,
    typename TTypes<T>::ConstVec wco, typename TTypes<T>::ConstVec b,
    typename TTypes<T>::Matrix xh, typename TTypes<T>::Matrix i,
    typename TTypes<T>::Matrix cs, typename TTypes<T>::Matrix f,
    typename TTypes<T>::Matrix o, typename TTypes<T>::Matrix ci,
    typename TTypes<T>::Matrix co, typename TTypes<T>::Matrix icfo,
    typename TTypes<T>::Matrix h, int batch_size, int cell_size,
    int input_size) {
  const cudaStream_t& cu_stream = GetCudaStream(ctx);

  // Concatenate xh = [x, h].
  //
  // Each block is assigned 128 threads. Good values are in [128, 1024] and are
  // divisible by 32 (the size of a warp). The number of blocks is such that
  // there are enough to process all the data.
  const int block_dim = 128;
  const int grid_dim =
      Eigen::divup(batch_size * (cell_size + input_size), block_dim);
  TF_CHECK_OK(CudaLaunchKernel(concat_xh<T>, grid_dim, block_dim, 0, cu_stream,
                               xh.data(), x.data(), h_prev.data(), batch_size,
                               cell_size, input_size));

  // states1 = xh * w
  typename TTypes<T>::ConstMatrix const_xh(xh.data(), xh.dimensions());
  TensorBlasGemm<GPUDevice, T, true /* USE_CUBLAS */>::compute(
      ctx, d, false, false, typename gemm_compute_type<T>::type(1.f), const_xh,
      w, typename gemm_compute_type<T>::type(0.f), icfo);

  // Add bias, apply non-linearities and gating.
  //
  // Use 2D blocks. The number of threads per block is equal to x * y, where x =
  // min(batch_size, 8) and y = 32. See above for guidance on number of
  // threads.
  dim3 block_dim_2d(std::min(batch_size, 8), 32);
  dim3 grid_dim_2d(Eigen::divup(batch_size, static_cast<int>(block_dim_2d.x)),
                   Eigen::divup(cell_size, static_cast<int>(block_dim_2d.y)));

  if (use_peephole) {
    TF_CHECK_OK(CudaLaunchKernel(
        lstm_gates<T, true>, grid_dim_2d, block_dim_2d, 0, cu_stream,
        icfo.data(), b.data(), cs_prev.data(), wci.data(), wcf.data(),
        wco.data(), o.data(), h.data(), ci.data(), cs.data(), co.data(),
        i.data(), f.data(), forget_bias, cell_clip, batch_size, cell_size));
  } else {
    TF_CHECK_OK(CudaLaunchKernel(
        lstm_gates<T, false>, grid_dim_2d, block_dim_2d, 0, cu_stream,
        icfo.data(), b.data(), cs_prev.data(), wci.data(), wcf.data(),
        wco.data(), o.data(), h.data(), ci.data(), cs.data(), co.data(),
        i.data(), f.data(), forget_bias, cell_clip, batch_size, cell_size));
  }
}

template <typename T>
__global__ void lstm_gates_bprop(
    const T* cs_prev,  // [batch_size, cell_size]
    const T* h_prev,   // [batch_size, cell_size]
    const T* w,        // [input_size + cell_size, 4 * cell_size]
    const T* wci,      // [cell_size]
    const T* wcf,      // [cell_size]
    const T* wco,      // [cell_size]
    const T* b,        // [4 * cell_size]
    const T* i,        // [batch_size, cell_size]
    const T* cs,       // [batch_size, cell_size]
    const T* f,        // [batch_size, cell_size]
    const T* o,        // [batch_size, cell_size]
    const T* ci,       // [batch_size, cell_size]
    const T* co,       // [batch_size, cell_size]
    const T* cs_grad,  // [batch_size, cell_size]
    const T* h_grad,   // [batch_size, cell_size]
    T* do_,            // [batch_size, cell_size]
    T* dcs,            // [batch_size, cell_size]
    T* dci,            // [batch_size, cell_size]
    T* df,             // [batch_size, cell_size]
    T* di,             // [batch_size, cell_size]
    T* dicfo,          // [input_size + cell_size, 4 * cell_size]
    T* cs_prev_grad,   // [batch_size, cell_size]
    const int batch_size, const int cell_size, const bool use_peephole) {
  const int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int act_id = blockIdx.y * blockDim.y + threadIdx.y;

  if (batch_id >= batch_size || act_id >= cell_size) return;

  const int gid = batch_id * cell_size * 4 + act_id;
  const int cid = batch_id * cell_size + act_id;

  const T one = static_cast<T>(1.0f);

  // do[t] = sigm'(o[t]) .* dh[t] .* co[t]
  const T o_local = o[cid];
  const T h_grad_local = h_grad[cid];
  const T co_local = co[cid];
  const T ci_local = ci[cid];
  const T do_local = o_local * (one - o_local) * h_grad_local * co_local;
  const T i_local = i[cid];
  const T f_local = f[cid];

  do_[cid] = do_local;

  // dcs[t] += tanh'(cs[t]) .* dh[t] .* o[t] + dcs[t + 1] .* f[t + 1]
  T dcs_local =
      (one - co_local * co_local) * h_grad_local * o_local + cs_grad[cid];
  if (use_peephole) {
    dcs_local += do_local * wco[act_id];
  }
  dcs[cid] = dcs_local;

  // dci[t] = tanh'(ci[t]) dcs[t] i[t]
  const T dci_local = (one - ci_local * ci_local) * dcs_local * i_local;
  dci[cid] = dci_local;

  // df[t] = sigm'(f[t]) dcs[t] cs[t - 1]
  const T df_local = f_local * (one - f_local) * dcs_local * cs_prev[cid];
  df[cid] = df_local;

  // di[t] = sigm'(i[t]) dcs[t] ci[t]
  const T di_local = i_local * (one - i_local) * dcs_local * ci_local;
  di[cid] = di_local;

  dicfo[gid + 0 * cell_size] = di_local;
  dicfo[gid + 1 * cell_size] = dci_local;
  dicfo[gid + 2 * cell_size] = df_local;
  dicfo[gid + 3 * cell_size] = do_local;

  cs_prev_grad[cid] = dcs_local * f_local;
  if (use_peephole) {
    cs_prev_grad[cid] += di_local * wci[act_id] + df_local * wcf[act_id];
  }
}

template <typename T>
void LSTMBlockCellBpropWithCUDA(
    OpKernelContext* ctx, const GPUDevice& d, typename TTypes<T>::ConstMatrix x,
    typename TTypes<T>::ConstMatrix cs_prev,
    typename TTypes<T>::ConstMatrix h_prev, typename TTypes<T>::ConstMatrix w,
    typename TTypes<T>::ConstVec wci, typename TTypes<T>::ConstVec wcf,
    typename TTypes<T>::ConstVec wco, typename TTypes<T>::ConstVec b,
    typename TTypes<T>::ConstMatrix i, typename TTypes<T>::ConstMatrix cs,
    typename TTypes<T>::ConstMatrix f, typename TTypes<T>::ConstMatrix o,
    typename TTypes<T>::ConstMatrix ci, typename TTypes<T>::ConstMatrix co,
    typename TTypes<T>::ConstMatrix cs_grad,
    typename TTypes<T>::ConstMatrix h_grad, typename TTypes<T>::Matrix do_,
    typename TTypes<T>::Matrix dcs, typename TTypes<T>::Matrix dci,
    typename TTypes<T>::Matrix df, typename TTypes<T>::Matrix di,
    typename TTypes<T>::Matrix dicfo, typename TTypes<T>::Matrix cs_prev_grad,
    typename TTypes<T>::Vec wci_grad, typename TTypes<T>::Vec wcf_grad,
    typename TTypes<T>::Vec wco_grad, const int batch_size, const int cell_size,
    const bool use_peephole) {
  const cudaStream_t& cu_stream = GetCudaStream(ctx);

  dim3 block_dim_2d(std::min(batch_size, 8), 32);
  dim3 grid_dim_2d(Eigen::divup(batch_size, static_cast<int>(block_dim_2d.x)),
                   Eigen::divup(cell_size, static_cast<int>(block_dim_2d.y)));

  TF_CHECK_OK(CudaLaunchKernel(
      lstm_gates_bprop<T>, grid_dim_2d, block_dim_2d, 0, cu_stream,
      cs_prev.data(), h_prev.data(), w.data(), wci.data(), wcf.data(),
      wco.data(), b.data(), i.data(), cs.data(), f.data(), o.data(), ci.data(),
      co.data(), cs_grad.data(), h_grad.data(), do_.data(), dcs.data(),
      dci.data(), df.data(), di.data(), dicfo.data(), cs_prev_grad.data(),
      batch_size, cell_size, use_peephole));

  if (use_peephole) {
    Eigen::array<Eigen::DenseIndex, 2> p_shape({1, cell_size});
    Eigen::array<Eigen::DenseIndex, 2> p_broadcast_shape({batch_size, 1});
    cs_prev_grad.device(d) =
        cs_prev_grad + di * wci.reshape(p_shape).broadcast(p_broadcast_shape) +
        df * wcf.reshape(p_shape).broadcast(p_broadcast_shape);
    wci_grad.device(d) = (di * cs_prev).sum(Eigen::array<int, 1>({0}));
    wcf_grad.device(d) = (df * cs_prev).sum(Eigen::array<int, 1>({0}));
    wco_grad.device(d) = (do_ * cs).sum(Eigen::array<int, 1>({0}));
  }
}

}  // namespace

#define DEFINE_GPU_SPECS(T)                                                    \
  template struct TensorZero<GPUDevice, T>;                                    \
  template struct TensorUnalignedZero<GPUDevice, T>;                           \
  template struct TensorCopy<GPUDevice, T>;                                    \
  template struct TensorCopyUnaligned<GPUDevice, T>;                           \
  template struct TensorCopyToUnaligned<GPUDevice, T>;                         \
  template struct TensorAdd<GPUDevice, T>;                                     \
  template <>                                                                  \
  void LSTMBlockCellFprop<GPUDevice, T, true /* USE_CUBLAS */>::operator()(    \
      OpKernelContext* ctx, const GPUDevice& d, const float forget_bias,       \
      const float cell_clip, bool use_peephole,                                \
      typename TTypes<T>::ConstMatrix x,                                       \
      typename TTypes<T>::ConstMatrix cs_prev,                                 \
      typename TTypes<T>::ConstMatrix h_prev,                                  \
      typename TTypes<T>::ConstMatrix w, typename TTypes<T>::ConstVec wci,     \
      typename TTypes<T>::ConstVec wcf, typename TTypes<T>::ConstVec wco,      \
      typename TTypes<T>::ConstVec b, typename TTypes<T>::Matrix xh,           \
      typename TTypes<T>::Matrix i, typename TTypes<T>::Matrix cs,             \
      typename TTypes<T>::Matrix f, typename TTypes<T>::Matrix o,              \
      typename TTypes<T>::Matrix ci, typename TTypes<T>::Matrix co,            \
      typename TTypes<T>::Matrix icfo, typename TTypes<T>::Matrix h) {         \
    LSTMBlockCellFpropWithCUDA<T>(ctx, d, forget_bias, cell_clip,              \
                                  use_peephole, x, cs_prev, h_prev, w, wci,    \
                                  wcf, wco, b, xh, i, cs, f, o, ci, co, icfo,  \
                                  h, batch_size_, cell_size_, input_size_);    \
  }                                                                            \
  template <>                                                                  \
  void LSTMBlockCellBprop<GPUDevice, T, true /* USE_CUBLAS */>::operator()(    \
      OpKernelContext* ctx, const GPUDevice& d, bool use_peephole,             \
      typename TTypes<T>::ConstMatrix x,                                       \
      typename TTypes<T>::ConstMatrix cs_prev,                                 \
      typename TTypes<T>::ConstMatrix h_prev,                                  \
      typename TTypes<T>::ConstMatrix w, typename TTypes<T>::ConstVec wci,     \
      typename TTypes<T>::ConstVec wcf, typename TTypes<T>::ConstVec wco,      \
      typename TTypes<T>::ConstVec b, typename TTypes<T>::ConstMatrix i,       \
      typename TTypes<T>::ConstMatrix cs, typename TTypes<T>::ConstMatrix f,   \
      typename TTypes<T>::ConstMatrix o, typename TTypes<T>::ConstMatrix ci,   \
      typename TTypes<T>::ConstMatrix co,                                      \
      typename TTypes<T>::ConstMatrix cs_grad,                                 \
      typename TTypes<T>::ConstMatrix h_grad, typename TTypes<T>::Matrix do_,  \
      typename TTypes<T>::Matrix dcs, typename TTypes<T>::Matrix dci,          \
      typename TTypes<T>::Matrix df, typename TTypes<T>::Matrix di,            \
      typename TTypes<T>::Matrix dicfo,                                        \
      typename TTypes<T>::Matrix cs_prev_grad,                                 \
      typename TTypes<T>::Vec wci_grad, typename TTypes<T>::Vec wcf_grad,      \
      typename TTypes<T>::Vec wco_grad) {                                      \
    LSTMBlockCellBpropWithCUDA<T>(                                             \
        ctx, d, x, cs_prev, h_prev, w, wci, wcf, wco, b, i, cs, f, o, ci, co,  \
        cs_grad, h_grad, do_, dcs, dci, df, di, dicfo, cs_prev_grad, wci_grad, \
        wcf_grad, wco_grad, batch_size_, cell_size_, use_peephole);            \
  }                                                                            \
  template struct LSTMBlockCellFprop<GPUDevice, T, true /* USE_CUBLAS */>;     \
  template struct LSTMBlockCellBprop<GPUDevice, T, true /* USE_CUBLAS */>;     \
  template struct BlockLSTMBprop<GPUDevice, T, true /* USE_CUBLAS */>;

DEFINE_GPU_SPECS(float);
DEFINE_GPU_SPECS(Eigen::half);
// DEFINE_GPU_SPECS(double);
#undef DEFINE_GPU_SPECS

}  // end namespace functor
}  // end namespace tensorflow
#endif  // GOOGLE_CUDA
