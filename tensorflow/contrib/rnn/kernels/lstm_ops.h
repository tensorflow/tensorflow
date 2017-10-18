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

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_RNN_KERNELS_LSTM_OPS_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_RNN_KERNELS_LSTM_OPS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/contrib/rnn/kernels/blas_gemm.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/eigen_activations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
class OpKernelContext;

namespace functor {

template <typename Device, typename T>
struct TensorZero {
  void operator()(const Device& d, typename TTypes<T>::Flat t) {
    t.device(d) = t.constant(T(0));
  }
};

template <typename Device, typename T>
struct TensorUnalignedZero {
  void operator()(const Device& d, typename TTypes<T>::UnalignedFlat t) {
    t.device(d) = t.constant(T(0));
  }
};

template <typename Device, typename T>
struct TensorCopy {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat src,
                  typename TTypes<T>::Flat dst) {
    dst.device(d) = src;
  }
};

template <typename Device, typename T>
struct TensorCopyUnaligned {
  void operator()(const Device& d, typename TTypes<T>::UnalignedConstFlat src,
                  typename TTypes<T>::Flat dst) {
    dst.device(d) = src;
  }
};

template <typename Device, typename T>
struct TensorCopyToUnaligned {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat src,
                  typename TTypes<T>::UnalignedFlat dst) {
    dst.device(d) = src;
  }
};

template <typename Device, typename T>
struct TensorAdd {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat a,
                  typename TTypes<T>::ConstFlat b, typename TTypes<T>::Flat c) {
    c.device(d) = a + b;
  }
};

template <typename Device, typename T>
struct TensorZeroPadding {
  void operator()(const Device& d, const int64 time_idx,
                  typename TTypes<int64>::ConstVec seq_len,
                  typename TTypes<float>::Vec mask,
                  typename TTypes<float>::Matrix m) {
    // mask is shape [batch_size].
    mask.device(d) = seq_len.constant(time_idx) < seq_len;

    // m_shape is [batch_size, 1].
    Eigen::array<Eigen::DenseIndex, 2> m_shape({m.dimensions()[0], 1});
    // broadcast_shape is [1, units].
    Eigen::array<Eigen::DenseIndex, 2> broadcast_shape({1, m.dimensions()[1]});

    // m is shape [batch_size, units].
    m.device(d) = m * mask.reshape(m_shape).broadcast(broadcast_shape);
  }
};


struct LSTMBlockCell {
  LSTMBlockCell(const int batch_size, const int input_size, const int cell_size)
      : batch_size_(batch_size),
        input_size_(input_size),
        cell_size_(cell_size) {}

  int batch_size() const { return batch_size_; }

  int input_size() const { return input_size_; }

  int cell_size() const { return cell_size_; }

  inline Eigen::array<Eigen::DenseIndex, 2> icfo_i_offsets() const {
    return {0, 0};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> icfo_c_offsets() const {
    return {0, cell_size_};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> icfo_f_offsets() const {
    return {0, cell_size_ * 2};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> icfo_o_offsets() const {
    return {0, cell_size_ * 3};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> cell_extents() const {
    return {batch_size_, cell_size_};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> xh_x_offsets() const {
    return {0, 0};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> xh_x_extents() const {
    return {batch_size_, input_size_};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> xh_h_offsets() const {
    return {0, input_size_};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> xh_h_extents() const {
    return {batch_size_, cell_size_};
  }

 protected:
  const int batch_size_;
  const int input_size_;
  const int cell_size_;
};

// See lstm_ops.cc for CPUDevice implementation and lstm_ops_gpu.cu.cc for
// GPUDevice implementation.
template <typename Device, typename T, bool USE_CUBLAS>
struct LSTMBlockCellFprop : public LSTMBlockCell {
  LSTMBlockCellFprop(const int batch_size, const int input_size,
                     const int cell_size)
      : LSTMBlockCell(batch_size, input_size, cell_size) {}

  void operator()(
      OpKernelContext* ctx, const Device& d, const T forget_bias,
      const T cell_clip, bool use_peephole, typename TTypes<T>::ConstMatrix x,
      typename TTypes<T>::ConstMatrix cs_prev,
      typename TTypes<T>::ConstMatrix h_prev, typename TTypes<T>::ConstMatrix w,
      typename TTypes<T>::ConstVec wci, typename TTypes<T>::ConstVec wcf,
      typename TTypes<T>::ConstVec wco, typename TTypes<T>::ConstVec b,
      typename TTypes<T>::Matrix xh, typename TTypes<T>::Matrix i,
      typename TTypes<T>::Matrix cs, typename TTypes<T>::Matrix f,
      typename TTypes<T>::Matrix o, typename TTypes<T>::Matrix ci,
      typename TTypes<T>::Matrix co, typename TTypes<T>::Matrix icfo,
      typename TTypes<T>::Matrix h);
};

// See lstm_ops.cc for CPUDevice implementation and lstm_ops_gpu.cu.cc for
// GPUDevice implementation.
template <typename Device, typename T, bool USE_CUBLAS>
struct LSTMBlockCellBprop : public LSTMBlockCell {
  LSTMBlockCellBprop(const int batch_size, const int input_size,
                     const int cell_size)
      : LSTMBlockCell(batch_size, input_size, cell_size) {}

  void operator()(
      OpKernelContext* ctx, const Device& d, bool use_peephole,
      typename TTypes<T>::ConstMatrix x,
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
      typename TTypes<T>::Vec wco_grad);
};

template <typename Device, typename T, bool USE_CUBLAS>
struct BlockLSTMBprop : public LSTMBlockCell {
  BlockLSTMBprop(const int batch_size, const int input_size,
                 const int cell_size)
      : LSTMBlockCell(batch_size, input_size, cell_size) {}

  void operator()(
      OpKernelContext* ctx, const Device& d, bool use_peephole,
      typename TTypes<T>::ConstMatrix x,
      typename TTypes<T>::ConstMatrix cs_prev,
      typename TTypes<T>::ConstMatrix h_prev, typename TTypes<T>::ConstMatrix w,
      typename TTypes<T>::ConstVec wci, typename TTypes<T>::ConstVec wcf,
      typename TTypes<T>::ConstVec wco, typename TTypes<T>::ConstVec b,
      typename TTypes<T>::Matrix xh, typename TTypes<T>::ConstMatrix i,
      typename TTypes<T>::ConstMatrix cs, typename TTypes<T>::ConstMatrix f,
      typename TTypes<T>::ConstMatrix o, typename TTypes<T>::ConstMatrix ci,
      typename TTypes<T>::ConstMatrix co,
      typename TTypes<T>::ConstMatrix cs_grad,
      typename TTypes<T>::ConstMatrix h_grad, typename TTypes<T>::Matrix do_,
      typename TTypes<T>::Matrix dcs, typename TTypes<T>::Matrix dci,
      typename TTypes<T>::Matrix df, typename TTypes<T>::Matrix di,
      typename TTypes<T>::Matrix dicfo, typename TTypes<T>::Matrix cs_prev_grad,
      typename TTypes<T>::Matrix h_prev_grad,
      typename TTypes<T>::Matrix xh_grad, typename TTypes<T>::Matrix x_grad,
      typename TTypes<T>::Matrix w_grad, typename TTypes<T>::Vec wci_grad,
      typename TTypes<T>::Vec wcf_grad, typename TTypes<T>::Vec wco_grad,
      typename TTypes<T>::Vec b_grad) {
    // do[t] = sigm'(o[t]) .* dh[t] .* co[t]
    do_.device(d) = o * (o.constant(T(1)) - o) * h_grad * co;

    // dcs[t] += tanh'(cs[t]) .* dh[t] .* o[t] + dcs[t + 1] .* f[t + 1]
    dcs.device(d) = (co.constant(T(1)) - co * co) * h_grad * o + cs_grad;

    Eigen::array<Eigen::DenseIndex, 2> p_shape({1, cell_size_});
    Eigen::array<Eigen::DenseIndex, 2> p_broadcast_shape({batch_size_, 1});
    if (use_peephole) {
      dcs.device(d) =
          dcs + do_ * wco.reshape(p_shape).broadcast(p_broadcast_shape);
    }

    // dci[t] = tanh'(ci[t]) dcs[t] i[t]
    dci.device(d) = (ci.constant(T(1)) - ci * ci) * dcs * i;

    // df[t] = sigm'(f[t]) dcs[t] cs[t - 1]
    df.device(d) = f * (f.constant(T(1)) - f) * dcs * cs_prev;

    // di[t] = sigm'(i[t]) dcs[t] ci[t]
    di.device(d) = i * (i.constant(T(1)) - i) * dcs * ci;

    dicfo.slice(icfo_i_offsets(), cell_extents()).device(d) = di;
    dicfo.slice(icfo_c_offsets(), cell_extents()).device(d) = dci;
    dicfo.slice(icfo_f_offsets(), cell_extents()).device(d) = df;
    dicfo.slice(icfo_o_offsets(), cell_extents()).device(d) = do_;

    cs_prev_grad.device(d) = dcs * f;
    if (use_peephole) {
      cs_prev_grad.device(d) =
          cs_prev_grad +
          di * wci.reshape(p_shape).broadcast(p_broadcast_shape) +
          df * wcf.reshape(p_shape).broadcast(p_broadcast_shape);
    }

    // xh_grad.
    typename TTypes<T>::ConstMatrix const_dicfo(dicfo.data(),
                                                dicfo.dimensions());
    TensorBlasGemm<Device, T, USE_CUBLAS>::compute(
        ctx, d, false, true, T(1), const_dicfo, w, T(0), xh_grad);

    // xh.
    xh.slice(xh_x_offsets(), xh_x_extents()).device(d) = x;
    xh.slice(xh_h_offsets(), xh_h_extents()).device(d) = h_prev;
    typename TTypes<T>::ConstMatrix const_xh(xh.data(), xh.dimensions());

    // x_grad.
    x_grad.device(d) = xh_grad.slice(xh_x_offsets(), xh_x_extents());
    h_prev_grad.device(d) = xh_grad.slice(xh_h_offsets(), xh_h_extents());

    // w_grad.
    TensorBlasGemm<Device, T, USE_CUBLAS>::compute(
        ctx, d, true, false, T(1), const_xh, const_dicfo, T(1), w_grad);

    // b_grad.
    b_grad.device(d) += dicfo.sum(Eigen::array<int, 1>({0}));

    if (use_peephole) {
      wci_grad.device(d) += (di * cs_prev).sum(Eigen::array<int, 1>({0}));
      wcf_grad.device(d) += (df * cs_prev).sum(Eigen::array<int, 1>({0}));
      wco_grad.device(d) += (do_ * cs).sum(Eigen::array<int, 1>({0}));
    }
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_RNN_KERNELS_LSTM_OPS_H_
