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

#ifndef TENSORFLOW_CORE_KERNELS_RNN_GRU_OPS_H_
#define TENSORFLOW_CORE_KERNELS_RNN_GRU_OPS_H_

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/rnn/blas_gemm.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class OpKernelContext;

namespace functor {

struct GRUCell {
  GRUCell(const int batch_size, const int input_size, const int cell_size)
      : batch_size_(batch_size),
        input_size_(input_size),
        cell_size_(cell_size) {}

  inline Eigen::array<Eigen::DenseIndex, 2> x_offsets() const { return {0, 0}; }

  inline Eigen::array<Eigen::DenseIndex, 2> x_extends() const {
    return {batch_size_, input_size_};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> h_offsets() const {
    return {0, input_size_};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> h_extends() const {
    return {batch_size_, cell_size_};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> ru_r_offset() const {
    return {0, 0};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> ru_u_offset() const {
    return {0, cell_size_};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> cell_extents() const {
    return {batch_size_, cell_size_};
  }

 protected:
  const int batch_size_;
  const int input_size_;
  const int cell_size_;
};

template <typename Device, typename T, bool USE_CUBLAS>
struct GRUBlockCellFprop : public GRUCell {
  GRUBlockCellFprop(const int batch_size, const int input_size,
                    const int cell_size)
      : GRUCell(batch_size, input_size, cell_size) {}

  void operator()(
      OpKernelContext* ctx, const Device& d, typename TTypes<T>::ConstMatrix x,
      typename TTypes<T>::ConstMatrix h_prev,
      typename TTypes<T>::ConstMatrix w_ru, typename TTypes<T>::ConstMatrix w_c,
      typename TTypes<T>::ConstVec b_ru, typename TTypes<T>::ConstVec b_c,
      typename TTypes<T>::Matrix r_u_bar, typename TTypes<T>::Matrix r,
      typename TTypes<T>::Matrix u, typename TTypes<T>::Matrix c,
      typename TTypes<T>::Matrix h, typename TTypes<T>::Matrix x_h_prev,
      typename TTypes<T>::Matrix x_h_prevr) {
    // Concat x_h_prev = [x, h_prev].
    x_h_prev.slice(x_offsets(), x_extends()).device(d) = x;
    x_h_prev.slice(h_offsets(), h_extends()).device(d) = h_prev;

    // r_u_bar = x_h_prev * w_ru + b_ru
    typename TTypes<T>::ConstMatrix const_x_h_prev(x_h_prev.data(),
                                                   x_h_prev.dimensions());
    TensorBlasGemm<Device, T, USE_CUBLAS>::compute(
        ctx, d, false, false, typename gemm_compute_type<T>::type(1.f),
        const_x_h_prev, w_ru, typename gemm_compute_type<T>::type(0.f),
        r_u_bar);

    // Creating a bias matrix for adding by broadcasting 'b_ru'
    Eigen::array<Eigen::DenseIndex, 2> broadcast_shape({batch_size_, 1});
    Eigen::array<Eigen::DenseIndex, 2> b_ru_shape({1, b_ru.dimensions()[0]});
    r_u_bar.device(d) += b_ru.reshape(b_ru_shape).broadcast(broadcast_shape);

    // Slice r_u_bar into r, u and apply the sigmoid.
    r.device(d) = (r_u_bar.slice(ru_r_offset(), cell_extents())).sigmoid();
    u.device(d) = (r_u_bar.slice(ru_u_offset(), cell_extents())).sigmoid();

    // Concat x_h_prevr = [x,h_prev*r]
    x_h_prevr.slice(x_offsets(), x_extends()).device(d) = x;
    x_h_prevr.slice(h_offsets(), h_extends()).device(d) = h_prev * r;

    // c = tanh(x_h_prevr*w_c+b_c), Note b_c is broadcasted before adding.
    typename TTypes<T>::ConstMatrix const_x_h_prevr(x_h_prevr.data(),
                                                    x_h_prevr.dimensions());
    TensorBlasGemm<Device, T, USE_CUBLAS>::compute(
        ctx, d, false, false, typename gemm_compute_type<T>::type(1.f),
        const_x_h_prevr, w_c, typename gemm_compute_type<T>::type(0.f), c);

    Eigen::array<Eigen::DenseIndex, 2> b_c_shape({1, b_c.dimensions()[0]});
    c.device(d) += (b_c.reshape(b_c_shape).broadcast(broadcast_shape));
    c.device(d) = c.tanh();

    // h= u*h_prev + (1-u)*c
    h.device(d) = u * (h_prev - c) + c;
  }
};

template <typename Device, typename T, bool USE_CUBLAS>
struct GRUBlockCellBprop : public GRUCell {
  GRUBlockCellBprop(const int batch_size, const int input_size,
                    const int cell_size)
      : GRUCell(batch_size, input_size, cell_size) {}

  void operator()(
      OpKernelContext* ctx, const Device& d, typename TTypes<T>::ConstMatrix x,
      typename TTypes<T>::ConstMatrix h_prev,
      typename TTypes<T>::ConstMatrix w_ru, typename TTypes<T>::ConstMatrix w_c,
      typename TTypes<T>::ConstVec b_ru, typename TTypes<T>::ConstVec b_c,
      typename TTypes<T>::ConstMatrix r, typename TTypes<T>::ConstMatrix u,
      typename TTypes<T>::ConstMatrix c, typename TTypes<T>::ConstMatrix d_h,
      typename TTypes<T>::Matrix d_x, typename TTypes<T>::Matrix d_h_prev,
      typename TTypes<T>::Matrix d_c_bar,
      typename TTypes<T>::Matrix d_r_bar_u_bar,
      typename TTypes<T>::Matrix d_r_bar, typename TTypes<T>::Matrix d_u_bar,
      typename TTypes<T>::Matrix d_hr,
      typename TTypes<T>::Matrix d_x_comp1_and_h_prev_comp1,
      typename TTypes<T>::Matrix d_x_comp2_and_h_prevr) {
    // d_c_bar = d_h*(1-u)*(1-(c*c))
    d_c_bar.device(d) =
        ((d_h * (u.constant(T(1)) - u)) * (c.constant(T(1)) - c * c));

    // d_u_bar = d_h*(h-c)*(u*(1-u))
    d_u_bar.device(d) = d_h * (h_prev - c) * u * (u.constant(T(1)) - u);

    // [2nd_component_of_d_x d_h_prevr] = d_c_bar X w_c^T
    typename TTypes<T>::ConstMatrix const_d_c_bar(d_c_bar.data(),
                                                  d_c_bar.dimensions());
    TensorBlasGemm<Device, T, USE_CUBLAS>::compute(
        ctx, d, false, true, typename gemm_compute_type<T>::type(1.f),
        const_d_c_bar, w_c, typename gemm_compute_type<T>::type(0.f),
        d_x_comp2_and_h_prevr);

    d_hr.device(d) = d_x_comp2_and_h_prevr.slice(h_offsets(), h_extends());
    d_r_bar.device(d) = (d_hr * h_prev * r) * (r.constant(T(1)) - r);

    // d_r_bar_u_bar = concatenate(d_r_bar, d_u_bar) along axis = 1.
    d_r_bar_u_bar.slice(ru_r_offset(), cell_extents()).device(d) = d_r_bar;
    d_r_bar_u_bar.slice(ru_u_offset(), cell_extents()).device(d) = d_u_bar;

    // [1st_component_of_d_x 1st_component_of_d_h_prev] = [d_r_bar d_u_bar] X
    // w_ru^T
    typename TTypes<T>::ConstMatrix const_d_r_bar_u_bar(
        d_r_bar_u_bar.data(), d_r_bar_u_bar.dimensions());
    TensorBlasGemm<Device, T, USE_CUBLAS>::compute(
        ctx, d, false, true, typename gemm_compute_type<T>::type(1.f),
        const_d_r_bar_u_bar, w_ru, typename gemm_compute_type<T>::type(0.f),
        d_x_comp1_and_h_prev_comp1);

    // d_x = d_x_comp1 + d_x_comp2
    d_x.device(d) = (d_x_comp1_and_h_prev_comp1 + d_x_comp2_and_h_prevr)
                        .slice(x_offsets(), x_extends());

    // d_h_prev = d_h_comp1 + d_hr*r + d_h*u
    d_h_prev.device(d) =
        d_x_comp1_and_h_prev_comp1.slice(h_offsets(), h_extends()) +
        (d_hr * r) + (d_h * u);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_RNN_GRU_OPS_H_
