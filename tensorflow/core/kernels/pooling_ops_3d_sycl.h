/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#if !TENSORFLOW_USE_SYCL
#error This file must only be included when building with SYCL support
#endif

#ifndef TENSORFLOW_CORE_KERNELS_POOLING_OP_3D_SYCL_H_
#define TENSORFLOW_CORE_KERNELS_POOLING_OP_3D_SYCL_H_

#include "tensorflow/core/kernels/pooling_ops_3d.h"

namespace tensorflow {

typedef Eigen::SyclDevice SYCLDevice;

// Helper struct to contain the various pool parameters used in the SYCL
// pooling kernels. Similar to the Pool3dParameters, but with a number of
// convenient constructors.
struct SYCL3DPoolParams {
  SYCL3DPoolParams(const int depth, const int batch, const int in_planes,
                   const int in_rows, const int in_cols, const int out_planes,
                   const int out_rows, const int out_cols,
                   const std::array<int64, 3>& window,
                   const std::array<int64, 3>& stride,
                   const std::array<int64, 3>& padding)
      : depth_(depth),
        batch_(batch),
        in_planes_(in_planes),
        in_rows_(in_rows),
        in_cols_(in_cols),
        window_planes_(window[2]),
        window_rows_(window[1]),
        window_cols_(window[0]),
        stride_planes_(stride[2]),
        stride_rows_(stride[1]),
        stride_cols_(stride[0]),
        out_planes_(out_planes),
        out_rows_(out_rows),
        out_cols_(out_cols),
        pad_planes_(padding[2]),
        pad_rows_(padding[1]),
        pad_cols_(padding[0]) {}

  SYCL3DPoolParams(const int depth, const int batch, const int in_planes,
                   const int in_rows, const int in_cols,
                   const std::array<int64, 3>& out_shape,
                   const std::array<int64, 3>& window,
                   const std::array<int64, 3>& stride,
                   const std::array<int64, 3>& padding)
      : SYCL3DPoolParams(depth, batch, in_planes, in_rows, in_cols,
                         out_shape[2], out_shape[1], out_shape[0], window,
                         stride, padding) {}

  SYCL3DPoolParams(const Pool3dParameters& params)
      : depth_(params.depth),
        batch_(params.tensor_in_batch),
        in_planes_(params.tensor_in_planes),
        in_rows_(params.tensor_in_rows),
        in_cols_(params.tensor_in_cols),
        window_planes_(params.window_planes),
        window_rows_(params.window_rows),
        window_cols_(params.window_cols),
        stride_planes_(params.plane_stride),
        stride_rows_(params.row_stride),
        stride_cols_(params.col_stride),
        out_planes_(params.out_plane),
        out_rows_(params.out_height),
        out_cols_(params.out_width),
        pad_planes_(params.pad_planes),
        pad_rows_(params.pad_rows),
        pad_cols_(params.pad_cols) {}

  const int depth_;
  const int batch_;
  const int in_planes_;
  const int in_rows_;
  const int in_cols_;

  const int window_planes_;
  const int window_rows_;
  const int window_cols_;

  const int stride_planes_;
  const int stride_rows_;
  const int stride_cols_;

  const int out_planes_;
  const int out_rows_;
  const int out_cols_;

  const int pad_planes_;
  const int pad_rows_;
  const int pad_cols_;
};
// MaxPool3d SYCL kernel. Expects the number of threads to be equal to the
// number of elements in the output tensor.
//
// For each output element, find the corresponding input window and run over
// all values in the window to find the maximum value. This value is then
// copied into that output element.
template <typename T>
class MaxPool3DSYCL {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;

 public:
  MaxPool3DSYCL(const int depth, const int batch, const int in_planes,
                const int in_rows, const int in_cols, const int out_planes,
                const int out_rows, const int out_cols,
                const std::array<int64, 3>& window,
                const std::array<int64, 3>& stride,
                const std::array<int64, 3>& padding,
                const read_accessor input_accessor,
                write_accessor output_accessor)
      : p_(depth, batch, in_planes, in_rows, in_cols, out_planes, out_rows,
           out_cols, window, stride, padding),
        input_accessor_(input_accessor),
        output_accessor_(output_accessor) {}
  void operator()(cl::sycl::item<1> item) {
    T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

    int index = item.get_linear_id();
    int n = index;
    int d = n % p_.depth_;
    n /= p_.depth_;
    int cstart = (n % p_.out_cols_) * p_.stride_cols_ - p_.pad_cols_;
    int cend = std::min(cstart + p_.window_cols_, p_.in_cols_);
    cstart = std::max(cstart, 0);
    n /= p_.out_cols_;
    int rstart = (n % p_.out_rows_) * p_.stride_rows_ - p_.pad_rows_;
    int rend = std::min(rstart + p_.window_rows_, p_.in_rows_);
    rstart = std::max(rstart, 0);
    n /= p_.out_rows_;
    int pstart = (n % p_.out_planes_) * p_.stride_planes_ - p_.pad_planes_;
    int pend = std::min(pstart + p_.window_planes_, p_.in_planes_);
    pstart = std::max(pstart, 0);
    n /= p_.out_planes_;
    T maxval = Eigen::NumTraits<T>::lowest();
    const T* input_data_n =
        input_data + n * p_.in_planes_ * p_.in_cols_ * p_.in_rows_ * p_.depth_;
    for (int p = pstart; p < pend; ++p) {
      for (int r = rstart; r < rend; ++r) {
        for (int c = cstart; c < cend; ++c) {
          int idx = ((p * p_.in_rows_ + r) * p_.in_cols_ + c) * p_.depth_ + d;
          if (input_data_n[idx] > maxval) {
            maxval = input_data_n[idx];
          }
        }
      }
    }
    output_data[index] = maxval;
  }

 private:
  const SYCL3DPoolParams p_;
  const read_accessor input_accessor_;
  write_accessor output_accessor_;
};
template <typename T>
struct LaunchPoolingOp<SYCLDevice, T, MAX> {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Padding padding_type,
                     Tensor* output) {
    const SYCLDevice& device = context->eigen_device<SYCLDevice>();
    const int out_planes = GetTensorDim(*output, data_format, '0');
    const int out_rows = GetTensorDim(*output, data_format, '1');
    const int out_cols = GetTensorDim(*output, data_format, '2');
    const int batch = GetTensorDim(tensor_in, data_format, 'N');
    const int in_planes = GetTensorDim(tensor_in, data_format, '0');
    const int in_rows = GetTensorDim(tensor_in, data_format, '1');
    const int in_cols = GetTensorDim(tensor_in, data_format, '2');
    const int depth = GetTensorDim(tensor_in, data_format, 'C');

    const int num_threads = output->NumElements();

    auto input_buffer =
        device.get_sycl_buffer(tensor_in.template flat<T>().data());
    auto output_buffer =
        device.get_sycl_buffer(output->template flat<T>().data());

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_access =
          input_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
      auto output_access =
          output_buffer.template get_access<cl::sycl::access::mode::write>(cgh);
      MaxPool3DSYCL<T> max_pool(depth, batch, in_planes, in_rows, in_cols,
                                out_planes, out_rows, out_cols, window, stride,
                                padding, input_access, output_access);

      cgh.parallel_for(cl::sycl::range<1>(num_threads), max_pool);
    });
  }
};
// MaxPool3DGrad SYCL kernel. Expects the number of threads to be equal to the
// number of elements in the output backprop tenor (i.e. the number of elements
// in the input data tensor).
//
// For each output backprop element we compute the possible window of values in
// the input backprop tensor which might contribute to this element. Then for
// each error in this window, compute the corresponding input window which was
// pooled into that element in the output. Walk through this input window to
// determine whether the input value is the first maximum value, and so the
// error should be propagated back to the corresponding backprop element.
template <typename T>
class MaxPool3DGradSYCL {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;

 public:
  MaxPool3DGradSYCL(const int depth, const int batch, const int in_planes,
                    const int in_rows, const int in_cols,
                    const std::array<int64, 3>& output_shape,
                    const std::array<int64, 3>& window,
                    const std::array<int64, 3>& stride,
                    const std::array<int64, 3>& padding,
                    const read_accessor input_data_accessor,
                    const read_accessor output_data_accessor,
                    const read_accessor input_backprop_accessor,
                    write_accessor output_backprop_accessor)
      : p_(depth, batch, in_planes, in_rows, in_cols, output_shape, window,
           stride, padding),
        input_data_accessor_(input_data_accessor),
        output_data_accessor_(output_data_accessor),
        input_backprop_accessor_(input_backprop_accessor),
        output_backprop_accessor_(output_backprop_accessor) {}
  void operator()(cl::sycl::item<1> item) {
    T* input_data = ConvertToActualTypeSycl(T, input_data_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_data_accessor_);
    T* input_backprop = ConvertToActualTypeSycl(T, input_backprop_accessor_);
    T* output_backprop = ConvertToActualTypeSycl(T, output_backprop_accessor_);

    const int index = item.get_linear_id();
    T output_value = 0;
    int n = index;
    const int d = n % p_.depth_;
    n /= p_.depth_;
    const int c = (n % p_.in_cols_) + p_.pad_cols_;
    const int poolcstart =
        (c < p_.window_cols_) ? 0 : (c - p_.window_cols_) / p_.stride_cols_ + 1;
    const int poolcend = std::min(c / p_.stride_cols_ + 1, p_.out_cols_);
    n /= p_.in_cols_;
    const int r = (n % p_.in_rows_) + p_.pad_rows_;
    const int poolrstart =
        (r < p_.window_rows_) ? 0 : (r - p_.window_rows_) / p_.stride_rows_ + 1;
    const int poolrend = std::min(r / p_.stride_rows_ + 1, p_.out_rows_);
    n /= p_.in_rows_;
    const int p = (n % p_.in_planes_) + p_.pad_planes_;
    const int poolpstart =
        (p < p_.window_planes_)
            ? 0
            : (p - p_.window_planes_) / p_.stride_planes_ + 1;
    const int poolpend = std::min(p / p_.stride_planes_ + 1, p_.out_planes_);
    n /= p_.in_planes_;
    const int index_no_n =
        index - n * p_.in_planes_ * p_.in_cols_ * p_.in_rows_ * p_.depth_;

    const T* input_data_n =
        input_data + n * p_.in_planes_ * p_.in_cols_ * p_.in_rows_ * p_.depth_;
    const T* output_data_n =
        output_data +
        n * p_.out_planes_ * p_.out_cols_ * p_.out_rows_ * p_.depth_;
    const T* input_backprop_n =
        input_backprop +
        n * p_.out_planes_ * p_.out_cols_ * p_.out_rows_ * p_.depth_;
    for (int poolp = poolpstart; poolp < poolpend; ++poolp) {
      int pstart = poolp * p_.stride_planes_ - p_.pad_planes_;
      const int pend = std::min(pstart + p_.window_planes_, p_.in_planes_);
      pstart = std::max(pstart, 0);

      for (int poolr = poolrstart; poolr < poolrend; ++poolr) {
        int rstart = poolr * p_.stride_rows_ - p_.pad_rows_;
        const int rend = std::min(rstart + p_.window_rows_, p_.in_rows_);
        rstart = std::max(rstart, 0);

        for (int poolc = poolcstart; poolc < poolcend; ++poolc) {
          int cstart = poolc * p_.stride_cols_ - p_.pad_cols_;
          const int cend = std::min(cstart + p_.window_cols_, p_.in_cols_);
          cstart = std::max(cstart, 0);

          const int output_data_idx =
              ((poolp * p_.out_rows_ + poolr) * p_.out_cols_ + poolc) *
                  p_.depth_ +
              d;
          bool should_continue = true;
          bool is_max = (input_data[index] == output_data_n[output_data_idx]);
          for (int win_p = pstart; win_p < pend && should_continue; ++win_p) {
            for (int win_r = rstart; win_r < rend && should_continue; ++win_r) {
              for (int win_c = cstart; win_c < cend && should_continue;
                   ++win_c) {
                const int input_data_idx =
                    ((win_p * p_.in_rows_ + win_r) * p_.in_cols_ + win_c) *
                        p_.depth_ +
                    d;
                if (input_data_idx == index_no_n) {
                  should_continue = false;
                } else if (input_data_n[input_data_idx] ==
                           output_data_n[output_data_idx]) {
                  should_continue = false;
                  is_max = false;
                }
              }
            }
          }
          if (is_max) {
            output_value += input_backprop_n[output_data_idx];
          }
        }
      }
    }
    output_backprop[index] = output_value;
  }

 private:
  const SYCL3DPoolParams p_;

  const read_accessor input_data_accessor_;
  const read_accessor output_data_accessor_;
  const read_accessor input_backprop_accessor_;
  write_accessor output_backprop_accessor_;
};
template <typename T>
struct LaunchMaxPooling3dGradOp<SYCLDevice, T> {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const Tensor& tensor_out, const Tensor& out_backprop,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& out,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Tensor* output) {
    const SYCLDevice& device = context->eigen_device<SYCLDevice>();
    const int batch = GetTensorDim(tensor_in, data_format, 'N');
    const int in_planes = GetTensorDim(tensor_in, data_format, '0');
    const int in_rows = GetTensorDim(tensor_in, data_format, '1');
    const int in_cols = GetTensorDim(tensor_in, data_format, '2');
    const int depth = GetTensorDim(tensor_in, data_format, 'C');

    const int output_size = output->NumElements();

    auto input_data_buffer =
        device.get_sycl_buffer(tensor_in.template flat<T>().data());
    auto output_data_buffer =
        device.get_sycl_buffer(tensor_out.template flat<T>().data());
    auto input_backprop_buffer =
        device.get_sycl_buffer(out_backprop.template flat<T>().data());
    auto output_backprop_buffer =
        device.get_sycl_buffer(output->template flat<T>().data());

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_data_access =
          input_data_buffer.template get_access<cl::sycl::access::mode::read>(
              cgh);
      auto output_data_access =
          output_data_buffer.template get_access<cl::sycl::access::mode::read>(
              cgh);
      auto input_backprop_access =
          input_backprop_buffer
              .template get_access<cl::sycl::access::mode::read>(cgh);
      auto output_backprop_access =
          output_backprop_buffer
              .template get_access<cl::sycl::access::mode::write>(cgh);
      MaxPool3DGradSYCL<T> max_pool(
          depth, batch, in_planes, in_rows, in_cols, out, window, stride,
          padding, input_data_access, output_data_access, input_backprop_access,
          output_backprop_access);

      cgh.parallel_for(cl::sycl::range<1>(output_size), max_pool);
    });
  }
};
// MaxPool3DGradGrad SYCL kernel. Expects the number of threads to be equal to
// the number of elements in the output backprop tensor, i.e. the number of
// elements in the output tensor.
//
// For each element in the output backprop tensor, find the corresponding input
// window, and compare the input and output data to find the index of the
// maximum value in the input tensor. This is then the index of the gradient to
// pass through to the output backprop tensor.
template <typename T>
class MaxPool3DGradGradSYCL {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;

 public:
  MaxPool3DGradGradSYCL(const Pool3dParameters& params,
                        const read_accessor input_data_accessor,
                        const read_accessor output_data_accessor,
                        const read_accessor input_backprop_accessor,
                        write_accessor output_backprop_accessor)
      : p_(params),
        input_data_accessor_(input_data_accessor),
        output_data_accessor_(output_data_accessor),
        input_backprop_accessor_(input_backprop_accessor),
        output_backprop_accessor_(output_backprop_accessor) {}
  void operator()(cl::sycl::item<1> item) {
    T* input_data = ConvertToActualTypeSycl(T, input_data_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_data_accessor_);
    T* input_backprop = ConvertToActualTypeSycl(T, input_backprop_accessor_);
    T* output_backprop = ConvertToActualTypeSycl(T, output_backprop_accessor_);

    int index = item.get_linear_id();
    int n = index;
    int d = n % p_.depth_;
    n /= p_.depth_;
    int cstart = (n % p_.out_cols_) * p_.stride_cols_ - p_.pad_cols_;
    int cend = std::min(cstart + p_.window_cols_, p_.in_cols_);
    cstart = std::max(cstart, 0);
    n /= p_.out_cols_;
    int rstart = (n % p_.out_rows_) * p_.stride_rows_ - p_.pad_rows_;
    int rend = std::min(rstart + p_.window_rows_, p_.in_rows_);
    rstart = std::max(rstart, 0);
    n /= p_.out_rows_;
    int pstart = (n % p_.out_planes_) * p_.stride_planes_ - p_.pad_planes_;
    int pend = std::min(pstart + p_.window_planes_, p_.in_planes_);
    pstart = std::max(pstart, 0);
    n /= p_.out_planes_;
    int maxidx = -1;
    bool should_stop = false;
    const T* input_data_n =
        input_data + n * p_.in_planes_ * p_.in_cols_ * p_.in_rows_ * p_.depth_;
    for (int p = pstart; p < pend && !should_stop; ++p) {
      for (int r = rstart; r < rend && !should_stop; ++r) {
        for (int c = cstart; c < cend && !should_stop; ++c) {
          int idx = ((p * p_.in_rows_ + r) * p_.in_cols_ + c) * p_.depth_ + d;
          if (output_data[index] == input_data_n[idx]) {
            maxidx = idx;
            should_stop = true;
          }
        }
      }
    }
    if (maxidx != -1) {
      output_backprop[index] = input_backprop[n * p_.in_planes_ * p_.in_rows_ *
                                                  p_.in_cols_ * p_.depth_ +
                                              maxidx];
    }
  }

 private:
  const SYCL3DPoolParams p_;

  const read_accessor input_data_accessor_;
  const read_accessor output_data_accessor_;
  const read_accessor input_backprop_accessor_;
  write_accessor output_backprop_accessor_;
};
template <typename T>
struct LaunchMaxPooling3dGradGradOp<SYCLDevice, T> {
  static void launch(OpKernelContext* context, const Pool3dParameters& params,
                     const Tensor& tensor_in, const Tensor& tensor_out,
                     const Tensor& out_backprop, Tensor* output) {
    const SYCLDevice& device = context->eigen_device<SYCLDevice>();

    const int num_threads = output->NumElements();

    auto input_data_buffer =
        device.get_sycl_buffer(tensor_in.template flat<T>().data());
    auto output_data_buffer =
        device.get_sycl_buffer(tensor_out.template flat<T>().data());
    auto input_backprop_buffer =
        device.get_sycl_buffer(out_backprop.template flat<T>().data());
    auto output_backprop_buffer =
        device.get_sycl_buffer(output->template flat<T>().data());

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_data_access =
          input_data_buffer.template get_access<cl::sycl::access::mode::read>(
              cgh);
      auto output_data_access =
          output_data_buffer.template get_access<cl::sycl::access::mode::read>(
              cgh);
      auto input_backprop_access =
          input_backprop_buffer
              .template get_access<cl::sycl::access::mode::read>(cgh);
      auto output_backprop_access =
          output_backprop_buffer
              .template get_access<cl::sycl::access::mode::write>(cgh);
      MaxPool3DGradGradSYCL<T> functor(
          params, input_data_access, output_data_access, input_backprop_access,
          output_backprop_access);

      cgh.parallel_for(cl::sycl::range<1>(num_threads), functor);
    });
  }
};
// AvgPool3D SYCL kernel. Expects the number of threads to be equal to the
// number of elements in the output tensor.
//
// For each output value find the corresponding input window, and run through
// the window accumulating the values to form an average. We divide each value
// before accumulating to prevent the accumulator from becoming significantly
// bigger than the values we are adding and so decrease any errors.
template <typename T>
class AvgPool3DSYCL {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;

 public:
  AvgPool3DSYCL(const int depth, const int batch, const int in_planes,
                const int in_rows, const int in_cols, const int out_planes,
                const int out_rows, const int out_cols,
                const std::array<int64, 3>& window,
                const std::array<int64, 3>& stride,
                const std::array<int64, 3>& padding,
                const read_accessor input_accessor,
                write_accessor output_accessor)
      : p_(depth, batch, in_planes, in_rows, in_cols, out_planes, out_rows,
           out_cols, window, stride, padding),
        input_accessor_(input_accessor),
        output_accessor_(output_accessor) {}
  void operator()(cl::sycl::item<1> item) {
    T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

    int index = item.get_linear_id();
    int n = index;
    int d = n % p_.depth_;
    n /= p_.depth_;
    int cstart = (n % p_.out_cols_) * p_.stride_cols_ - p_.pad_cols_;
    int cend = std::min(cstart + p_.window_cols_, p_.in_cols_);
    cstart = std::max(cstart, 0);
    n /= p_.out_cols_;
    int rstart = (n % p_.out_rows_) * p_.stride_rows_ - p_.pad_rows_;
    int rend = std::min(rstart + p_.window_rows_, p_.in_rows_);
    rstart = std::max(rstart, 0);
    n /= p_.out_rows_;
    int pstart = (n % p_.out_planes_) * p_.stride_planes_ - p_.pad_planes_;
    int pend = std::min(pstart + p_.window_planes_, p_.in_planes_);
    pstart = std::max(pstart, 0);
    n /= p_.out_planes_;
    T accum = T(0);
    T count =
        static_cast<T>((pend - pstart) * (rend - rstart) * (cend - cstart));
    const T* input_data_n =
        input_data + n * p_.in_planes_ * p_.in_cols_ * p_.in_rows_ * p_.depth_;
    for (int p = pstart; p < pend; ++p) {
      for (int r = rstart; r < rend; ++r) {
        for (int c = cstart; c < cend; ++c) {
          int idx = ((p * p_.in_rows_ + r) * p_.in_cols_ + c) * p_.depth_ + d;
          accum += input_data_n[idx] / count;
        }
      }
    }
    output_data[index] = accum;
  }

 private:
  const SYCL3DPoolParams p_;
  const read_accessor input_accessor_;
  write_accessor output_accessor_;
};
template <typename T>
struct LaunchPoolingOp<SYCLDevice, T, AVG> {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Padding padding_type,
                     Tensor* output) {
    const SYCLDevice& device = context->eigen_device<SYCLDevice>();
    const int out_planes = GetTensorDim(*output, data_format, '0');
    const int out_rows = GetTensorDim(*output, data_format, '1');
    const int out_cols = GetTensorDim(*output, data_format, '2');
    const int batch = GetTensorDim(tensor_in, data_format, 'N');
    const int in_planes = GetTensorDim(tensor_in, data_format, '0');
    const int in_rows = GetTensorDim(tensor_in, data_format, '1');
    const int in_cols = GetTensorDim(tensor_in, data_format, '2');
    const int depth = GetTensorDim(tensor_in, data_format, 'C');

    const int num_threads = output->NumElements();

    auto input_buffer =
        device.get_sycl_buffer(tensor_in.template flat<T>().data());
    auto output_buffer =
        device.get_sycl_buffer(output->template flat<T>().data());

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_access =
          input_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
      auto output_access =
          output_buffer.template get_access<cl::sycl::access::mode::write>(cgh);
      AvgPool3DSYCL<T> avg_pool(depth, batch, in_planes, in_rows, in_cols,
                                out_planes, out_rows, out_cols, window, stride,
                                padding, input_access, output_access);

      cgh.parallel_for(cl::sycl::range<1>(num_threads), avg_pool);
    });
  }
};
// AvgPool3DGrad SYCL kernel. Expects the number of threads to be equal to the
// number of elements in the output backprop tensor, i.e. the number of
// elements in the input tensor.
//
// For each output backprop index find a window in the input backprop tensor
// which corresponds to all the values of the output which were affected by the
// input value at this index. Then for each gradient in this window, compute
// the size of the input window which was averaged to give this output, and use
// this size to scale the gradient accordingly. Add this scaled gradient to the
// output backprop value.
template <typename T>
class AvgPool3DGradSYCL {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;

 public:
  AvgPool3DGradSYCL(const int depth, const int batch, const int in_planes,
                    const int in_rows, const int in_cols,
                    const std::array<int64, 3>& out_shape,
                    const std::array<int64, 3>& window,
                    const std::array<int64, 3>& stride,
                    const std::array<int64, 3>& padding,
                    const read_accessor input_backprop_accessor,
                    write_accessor output_backprop_accessor)
      : p_(depth, batch, in_planes, in_rows, in_cols, out_shape, window, stride,
           padding),
        input_backprop_accessor_(input_backprop_accessor),
        output_backprop_accessor_(output_backprop_accessor) {}
  void operator()(cl::sycl::item<1> item) {
    T* input_backprop = ConvertToActualTypeSycl(T, input_backprop_accessor_);
    T* output_backprop = ConvertToActualTypeSycl(T, output_backprop_accessor_);

    const int index = item.get_linear_id();
    int n = index;
    const int d = n % p_.depth_;
    n /= p_.depth_;
    const int c = (n % p_.in_cols_) + p_.pad_cols_;
    const int poolcstart =
        (c < p_.window_cols_) ? 0 : (c - p_.window_cols_) / p_.stride_cols_ + 1;
    const int poolcend = std::min(c / p_.stride_cols_ + 1, p_.out_cols_);
    n /= p_.in_cols_;
    const int r = (n % p_.in_rows_) + p_.pad_rows_;
    const int poolrstart =
        (r < p_.window_rows_) ? 0 : (r - p_.window_rows_) / p_.stride_rows_ + 1;
    const int poolrend = std::min(r / p_.stride_rows_ + 1, p_.out_rows_);
    n /= p_.in_rows_;
    const int p = (n % p_.in_planes_) + p_.pad_planes_;
    const int poolpstart =
        (p < p_.window_planes_)
            ? 0
            : (p - p_.window_planes_) / p_.stride_planes_ + 1;
    const int poolpend = std::min(p / p_.stride_planes_ + 1, p_.out_planes_);
    n /= p_.in_planes_;

    T gradient = T(0);
    const T* input_backprop_n =
        input_backprop +
        n * p_.out_planes_ * p_.out_cols_ * p_.out_rows_ * p_.depth_;
    for (int poolp = poolpstart; poolp < poolpend; ++poolp) {
      int pstart = poolp * p_.stride_planes_ - p_.pad_planes_;
      const int pend = std::min(pstart + p_.window_planes_, p_.in_planes_);
      pstart = std::max(pstart, 0);
      const int plane_window_size = pend - pstart;
      for (int poolr = poolrstart; poolr < poolrend; ++poolr) {
        int rstart = poolr * p_.stride_rows_ - p_.pad_rows_;
        const int rend = std::min(rstart + p_.window_rows_, p_.in_rows_);
        rstart = std::max(rstart, 0);
        const int row_window_size = rend - rstart;
        for (int poolc = poolcstart; poolc < poolcend; ++poolc) {
          const int idx =
              ((poolp * p_.out_rows_ + poolr) * p_.out_cols_ + poolc) *
                  p_.depth_ +
              d;
          int cstart = poolc * p_.stride_cols_ - p_.pad_cols_;
          const int cend = std::min(cstart + p_.window_cols_, p_.in_cols_);
          cstart = std::max(cstart, 0);
          const int col_window_size = cend - cstart;
          const int window_size =
              plane_window_size * row_window_size * col_window_size;
          gradient += input_backprop_n[idx] / static_cast<T>(window_size);
        }
      }
    }
    output_backprop[index] = gradient;
  }

 private:
  const SYCL3DPoolParams p_;
  const read_accessor input_backprop_accessor_;
  write_accessor output_backprop_accessor_;
};
template <typename T>
struct LaunchAvgPooling3dGradOp<SYCLDevice, T> {
  static void launch(OpKernelContext* context,
                     const TensorShape& tensor_in_shape,
                     const Tensor& out_backprop,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& output_shape,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Tensor* output) {
    const SYCLDevice& device = context->eigen_device<SYCLDevice>();
    const int batch = GetTensorDim(tensor_in_shape, data_format, 'N');
    const int in_planes = GetTensorDim(tensor_in_shape, data_format, '0');
    const int in_rows = GetTensorDim(tensor_in_shape, data_format, '1');
    const int in_cols = GetTensorDim(tensor_in_shape, data_format, '2');
    const int depth = GetTensorDim(tensor_in_shape, data_format, 'C');

    const int num_threads = output->NumElements();

    auto input_backprop_buffer =
        device.get_sycl_buffer(out_backprop.template flat<T>().data());
    auto output_backprop_buffer =
        device.get_sycl_buffer(output->template flat<T>().data());

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_backprop_access =
          input_backprop_buffer
              .template get_access<cl::sycl::access::mode::read>(cgh);
      auto output_backprop_access =
          output_backprop_buffer
              .template get_access<cl::sycl::access::mode::write>(cgh);
      AvgPool3DGradSYCL<T> functor(
          depth, batch, in_planes, in_rows, in_cols, output_shape, window,
          stride, padding, input_backprop_access, output_backprop_access);

      cgh.parallel_for(cl::sycl::range<1>(num_threads), functor);
    });
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_POOLING_OP_3D_SYCL_H_
