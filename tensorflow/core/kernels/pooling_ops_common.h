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

#ifndef TENSORFLOW_CORE_KERNELS_POOLING_OPS_COMMON_H_
#define TENSORFLOW_CORE_KERNELS_POOLING_OPS_COMMON_H_

#include <vector>

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/avgpooling_op.h"
#include "tensorflow/core/kernels/maxpooling_op.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/work_sharder.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/maxpooling_op_gpu.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// A helper class to manage sizes and shapes for pooling operations.
struct PoolParameters {
  // Updates context->status if there is an invalid input.
  // explicit_paddings has eight elements if padding==EXPLIICT, and zero
  // elements otherwise.
  PoolParameters(OpKernelContext* context, const std::vector<int32>& ksize,
                 const std::vector<int32>& stride, Padding padding,
                 std::vector<int64_t> explicit_paddings,
                 TensorFormat data_format, const TensorShape& tensor_in_shape);

  // Returns the shape of the output for "forward" pooling operations.
  absl::Status forward_output_shape(TensorShape* shape);

  int depth;

  int tensor_in_cols;
  int tensor_in_rows;
  int tensor_in_batch;

  int window_rows;
  int window_cols;
  int depth_window;

  int row_stride;
  int col_stride;
  int depth_stride;

  int64_t out_height;
  int64_t out_width;
  int out_depth;

  int64_t pad_top;
  int64_t pad_bottom;
  int64_t pad_left;
  int64_t pad_right;

  int pad_depth;

  TensorFormat data_format;
};

// An implementation of MaxPooling (forward).
// TODO (yongtang): Remove MaxPoolingOp and use MaxPoolingV2Op,
//     QuantizedMaxPoolingOp depends on MaxPoolingOp so keep intact for now
template <typename Device, typename T>
class MaxPoolingOp : public OpKernel {
 public:
  explicit MaxPoolingOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    auto status = context->GetAttr("data_format", &data_format);
    if (status.ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
      OP_REQUIRES(
          context, data_format_ == FORMAT_NHWC,
          errors::InvalidArgument("Default MaxPoolingOp only supports NHWC ",
                                  "on device type ",
                                  DeviceTypeString(context->device_type())));
    } else {
      data_format_ = FORMAT_NHWC;
    }
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(
        context,
        ksize_[0] > 0 && ksize_[1] > 0 && ksize_[2] > 0 && ksize_[3] > 0,
        errors::InvalidArgument(
            absl::StrCat("Sliding window ksize must be positive. The "
                         "specified or inferred ksize is: ",
                         absl::StrJoin(ksize_, ","))));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    if (padding_ == Padding::EXPLICIT) {
      OP_REQUIRES_OK(
          context, context->GetAttr("explicit_paddings", &explicit_paddings_));
    }
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    PoolParameters params{
        context,     ksize_,           stride_, padding_, explicit_paddings_,
        FORMAT_NHWC, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    Tensor* output = nullptr;
    TensorShape params_forward_output_shape;
    OP_REQUIRES_OK(context,
                   params.forward_output_shape(&params_forward_output_shape));
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, params_forward_output_shape, &output));

    if (params.depth_window > 1) {
      // Validate spec against the current implementation.  A
      // relaxation of these requirements would be ideal.
      OP_REQUIRES(context, params.depth % params.depth_window == 0,
                  errors::Unimplemented(
                      "Depthwise max pooling requires "
                      "the depth window to evenly divide the input depth."));
      OP_REQUIRES(
          context, params.depth_window == params.depth_stride,
          errors::Unimplemented("Depthwise max pooling requires "
                                "the depth window to equal the depth stride."));
      OP_REQUIRES(
          context, padding_ != EXPLICIT,
          errors::Unimplemented("Depthwise max pooling does not support "
                                "explicit padding."));

      DepthwiseMaxPool(context, output, tensor_in, params);
    } else {
      // MaxPoolingOp is only called on the GPU when the eigen_tensor label
      // is used. In this case, explicit padding is not supported
      if (std::is_same<Device, GPUDevice>::value &&
          padding_ == Padding::EXPLICIT) {
        context->SetStatus(errors::Unimplemented(
            "MaxPoolingOp does not support explicit padding."));
        return;
      }
      SpatialMaxPool(context, output, tensor_in, params, padding_);
    }
  }

 private:
  // Single-threaded implementation of DepthwiseMaxPool which
  // does not handle all of the same options as SpatialMaxPool
  // (strict assumptions on no padding, stride).
  //
  // TODO(vrv): implement a more general depthwise-max pool that works
  // on GPU as well.
  void DepthwiseMaxPool(OpKernelContext* context, Tensor* output,
                        const Tensor& tensor_in, const PoolParameters& params) {
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
        in_by_pool(tensor_in.flat<T>().data(), params.depth_window,
                   tensor_in.NumElements() / params.depth_window);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> out_by_pool(
        output->flat<T>().data(), 1, output->NumElements());
    out_by_pool = in_by_pool.colwise().maxCoeff();
  }

  void SpatialMaxPool(OpKernelContext* context, Tensor* output,
                      const Tensor& tensor_in, const PoolParameters& params,
                      const Padding& padding) {
    if (output->NumElements() == 0) {
      return;
    }
    // On GPU, use Eigen's Spatial Max Pooling.  On CPU, use an
    // EigenMatrix version that is currently faster than Eigen's
    // Spatial MaxPooling implementation.
    //
    // TODO(vrv): Remove this once we no longer need it.
    if (std::is_same<Device, GPUDevice>::value) {
      Eigen::PaddingType pt = BrainPadding2EigenPadding(padding);
      functor::SpatialMaxPooling<Device, T>()(
          context->eigen_device<Device>(), output->tensor<T, 4>(),
          tensor_in.tensor<T, 4>(), params.window_rows, params.window_cols,
          params.row_stride, params.col_stride, pt);
    } else {
      typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
          ConstEigenMatrixMap;
      typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
          EigenMatrixMap;

      ConstEigenMatrixMap in_mat(tensor_in.flat<T>().data(), params.depth,
                                 params.tensor_in_cols * params.tensor_in_rows *
                                     params.tensor_in_batch);
      EigenMatrixMap out_mat(
          output->flat<T>().data(), params.depth,
          params.out_width * params.out_height * params.tensor_in_batch);

      const DeviceBase::CpuWorkerThreads& worker_threads =
          *(context->device()->tensorflow_cpu_worker_threads());

      // The following code basically does the following:
      // 1. Flattens the input and output tensors into two dimensional arrays.
      //    tensor_in_as_matrix:
      //      depth by (tensor_in_cols * tensor_in_rows * tensor_in_batch)
      //    output_as_matrix:
      //      depth by (out_width * out_height * tensor_in_batch)
      //
      // 2. Walks through the set of columns in the flattened
      // tensor_in_as_matrix,
      //    and updates the corresponding column(s) in output_as_matrix with the
      //    max value.
      auto shard = [&params, &in_mat, &out_mat](int64_t start, int64_t limit) {
        const int32_t in_rows = params.tensor_in_rows;
        const int32_t in_cols = params.tensor_in_cols;
        const int32_t pad_top = params.pad_top;
        const int32_t pad_left = params.pad_left;
        const int32_t window_rows = params.window_rows;
        const int32_t window_cols = params.window_cols;
        const int32_t row_stride = params.row_stride;
        const int32_t col_stride = params.col_stride;
        const int32_t out_height = params.out_height;
        const int32_t out_width = params.out_width;

        {
          // Initializes the output tensor with MIN<T>.
          const int32_t output_image_size =
              out_height * out_width * params.depth;
          EigenMatrixMap out_shard(out_mat.data() + start * output_image_size,
                                   1, (limit - start) * output_image_size);
          out_shard.setConstant(Eigen::NumTraits<T>::lowest());
        }

        for (int32_t b = start; b < limit; ++b) {
          const int32_t out_offset_batch = b * out_height;
          for (int32_t h = 0; h < in_rows; ++h) {
            for (int32_t w = 0; w < in_cols; ++w) {
              // (h_start, h_end) * (w_start, w_end) is the range that the input
              // vector projects to.
              const int32_t hpad = h + pad_top;
              const int32_t wpad = w + pad_left;
              const int32_t h_start =
                  (hpad < window_rows) ? 0
                                       : (hpad - window_rows) / row_stride + 1;
              const int32_t h_end = std::min(hpad / row_stride + 1, out_height);
              const int32_t w_start =
                  (wpad < window_cols) ? 0
                                       : (wpad - window_cols) / col_stride + 1;
              const int32_t w_end = std::min(wpad / col_stride + 1, out_width);
              // compute elementwise max
              const int32_t in_offset = (b * in_rows + h) * in_cols + w;
              for (int32_t ph = h_start; ph < h_end; ++ph) {
                const int32_t out_offset_base =
                    (out_offset_batch + ph) * out_width;
                for (int32_t pw = w_start; pw < w_end; ++pw) {
                  const int32_t out_offset = out_offset_base + pw;
                  out_mat.col(out_offset) =
                      out_mat.col(out_offset).cwiseMax(in_mat.col(in_offset));
                }
              }
            }
          }
        }
      };

      // TODO(andydavis) Consider sharding across batch x rows x cols.
      // TODO(andydavis) Consider a higher resolution shard cost model.
      const int64_t shard_cost =
          params.tensor_in_rows * params.tensor_in_cols * params.depth;
      Shard(worker_threads.num_threads, worker_threads.workers,
            params.tensor_in_batch, shard_cost, shard);
    }
  }

  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  std::vector<int64_t> explicit_paddings_;
  TensorFormat data_format_;
};

template <typename Device>
struct LaunchMaxPoolingNoMask_NCHW_VECT_C;

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <>
struct LaunchMaxPoolingNoMask_NCHW_VECT_C<Eigen::GpuDevice> {
  static void launch(OpKernelContext* context, const PoolParameters& params,
                     const Tensor& input, Tensor* output) {
#if GOOGLE_CUDA
    bool status = functor::MaxPoolForwardNoMask_NCHW_VECT_C()(
        reinterpret_cast<const int32*>(input.flat<qint8>().data()),
        params.tensor_in_batch, params.tensor_in_rows, params.tensor_in_cols,
        params.depth, params.out_height, params.out_width, params.window_rows,
        params.window_cols, params.row_stride, params.col_stride,
        params.pad_top, params.pad_left,
        reinterpret_cast<int32*>(output->flat<qint8>().data()),
        context->eigen_gpu_device());
    if (!status) {
      context->SetStatus(errors::Internal(
          "Failed launching LaunchMaxPoolingNoMask_NCHW_VECT_C"));
    }
#else
    // ROCm TODO: add support __vmaxs4 on ROCm
    context->SetStatus(errors::Internal(
        "Failed launching LaunchMaxPoolingNoMask_NCHW_VECT_C"));
#endif  // GOOGLE_CUDA
  }
};
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename Device, typename T>
class MaxPoolingV2Op : public OpKernel {
 public:
  explicit MaxPoolingV2Op(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    auto status = context->GetAttr("data_format", &data_format);
    if (status.ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
      OP_REQUIRES(
          context,
          data_format_ == FORMAT_NHWC || data_format_ == FORMAT_NCHW_VECT_C,
          errors::InvalidArgument(
              "MaxPoolingV2Op only supports NHWC or NCHW_VECT_C. Got: ",
              data_format));
    } else {
      data_format_ = FORMAT_NHWC;
    }
    if (context->num_inputs() == 1) {
      OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
      OP_REQUIRES(context, ksize_.size() == 4,
                  errors::InvalidArgument("Sliding window ksize field must "
                                          "specify 4 dimensions"));
      OP_REQUIRES(
          context,
          ksize_[0] > 0 && ksize_[1] > 0 && ksize_[2] > 0 && ksize_[3] > 0,
          errors::InvalidArgument("Sliding window ksize must be positive."));
      OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
      OP_REQUIRES(context, stride_.size() == 4,
                  errors::InvalidArgument("Sliding window stride field must "
                                          "specify 4 dimensions"));
      OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                  errors::Unimplemented(
                      "Pooling is not yet supported on the batch dimension."));
    }
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);

    std::vector<int32> ksize = ksize_;
    std::vector<int32> stride = stride_;

    if (context->num_inputs() != 1) {
      const Tensor& tensor_ksize = context->input(1);
      auto value_ksize = tensor_ksize.flat<int32>();
      ksize.resize(tensor_ksize.shape().num_elements());
      std::copy_n(&value_ksize(0), ksize.size(), ksize.begin());

      const Tensor& tensor_stride = context->input(2);
      auto value_stride = tensor_stride.flat<int32>();
      stride.resize(tensor_stride.shape().num_elements());
      std::copy_n(&value_stride(0), stride.size(), stride.begin());
    }

    OP_REQUIRES(context, ksize.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(
        context, ksize[0] > 0 && ksize[1] > 0 && ksize[2] > 0 && ksize[3] > 0,
        errors::InvalidArgument("Sliding window ksize must be positive."));
    OP_REQUIRES(context, stride.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, ksize[0] == 1 && stride[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));

    PoolParameters params{
        context,
        ksize,
        stride,
        padding_,
        /*explicit_paddings=*/{},
        data_format_,
        tensor_in.shape(),
    };
    if (!context->status().ok()) {
      return;
    }

    Tensor* output = nullptr;
    TensorShape params_forward_output_shape;
    OP_REQUIRES_OK(context,
                   params.forward_output_shape(&params_forward_output_shape));
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, params_forward_output_shape, &output));

    if (params.depth_window > 1) {
      // Validate spec against the current implementation.  A
      // relaxation of these requirements would be ideal.
      OP_REQUIRES(context, params.depth % params.depth_window == 0,
                  errors::Unimplemented(
                      "Depthwise max pooling requires "
                      "the depth window to evenly divide the input depth."));
      OP_REQUIRES(
          context, params.depth_window == params.depth_stride,
          errors::Unimplemented("Depthwise max pooling requires "
                                "the depth window to equal the depth stride."));

      DepthwiseMaxPool(context, output, tensor_in, params);
    } else {
      SpatialMaxPool(context, output, tensor_in, params, padding_);
    }
  }

 private:
  // Single-threaded implementation of DepthwiseMaxPool which
  // does not handle all of the same options as SpatialMaxPool
  // (strict assumptions on no padding, stride).
  //
  // TODO(vrv): implement a more general depthwise-max pool that works
  // on GPU as well.
  void DepthwiseMaxPool(OpKernelContext* context, Tensor* output,
                        const Tensor& tensor_in, const PoolParameters& params) {
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
        in_by_pool(tensor_in.flat<T>().data(), params.depth_window,
                   tensor_in.NumElements() / params.depth_window);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> out_by_pool(
        output->flat<T>().data(), 1, output->NumElements());
    out_by_pool = in_by_pool.colwise().maxCoeff();
  }

  void SpatialMaxPool(OpKernelContext* context, Tensor* output,
                      const Tensor& tensor_in, const PoolParameters& params,
                      const Padding& padding) {
    if (output->NumElements() == 0) {
      return;
    }
    // On GPU, use Eigen's Spatial Max Pooling.  On CPU, use an
    // EigenMatrix version that is currently faster than Eigen's
    // Spatial MaxPooling implementation.
    //
    // TODO(vrv): Remove this once we no longer need it.
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    if (std::is_same<Device, GPUDevice>::value) {
      Eigen::PaddingType pt = BrainPadding2EigenPadding(padding);
      if (std::is_same<T, qint8>::value) {
        LaunchMaxPoolingNoMask_NCHW_VECT_C<GPUDevice>::launch(
            context, params, tensor_in, output);
      } else {
        functor::SpatialMaxPooling<Device, T>()(
            context->eigen_device<Device>(), output->tensor<T, 4>(),
            tensor_in.tensor<T, 4>(), params.window_rows, params.window_cols,
            params.row_stride, params.col_stride, pt);
      }
    } else
#endif
    {
      typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
          ConstEigenMatrixMap;
      typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
          EigenMatrixMap;

      ConstEigenMatrixMap in_mat(tensor_in.flat<T>().data(), params.depth,
                                 params.tensor_in_cols * params.tensor_in_rows *
                                     params.tensor_in_batch);
      EigenMatrixMap out_mat(
          output->flat<T>().data(), params.depth,
          params.out_width * params.out_height * params.tensor_in_batch);

      const DeviceBase::CpuWorkerThreads& worker_threads =
          *(context->device()->tensorflow_cpu_worker_threads());

      // The following code basically does the following:
      // 1. Flattens the input and output tensors into two dimensional arrays.
      //    tensor_in_as_matrix:
      //      depth by (tensor_in_cols * tensor_in_rows * tensor_in_batch)
      //    output_as_matrix:
      //      depth by (out_width * out_height * tensor_in_batch)
      //
      // 2. Walks through the set of columns in the flattened
      // tensor_in_as_matrix,
      //    and updates the corresponding column(s) in output_as_matrix with the
      //    max value.
      auto shard = [&params, &in_mat, &out_mat](int64_t start, int64_t limit) {
        const int32_t in_rows = params.tensor_in_rows;
        const int32_t in_cols = params.tensor_in_cols;
        const int32_t pad_top = params.pad_top;
        const int32_t pad_left = params.pad_left;
        const int32_t window_rows = params.window_rows;
        const int32_t window_cols = params.window_cols;
        const int32_t row_stride = params.row_stride;
        const int32_t col_stride = params.col_stride;
        const int32_t out_height = params.out_height;
        const int32_t out_width = params.out_width;

        {
          // Initializes the output tensor with MIN<T>.
          const int32_t output_image_size =
              out_height * out_width * params.depth;
          EigenMatrixMap out_shard(out_mat.data() + start * output_image_size,
                                   1, (limit - start) * output_image_size);
          out_shard.setConstant(Eigen::NumTraits<T>::lowest());
        }

        for (int32_t b = start; b < limit; ++b) {
          const int32_t out_offset_batch = b * out_height;
          for (int32_t h = 0; h < in_rows; ++h) {
            for (int32_t w = 0; w < in_cols; ++w) {
              // (h_start, h_end) * (w_start, w_end) is the range that the input
              // vector projects to.
              const int32_t hpad = h + pad_top;
              const int32_t wpad = w + pad_left;
              const int32_t h_start =
                  (hpad < window_rows) ? 0
                                       : (hpad - window_rows) / row_stride + 1;
              const int32_t h_end = std::min(hpad / row_stride + 1, out_height);
              const int32_t w_start =
                  (wpad < window_cols) ? 0
                                       : (wpad - window_cols) / col_stride + 1;
              const int32_t w_end = std::min(wpad / col_stride + 1, out_width);
              // compute elementwise max
              const int32_t in_offset = (b * in_rows + h) * in_cols + w;
              for (int32_t ph = h_start; ph < h_end; ++ph) {
                const int32_t out_offset_base =
                    (out_offset_batch + ph) * out_width;
                for (int32_t pw = w_start; pw < w_end; ++pw) {
                  const int32_t out_offset = out_offset_base + pw;
                  out_mat.col(out_offset) =
                      out_mat.col(out_offset).cwiseMax(in_mat.col(in_offset));
                }
              }
            }
          }
        }
      };

      // TODO(andydavis) Consider sharding across batch x rows x cols.
      // TODO(andydavis) Consider a higher resolution shard cost model.
      const int64_t shard_cost =
          params.tensor_in_rows * params.tensor_in_cols * params.depth;
      Shard(worker_threads.num_threads, worker_threads.workers,
            params.tensor_in_batch, shard_cost, shard);
    }
  }

  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

template <typename Device, typename T>
void SpatialAvgPool(OpKernelContext* context, Tensor* output,
                    const Tensor& input, const PoolParameters& params,
                    const Padding& padding) {
  if (output->NumElements() == 0) {
    return;
  }
  typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      ConstEigenMatrixMap;
  typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      EigenMatrixMap;

  auto in_flat = input.flat<T>();
  auto out_flat = output->flat<T>();

  auto shard = [&params, &in_flat, &out_flat](int64_t start, int64_t limit) {
    // Calculate indices for this shards chunk of work.
    const int64_t input_image_size =
        params.tensor_in_rows * params.tensor_in_cols * params.depth;
    const int64_t output_image_size =
        params.out_width * params.out_height * params.depth;
    const int64_t shard_batch_size = limit - start;

    ConstEigenMatrixMap in_mat(
        in_flat.data() + start * input_image_size, params.depth,
        params.tensor_in_cols * params.tensor_in_rows * shard_batch_size);
    EigenMatrixMap out_mat(
        out_flat.data() + start * output_image_size, params.depth,
        params.out_width * params.out_height * shard_batch_size);
    Eigen::Matrix<T, Eigen::Dynamic, 1> out_count(out_mat.cols());
    out_count.setZero();

    // Initializes output to zero.
    out_mat.setZero();

    // The following code basically does the following:
    // 1. Flattens the input and output tensors into two dimensional arrays.
    //    tensor_in_as_matrix:
    //      depth by (tensor_in_cols * tensor_in_rows * tensor_in_batch)
    //    output_as_matrix:
    //      depth by (out_width * out_height * tensor_in_batch)
    //
    // 2. Walks through the set of columns in the flattened
    // tensor_in_as_matrix,
    //    and updates the corresponding column(s) in output_as_matrix with the
    //    average value.
    for (int b = 0; b < shard_batch_size; ++b) {
      for (int h = 0; h < params.tensor_in_rows; ++h) {
        for (int w = 0; w < params.tensor_in_cols; ++w) {
          // (h_start, h_end) * (w_start, w_end) is the range that the input
          // vector projects to.
          const int hpad = h + params.pad_top;
          const int wpad = w + params.pad_left;
          const int h_start =
              (hpad < params.window_rows)
                  ? 0
                  : (hpad - params.window_rows) / params.row_stride + 1;
          const int h_end =
              std::min<int>(hpad / params.row_stride + 1, params.out_height);
          const int w_start =
              (wpad < params.window_cols)
                  ? 0
                  : (wpad - params.window_cols) / params.col_stride + 1;
          const int w_end =
              std::min<int>(wpad / params.col_stride + 1, params.out_width);
          const int in_offset =
              (b * params.tensor_in_rows + h) * params.tensor_in_cols + w;
          Eigen::DSizes<Eigen::DenseIndex, 2> in_indices(0, in_offset);
          for (int ph = h_start; ph < h_end; ++ph) {
            for (int pw = w_start; pw < w_end; ++pw) {
              const int out_offset =
                  (b * params.out_height + ph) * params.out_width + pw;
              out_mat.col(out_offset) += in_mat.col(in_offset);
              out_count(out_offset) += T(1);
            }
          }
        }
      }
    }

    DCHECK_GT(out_count.minCoeff(), T(0));
    out_mat.array().rowwise() /= out_count.transpose().array();
  };

  const int64_t work_unit_size =
      params.tensor_in_rows * params.tensor_in_cols * params.depth;
  // NOTE: Constants in calculation below were estimated based on benchmarking.
  // Nanoseconds/work_unit for benchmarks ranged from 0.01 to 0.001, and
  // so the factor 0.01 (i.e. 1/100) with a max of 10000, was chosen to limit
  // the work unit cost to an operating range in which it empirically performed
  // best.
  const int64_t work_unit_cost = std::max(int64_t{10000}, work_unit_size / 100);
  const DeviceBase::CpuWorkerThreads& worker_threads =
      *(context->device()->tensorflow_cpu_worker_threads());
  Shard(worker_threads.num_threads, worker_threads.workers,
        params.tensor_in_batch, work_unit_cost, shard);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_POOLING_OPS_COMMON_H_
