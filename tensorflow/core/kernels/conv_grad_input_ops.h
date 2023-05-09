/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/nn_ops.cc.

#ifndef TENSORFLOW_CORE_KERNELS_CONV_GRAD_INPUT_OPS_H_
#define TENSORFLOW_CORE_KERNELS_CONV_GRAD_INPUT_OPS_H_

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include <algorithm>
#include <limits>
#include <vector>

#include "absl/base/dynamic_annotations.h"
#include "absl/status/status.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/conv_grad_ops.h"
#include "tensorflow/core/kernels/conv_grad_shape_utils.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "tensorflow/core/util/work_sharder.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/tsl/framework/contraction/eigen_contraction_kernel.h"
#endif

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/proto/proto_utils.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#if GOOGLE_CUDA
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_asm_opts.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/redzone_allocator.h"
#include "tensorflow/compiler/xla/stream_executor/tf_allocator_adapter.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Returns in 'im_data' (assumes to be zero-initialized) image patch in storage
// order (height, width, depth), constructed from patches in 'col_data', which
// is required to be in storage order (out_height * out_width, filter_height,
// filter_width, in_depth).  Implementation by Yangqing Jia (jiayq).
template <typename T>
void Col2im(const T* col_data, const int depth, const int height,
            const int width, const int filter_h, const int filter_w,
            const int pad_t, const int pad_l, const int pad_b, const int pad_r,
            const int stride_h, const int stride_w, T* __restrict im_data) {
  int height_col = (height + pad_t + pad_b - filter_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - filter_w) / stride_w + 1;
  int h_pad = -pad_t;
  for (int h = 0; h < height_col; ++h) {
    int w_pad = -pad_l;
    for (int w = 0; w < width_col; ++w) {
      T* im_patch_data = im_data + (h_pad * width + w_pad) * depth;
      for (int ih = h_pad; ih < h_pad + filter_h; ++ih) {
        for (int iw = w_pad; iw < w_pad + filter_w; ++iw) {
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            for (int i = 0; i < depth; ++i) {
              im_patch_data[i] += col_data[i];
            }
          }
          im_patch_data += depth;
          col_data += depth;
        }
        // Jump over remaining number of depth.
        im_patch_data += depth * (width - filter_w);
      }
      w_pad += stride_w;
    }
    h_pad += stride_h;
  }
}

// Computes backprop input using Eigen::SpatialConvolutionBackwardInput on CPU
// and GPU (for int32 only).
template <typename Device, typename T>
struct LaunchConv2DBackpropInputOpImpl {
  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& out_backprop, const Tensor& filter,
                  int row_dilation, int col_dilation, int row_stride,
                  int col_stride, const Padding& padding,
                  const std::vector<int64_t>& explicit_paddings,
                  Tensor* in_backprop, TensorFormat data_format) {
    std::vector<int32> strides(4, 1);
    std::vector<int32> dilations(4, 1);

    auto input_h = GetTensorDimIndex(data_format, 'H');
    auto input_w = GetTensorDimIndex(data_format, 'W');
    strides[input_h] = row_stride;
    strides[input_w] = col_stride;
    dilations[input_h] = row_dilation;
    dilations[input_w] = col_dilation;

    const TensorShape& input_shape = in_backprop->shape();
    const TensorShape& filter_shape = filter.shape();

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(
        ctx, ConvBackpropComputeDimensionsV2(
                 "Conv2DBackpropInput", /*num_spatial_dims=*/2, input_shape,
                 filter_shape, out_backprop.shape(), dilations, strides,
                 padding, explicit_paddings, data_format, &dims));

    int64_t padding_top = -1, padding_bottom = -1;
    int64_t padding_left = -1, padding_right = -1;
    if (padding == EXPLICIT) {
      GetExplicitPaddingForDim(explicit_paddings, data_format, 'H',
                               &padding_top, &padding_bottom);
      GetExplicitPaddingForDim(explicit_paddings, data_format, 'W',
                               &padding_left, &padding_right);
    }

    int64_t expected_out_rows, expected_out_cols;
    // The function is guaranteed to succeed because we checked the output and
    // padding was valid earlier.
    TF_CHECK_OK(GetWindowedOutputSizeVerboseV2(
        dims.spatial_dims[0].input_size, dims.spatial_dims[0].filter_size,
        row_dilation, row_stride, padding, &expected_out_rows, &padding_top,
        &padding_bottom));
    DCHECK_EQ(dims.spatial_dims[0].output_size, expected_out_rows);

    TF_CHECK_OK(GetWindowedOutputSizeVerboseV2(
        dims.spatial_dims[1].input_size, dims.spatial_dims[1].filter_size,
        col_dilation, col_stride, padding, &expected_out_cols, &padding_left,
        &padding_right));
    DCHECK_EQ(dims.spatial_dims[1].output_size, expected_out_cols);

    if (std::is_same<Device, GPUDevice>::value) {
      int64_t size = 1;
#define REQUIRES_32BIT(x)                                                   \
  size *= x;                                                                \
  OP_REQUIRES(ctx,                                                          \
              FastBoundsCheck(x, std::numeric_limits<int32>::max()) &&      \
                  FastBoundsCheck(size, std::numeric_limits<int32>::max()), \
              absl::InvalidArgumentError(absl::StrCat("Tensor too large")))

      REQUIRES_32BIT(in_backprop->dim_size(0));
      REQUIRES_32BIT(in_backprop->dim_size(1) + padding_top + padding_bottom);
      REQUIRES_32BIT(in_backprop->dim_size(2) + padding_left + padding_right);
      REQUIRES_32BIT(in_backprop->dim_size(3));
#undef REQUIRES_32BIT
    }

    auto in_backprop_t = in_backprop->tensor<T, 4>();
    auto out_backprop_t = out_backprop.tensor<T, 4>();
    auto filter_t = filter.tensor<T, 4>();

    // WARNING: Need to swap row/col, padding_top/padding_left, and
    // padding_bottom/padding_right when calling Eigen. Eigen expects tensors
    // in NWHC format, but Tensorflow uses NHWC.

    if (padding != EXPLICIT) {
      // If padding was not explicitly defined, Eigen spatial convolution
      // backward input will infer correct forward paddings from input tensors.
      functor::SpatialConvolutionBackwardInputFunc<Device, T>()(
          ctx->eigen_device<Device>(), in_backprop_t, filter_t, out_backprop_t,
          col_stride, row_stride, col_dilation, row_dilation);
    } else {
      functor::SpatialConvolutionBackwardInputWithExplicitPaddingFunc<Device,
                                                                      T>()(
          ctx->eigen_device<Device>(), in_backprop_t, filter_t, out_backprop_t,
          in_backprop_t.dimension(2) + (padding_left + padding_right),
          in_backprop_t.dimension(1) + (padding_top + padding_bottom),
          col_stride, row_stride, col_dilation, row_dilation, padding_top,
          padding_left);
    }
  }
};

// Computes backprop input using Eigen::SpatialConvolutionBackwardInput on CPU.
template <typename T>
struct LaunchConv2DBackpropInputOp<CPUDevice, T> {
  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& out_backprop, const Tensor& filter,
                  int row_dilation, int col_dilation, int row_stride,
                  int col_stride, const Padding& padding,
                  const std::vector<int64_t>& explicit_paddings,
                  Tensor* in_backprop, TensorFormat data_format) {
    LaunchConv2DBackpropInputOpImpl<CPUDevice, T> launcher;
    launcher(ctx, use_cudnn, cudnn_use_autotune, out_backprop, filter,
             row_dilation, col_dilation, row_stride, col_stride, padding,
             explicit_paddings, in_backprop, data_format);
  }
};

template <typename T>
struct Conv2DCustomBackpropInputMatMulFunctor {
  using MatrixMap = Eigen::Map<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using ConstMatrixMap = Eigen::Map<
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

  void operator()(OpKernelContext* ctx, const T* out_data, const T* filter_data,
                  const int filter_total_size, const int output_image_size,
                  const int dims_out_depth, T* im2col_buf) {
    // Compute gradient into 'im2col_buf'.
    MatrixMap C(im2col_buf, output_image_size, filter_total_size);

    ConstMatrixMap A(out_data, output_image_size, dims_out_depth);
    ConstMatrixMap B(filter_data, filter_total_size, dims_out_depth);

    C.noalias() = A * B.transpose();
  }
};

#if defined(TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL)
template <>
struct Conv2DCustomBackpropInputMatMulFunctor<float> {
  using T = float;

  void operator()(OpKernelContext* ctx, const T* out_data, const T* filter_data,
                  const int filter_total_size, const int output_image_size,
                  const int dims_out_depth, T* im2col_buf) {
    // Inputs are in RowMajor order.
    //   im2col      = out_data    * filter_data^T
    //   [ois x fts] = [ois x dod] * [fts x dod]^T
    //
    // Dimension names:
    //   out_image_size    -> ois
    //   filter_total_size -> fts
    //   dims_out_depth    -> dod

    const int m = output_image_size;
    const int n = filter_total_size;
    const int k = dims_out_depth;  // contraction dim

    const char transposeA = 'N';  // sgemm(A) == filter_data
    const char transposeB = 'T';  // sgemm(B) == out_data

    const int ldA = dims_out_depth;
    const int ldB = dims_out_depth;
    const int ldC = filter_total_size;

    const float alpha = 1.0;
    const float beta = 0.0;

    // dnnl_sgemm code can't be instrumented with msan.
    ANNOTATE_MEMORY_IS_INITIALIZED(
        im2col_buf, filter_total_size * output_image_size * sizeof(T));

    dnnl_status_t st =
        dnnl_sgemm(transposeA, transposeB, m, n, k, alpha, out_data, ldA,
                   filter_data, ldB, beta, im2col_buf, ldC);

    OP_REQUIRES(ctx, st == 0,
                absl::InternalError(absl::StrCat(
                    "Failed to call dnnl_sgemm. Error code: ", st)));
  }
};
#endif

template <typename Device, class T>
class Conv2DBackpropInputOp : public OpKernel {
 public:
  explicit Conv2DBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(
        context, FormatFromString(data_format, &data_format_),
        absl::InvalidArgumentError(absl::StrCat("Invalid data format")));

    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                absl::InvalidArgumentError(
                    absl::StrCat("Sliding window strides field must "
                                 "specify 4 dimensions")));
    int stride_n = GetTensorDim(strides_, data_format_, 'N');
    int stride_c = GetTensorDim(strides_, data_format_, 'C');
    int stride_h = GetTensorDim(strides_, data_format_, 'H');
    int stride_w = GetTensorDim(strides_, data_format_, 'W');
    OP_REQUIRES(
        context, (stride_n == 1 && stride_c == 1),
        absl::UnimplementedError("Current implementation does not yet support "
                                 "strides in the batch and depth dimensions."));
    OP_REQUIRES(context, stride_h > 0 && stride_w > 0,
                absl::InvalidArgumentError(absl::StrCat(
                    "Row and column strides should be larger than 0.")));

    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));
    OP_REQUIRES(context, dilations_.size() == 4,
                absl::InvalidArgumentError(
                    absl::StrCat("Sliding window dilations field must "
                                 "specify 4 dimensions")));
    int dilation_n = GetTensorDim(dilations_, data_format_, 'N');
    int dilation_c = GetTensorDim(dilations_, data_format_, 'C');
    int dilation_h = GetTensorDim(dilations_, data_format_, 'H');
    int dilation_w = GetTensorDim(dilations_, data_format_, 'W');
    OP_REQUIRES(context, (dilation_n == 1 && dilation_c == 1),
                absl::UnimplementedError(
                    "Current implementation does not yet support "
                    "dilations in the batch and depth dimensions."));
    OP_REQUIRES(context, dilation_h > 0 && dilation_w > 0,
                absl::InvalidArgumentError(
                    absl::StrCat("Dilated rates should be larger than 0.")));

    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("explicit_paddings", &explicit_paddings_));
    OP_REQUIRES_OK(context, CheckValidPadding(padding_, explicit_paddings_,
                                              /*num_dims=*/4, data_format_));

    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    cudnn_use_autotune_ = CudnnUseAutotune();

    if (std::is_same<Device, CPUDevice>::value ||
        std::is_same<T, int32>::value) {
      OP_REQUIRES(context, data_format_ == FORMAT_NHWC,
                  absl::InvalidArgumentError(
                      "Conv2DBackpropInputOp [CPU or GPU(int32)] "
                      "only supports NHWC data format."));

      // TODO(yangzihao): Add a CPU implementation for dilated convolution.
      OP_REQUIRES(
          context, (dilation_h == 1 && dilation_w == 1),
          absl::InvalidArgumentError(
              "Conv2DBackpropInputOp [CPU or GPU(int32)] not yet support "
              "dilation rates larger than 1."));
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_sizes = context->input(0);
    const Tensor& filter = context->input(1);
    const Tensor& out_backprop = context->input(2);

    OP_REQUIRES(
        context, out_backprop.dims() == 4,
        absl::InvalidArgumentError(absl::StrCat(
            "input_sizes must be 4-dimensional, got: ", out_backprop.dims())));

    TensorShape input_shape;
    OP_REQUIRES_OK(context,
                   Conv2DBackpropComputeInputShape(input_sizes, filter.shape(),
                                                   out_backprop.shape(),
                                                   data_format_, &input_shape));

    Tensor* in_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));

    // If there is nothing to compute, return.
    if (input_shape.num_elements() == 0) {
      return;
    }

    // If shapes are valid but `out_backprop` is empty, in_backprop should be
    // set to all zeros.  Otherwise, cudnn/dnnl fail with an empty input.
    if (out_backprop.NumElements() == 0) {
      functor::SetZeroFunctor<Device, T> set_zero;
      set_zero(context->eigen_device<Device>(),
               in_backprop->template flat<T>());
      return;
    }

    // For now we take the stride from the second and third dimensions only (we
    // do not support striding on the batch or depth dimension).
    const int stride_rows = GetTensorDim(strides_, data_format_, 'H');
    const int stride_cols = GetTensorDim(strides_, data_format_, 'W');
    const int dilation_rows = GetTensorDim(dilations_, data_format_, 'H');
    const int dilation_cols = GetTensorDim(dilations_, data_format_, 'W');

    VLOG(2) << "Conv2DBackpropInput:"
            << " input: " << input_shape.DebugString()
            << " filter:" << filter.shape().DebugString()
            << " out_backprop: " << out_backprop.shape().DebugString()
            << " strides: [" << stride_rows << ", " << stride_cols << "]"
            << " dilations: [" << dilation_rows << ", " << dilation_cols << "]";

    LaunchConv2DBackpropInputOp<Device, T> launch;
    launch(context, use_cudnn_, cudnn_use_autotune_, out_backprop, filter,
           dilation_rows, dilation_cols, stride_rows, stride_cols, padding_,
           explicit_paddings_, in_backprop, data_format_);
  }

 private:
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  TensorFormat data_format_;
  Padding padding_;
  std::vector<int64_t> explicit_paddings_;

  bool use_cudnn_ = false;
  bool cudnn_use_autotune_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DBackpropInputOp);
};

// Based on implementation written by Yangqing Jia (jiayq).
template <typename Device, class T>
class Conv2DCustomBackpropInputOp : public OpKernel {
 public:
  explicit Conv2DCustomBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                absl::InvalidArgumentError("Invalid data format"));
    OP_REQUIRES(context, data_format_ == FORMAT_NHWC,
                absl::InvalidArgumentError(
                    "Conv2DCustomBackpropInputOp only supports NHWC."));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                absl::InvalidArgumentError("Sliding window strides field must "
                                           "specify 4 dimensions"));
    OP_REQUIRES(
        context, (strides_[0] == 1 && strides_[3] == 1),
        absl::UnimplementedError("Current implementation does not yet support "
                                 "strides in the batch and depth dimensions."));
    OP_REQUIRES(context, strides_[1] > 0 && strides_[2] > 0,
                absl::InvalidArgumentError(
                    "Row and column strides should be larger than 0."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));
    OP_REQUIRES(
        context, dilations_.size() == 4,
        absl::InvalidArgumentError("Sliding window dilations field must "
                                   "specify 4 dimensions"));
    OP_REQUIRES(context, (dilations_[0] == 1 && dilations_[3] == 1),
                absl::UnimplementedError(
                    "Current implementation does not yet support "
                    "dilations in the batch and depth dimensions."));
    // TODO(yangzihao): Add a CPU implementation for dilated convolution.
    OP_REQUIRES(
        context, (dilations_[1] == 1 && dilations_[2] == 1),
        absl::InvalidArgumentError("Current CPU implementations do not yet "
                                   "support dilation rates larger than 1."));
    OP_REQUIRES_OK(context,
                   context->GetAttr("explicit_paddings", &explicit_paddings_));
    OP_REQUIRES_OK(context, CheckValidPadding(padding_, explicit_paddings_,
                                              /*num_dims=*/4, data_format_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_sizes = context->input(0);
    const Tensor& filter = context->input(1);
    const Tensor& out_backprop = context->input(2);
    OP_REQUIRES(
        context, out_backprop.dims() == 4,
        absl::InvalidArgumentError(absl::StrCat(
            "input_sizes must be 4-dimensional, got: ", out_backprop.dims())));

    TensorShape input_shape;
    OP_REQUIRES_OK(context,
                   Conv2DBackpropComputeInputShape(input_sizes, filter.shape(),
                                                   out_backprop.shape(),
                                                   data_format_, &input_shape));

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(context,
                   ConvBackpropComputeDimensionsV2(
                       "Conv2DCustomBackpropInput", /*num_spatial_dims=*/2,
                       input_shape, filter.shape(), out_backprop.shape(),
                       /*dilations=*/{1, 1, 1, 1}, strides_, padding_,
                       explicit_paddings_, data_format_, &dims));

    OP_REQUIRES(context, dims.in_depth == filter.shape().dim_size(2),
                absl::InvalidArgumentError(absl::StrCat(
                    "Gradients for grouped convolutions are not "
                    "supported on CPU. Please file a feature request if you "
                    "run into this issue. Computed input depth ",
                    dims.in_depth, " doesn't match filter input depth ",
                    filter.shape().dim_size(2))));
    OP_REQUIRES(context, dims.out_depth == filter.shape().dim_size(3),
                absl::InvalidArgumentError(
                    absl::StrCat("Computed output depth ", dims.out_depth,
                                 " doesn't match filter output depth ",
                                 filter.shape().dim_size(3))));

    Tensor* in_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));

    // If there is nothing to compute, return.
    if (input_shape.num_elements() == 0) {
      return;
    }

    // If shapes are valid but `out_backprop` is empty, in_backprop should be
    // set to all zeros.  Otherwise, cudnn/dnnl fail with an empty input.
    if (out_backprop.NumElements() == 0) {
      functor::SetZeroFunctor<Device, T> set_zero;
      set_zero(context->eigen_device<Device>(),
               in_backprop->template flat<T>());
      return;
    }

    int64_t pad_top, pad_bottom;
    int64_t pad_left, pad_right;

    if (padding_ == Padding::EXPLICIT) {
      pad_top = explicit_paddings_[2];
      pad_bottom = explicit_paddings_[3];
      pad_left = explicit_paddings_[4];
      pad_right = explicit_paddings_[5];
    }
    OP_REQUIRES_OK(
        context,
        GetWindowedOutputSizeVerbose(
            dims.spatial_dims[0].input_size, dims.spatial_dims[0].filter_size,
            dims.spatial_dims[0].stride, padding_,
            &dims.spatial_dims[0].output_size, &pad_top, &pad_bottom));
    OP_REQUIRES_OK(
        context,
        GetWindowedOutputSizeVerbose(
            dims.spatial_dims[1].input_size, dims.spatial_dims[1].filter_size,
            dims.spatial_dims[1].stride, padding_,
            &dims.spatial_dims[1].output_size, &pad_left, &pad_right));

    // The total dimension size of each kernel.
    const int filter_total_size = dims.spatial_dims[0].filter_size *
                                  dims.spatial_dims[1].filter_size *
                                  dims.in_depth;
    // The output image size is the spatial size of the output.
    const int output_image_size =
        dims.spatial_dims[0].output_size * dims.spatial_dims[1].output_size;

    // TODO(andydavis) Get L2/L3 cache sizes from device.
    const size_t l2_cache_size = 256LL << 10;
    const size_t l3_cache_size = 30LL << 20;

    // Use L3 cache size as target working set size.
    const size_t target_working_set_size = l3_cache_size / sizeof(T);

    // Calculate size of matrices involved in MatMul: C = A x B.
    const size_t size_A = output_image_size * dims.out_depth;

    const size_t size_B = filter_total_size * dims.out_depth;

    const size_t size_C = output_image_size * filter_total_size;

    const size_t work_unit_size = size_A + size_B + size_C;

    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());

    // Calculate per-thread work unit size.
    const size_t thread_work_unit_size =
        work_unit_size / worker_threads.num_threads;

    // Set minimum per-thread work unit size to size of L2 cache.
    const size_t min_thread_work_unit_size = l2_cache_size / sizeof(T);

    // Use parallel tensor contractions if there is no batching, or if the
    // minimum per-thread work unit size threshold has been exceeded.
    // Otherwise, revert to multiple single-threaded matmul ops running in
    // parallel to keep all threads busy.
    // TODO(andydavis) Explore alternatives to branching the code in this way
    // (i.e. run multiple, parallel tensor contractions in another thread pool).
    const bool use_parallel_contraction =
        dims.batch_size == 1 ||
        thread_work_unit_size >= min_thread_work_unit_size;

    OP_REQUIRES(context, work_unit_size > 0,
                absl::InvalidArgumentError(
                    "input, filter_sizes and out_backprop tensors "
                    "must all have at least 1 element"));

    const size_t shard_size =
        use_parallel_contraction
            ? 1
            : (target_working_set_size + work_unit_size - 1) / work_unit_size;

    Tensor col_buffer;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(
                       DataTypeToEnum<T>::value,
                       TensorShape({static_cast<int64_t>(shard_size),
                                    static_cast<int64_t>(output_image_size),
                                    static_cast<int64_t>(filter_total_size)}),
                       &col_buffer));

    // The input offset corresponding to a single input image.
    const int input_offset = dims.spatial_dims[0].input_size *
                             dims.spatial_dims[1].input_size * dims.in_depth;
    // The output offset corresponding to a single output image.
    const int output_offset = dims.spatial_dims[0].output_size *
                              dims.spatial_dims[1].output_size * dims.out_depth;

    const T* filter_data = filter.template flat<T>().data();
    T* col_buffer_data = col_buffer.template flat<T>().data();
    const T* out_backprop_data = out_backprop.template flat<T>().data();

    auto in_backprop_flat = in_backprop->template flat<T>();
    T* input_backprop_data = in_backprop_flat.data();
    in_backprop_flat.device(context->eigen_device<Device>()) =
        in_backprop_flat.constant(T(0));

    if (use_parallel_contraction) {
      typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>,
                               Eigen::Unaligned>
          TensorMap;
      typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>,
                               Eigen::Unaligned>
          ConstTensorMap;

      // Initialize contraction dims (we need to transpose 'B' below).
      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_dims;
      contract_dims[0].first = 1;
      contract_dims[0].second = 1;

      for (int image_id = 0; image_id < dims.batch_size; ++image_id) {
        // Compute gradient into col_buffer.
        TensorMap C(col_buffer_data, output_image_size, filter_total_size);

        ConstTensorMap A(out_backprop_data + output_offset * image_id,
                         output_image_size, dims.out_depth);
        ConstTensorMap B(filter_data, filter_total_size, dims.out_depth);

        C.device(context->eigen_cpu_device()) = A.contract(B, contract_dims);

        Col2im<T>(
            col_buffer_data, dims.in_depth, dims.spatial_dims[0].input_size,
            dims.spatial_dims[1].input_size, dims.spatial_dims[0].filter_size,
            dims.spatial_dims[1].filter_size, pad_top, pad_left, pad_bottom,
            pad_right, dims.spatial_dims[0].stride, dims.spatial_dims[1].stride,
            input_backprop_data);

        input_backprop_data += input_offset;
      }
    } else {
      for (int image_id = 0; image_id < dims.batch_size;
           image_id += shard_size) {
        const int shard_limit =
            std::min(static_cast<int>(shard_size),
                     static_cast<int>(dims.batch_size) - image_id);

        auto shard = [&context, &dims, &pad_top, &pad_left, &pad_bottom,
                      &pad_right, &output_image_size, &filter_total_size,
                      &input_backprop_data, &col_buffer_data,
                      &out_backprop_data, &filter_data, &input_offset,
                      &output_offset, &size_C](int64_t start, int64_t limit) {
          for (int shard_id = start; shard_id < limit; ++shard_id) {
            T* im2col_buf = col_buffer_data + shard_id * size_C;
            T* input_data = input_backprop_data + shard_id * input_offset;
            const T* out_data = out_backprop_data + shard_id * output_offset;

            Conv2DCustomBackpropInputMatMulFunctor<T>()(
                context, out_data, filter_data, filter_total_size,
                output_image_size, dims.out_depth, im2col_buf);

            Col2im<T>(im2col_buf, dims.in_depth,
                      dims.spatial_dims[0].input_size,
                      dims.spatial_dims[1].input_size,
                      dims.spatial_dims[0].filter_size,
                      dims.spatial_dims[1].filter_size, pad_top, pad_left,
                      pad_bottom, pad_right, dims.spatial_dims[0].stride,
                      dims.spatial_dims[1].stride, input_data);
          }
        };
        Shard(worker_threads.num_threads, worker_threads.workers, shard_limit,
              work_unit_size, shard);

        input_backprop_data += input_offset * shard_limit;
        out_backprop_data += output_offset * shard_limit;
      }
    }
  }

 private:
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  Padding padding_;
  std::vector<int64_t> explicit_paddings_;
  TensorFormat data_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DCustomBackpropInputOp);
};

// TODO(ezhulenev): Add a cost model to switch between custom/Eigen ops.
#define DEFAULT_CONV_2D_BACKPROP_CPU_OP Conv2DCustomBackpropInputOp

#define REGISTER_CONV_2D_BACKPROP_CPU_KERNELS(T)                             \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("Conv2DBackpropInput").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DEFAULT_CONV_2D_BACKPROP_CPU_OP<CPUDevice, T>);                        \
  REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")                        \
                              .Device(DEVICE_CPU)                            \
                              .Label("custom")                               \
                              .TypeConstraint<T>("T"),                       \
                          Conv2DCustomBackpropInputOp<CPUDevice, T>);        \
  REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")                        \
                              .Device(DEVICE_CPU)                            \
                              .Label("eigen_tensor")                         \
                              .TypeConstraint<T>("T"),                       \
                          Conv2DBackpropInputOp<CPUDevice, T>);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONV_GRAD_INPUT_OPS_H_
