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

// See docs in ../ops/nn_ops.cc.

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/conv_grad_ops.h"

#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/fill_functor.h"
#ifdef TENSORFLOW_USE_LIBXSMM_CONVOLUTIONS
#include "tensorflow/core/kernels/xsmm_conv2d.h"
#endif
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "tensorflow/core/util/work_sharder.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"
#endif

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/protobuf/autotuning.pb.h"
#include "tensorflow/core/util/proto/proto_utils.h"
#endif  // GOOGLE_CUDA

namespace {

// Returns in 'col_data', image patches in storage order (height, width, depth)
// extracted from image at 'input_data', which is required to be in storage
// order (batch, height, width, depth).
// Implementation written by Yangqing Jia (jiayq).
template <typename T>
void Im2col(const T* input_data, const int depth, const int height,
            const int width, const int filter_h, const int filter_w,
            const int pad_t, const int pad_l, const int pad_b, const int pad_r,
            const int stride_h, const int stride_w, T* col_data) {
  int height_col = (height + pad_t + pad_b - filter_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - filter_w) / stride_w + 1;

  int h_pad = -pad_t;
  for (int h = 0; h < height_col; ++h) {
    int w_pad = -pad_l;
    for (int w = 0; w < width_col; ++w) {
      for (int ih = h_pad; ih < h_pad + filter_h; ++ih) {
        for (int iw = w_pad; iw < w_pad + filter_w; ++iw) {
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            memcpy(col_data, input_data + (ih * width + iw) * depth,
                   sizeof(T) * depth);
          } else {
            // This should be simply padded with zero.
            memset(col_data, 0, sizeof(T) * depth);
          }
          col_data += depth;
        }
      }
      w_pad += stride_w;
    }
    h_pad += stride_h;
  }
}

}  // namespace

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename T>
struct LaunchConv2DBackpropFilterOp<CPUDevice, T> {
  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& out_backprop, const Tensor& input,
                  int row_dilation, int col_dilation, int row_stride,
                  int col_stride, const Padding& padding,
                  const std::vector<int64>& explicit_paddings,
                  Tensor* filter_backprop, TensorFormat data_format) {
    const CPUDevice& d = ctx->eigen_device<CPUDevice>();
    functor::SpatialConvolutionBackwardFilter<CPUDevice, T>()(
        d, filter_backprop->tensor<T, 4>(), input.tensor<T, 4>(),
        out_backprop.tensor<T, 4>(), row_stride, col_stride,
        /*row_dilation=*/1, /*col_dilation=*/1);
  }
};

#ifdef TENSORFLOW_USE_LIBXSMM_CONVOLUTIONS
template <typename Device, class T>
struct LaunchXsmmBackwardFilter {
  bool operator()(OpKernelContext* context, const Device& d,
                  typename TTypes<T, 4>::ConstTensor input_backward,
                  typename TTypes<T, 4>::Tensor kernel,
                  typename TTypes<T, 4>::ConstTensor output_backward,
                  int input_rows, int input_cols, int row_stride,
                  int col_stride, int pad_h, int pad_w,
                  TensorFormat data_format) const {
    return false;
  }
};

template <>
struct LaunchXsmmBackwardFilter<CPUDevice, float> {
  bool operator()(OpKernelContext* context, const CPUDevice& d,
                  typename TTypes<float, 4>::ConstTensor input,
                  typename TTypes<float, 4>::Tensor filter,
                  typename TTypes<float, 4>::ConstTensor output, int input_rows,
                  int input_cols, int row_stride, int col_stride, int pad_h,
                  int pad_w, TensorFormat data_format) const {
    auto batch = input.dimension(0);
    auto in_depth = input.dimension(3);
    auto out_depth = output.dimension(3);
    auto filter_rows = filter.dimension(0);
    auto filter_cols = filter.dimension(1);

    auto num_threads =
        context->device()->tensorflow_cpu_worker_threads()->num_threads;
    // See libxsmm_dnn.h for this struct definition.
    libxsmm_dnn_conv_desc desc;
    desc.N = batch;
    desc.C = in_depth;
    desc.H = input_rows;
    desc.W = input_cols;
    desc.K = out_depth;
    desc.R = filter_rows;
    desc.S = filter_cols;
    desc.u = row_stride;
    desc.v = col_stride;
    desc.pad_h = pad_h;
    desc.pad_w = pad_w;
    desc.pad_h_in = 0;  // pad_rows;  // ignored by libxsmm for now.
    desc.pad_w_in = 0;  // pad_cols;  // ignored by libxsmm for now.
    desc.pad_h_out = 0;
    desc.pad_w_out = 0;
    desc.threads = num_threads;
    desc.algo = LIBXSMM_DNN_CONV_ALGO_DIRECT;
    desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_NHWC;
    desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_RSCK;
    desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
    desc.options = LIBXSMM_DNN_CONV_OPTION_NONE;
    desc.datatype = LIBXSMM_DNN_DATATYPE_F32;

    if (!CanUseXsmmConv2D(desc, data_format)) {
      return false;
    }

    auto input_ptr = input.data();
    auto filter_ptr = filter.data();
    auto output_ptr = output.data();
    bool success = functor::XsmmBkwFilterConv2D<CPUDevice, float>()(
        context, desc, input_ptr, filter_ptr, output_ptr);
    return success;
  }
};
#endif

// Based on implementation written by Yangqing Jia (jiayq).
template <typename Device, class T>
class Conv2DCustomBackpropFilterOp : public OpKernel {
 public:
  explicit Conv2DCustomBackpropFilterOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, data_format_ == FORMAT_NHWC,
                errors::InvalidArgument(
                    "Conv2DCustomBackpropFilterOp only supports NHWC."));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(
        context, (strides_[0] == 1 && strides_[3] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES(context, strides_[1] > 0 && strides_[2] > 0,
                errors::InvalidArgument(
                    "Row and column strides should be larger than 0."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(
        context, padding_ != Padding::EXPLICIT,
        errors::Unimplemented("Current CPU implementation does not support "
                              "EXPLICIT padding yet."));
    std::vector<int64> explicit_paddings;
    OP_REQUIRES_OK(context,
                   context->GetAttr("explicit_paddings", &explicit_paddings));
    OP_REQUIRES_OK(context, CheckValidPadding(padding_, explicit_paddings,
                                              /*num_dims=*/4, data_format_));
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));
    OP_REQUIRES(context, dilations_.size() == 4,
                errors::InvalidArgument("Sliding window dilations field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, (dilations_[0] == 1 && dilations_[3] == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilations in the batch and depth dimensions."));
    // TODO(yangzihao): Add a CPU implementation for dilated convolution.
    OP_REQUIRES(context, (dilations_[1] == 1 && dilations_[2] == 1),
                errors::InvalidArgument(
                    "Current libxsmm and customized CPU implementations do "
                    "not yet support dilation rates larger than 1."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& filter_sizes = context->input(1);
    const Tensor& out_backprop = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(filter_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DCustomBackpropFilter: filter_sizes input must be 1-dim, "
            "not ",
            filter_sizes.dims()));
    TensorShape filter_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                filter_sizes.vec<int32>(), &filter_shape));

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(context,
                   ConvBackpropComputeDimensions(
                       "Conv2DCustomBackpropFilter", /*num_spatial_dims=*/2,
                       input.shape(), filter_shape, out_backprop.shape(),
                       strides_, padding_, data_format_, &dims));

    Tensor* filter_backprop;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));

    // If there is nothing to compute, return.
    if (filter_shape.num_elements() == 0) {
      return;
    }

    int64 pad_top, pad_bottom;
    int64 pad_left, pad_right;
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
#if defined TENSORFLOW_USE_LIBXSMM_CONVOLUTIONS && \
    defined TENSORFLOW_USE_LIBXSMM_BACKWARD_CONVOLUTIONS
    if (pad_left == pad_right && pad_top == pad_bottom) {
      if (LaunchXsmmBackwardFilter<Device, T>()(
              context, context->eigen_device<Device>(), input.tensor<T, 4>(),
              filter_backprop->tensor<T, 4>(), out_backprop.tensor<T, 4>(),
              dims.spatial_dims[0].input_size, dims.spatial_dims[1].input_size,
              static_cast<int>(dims.spatial_dims[0].stride),
              static_cast<int>(dims.spatial_dims[1].stride),
              static_cast<int>(pad_top), static_cast<int>(pad_left),
              data_format_)) {
        return;
      }
    }
#endif

    // The total dimension size of each kernel.
    const int filter_total_size = dims.spatial_dims[0].filter_size *
                                  dims.spatial_dims[1].filter_size *
                                  dims.in_depth;
    // The output image size is the spatial size of the output.
    const int output_image_size =
        dims.spatial_dims[0].output_size * dims.spatial_dims[1].output_size;

    // Shard 'batch' images into 'shard_size' groups of images to be fed
    // into the parallel matmul. Calculate 'shard_size' by dividing the L3 cache
    // size ('target_working_set_size') by the matmul size of an individual
    // image ('work_unit_size').

    // TODO(andydavis)
    // *) Get L3 cache size from device at runtime (30MB is from ivybridge).
    // *) Consider reducing 'target_working_set_size' if L3 is shared by
    //    other concurrently running tensorflow ops.
    const size_t target_working_set_size = (30LL << 20) / sizeof(T);

    const size_t size_A = output_image_size * filter_total_size;

    const size_t size_B = output_image_size * dims.out_depth;

    const size_t size_C = filter_total_size * dims.out_depth;

    const size_t work_unit_size = size_A + size_B + size_C;

    const size_t shard_size =
        (target_working_set_size + work_unit_size - 1) / work_unit_size;

    Tensor col_buffer;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(
                       DataTypeToEnum<T>::value,
                       TensorShape({static_cast<int64>(shard_size),
                                    static_cast<int64>(output_image_size),
                                    static_cast<int64>(filter_total_size)}),
                       &col_buffer));

    // The input offset corresponding to a single input image.
    const int input_offset = dims.spatial_dims[0].input_size *
                             dims.spatial_dims[1].input_size * dims.in_depth;
    // The output offset corresponding to a single output image.
    const int output_offset = dims.spatial_dims[0].output_size *
                              dims.spatial_dims[1].output_size * dims.out_depth;

    const T* input_data = input.template flat<T>().data();
    T* col_buffer_data = col_buffer.template flat<T>().data();
    const T* out_backprop_data = out_backprop.template flat<T>().data();
    T* filter_backprop_data = filter_backprop->template flat<T>().data();

    typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>,
                             Eigen::Unaligned>
        TensorMap;
    typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>,
                             Eigen::Unaligned>
        ConstTensorMap;

    TensorMap C(filter_backprop_data, filter_total_size, dims.out_depth);
    C.setZero();

    // Initialize contraction dims (we need to transpose 'A' below).
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_dims;
    contract_dims[0].first = 0;
    contract_dims[0].second = 0;

    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());

    for (int image_id = 0; image_id < dims.batch_size; image_id += shard_size) {
      const int shard_limit =
          std::min(static_cast<int>(shard_size),
                   static_cast<int>(dims.batch_size) - image_id);

      auto shard = [&input_data, &col_buffer_data, &dims, &pad_top, &pad_left,
                    &pad_bottom, &pad_right, &input_offset,
                    &size_A](int64 start, int64 limit) {
        for (int shard_id = start; shard_id < limit; ++shard_id) {
          const T* input_data_shard = input_data + shard_id * input_offset;
          T* col_data_shard = col_buffer_data + shard_id * size_A;

          // When we compute the gradient with respect to the filters, we need
          // to do im2col to allow gemm-type computation.
          Im2col<T>(
              input_data_shard, dims.in_depth, dims.spatial_dims[0].input_size,
              dims.spatial_dims[1].input_size, dims.spatial_dims[0].filter_size,
              dims.spatial_dims[1].filter_size, pad_top, pad_left, pad_bottom,
              pad_right, dims.spatial_dims[0].stride,
              dims.spatial_dims[1].stride, col_data_shard);
        }
      };
      Shard(worker_threads.num_threads, worker_threads.workers, shard_limit,
            size_A, shard);

      ConstTensorMap A(col_buffer_data, output_image_size * shard_limit,
                       filter_total_size);
      ConstTensorMap B(out_backprop_data, output_image_size * shard_limit,
                       dims.out_depth);

      // Gradient with respect to filter.
      C.device(context->eigen_cpu_device()) += A.contract(B, contract_dims);

      input_data += input_offset * shard_limit;
      out_backprop_data += output_offset * shard_limit;
    }
  }

 private:
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DCustomBackpropFilterOp);
};

#define REGISTER_CPU_KERNELS(T)                                               \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("Conv2DBackpropFilter").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      Conv2DCustomBackpropFilterOp<CPUDevice, T>);                            \
  REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropFilter")                        \
                              .Device(DEVICE_CPU)                             \
                              .Label("custom")                                \
                              .TypeConstraint<T>("T"),                        \
                          Conv2DCustomBackpropFilterOp<CPUDevice, T>);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

// To be used inside depthwise_conv_grad_op.cc.
template struct LaunchConv2DBackpropFilterOp<CPUDevice, Eigen::half>;
template struct LaunchConv2DBackpropFilterOp<CPUDevice, float>;
template struct LaunchConv2DBackpropFilterOp<CPUDevice, double>;

// GPU definitions.
#if GOOGLE_CUDA
// The slow version (but compiles for GPU)

// A dummy type to group forward backward filter autotune results together.
struct ConvBackwardFilterAutoTuneGroup {
  static string name() { return "ConvBwdFilter"; }
};
typedef AutoTuneSingleton<ConvBackwardFilterAutoTuneGroup, ConvParameters,
                          se::dnn::AlgorithmConfig>
    AutoTuneConvBwdFilter;

// Backprop for filter.
template <typename Device, class T>
class Conv2DSlowBackpropFilterOp : public OpKernel {
 public:
  explicit Conv2DSlowBackpropFilterOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    int stride_n = GetTensorDim(strides_, data_format_, 'N');
    int stride_c = GetTensorDim(strides_, data_format_, 'C');
    int stride_h = GetTensorDim(strides_, data_format_, 'H');
    int stride_w = GetTensorDim(strides_, data_format_, 'W');
    OP_REQUIRES(
        context, (stride_n == 1 && stride_c == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES(context, stride_h > 0 && stride_w > 0,
                errors::InvalidArgument(
                    "Row and column strides should be larger than 0."));
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));
    OP_REQUIRES(context, dilations_.size() == 4,
                errors::InvalidArgument("Sliding window dilations field must "
                                        "specify 4 dimensions"));
    int dilation_n = GetTensorDim(dilations_, data_format_, 'N');
    int dilation_c = GetTensorDim(dilations_, data_format_, 'C');
    int dilation_h = GetTensorDim(dilations_, data_format_, 'H');
    int dilation_w = GetTensorDim(dilations_, data_format_, 'W');
    OP_REQUIRES(context, dilation_n == 1 && dilation_c == 1,
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilations in the batch and depth dimensions."));
    OP_REQUIRES(
        context, dilation_h > 0 && dilation_w > 0,
        errors::InvalidArgument("Dilated rates should be larger than 0."));
    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    use_cudnn_ &= CanUseCudnn();
    cudnn_use_autotune_ = CudnnUseAutotune();
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("explicit_paddings", &explicit_paddings_));
    OP_REQUIRES_OK(context, CheckValidPadding(padding_, explicit_paddings_,
                                              /*num_dims=*/4, data_format_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& filter_sizes = context->input(1);
    const Tensor& out_backprop = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(filter_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropFilter: filter_sizes input must be 1-dim, not ",
            filter_sizes.dims()));
    TensorShape filter_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                filter_sizes.vec<int32>(), &filter_shape));

    Tensor* filter_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));

    // If there is nothing to compute, return.
    if (filter_shape.num_elements() == 0) {
      return;
    }
    // If input is empty, set gradients to zero.
    if (input.shape().num_elements() == 0) {
      functor::SetZeroFunctor<Device, T> f;
      f(context->eigen_device<Device>(), filter_backprop->flat<T>());
      return;
    }

    // For now we take the stride from the second and third dimensions only (we
    // do not support striding on the batch or depth dimension).
    const int stride_rows = GetTensorDim(strides_, data_format_, 'H');
    const int stride_cols = GetTensorDim(strides_, data_format_, 'W');
    const int dilation_rows = GetTensorDim(dilations_, data_format_, 'H');
    const int dilation_cols = GetTensorDim(dilations_, data_format_, 'W');

    launcher_(context, use_cudnn_, cudnn_use_autotune_, out_backprop, input,
              dilation_rows, dilation_cols, stride_rows, stride_cols, padding_,
              explicit_paddings_, filter_backprop, data_format_);
  }

 private:
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  Padding padding_;
  std::vector<int64> explicit_paddings_;
  bool use_cudnn_;
  TensorFormat data_format_;
  LaunchConv2DBackpropFilterOp<Device, T> launcher_;
  bool cudnn_use_autotune_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DSlowBackpropFilterOp);
};

template <typename T>
void LaunchConv2DBackpropFilterOp<Eigen::GpuDevice, T>::operator()(
    OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
    const Tensor& out_backprop, const Tensor& input, int row_dilation,
    int col_dilation, int row_stride, int col_stride, const Padding& padding,
    const std::vector<int64>& explicit_paddings, Tensor* filter_backprop,
    TensorFormat data_format) {
  using se::dnn::AlgorithmConfig;
  using se::dnn::AlgorithmDesc;
  using se::dnn::ProfileResult;

  std::vector<int32> dilations(4, 1);
  dilations[GetTensorDimIndex(data_format, 'H')] = row_dilation;
  dilations[GetTensorDimIndex(data_format, 'W')] = col_dilation;

  std::vector<int32> strides(4, 1);
  strides[GetTensorDimIndex(data_format, 'H')] = row_stride;
  strides[GetTensorDimIndex(data_format, 'W')] = col_stride;
  TensorShape filter_shape = filter_backprop->shape();

  ConvBackpropDimensions dims;
  OP_REQUIRES_OK(
      ctx, ConvBackpropComputeDimensionsV2(
               "Conv2DSlowBackpropFilter", /*num_spatial_dims=*/2,
               input.shape(), filter_shape, out_backprop.shape(), dilations,
               strides, padding, explicit_paddings, data_format, &dims));

  int64 padding_top = -1, padding_bottom = -1;
  int64 padding_left = -1, padding_right = -1;
  if (padding == EXPLICIT) {
    GetExplicitPaddingForDim(explicit_paddings, data_format, 'H', &padding_top,
                             &padding_bottom);
    GetExplicitPaddingForDim(explicit_paddings, data_format, 'W', &padding_left,
                             &padding_right);
  }
  int64 expected_out_rows, expected_out_cols;
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

  auto* stream = ctx->op_device_context()->stream();
  OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

  if (!use_cudnn) {
    ctx->SetStatus(errors::Unimplemented(
        "Conv2DBackprop for GPU is not currently supported "
        "without cudnn"));
    return;
  }

  // If the filter in-depth (filter_shape.dim_size(2)) is 1 and smaller than the
  // input depth, it's a depthwise convolution. More generally, if the filter
  // in-depth divides but is smaller than the input depth, it is a grouped
  // convolution.
  bool is_grouped_convolution = filter_shape.dim_size(2) != dims.in_depth;
  bool cudnn_disable_conv_1x1_optimization_ = CudnnDisableConv1x1Optimization();
  if (!cudnn_disable_conv_1x1_optimization_ &&
      dims.spatial_dims[0].filter_size == 1 &&
      dims.spatial_dims[1].filter_size == 1 && !is_grouped_convolution &&
      dims.spatial_dims[0].stride == 1 && dims.spatial_dims[1].stride == 1 &&
      data_format == FORMAT_NHWC && (padding == VALID || padding == SAME)) {
    const uint64 m = dims.in_depth;
    const uint64 k = dims.batch_size * dims.spatial_dims[0].input_size *
                     dims.spatial_dims[1].input_size;
    const uint64 n = dims.out_depth;

    // The shape of output backprop is
    //   [batch, out_rows, out_cols, out_depth]
    //   From cublas's perspective, it is: n x k
    auto a_ptr = AsDeviceMemory(out_backprop.template flat<T>().data(),
                                out_backprop.template flat<T>().size());

    // The shape of input is
    //   [batch, in_rows, in_cols, in_depth],
    //   From cublas's perspective, it is: m x k
    auto b_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                input.template flat<T>().size());

    // the shape of the filter backprop from the conv_2d should be
    //   [1, 1, in_depth, out_depth]
    //   From cublas's perspective, it is: n x m
    auto c_ptr = AsDeviceMemory(filter_backprop->template flat<T>().data(),
                                filter_backprop->template flat<T>().size());

    bool blas_launch_status =
        stream
            ->ThenBlasGemm(se::blas::Transpose::kNoTranspose,
                           se::blas::Transpose::kTranspose, n, m, k, 1.0f,
                           a_ptr, n, b_ptr, m, 0.0f, &c_ptr, n)
            .ok();
    if (!blas_launch_status) {
      ctx->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                      ", n=", n, ", k=", k));
    }
    return;
  } else if (dims.spatial_dims[0].filter_size ==
                 dims.spatial_dims[0].input_size &&
             dims.spatial_dims[1].filter_size ==
                 dims.spatial_dims[1].input_size &&
             !is_grouped_convolution && padding == VALID &&
             data_format == FORMAT_NHWC) {
    // The input data and filter have the same height/width, and we are not
    // using grouped convolution, so call cublas directly.
    const uint64 m = dims.spatial_dims[0].input_size *
                     dims.spatial_dims[1].input_size * dims.in_depth;
    const uint64 k = dims.batch_size;
    const uint64 n = dims.out_depth;

    auto a_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                input.template flat<T>().size());
    auto b_ptr = AsDeviceMemory(out_backprop.template flat<T>().data(),
                                out_backprop.template flat<T>().size());
    auto c_ptr = AsDeviceMemory(filter_backprop->template flat<T>().data(),
                                filter_backprop->template flat<T>().size());

    bool blas_launch_status =
        stream
            ->ThenBlasGemm(se::blas::Transpose::kNoTranspose,
                           se::blas::Transpose::kTranspose, n, m, k, 1.0f,
                           b_ptr, n, a_ptr, m, 0.0f, &c_ptr, n)
            .ok();
    if (!blas_launch_status) {
      ctx->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                      ", n=", n, ", k=", k));
    }
    return;
  }

  const int64 common_padding_rows = std::min(padding_top, padding_bottom);
  const int64 common_padding_cols = std::min(padding_left, padding_right);
  Tensor compatible_input;
  if (padding_top != padding_bottom || padding_left != padding_right) {
    // Pad the input in the same way we did during the forward pass, so that
    // cuDNN receives the same input during the backward pass function as it did
    // during the forward pass function.
    const int64 padding_rows_diff = std::abs(padding_bottom - padding_top);
    const int64 padding_cols_diff = std::abs(padding_right - padding_left);
    const int64 new_in_rows =
        dims.spatial_dims[0].input_size + padding_rows_diff;
    const int64 new_in_cols =
        dims.spatial_dims[1].input_size + padding_cols_diff;
    const int64 input_pad_top = padding_top - common_padding_rows;
    const int64 input_pad_bottom = padding_bottom - common_padding_rows;
    const int64 input_pad_left = padding_left - common_padding_cols;
    const int64 input_pad_right = padding_right - common_padding_cols;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(
                 DataTypeToEnum<T>::value,
                 ShapeFromFormat(data_format, dims.batch_size, new_in_rows,
                                 new_in_cols, dims.in_depth),
                 &compatible_input));

    functor::PadInput<GPUDevice, T, int, 4>()(
        ctx->template eigen_device<GPUDevice>(), To32Bit(input.tensor<T, 4>()),
        {{static_cast<int>(input_pad_top), static_cast<int>(input_pad_left)}},
        {{static_cast<int>(input_pad_bottom),
          static_cast<int>(input_pad_right)}},
        To32Bit(compatible_input.tensor<T, 4>()), data_format);
  } else {
    compatible_input = input;
  }

  CHECK(common_padding_rows >= 0 && common_padding_cols >= 0)  // Crash OK
      << "Negative row or col paddings: (" << common_padding_rows << ", "
      << common_padding_cols << ")";
  se::dnn::BatchDescriptor input_desc;
  input_desc.set_count(dims.batch_size)
      .set_height(GetTensorDim(compatible_input, data_format, 'H'))
      .set_width(GetTensorDim(compatible_input, data_format, 'W'))
      .set_feature_map_count(dims.in_depth)
      .set_layout(se::dnn::DataLayout::kBatchDepthYX);
  se::dnn::BatchDescriptor output_desc;
  output_desc.set_count(dims.batch_size)
      .set_height(dims.spatial_dims[0].output_size)
      .set_width(dims.spatial_dims[1].output_size)
      .set_feature_map_count(dims.out_depth)
      .set_layout(se::dnn::DataLayout::kBatchDepthYX);
  se::dnn::FilterDescriptor filter_desc;
  filter_desc.set_input_filter_height(dims.spatial_dims[0].filter_size)
      .set_input_filter_width(dims.spatial_dims[1].filter_size)
      .set_input_feature_map_count(filter_shape.dim_size(2))
      .set_output_feature_map_count(filter_shape.dim_size(3));
  se::dnn::ConvolutionDescriptor conv_desc;
  conv_desc.set_vertical_dilation_rate(dims.spatial_dims[0].dilation)
      .set_horizontal_dilation_rate(dims.spatial_dims[1].dilation)
      .set_vertical_filter_stride(dims.spatial_dims[0].stride)
      .set_horizontal_filter_stride(dims.spatial_dims[1].stride)
      .set_zero_padding_height(common_padding_rows)
      .set_zero_padding_width(common_padding_cols)
      .set_group_count(dims.in_depth / filter_shape.dim_size(2));

  // NOTE(zhengxq):
  // cuDNN only supports the filter layouts: OD x FD x R x C
  // Whereas, we have: R x C x ID x OD
  // TransformFilter performs (R x C x FD x OD) => (OD x FD x R x C)

  Tensor pre_transformed_filter_backprop;
  OP_REQUIRES_OK(
      ctx,
      ctx->allocate_temp(
          DataTypeToEnum<T>::value,
          TensorShape({filter_shape.dim_size(3), filter_shape.dim_size(2),
                       filter_shape.dim_size(0), filter_shape.dim_size(1)}),
          &pre_transformed_filter_backprop));

  Tensor transformed_out_backprop;
  if (data_format == FORMAT_NHWC) {
    TensorShape nchw_shape = ShapeFromFormat(
        FORMAT_NCHW, dims.batch_size, dims.spatial_dims[0].output_size,
        dims.spatial_dims[1].output_size, dims.out_depth);
    if (dims.out_depth > 1) {
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DataTypeToEnum<T>::value, nchw_shape,
                                        &transformed_out_backprop));
      functor::NHWCToNCHW<GPUDevice, T, 4>()(
          ctx->eigen_device<GPUDevice>(), out_backprop.tensor<T, 4>(),
          transformed_out_backprop.tensor<T, 4>());
    } else {
      // If depth <= 1, just reshape.
      CHECK(transformed_out_backprop.CopyFrom(out_backprop, nchw_shape));
    }
  } else {
    transformed_out_backprop = out_backprop;
  }

  Tensor transformed_input;
  if (data_format == FORMAT_NHWC) {
    TensorShape nchw_shape = ShapeFromFormat(
        FORMAT_NCHW, GetTensorDim(compatible_input, data_format, 'N'),
        GetTensorDim(compatible_input, data_format, 'H'),
        GetTensorDim(compatible_input, data_format, 'W'),
        GetTensorDim(compatible_input, data_format, 'C'));
    if (nchw_shape.dim_size(1) > 1) {
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                             nchw_shape, &transformed_input));
      functor::NHWCToNCHW<GPUDevice, T, 4>()(
          ctx->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(compatible_input).tensor<T, 4>(),
          transformed_input.tensor<T, 4>());
    } else {
      // If depth <= 1, just reshape.
      CHECK(transformed_input.CopyFrom(compatible_input, nchw_shape));
    }
  } else {
    transformed_input = compatible_input;
  }

  auto out_backprop_ptr =
      AsDeviceMemory(transformed_out_backprop.template flat<T>().data(),
                     transformed_out_backprop.template flat<T>().size());
  auto filter_backprop_ptr =
      AsDeviceMemory(pre_transformed_filter_backprop.template flat<T>().data(),
                     pre_transformed_filter_backprop.template flat<T>().size());
  auto input_ptr = AsDeviceMemory(transformed_input.template flat<T>().data(),
                                  transformed_input.template flat<T>().size());

  static int64 ConvolveBackwardFilterScratchSize = GetDnnWorkspaceLimit(
      "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB by default
  );
  int device_id = stream->parent()->device_ordinal();
  DataType dtype = input.dtype();
  ConvParameters conv_parameters = {
      dims.batch_size,                     // batch
      dims.in_depth,                       // in_depths
      {{input_desc.height(),               // in_rows
        input_desc.width()}},              // in_cols
      FORMAT_NCHW,                         // compute_data_format
      dims.out_depth,                      // out_depths
      {{dims.spatial_dims[0].filter_size,  // filter_rows
        dims.spatial_dims[1].filter_size,  // filter_cols
        filter_shape.dim_size(2)}},        // filter_depth
      {{dims.spatial_dims[0].dilation,     // dilation_rows
        dims.spatial_dims[1].dilation}},   // dilation_cols
      {{dims.spatial_dims[0].stride,       // stride_rows
        dims.spatial_dims[1].stride}},     // stride_cols
      {{common_padding_rows,               // padding_rows
        common_padding_cols}},             // padding_cols
      dtype,                               // tensor datatype
      device_id,                           // device_id
  };
  AlgorithmConfig algorithm_config;
  if (cudnn_use_autotune && !AutoTuneConvBwdFilter::GetInstance()->Find(
                                conv_parameters, &algorithm_config)) {
    std::vector<AlgorithmDesc> algorithms;
    CHECK(stream->parent()->GetConvolveBackwardFilterAlgorithms(
        conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(stream->parent()),
        &algorithms));
    std::vector<tensorflow::AutotuneResult> results;
    for (auto profile_algorithm : algorithms) {
      // TODO(zhengxq): profile each algorithm multiple times to better
      // accuracy.
      DnnScratchAllocator scratch_allocator(ConvolveBackwardFilterScratchSize,
                                            ctx);
      ProfileResult profile_result;
      bool cudnn_launch_status =
          stream
              ->ThenConvolveBackwardFilterWithAlgorithm(
                  input_desc, input_ptr, output_desc, out_backprop_ptr,
                  conv_desc, filter_desc, &filter_backprop_ptr,
                  &scratch_allocator, AlgorithmConfig(profile_algorithm),
                  &profile_result)
              .ok();
      if (cudnn_launch_status && profile_result.is_valid()) {
        results.emplace_back();
        auto& result = results.back();
        result.mutable_conv()->set_algorithm(profile_algorithm.algo_id());
        result.mutable_conv()->set_tensor_ops_enabled(
            profile_algorithm.tensor_ops_enabled());
        result.set_scratch_bytes(scratch_allocator.TotalByteSize());
        *result.mutable_run_time() = proto_utils::ToDurationProto(
            absl::Milliseconds(profile_result.elapsed_time_in_ms()));
      }
    }
    LogConvAutotuneResults(ctx->op_kernel().def(), transformed_input,
                           pre_transformed_filter_backprop,
                           transformed_out_backprop, stream->parent(), results);
    OP_REQUIRES_OK(ctx, BestCudnnConvAlgorithm(results, &algorithm_config));
    AutoTuneConvBwdFilter::GetInstance()->Insert(conv_parameters,
                                                 algorithm_config);
  }
  DnnScratchAllocator scratch_allocator(ConvolveBackwardFilterScratchSize, ctx);
  bool cudnn_launch_status =
      stream
          ->ThenConvolveBackwardFilterWithAlgorithm(
              input_desc, input_ptr, output_desc, out_backprop_ptr, conv_desc,
              filter_desc, &filter_backprop_ptr, &scratch_allocator,
              algorithm_config, nullptr)
          .ok();

  if (!cudnn_launch_status) {
    ctx->SetStatus(errors::Internal(
        "cuDNN Backward Filter function launch failure : input shape(",
        input.shape().DebugString(), ") filter shape(",
        filter_shape.DebugString(), ")"));
    return;
  }

  auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
  functor::ReverseTransformFilter<GPUDevice, T, 4>()(
      ctx->eigen_device<GPUDevice>(),
      toConstTensor(pre_transformed_filter_backprop).template tensor<T, 4>(),
      filter_backprop->tensor<T, 4>());
}

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                              \
  template <>                                                            \
  void ShuffleAndReverse<GPUDevice, T, 4, int>::operator()(              \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor input, \
      const Eigen::DSizes<int, 4>& order,                                \
      const Eigen::array<bool, 4>& reverse_dims,                         \
      typename TTypes<T, 4, int>::Tensor output);                        \
  extern template struct ShuffleAndReverse<GPUDevice, T, 4, int>;        \
  template <>                                                            \
  void InflatePadAndShuffle<GPUDevice, T, 4, int>::operator()(           \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor input, \
      const Eigen::DSizes<int, 4>& strides,                              \
      const Eigen::array<Eigen::IndexPair<int>, 4>& pad_dims,            \
      const Eigen::DSizes<int, 4>& order,                                \
      typename TTypes<T, 4, int>::Tensor output);                        \
  extern template struct InflatePadAndShuffle<GPUDevice, T, 4, int>;     \
  template <>                                                            \
  void TransformFilter<GPUDevice, T, int, 4>::operator()(                \
      const GPUDevice& d, FilterTensorFormat dst_filter_format,          \
      typename TTypes<T, 4, int>::ConstTensor in,                        \
      typename TTypes<T, 4, int>::Tensor out);                           \
  extern template struct TransformFilter<GPUDevice, T, int, 4>;          \
  template <>                                                            \
  void TransformDepth<GPUDevice, T, int>::operator()(                    \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,    \
      const Eigen::DSizes<int, 4>& shuffle,                              \
      typename TTypes<T, 4, int>::Tensor out);                           \
  extern template struct TransformDepth<GPUDevice, T, int>;              \
  template <>                                                            \
  void PadInput<GPUDevice, T, int, 4>::operator()(                       \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,    \
      const std::array<int, 2>& padding_left,                            \
      const std::array<int, 2>& padding_right,                           \
      typename TTypes<T, 4, int>::Tensor out, TensorFormat data_format); \
  extern template struct PadInput<GPUDevice, T, int, 4>;

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropFilter")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<double>("T")
                            .HostMemory("filter_sizes"),
                        Conv2DSlowBackpropFilterOp<GPUDevice, double>);
REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropFilter")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .HostMemory("filter_sizes"),
                        Conv2DSlowBackpropFilterOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropFilter")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<Eigen::half>("T")
                            .HostMemory("filter_sizes"),
                        Conv2DSlowBackpropFilterOp<GPUDevice, Eigen::half>);

// To be used inside depthwise_conv_grad_op.cc.
// TODO(reedwm): Move this and the definition to depthwise_conv_grad_op.cc.
template struct LaunchConv2DBackpropFilterOp<GPUDevice, float>;
template struct LaunchConv2DBackpropFilterOp<GPUDevice, Eigen::half>;
template struct LaunchConv2DBackpropFilterOp<GPUDevice, double>;

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
