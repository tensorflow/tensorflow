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
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "tensorflow/core/util/work_sharder.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
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

template <typename Device, class T>
class Conv2DFastBackpropFilterOp : public OpKernel {
 public:
  explicit Conv2DFastBackpropFilterOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, data_format_ == FORMAT_NHWC,
                errors::InvalidArgument(
                    "Conv2DFastBackpropFilterOp only supports NHWC."));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(
        context, (strides_[0] == 1 && strides_[3] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
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

    Conv2DBackpropDimensions dims;
    OP_REQUIRES_OK(context, Conv2DBackpropComputeDimensions(
                                "Conv2DFastBackpropFilter", input.shape(),
                                filter_shape, out_backprop.shape(), strides_,
                                padding_, data_format_, &dims));

    Tensor* filter_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));

    functor::SpatialConvolutionBackwardKernel<Device, T>()(
        context->eigen_device<Device>(), filter_backprop->tensor<T, 4>(),
        input.tensor<T, 4>(), out_backprop.tensor<T, 4>(),
        dims.rows.filter_size, dims.cols.filter_size, dims.rows.stride,
        dims.cols.stride);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DFastBackpropFilterOp);
};

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
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
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

    Conv2DBackpropDimensions dims;
    OP_REQUIRES_OK(context, Conv2DBackpropComputeDimensions(
                                "Conv2DCustomBackpropFilter", input.shape(),
                                filter_shape, out_backprop.shape(), strides_,
                                padding_, data_format_, &dims));

    Tensor* filter_backprop;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));

    int64 pad_top, pad_bottom;
    int64 pad_left, pad_right;
    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                dims.rows.input_size, dims.rows.filter_size,
                                dims.rows.stride, padding_,
                                &dims.rows.output_size, &pad_top, &pad_bottom));
    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                dims.cols.input_size, dims.cols.filter_size,
                                dims.cols.stride, padding_,
                                &dims.cols.output_size, &pad_left, &pad_right));

    // The total dimension size of each kernel.
    const int filter_total_size =
        dims.rows.filter_size * dims.cols.filter_size * dims.in_depth;
    // The output image size is the spatial size of the output.
    const int output_image_size = dims.rows.output_size * dims.cols.output_size;

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
    const int input_offset =
        dims.rows.input_size * dims.cols.input_size * dims.in_depth;
    // The output offset corresponding to a single output image.
    const int output_offset =
        dims.rows.output_size * dims.cols.output_size * dims.out_depth;

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
          Im2col<T>(input_data_shard, dims.in_depth, dims.rows.input_size,
                    dims.cols.input_size, dims.rows.filter_size,
                    dims.cols.filter_size, pad_top, pad_left, pad_bottom,
                    pad_right, dims.rows.stride, dims.cols.stride,
                    col_data_shard);
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
                          Conv2DCustomBackpropFilterOp<CPUDevice, T>);        \
  REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropFilter")                        \
                              .Device(DEVICE_CPU)                             \
                              .Label("eigen_tensor")                          \
                              .TypeConstraint<T>("T"),                        \
                          Conv2DFastBackpropFilterOp<CPUDevice, T>);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

// GPU definitions.
#if GOOGLE_CUDA
// The slow version (but compiles for GPU)

// A dummy type to group forward backward filter autotune results together.
struct ConvBackwardFilterAutoTuneGroup {};
typedef AutoTuneSingleton<ConvBackwardFilterAutoTuneGroup, ConvParameters,
                          perftools::gputools::dnn::AlgorithmConfig>
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
    OP_REQUIRES(
        context, (stride_n == 1 && stride_c == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    use_cudnn_ &= CanUseCudnn();
    cudnn_use_autotune_ = CudnnUseAutotune();
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    using perftools::gputools::dnn::AlgorithmConfig;
    using perftools::gputools::dnn::AlgorithmType;
    using perftools::gputools::dnn::ProfileResult;
    using perftools::gputools::dnn::kDefaultAlgorithm;
    const Tensor& input = context->input(0);
    const Tensor& filter_sizes = context->input(1);
    const Tensor& out_backprop = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(filter_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropFilter: filter_sizes input must be 1-dim, not ",
            filter_sizes.dims()));
    const TensorShape& input_shape = input.shape();
    TensorShape filter_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                filter_sizes.vec<int32>(), &filter_shape));

    Conv2DBackpropDimensions dims;
    OP_REQUIRES_OK(context, Conv2DBackpropComputeDimensions(
                                "Conv2DSlowBackpropFilter", input.shape(),
                                filter_shape, out_backprop.shape(), strides_,
                                padding_, data_format_, &dims));

    Tensor* filter_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));

    const int padding_rows =
        (padding_ == VALID)
            ? 0
            : std::max<int>(0, (dims.rows.output_size - 1) * dims.rows.stride +
                                   dims.rows.filter_size -
                                   dims.rows.input_size);
    const int padding_cols =
        (padding_ == VALID)
            ? 0
            : std::max<int>(0, (dims.cols.output_size - 1) * dims.cols.stride +
                                   dims.cols.filter_size -
                                   dims.cols.input_size);

    // TODO(zhengxq): cuDNN only supports equal padding on both sides, so only
    // calling it when that is true. Remove this check when (if?) cuDNN starts
    // supporting different padding.
    bool rows_odd = (padding_rows % 2 != 0);
    bool cols_odd = (padding_cols % 2 != 0);

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    if (!use_cudnn_) {
      context->SetStatus(errors::Unimplemented(
          "Conv2DBackprop for GPU is not currently supported "
          "without cudnn"));
      return;
    }

    if (dims.rows.filter_size == 1 && dims.cols.filter_size == 1 &&
        dims.rows.stride == 1 && dims.cols.stride == 1 &&
        data_format_ == FORMAT_NHWC) {
      const uint64 m = dims.in_depth;
      const uint64 k =
          dims.batch_size * dims.rows.input_size * dims.cols.input_size;
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
              ->ThenBlasGemm(perftools::gputools::blas::Transpose::kNoTranspose,
                             perftools::gputools::blas::Transpose::kTranspose,
                             n, m, k, 1.0f, a_ptr, n, b_ptr, m, 0.0f, &c_ptr, n)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                            ", n=", n, ", k=", k));
      }
      return;
    } else if (dims.rows.filter_size == dims.rows.input_size &&
               dims.cols.filter_size == dims.cols.input_size &&
               padding_ == VALID && data_format_ == FORMAT_NHWC) {
      // The input data and filter have the same height/width, so call cublas
      // directly.
      const uint64 m =
          dims.rows.input_size * dims.cols.input_size * dims.in_depth;
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
              ->ThenBlasGemm(perftools::gputools::blas::Transpose::kNoTranspose,
                             perftools::gputools::blas::Transpose::kTranspose,
                             n, m, k, 1.0f, b_ptr, n, a_ptr, m, 0.0f, &c_ptr, n)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                            ", n=", n, ", k=", k));
      }
      return;
    }

    Tensor compatible_input;
    if (rows_odd || cols_odd) {
      // If a padding dimension is odd, we have one more element on the right
      // side or the bottom side. This is unsupported in cudnn. Therefore,
      // we pad that extra element and make it compatible.
      OP_REQUIRES_OK(
          context,
          context->allocate_temp(
              DataTypeToEnum<T>::value,
              ShapeFromFormat(data_format_, dims.batch_size,
                              dims.rows.input_size + rows_odd,
                              dims.cols.input_size + cols_odd, dims.in_depth),
              &compatible_input));

      functor::PadInput<GPUDevice, T, int, 4>()(
          context->template eigen_device<GPUDevice>(),
          To32Bit(input.tensor<T, 4>()), {{0, 0}}, {{rows_odd, cols_odd}},
          To32Bit(compatible_input.tensor<T, 4>()), data_format_);
    } else {
      compatible_input = input;
    }

    CHECK(padding_rows >= 0 && padding_cols >= 0)
        << "Negative row or col paddings: (" << padding_rows << ", "
        << padding_cols << ")";
    perftools::gputools::dnn::BatchDescriptor input_desc;
    input_desc.set_count(dims.batch_size)
        .set_height(GetTensorDim(compatible_input, data_format_, 'H'))
        .set_width(GetTensorDim(compatible_input, data_format_, 'W'))
        .set_feature_map_count(dims.in_depth)
        .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
    perftools::gputools::dnn::BatchDescriptor output_desc;
    output_desc.set_count(dims.batch_size)
        .set_height(dims.rows.output_size)
        .set_width(dims.cols.output_size)
        .set_feature_map_count(dims.out_depth)
        .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
    perftools::gputools::dnn::FilterDescriptor filter_desc;
    filter_desc.set_input_filter_height(dims.rows.filter_size)
        .set_input_filter_width(dims.cols.filter_size)
        .set_input_feature_map_count(dims.in_depth)
        .set_output_feature_map_count(dims.out_depth);
    perftools::gputools::dnn::ConvolutionDescriptor conv_desc;
    conv_desc.set_vertical_filter_stride(dims.rows.stride)
        .set_horizontal_filter_stride(dims.cols.stride)
        .set_zero_padding_height(padding_rows / 2)
        .set_zero_padding_width(padding_cols / 2);

    // NOTE(zhengxq):
    // cuDNN only supports the following layouts :
    // Input  : B x D x R x C
    // Filter : OD x ID x R x C
    // Whereas, we have
    // Input  : B x R x C x D
    // Filter : R x C x ID x OD
    // TransformFilter performs (R x C x ID x OD) => (OD x ID x R x C)
    // The first TransformDepth performs
    // (B x R x C x D) => (B x D x R x C).
    // Since the tensor returned from cuDNN is B x D x R x C also,
    // the second TransformDepth performs
    // (B x D x R x C) => (B x R x C x D).

    Tensor pre_transformed_filter_backprop;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<T>::value,
                                TensorShape({dims.out_depth, dims.in_depth,
                                             dims.rows.filter_size,
                                             dims.cols.filter_size}),
                                &pre_transformed_filter_backprop));

    Tensor transformed_out_backprop;
    if (data_format_ == FORMAT_NHWC) {
      TensorShape nchw_shape =
          ShapeFromFormat(FORMAT_NCHW, dims.batch_size, dims.rows.output_size,
                          dims.cols.output_size, dims.out_depth);
      if (dims.out_depth > 1) {
        OP_REQUIRES_OK(context, context->allocate_temp(
                                    DataTypeToEnum<T>::value, nchw_shape,
                                    &transformed_out_backprop));
        functor::NHWCToNCHW<Device, T, 4>()(
            context->eigen_device<Device>(), out_backprop.tensor<T, 4>(),
            transformed_out_backprop.tensor<T, 4>());
      } else {
        // If depth <= 1, just reshape.
        CHECK(transformed_out_backprop.CopyFrom(out_backprop, nchw_shape));
      }
    } else {
      transformed_out_backprop = out_backprop;
    }

    Tensor transformed_input;
    if (data_format_ == FORMAT_NHWC) {
      TensorShape nchw_shape = ShapeFromFormat(
          FORMAT_NCHW, GetTensorDim(compatible_input, data_format_, 'N'),
          GetTensorDim(compatible_input, data_format_, 'H'),
          GetTensorDim(compatible_input, data_format_, 'W'),
          GetTensorDim(compatible_input, data_format_, 'C'));
      if (nchw_shape.dim_size(1) > 1) {
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<T>::value,
                                              nchw_shape, &transformed_input));
        functor::NHWCToNCHW<Device, T, 4>()(
            context->eigen_device<Device>(),
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
    auto filter_backprop_ptr = AsDeviceMemory(
        pre_transformed_filter_backprop.template flat<T>().data(),
        pre_transformed_filter_backprop.template flat<T>().size());
    auto input_ptr =
        AsDeviceMemory(transformed_input.template flat<T>().data(),
                       transformed_input.template flat<T>().size());

    static int64 ConvolveBackwardFilterScratchSize = GetCudnnWorkspaceLimit(
        "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB by default
        );
    int device_id = stream->parent()->device_ordinal();
    ConvParameters conv_parameters = {
        dims.batch_size,        // batch
        dims.in_depth,          // in_depths
        input_desc.height(),    // in_rows
        input_desc.width(),     // in_cols
        dims.out_depth,         // out_depths
        dims.rows.filter_size,  // filter_rows
        dims.cols.filter_size,  // filter_cols
        dims.rows.stride,       // stride_rows
        dims.cols.stride,       // stride_cols
        padding_rows,           // padding_rows
        padding_cols,           // padding_cols
        device_id,              // device_id
    };
    AlgorithmConfig algorithm_config;
    if (cudnn_use_autotune_ &&
        !AutoTuneConvBwdFilter::GetInstance()->Find(conv_parameters,
                                                    &algorithm_config)) {
      std::vector<AlgorithmType> algorithms;
      CHECK(stream->parent()->GetConvolveBackwardFilterAlgorithms(&algorithms));
      ProfileResult best_result;
      ProfileResult best_result_no_scratch;
      for (auto profile_algorithm : algorithms) {
        // TODO(zhengxq): profile each algorithm multiple times to better
        // accuracy.
        CudnnScratchAllocator scratch_allocator(
            ConvolveBackwardFilterScratchSize, context);
        ProfileResult profile_result;
        bool cudnn_launch_status =
            stream
                ->ThenConvolveBackwardFilterWithAlgorithm(
                    input_desc, input_ptr, output_desc, out_backprop_ptr,
                    conv_desc, filter_desc, &filter_backprop_ptr,
                    &scratch_allocator, AlgorithmConfig(profile_algorithm),
                    &profile_result)
                .ok();
        if (cudnn_launch_status) {
          if (profile_result.is_valid()) {
            if (profile_result.elapsed_time_in_ms() <
                best_result.elapsed_time_in_ms()) {
              best_result = profile_result;
            }
            if (scratch_allocator.TotalByteSize() == 0 &&
                profile_result.elapsed_time_in_ms() <
                    best_result_no_scratch.elapsed_time_in_ms()) {
              best_result_no_scratch = profile_result;
            }
          }
        }
      }
      OP_REQUIRES(context, best_result.is_valid() &&
                               best_result.algorithm() != kDefaultAlgorithm,
                  errors::NotFound("No algorithm worked!"));
      OP_REQUIRES(context,
                  best_result_no_scratch.is_valid() &&
                      best_result_no_scratch.algorithm() != kDefaultAlgorithm,
                  errors::NotFound("No algorithm without scratch worked!"));
      algorithm_config.set_algorithm(best_result.algorithm());
      algorithm_config.set_algorithm_no_scratch(
          best_result_no_scratch.algorithm());
      AutoTuneConvBwdFilter::GetInstance()->Insert(conv_parameters,
                                                   algorithm_config);
    }
    CudnnScratchAllocator scratch_allocator(ConvolveBackwardFilterScratchSize,
                                            context);
    bool cudnn_launch_status =
        stream
            ->ThenConvolveBackwardFilterWithAlgorithm(
                input_desc, input_ptr, output_desc, out_backprop_ptr, conv_desc,
                filter_desc, &filter_backprop_ptr, &scratch_allocator,
                algorithm_config, nullptr)
            .ok();

    if (!cudnn_launch_status) {
      context->SetStatus(errors::Internal(
          "cuDNN Backward Filter function launch failure : input shape(",
          input_shape.DebugString(), ") filter shape(",
          filter_shape.DebugString(), ")"));
      return;
    }

    auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
    functor::ReverseTransformFilter<Device, T, 4>()(
        context->eigen_device<Device>(),
        toConstTensor(pre_transformed_filter_backprop).template tensor<T, 4>(),
        filter_backprop->tensor<T, 4>());
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  bool use_cudnn_;
  TensorFormat data_format_;
  bool cudnn_use_autotune_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DSlowBackpropFilterOp);
};

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
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,    \
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
#undef DECLARE_GPU_SPEC
}  // namespace functor

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
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
