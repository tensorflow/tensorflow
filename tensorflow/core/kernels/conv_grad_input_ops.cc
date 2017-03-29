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
#ifdef TENSORFLOW_USE_LIBXSMM
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

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace {

// Returns in 'im_data' (assumes to be zero-initialized) image patch in storage
// order (height, width, depth), constructed from patches in 'col_data', which
// is required to be in storage order (out_height * out_width, filter_height,
// filter_width, in_depth).  Implementation by Yangqing Jia (jiayq).
template <typename T>
void Col2im(const T* col_data, const int depth, const int height,
            const int width, const int filter_h, const int filter_w,
            const int pad_t, const int pad_l, const int pad_b, const int pad_r,
            const int stride_h, const int stride_w, T* im_data) {
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
            // TODO(andydavis) Vectorize this loop (if compiler does not).
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

}  // namespace

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// The fast versions using eigen computations directly. They are only enabled
// for CPU for now since nvcc times out when trying to compile them.
// TODO(yangke): enable them for GPUs when we have a faster compiler.

template <typename Device, class T>
struct LaunchBackwardInputConvolution {
  bool operator()(OpKernelContext* context, const Device&,
                  typename TTypes<T, 4>::Tensor,
                  typename TTypes<T, 4>::ConstTensor,
                  typename TTypes<T, 4>::ConstTensor, int, int, int, int,
                  TensorFormat) const {
    return false;
  }
};

template <>
struct LaunchBackwardInputConvolution<CPUDevice, float> {
  bool operator()(OpKernelContext* context, const CPUDevice& d,
                  typename TTypes<float, 4>::Tensor input_backward,
                  typename TTypes<float, 4>::ConstTensor kernel,
                  typename TTypes<float, 4>::ConstTensor output_backward,
                  int input_rows, int input_cols, int row_stride,
                  int col_stride, TensorFormat data_format) const {
    functor::SpatialConvolutionBackwardInput<CPUDevice, float>()(
        d, input_backward, kernel, output_backward, input_rows, input_cols,
        row_stride, col_stride);
    return true;
  }
};

#ifdef TENSORFLOW_USE_LIBXSMM
template <typename Device, class T>
struct LaunchXsmmBackwardInputConvolution {
  bool operator()(OpKernelContext* context, const Device& d,
                  typename TTypes<T, 4>::Tensor input_backward,
                  typename TTypes<T, 4>::ConstTensor kernel,
                  typename TTypes<T, 4>::ConstTensor output_backward,
                  int input_rows, int input_cols, int row_stride,
                  int col_stride, TensorFormat data_format) const {
    return false;
  }
};

template <>
struct LaunchXsmmBackwardInputConvolution<CPUDevice, float> {
  bool operator()(OpKernelContext* context, const CPUDevice& d,
                  typename TTypes<float, 4>::Tensor input_backward,
                  typename TTypes<float, 4>::ConstTensor kernel,
                  typename TTypes<float, 4>::ConstTensor output_backward,
                  int input_rows, int input_cols, int row_stride,
                  int col_stride, TensorFormat data_format) const {
    auto batch = input_backward.dimension(0);
    auto in_depth = input_backward.dimension(3);
    auto out_depth = output_backward.dimension(3);
    auto filter_rows = kernel.dimension(0);
    auto filter_cols = kernel.dimension(1);
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
    desc.pad_h = 0;
    desc.pad_w = 0;
    desc.pad_h_in = 0;  // pad_rows;  // ignored by libxsmm for now.
    desc.pad_w_in = 0;  // pad_cols;  // ignored by libxsmm for now.
    desc.pad_h_out = 0;
    desc.pad_w_out = 0;
    desc.threads = num_threads;
    desc.algo = LIBXSMM_DNN_CONV_ALGO_DIRECT;
    desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_NHWC;
    desc.filter_format =
        LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;  // LIBXSMM_DNN_TENSOR_FORMAT_RSCK;
    desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
    desc.options = LIBXSMM_DNN_CONV_OPTION_NONE;
    desc.datatype = LIBXSMM_DNN_DATATYPE_F32;

    auto input_ptr = input_backward.data();
    auto filter_ptr = kernel.data();
    auto output_ptr = output_backward.data();

    bool success = functor::XsmmBkwInputConv2D<CPUDevice, float>()(
        context, desc, input_ptr, filter_ptr, output_ptr);
    return success;
  }
};
#endif

template <typename Device, class T>
class Conv2DFastBackpropInputOp : public OpKernel {
 public:
  explicit Conv2DFastBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, data_format_ == FORMAT_NHWC,
                errors::InvalidArgument(
                    "Eigen Conv2DFastBackpropInputOp only supports NHWC."));
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
    const Tensor& input_sizes = context->input(0);
    const Tensor& filter = context->input(1);
    const Tensor& out_backprop = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropInput: input_sizes input must be 1-dim, not ",
            input_sizes.dims()));
    TensorShape input_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                input_sizes.vec<int32>(), &input_shape));

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(context,
                   ConvBackpropComputeDimensions(
                       "Conv2DFastBackpropInput", /*num_spatial_dims=*/2,
                       input_shape, filter.shape(), out_backprop.shape(),
                       strides_, padding_, data_format_, &dims));

    Tensor* in_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));

#if defined TENSORFLOW_USE_LIBXSMM && defined TENSORFLOW_USE_LIBXSMM_BACKWARD
    if (LaunchXsmmBackwardInputConvolution<Device, T>()(
            context, context->eigen_device<Device>(),
            in_backprop->tensor<T, 4>(), filter.tensor<T, 4>(),
            out_backprop.tensor<T, 4>(), dims.spatial_dims[0].input_size,
            dims.spatial_dims[1].input_size, dims.spatial_dims[0].stride,
            dims.spatial_dims[1].stride, data_format_)) {
      return;
    }
#endif

    LaunchBackwardInputConvolution<Device, T>()(
        context, context->eigen_device<Device>(), in_backprop->tensor<T, 4>(),
        filter.tensor<T, 4>(), out_backprop.tensor<T, 4>(),
        dims.spatial_dims[0].input_size, dims.spatial_dims[1].input_size,
        dims.spatial_dims[0].stride, dims.spatial_dims[1].stride, data_format_);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DFastBackpropInputOp);
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
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, data_format_ == FORMAT_NHWC,
                errors::InvalidArgument(
                    "Conv2DCustomBackpropInputOp only supports NHWC."));
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
    const Tensor& input_sizes = context->input(0);
    const Tensor& filter = context->input(1);
    const Tensor& out_backprop = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropInput: input_sizes input must be 1-dim, not ",
            input_sizes.dims()));
    TensorShape input_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                input_sizes.vec<int32>(), &input_shape));

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(context,
                   ConvBackpropComputeDimensions(
                       "Conv2DCustomBackpropInput", /*num_spatial_dims=*/2,
                       input_shape, filter.shape(), out_backprop.shape(),
                       strides_, padding_, data_format_, &dims));

    Tensor* in_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));

#if defined TENSORFLOW_USE_LIBXSMM && defined TENSORFLOW_USE_LIBXSMM_BACKWARD
    if (LaunchXsmmBackwardInputConvolution<Device, T>()(
            context, context->eigen_device<Device>(),
            in_backprop->tensor<T, 4>(), filter.tensor<T, 4>(),
            out_backprop.tensor<T, 4>(), dims.spatial_dims[0].input_size,
            dims.spatial_dims[1].input_size, dims.spatial_dims[0].stride,
            dims.spatial_dims[1].stride, data_format_)) {
      return;
    }
#endif

    // TODO(andydavis) Consider moving code shared with
    // Conv2DCustomBackpropFilterOp into a shared helper function.
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

    const size_t shard_size =
        use_parallel_contraction
            ? 1
            : (target_working_set_size + work_unit_size - 1) / work_unit_size;

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
      typedef Eigen::Map<
          Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          MatrixMap;
      typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                             Eigen::RowMajor>>
          ConstMatrixMap;

      for (int image_id = 0; image_id < dims.batch_size;
           image_id += shard_size) {
        const int shard_limit =
            std::min(static_cast<int>(shard_size),
                     static_cast<int>(dims.batch_size) - image_id);

        auto shard = [&dims, &pad_top, &pad_left, &pad_bottom, &pad_right,
                      &output_image_size, &filter_total_size,
                      &input_backprop_data, &col_buffer_data,
                      &out_backprop_data, &filter_data, &input_offset,
                      &output_offset, &size_C](int64 start, int64 limit) {
          for (int shard_id = start; shard_id < limit; ++shard_id) {
            T* im2col_buf = col_buffer_data + shard_id * size_C;
            T* input_data = input_backprop_data + shard_id * input_offset;
            const T* out_data = out_backprop_data + shard_id * output_offset;

            // Compute gradient into 'im2col_buf'.
            MatrixMap C(im2col_buf, output_image_size, filter_total_size);

            ConstMatrixMap A(out_data, output_image_size, dims.out_depth);
            ConstMatrixMap B(filter_data, filter_total_size, dims.out_depth);

            C.noalias() = A * B.transpose();

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
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DCustomBackpropInputOp);
};

#define REGISTER_CPU_KERNELS(T)                                              \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("Conv2DBackpropInput").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      Conv2DCustomBackpropInputOp<CPUDevice, T>);                            \
  REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")                        \
                              .Device(DEVICE_CPU)                            \
                              .Label("custom")                               \
                              .TypeConstraint<T>("T"),                       \
                          Conv2DCustomBackpropInputOp<CPUDevice, T>);        \
  REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")                        \
                              .Device(DEVICE_CPU)                            \
                              .Label("eigen_tensor")                         \
                              .TypeConstraint<T>("T"),                       \
                          Conv2DFastBackpropInputOp<CPUDevice, T>);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

// GPU definitions.
#if GOOGLE_CUDA
// The slow version (but compiles for GPU)

// A dummy type to group forward backward data autotune results together.
struct ConvBackwardDataAutoTuneGroup {
  static string name() { return "ConvBwdData"; }
};
typedef AutoTuneSingleton<ConvBackwardDataAutoTuneGroup, ConvParameters,
                          perftools::gputools::dnn::AlgorithmConfig>
    AutoTuneConvBwdData;

// Backprop for input.
template <typename Device, class T>
class Conv2DSlowBackpropInputOp : public OpKernel {
 public:
  explicit Conv2DSlowBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
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
    const Tensor& input_sizes = context->input(0);
    const Tensor& filter = context->input(1);
    const Tensor& out_backprop = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropInput: input_sizes input must be 1-dim, not ",
            input_sizes.dims()));
    TensorShape input_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                input_sizes.vec<int32>(), &input_shape));
    const TensorShape& filter_shape = filter.shape();

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(
        context, ConvBackpropComputeDimensions(
                     "Conv2DSlowBackpropInput", /*num_spatial_dims=*/2,
                     input_shape, filter_shape, out_backprop.shape(), strides_,
                     padding_, data_format_, &dims));

    Tensor* in_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));

    const int padding_rows =
        (padding_ == VALID)
            ? 0
            : std::max<int>(0, (dims.spatial_dims[0].output_size - 1) *
                                       dims.spatial_dims[0].stride +
                                   dims.spatial_dims[0].filter_size -
                                   dims.spatial_dims[0].input_size);
    const int padding_cols =
        (padding_ == VALID)
            ? 0
            : std::max<int>(0, (dims.spatial_dims[1].output_size - 1) *
                                       dims.spatial_dims[1].stride +
                                   dims.spatial_dims[1].filter_size -
                                   dims.spatial_dims[1].input_size);

    // TODO(keveman): cuDNN only supports equal padding on both sides, so only
    // calling it when that is true. Remove this check when (if?) cuDNN starts
    // supporting different padding.
    bool rows_odd = (padding_rows % 2 != 0);
    bool cols_odd = (padding_cols % 2 != 0);

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    if (!use_cudnn_) {
      context->SetStatus(errors::Unimplemented(
          "Conv2DBackpropInput for GPU is not currently supported "
          "without cudnn"));
      return;
    }

    if (dims.spatial_dims[0].filter_size == 1 &&
        dims.spatial_dims[1].filter_size == 1 &&
        dims.spatial_dims[0].stride == 1 && dims.spatial_dims[1].stride == 1 &&
        data_format_ == FORMAT_NHWC) {
      // 1x1 filter, so call cublas directly.
      const uint64 m = dims.batch_size * dims.spatial_dims[0].input_size *
                       dims.spatial_dims[1].input_size;
      const uint64 k = dims.out_depth;
      const uint64 n = dims.in_depth;

      auto a_ptr = AsDeviceMemory(out_backprop.template flat<T>().data(),
                                  out_backprop.template flat<T>().size());
      auto b_ptr = AsDeviceMemory(filter.template flat<T>().data(),
                                  filter.template flat<T>().size());
      auto c_ptr = AsDeviceMemory(in_backprop->template flat<T>().data(),
                                  in_backprop->template flat<T>().size());

      auto transpose = perftools::gputools::blas::Transpose::kTranspose;
      auto no_transpose = perftools::gputools::blas::Transpose::kNoTranspose;

      bool blas_launch_status =
          stream
              ->ThenBlasGemm(transpose, no_transpose, n, m, k, 1.0f, b_ptr, k,
                             a_ptr, k, 0.0f, &c_ptr, n)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                            ", n=", n, ", k=", k));
      }
      return;
    } else if (dims.spatial_dims[0].filter_size ==
                   dims.spatial_dims[0].input_size &&
               dims.spatial_dims[1].filter_size ==
                   dims.spatial_dims[1].input_size &&
               padding_ == VALID && data_format_ == FORMAT_NHWC) {
      // The input data and filter have the same height/width, so call cublas
      // directly.
      const uint64 m = dims.batch_size;
      const uint64 k = dims.out_depth;
      const uint64 n = dims.spatial_dims[0].input_size *
                       dims.spatial_dims[1].input_size * dims.in_depth;

      auto a_ptr = AsDeviceMemory(out_backprop.template flat<T>().data(),
                                  out_backprop.template flat<T>().size());
      auto b_ptr = AsDeviceMemory(filter.template flat<T>().data(),
                                  filter.template flat<T>().size());
      auto c_ptr = AsDeviceMemory(in_backprop->template flat<T>().data(),
                                  in_backprop->template flat<T>().size());

      auto transpose = perftools::gputools::blas::Transpose::kTranspose;
      auto no_transpose = perftools::gputools::blas::Transpose::kNoTranspose;

      bool blas_launch_status =
          stream
              ->ThenBlasGemm(transpose, no_transpose, n, m, k, 1.0f, b_ptr, k,
                             a_ptr, k, 0.0f, &c_ptr, n)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                            ", n=", n, ", k=", k));
      }
      return;
    }

    TensorShape compatible_input_shape;
    if (rows_odd || cols_odd) {
      // If a padding dimension is odd, we have one more element on the right
      // side or the bottom side. This is unsupported in cudnn. Therefore,
      // we pad that extra element and make it compatible.
      compatible_input_shape = ShapeFromFormat(
          data_format_, dims.batch_size,
          dims.spatial_dims[0].input_size + rows_odd,
          dims.spatial_dims[1].input_size + cols_odd, dims.in_depth);
    } else {
      compatible_input_shape = input_shape;
    }

    CHECK(padding_rows >= 0 && padding_cols >= 0)
        << "Negative row or col paddings: (" << padding_rows << ", "
        << padding_cols << ")";
    perftools::gputools::dnn::BatchDescriptor input_desc;
    input_desc.set_count(dims.batch_size)
        .set_height(GetTensorDim(compatible_input_shape, data_format_, 'H'))
        .set_width(GetTensorDim(compatible_input_shape, data_format_, 'W'))
        .set_feature_map_count(dims.in_depth)
        .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
    perftools::gputools::dnn::BatchDescriptor output_desc;
    output_desc.set_count(dims.batch_size)
        .set_height(dims.spatial_dims[0].output_size)
        .set_width(dims.spatial_dims[1].output_size)
        .set_feature_map_count(dims.out_depth)
        .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
    perftools::gputools::dnn::FilterDescriptor filter_desc;
    filter_desc.set_input_filter_height(dims.spatial_dims[0].filter_size)
        .set_input_filter_width(dims.spatial_dims[1].filter_size)
        .set_input_feature_map_count(dims.in_depth)
        .set_output_feature_map_count(dims.out_depth);
    perftools::gputools::dnn::ConvolutionDescriptor conv_desc;
    conv_desc.set_vertical_filter_stride(dims.spatial_dims[0].stride)
        .set_horizontal_filter_stride(dims.spatial_dims[1].stride)
        .set_zero_padding_height(padding_rows / 2)
        .set_zero_padding_width(padding_cols / 2);

    // NOTE(keveman):
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
    Tensor transformed_filter;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<T>::value,
                                TensorShape({dims.out_depth, dims.in_depth,
                                             dims.spatial_dims[0].filter_size,
                                             dims.spatial_dims[1].filter_size}),
                                &transformed_filter));

    functor::TransformFilter<Device, T, int, 4>()(
        context->eigen_device<Device>(), To32Bit(filter.tensor<T, 4>()),
        To32Bit(transformed_filter.tensor<T, 4>()));

    Tensor transformed_out_backprop;
    if (data_format_ == FORMAT_NHWC) {
      TensorShape nchw_shape = ShapeFromFormat(
          FORMAT_NCHW, dims.batch_size, dims.spatial_dims[0].output_size,
          dims.spatial_dims[1].output_size, dims.out_depth);
      if (dims.out_depth > 1) {
        OP_REQUIRES_OK(context, context->allocate_temp(
                                    DataTypeToEnum<T>::value, nchw_shape,
                                    &transformed_out_backprop));
        functor::NHWCToNCHW<Device, T, 4>()(
            context->eigen_device<Device>(), out_backprop.tensor<T, 4>(),
            transformed_out_backprop.tensor<T, 4>());
      } else {
        // If depth <= 1, then just reshape.
        CHECK(transformed_out_backprop.CopyFrom(out_backprop, nchw_shape));
      }
    } else {
      transformed_out_backprop = out_backprop;
    }

    Tensor pre_transformed_in_backprop;
    OP_REQUIRES_OK(
        context,
        context->allocate_temp(
            DataTypeToEnum<T>::value,
            ShapeFromFormat(
                FORMAT_NCHW,
                GetTensorDim(compatible_input_shape, data_format_, 'N'),
                GetTensorDim(compatible_input_shape, data_format_, 'H'),
                GetTensorDim(compatible_input_shape, data_format_, 'W'),
                GetTensorDim(compatible_input_shape, data_format_, 'C')),
            &pre_transformed_in_backprop));

    auto out_backprop_ptr =
        AsDeviceMemory(transformed_out_backprop.template flat<T>().data(),
                       transformed_out_backprop.template flat<T>().size());
    auto filter_ptr =
        AsDeviceMemory(transformed_filter.template flat<T>().data(),
                       transformed_filter.template flat<T>().size());
    auto in_backprop_ptr =
        AsDeviceMemory(pre_transformed_in_backprop.template flat<T>().data(),
                       pre_transformed_in_backprop.template flat<T>().size());

    static int64 ConvolveBackwardDataScratchSize = GetCudnnWorkspaceLimit(
        "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB by default
        );
    CudnnScratchAllocator scratch_allocator(ConvolveBackwardDataScratchSize,
                                            context);
    int device_id = stream->parent()->device_ordinal();
    DataType dtype = out_backprop.dtype();
    ConvParameters conv_parameters = {
        dims.batch_size,                       // batch
        dims.in_depth,                         // in_depths
        {{input_desc.height(),                 // in_rows
          input_desc.width()}},                // in_cols
        dims.out_depth,                        // out_depths
        {{dims.spatial_dims[0].filter_size,    // filter_rows
          dims.spatial_dims[1].filter_size}},  // filter_cols
        {{dims.spatial_dims[0].stride,         // stride_rows
          dims.spatial_dims[1].stride}},       // stride_cols
        {{padding_rows,                        // padding_rows
          padding_cols}},                      // padding_cols
        dtype,                                 // tensor data type
        device_id,                             // device_id
    };
    AlgorithmConfig algorithm_config;
    if (cudnn_use_autotune_ && !AutoTuneConvBwdData::GetInstance()->Find(
                                   conv_parameters, &algorithm_config)) {
      std::vector<AlgorithmType> algorithms;
      CHECK(stream->parent()->GetConvolveBackwardDataAlgorithms(&algorithms));
      ProfileResult best_result;
      ProfileResult best_result_no_scratch;
      for (auto profile_algorithm : algorithms) {
        // TODO(zhengxq): profile each algorithm multiple times to better
        // accuracy.
        CudnnScratchAllocator scratch_allocator(ConvolveBackwardDataScratchSize,
                                                context);
        ProfileResult profile_result;
        bool cudnn_launch_status =
            stream
                ->ThenConvolveBackwardDataWithAlgorithm(
                    filter_desc, filter_ptr, output_desc, out_backprop_ptr,
                    conv_desc, input_desc, &in_backprop_ptr, &scratch_allocator,
                    AlgorithmConfig(profile_algorithm), &profile_result)
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
      OP_REQUIRES(context,
                  best_result.is_valid() &&
                      best_result.algorithm() != kDefaultAlgorithm,
                  errors::NotFound("No algorithm worked!"));
      OP_REQUIRES(context,
                  best_result_no_scratch.is_valid() &&
                      best_result_no_scratch.algorithm() != kDefaultAlgorithm,
                  errors::NotFound("No algorithm without scratch worked!"));
      algorithm_config.set_algorithm(best_result.algorithm());
      algorithm_config.set_algorithm_no_scratch(
          best_result_no_scratch.algorithm());
      AutoTuneConvBwdData::GetInstance()->Insert(conv_parameters,
                                                 algorithm_config);
    }
    bool cudnn_launch_status =
        stream
            ->ThenConvolveBackwardDataWithAlgorithm(
                filter_desc, filter_ptr, output_desc, out_backprop_ptr,
                conv_desc, input_desc, &in_backprop_ptr, &scratch_allocator,
                algorithm_config, nullptr)
            .ok();

    if (!cudnn_launch_status) {
      context->SetStatus(errors::Internal(
          "cuDNN Backward Data function launch failure : input shape(",
          input_shape.DebugString(), ") filter shape(",
          filter_shape.DebugString(), ")"));
      return;
    }

    if (rows_odd || cols_odd) {
      Tensor in_backprop_remove_padding;
      OP_REQUIRES_OK(
          context,
          context->allocate_temp(
              DataTypeToEnum<T>::value,
              ShapeFromFormat(FORMAT_NCHW,
                              GetTensorDim(input_shape, data_format_, 'N'),
                              GetTensorDim(input_shape, data_format_, 'H'),
                              GetTensorDim(input_shape, data_format_, 'W'),
                              GetTensorDim(input_shape, data_format_, 'C')),
              &in_backprop_remove_padding));

      // Remove the padding for odd rows or cols.
      functor::PadInput<GPUDevice, T, int, 4>()(
          context->template eigen_device<GPUDevice>(),
          To32Bit(const_cast<const Tensor&>(pre_transformed_in_backprop)
                      .tensor<T, 4>()),
          {{0, 0}}, {{-rows_odd, -cols_odd}},
          To32Bit(in_backprop_remove_padding.tensor<T, 4>()), FORMAT_NCHW);

      pre_transformed_in_backprop = in_backprop_remove_padding;
    }

    if (data_format_ == FORMAT_NHWC) {
      auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
      functor::NCHWToNHWC<Device, T, 4>()(
          context->eigen_device<Device>(),
          toConstTensor(pre_transformed_in_backprop).template tensor<T, 4>(),
          in_backprop->tensor<T, 4>());
    } else {
      *in_backprop = pre_transformed_in_backprop;
    }
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  bool use_cudnn_;
  TensorFormat data_format_;
  bool cudnn_use_autotune_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DSlowBackpropInputOp);
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

REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .HostMemory("input_sizes"),
                        Conv2DSlowBackpropInputOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<Eigen::half>("T")
                            .HostMemory("input_sizes"),
                        Conv2DSlowBackpropInputOp<GPUDevice, Eigen::half>);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
