/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include <vector>
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/conv_3d.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA


namespace tensorflow {


typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


template <typename Device, typename T>
struct LaunchGeneric3D {
  static void launch(OpKernelContext* ctx, const Tensor& input,
                     const Tensor& filter, int depth_stride, int row_stride,
                     int col_stride, const Eigen::PaddingType& padding,
                     Tensor* output, TensorFormat data_format) {
    CHECK(data_format == FORMAT_NDHWC) << "Generic 3d conv implementation only "
                                         "supports NDHWC tensor format.";

    functor::CuboidConvolution<Device, T>()(
        ctx->eigen_device<Device>(), output->tensor<T, 5>(),
        input.tensor<T, 5>(), filter.tensor<T, 5>(), depth_stride,
        row_stride, col_stride, padding);
  }
};

template <typename Device, typename T>
struct LaunchConv3DOp;

template <typename T>
struct LaunchConv3DOp<CPUDevice, T> {
  static void launch(OpKernelContext* ctx, bool use_cudnn, const Tensor& input,
                     const Tensor& filter, int depth_stride, int row_stride,
                     int col_stride, const Eigen::PaddingType& padding,
                     Tensor* output, TensorFormat data_format) {
    LaunchGeneric3D<CPUDevice, T>::launch(ctx, input, filter, depth_stride,
                                        row_stride, col_stride, padding,
                                        output, data_format);
  }
};


template <typename Device, typename T>
class Conv3DOp : public BinaryOp<T> {
 public:
  explicit Conv3DOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    use_cudnn_ &= CanUseCudnn();
    OP_REQUIRES(context, strides_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
    const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
    const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');
    OP_REQUIRES(
        context, stride_n == 1 && stride_c == 1,
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, D, H, W, in_chan ]
    const Tensor& input = context->input(0);
    // Input filter is of the following dimensions:
    // [ D, H, W, in_chan, out_chan]
    const Tensor& filter = context->input(1);

    // For 3D convolution, there should be 5 dimensions.
    OP_REQUIRES(context, input.dims() == 5,
                errors::InvalidArgument("Input must be 5-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, filter.dims() == 5,
                errors::InvalidArgument("Filter must be 5-dimensional: ",
                                        filter.shape().DebugString()));

    for (int i = 0; i < 4; i++) {
      OP_REQUIRES(context, FastBoundsCheck(filter.dim_size(i),
                                           std::numeric_limits<int>::max()),
                  errors::InvalidArgument("filter too large"));
    }

    // The last dimension for input is in_chan. It must be the same as the
    // filter's in_chan.
    const int64 in_chan = GetTensorDim(input, data_format_, 'C');
    OP_REQUIRES(
        context, in_chan == filter.dim_size(3),
        errors::InvalidArgument("input and filter must have the same in_chan: ",
                                in_chan, " vs ", filter.dim_size(3)));

    // The last dimension for filter is out_chan.
    const int out_chan = static_cast<int>(filter.dim_size(4));

    // The second dimension for input is depth.
    // The first dimension for filter is depth.
    const int64 input_depth_raw = GetTensorDim(input, data_format_, 'D');
    OP_REQUIRES(context, FastBoundsCheck(input_depth_raw,
                                         std::numeric_limits<int>::max()),
                errors::InvalidArgument("Input depth too large"));
    const int input_depth = static_cast<int>(input_depth_raw);
    const int filter_depth = static_cast<int>(filter.dim_size(0));

    // The third dimension for input is rows/height.
    // The second dimension for filter is rows/height.
    const int64 input_rows_raw = GetTensorDim(input, data_format_, 'H');
    OP_REQUIRES(context, FastBoundsCheck(input_rows_raw,
                                         std::numeric_limits<int>::max()),
                errors::InvalidArgument("Input rows too large"));
    const int input_rows = static_cast<int>(input_rows_raw);
    const int filter_rows = static_cast<int>(filter.dim_size(1));

    // The fourth dimension for input is columns/width.
    // The third dimension for filter is columns/width.
    const int64 input_cols_raw = GetTensorDim(input, data_format_, 'W');
    OP_REQUIRES(context, FastBoundsCheck(input_cols_raw,
                                         std::numeric_limits<int>::max()),
                errors::InvalidArgument("Input cols too large"));
    const int input_cols = static_cast<int>(input_cols_raw);
    const int filter_cols = static_cast<int>(filter.dim_size(2));


    // The first dimension for input is batch.
    const int64 batch_raw = GetTensorDim(input, data_format_, 'N');
    OP_REQUIRES(context,
                FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
                errors::InvalidArgument("batch is too large"));
    const int batch = static_cast<int>(batch_raw);

    // For now we take the stride from the second, third, fourth dimensions only
    // (we do not support striding on the batch or depth dimension).
    const int stride_depth = GetTensorDim(strides_, data_format_, 'D');
    const int stride_rows = GetTensorDim(strides_, data_format_, 'H');
    const int stride_cols = GetTensorDim(strides_, data_format_, 'W');

    int out_rows = 0, out_cols = 0, out_depth = 0;
    int pad_rows = 0, pad_cols = 0, pad_depth = 0;


    OP_REQUIRES_OK(
        context,
        Get3dOutputSize(input_rows, input_cols, input_depth,
                        filter_rows, filter_cols, filter_depth,
                        stride_rows, stride_cols, stride_depth,
                        padding_, &out_rows, &out_cols, &out_depth,
                        &pad_rows, &pad_cols, &pad_depth));

    TensorShape out_shape =
        ShapeFromFormat3D(data_format_, batch, out_rows, out_cols,
                        out_depth, out_chan);

    // Output tensor is of the following dimensions:
    // [ in_batch, out_depth, out_rows, out_cols, out_chan ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(2) << "Conv3D: in_chan = " << in_chan
            << ", input_cols = " << input_cols
            << ", filter_cols = " << filter_cols
            << ", input_rows = " << input_rows
            << ", filter_rows = " << filter_rows
            << ", input_depth = " << input_depth
            << ", filter_depth = " << filter_depth
            << ", stride_rows = " << stride_rows
            << ", stride_cols = " << stride_cols
            << ", stride_depth = " << stride_depth
            << ", out_chan = " << out_chan;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }
    LaunchConv3DOp<Device, T>::launch(
        context, use_cudnn_, input, filter, stride_depth, stride_rows,
        stride_cols, BrainPadding2EigenPadding(padding_),
        output, data_format_);
  }

 private:
  std::vector<int32> strides_;
  bool use_cudnn_;
  Padding padding_;
  TensorFormat data_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv3DOp);
};

REGISTER_KERNEL_BUILDER(
    Name("Conv3D").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    Conv3DOp<CPUDevice, float>);




// #if GOOGLE_CUDA

// GPU version remains to be implemented.

// #endif  // GOOGLE_CUDA

}  // namespace tensorflow
