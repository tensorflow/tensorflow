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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/maxpooling_op.h"

#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/eigen_pooling.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/pooling_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/maxpooling_op_gpu.h"
#include "tensorflow/core/kernels/pooling_ops_common_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

const int kInvalidMaxPoolingIndex = -1;

template <typename Device, typename T>
struct SpatialMaxPoolWithArgMaxHelper {
  static void Compute(Tensor* output, Tensor* output_arg_max,
                      const Tensor& tensor_in, const PoolParameters& params,
                      const Padding& padding) {
    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
        ConstEigenMatrixMap;
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
        EigenMatrixMap;
    typedef Eigen::Map<Eigen::Matrix<int64, Eigen::Dynamic, Eigen::Dynamic>>
        EigenIndexMatrixMap;

    ConstEigenMatrixMap in_mat(
        tensor_in.flat<T>().data(), params.depth,
        params.tensor_in_cols * params.tensor_in_rows * params.tensor_in_batch);
    EigenMatrixMap out_mat(
        output->flat<T>().data(), params.depth,
        params.out_width * params.out_height * params.tensor_in_batch);
    EigenIndexMatrixMap out_arg_max_mat(
        output_arg_max->flat<int64>().data(), params.depth,
        params.out_width * params.out_height * params.tensor_in_batch);

    // Initializes the output tensor with MIN<T>.
    output_arg_max->flat<int64>().setConstant(kInvalidMaxPoolingIndex);
    output->flat<T>().setConstant(Eigen::NumTraits<T>::lowest());

    // The following code basically does the following:
    // 1. Flattens the input and output tensors into two dimensional arrays.
    //    tensor_in_as_matrix:
    //      depth by (tensor_in_cols * tensor_in_rows * tensor_in_batch)
    //    output_as_matrix:
    //      depth by (out_width * out_height * tensor_in_batch)
    //
    // 2. Walks through the set of columns in the flattened tensor_in_as_matrix,
    //    and updates the corresponding column(s) in output_as_matrix with the
    //    max value.
    for (int b = 0; b < params.tensor_in_batch; ++b) {
      for (int h = 0; h < params.tensor_in_rows; ++h) {
        for (int w = 0; w < params.tensor_in_cols; ++w) {
          // (h_start, h_end) * (w_start, w_end) is the range that the input
          // vector projects to.
          const int hpad = h + params.pad_rows;
          const int wpad = w + params.pad_cols;
          const int h_start =
              (hpad < params.window_rows)
                  ? 0
                  : (hpad - params.window_rows) / params.row_stride + 1;
          const int h_end =
              std::min(hpad / params.row_stride + 1, params.out_height);
          const int w_start =
              (wpad < params.window_cols)
                  ? 0
                  : (wpad - params.window_cols) / params.col_stride + 1;
          const int w_end =
              std::min(wpad / params.col_stride + 1, params.out_width);
          // compute elementwise max
          const int in_index =
              (b * params.tensor_in_rows + h) * params.tensor_in_cols + w;
          for (int ph = h_start; ph < h_end; ++ph) {
            for (int pw = w_start; pw < w_end; ++pw) {
              const int out_index =
                  (b * params.out_height + ph) * params.out_width + pw;
              /// NOTES(zhengxq): not using the eigen matrix operation for now.
              /// May consider parallelizing the operations if needed.
              for (int d = 0; d < params.depth; ++d) {
                const T& input_ref = in_mat.coeffRef(d, in_index);
                T& output_ref = out_mat.coeffRef(d, out_index);
                int64& out_arg_max_ref = out_arg_max_mat.coeffRef(d, out_index);
                if (output_ref < input_ref ||
                    out_arg_max_ref == kInvalidMaxPoolingIndex) {
                  output_ref = input_ref;
                  int input_offset = in_index * params.depth + d;
                  out_arg_max_ref = input_offset;
                }
              }
            }
          }
        }
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MaxPool").Device(DEVICE_CPU),
                        MaxPoolingOp<CPUDevice, float>);

#if GOOGLE_CUDA
// Forward declarations for the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                            \
  template <>                                                          \
  void SpatialMaxPooling<Eigen::GpuDevice, T>::operator()(             \
      const Eigen::GpuDevice& d, typename TTypes<T, 4>::Tensor output, \
      typename TTypes<T, 4>::ConstTensor input, int window_rows,       \
      int window_cols, int row_stride, int col_stride,                 \
      const Eigen::PaddingType& padding);                              \
  extern template struct SpatialMaxPooling<Eigen::GpuDevice, T>;

DECLARE_GPU_SPEC(float);
#undef DECLARE_GPU_SPEC
}  // namespace functor

// Note(jiayq): Currently, the Caffe custom implementation is faster than the
// default Eigen implementation so we are using the custom kernel as the
// default. However, you can explicitly invoke the eigen version using
// kernel_label_map.
REGISTER_KERNEL_BUILDER(Name("MaxPool")
                            .Device(DEVICE_GPU)
                            .Label("eigen_tensor"),
                        MaxPoolingOp<Eigen::GpuDevice, float>);
#endif  // GOOGLE_CUDA

// The operation to compute MaxPool gradients.
// It takes three inputs:
//   - The original input tensor
//   - The original output tensor
//   - Backprop tensor for output
// It produces one output: backprop tensor for input.
template <class Device, class T>
class MaxPoolingGradOp : public OpKernel {
 public:
  explicit MaxPoolingGradOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, data_format_ == FORMAT_NHWC,
                errors::InvalidArgument(
                    "Default MaxPoolinGradgOp only supports NHWC."));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    OP_REQUIRES(
        context, ksize_[3] == 1 && stride_[3] == 1,
        errors::Unimplemented(
            "MaxPoolingGrad is not yet supported on the depth dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& tensor_out = context->input(1);
    const Tensor& out_backprop = context->input(2);

    // For maxpooling, tensor_in should have 4 dimensions.
    OP_REQUIRES(context, tensor_in.dims() == 4,
                errors::InvalidArgument("tensor_in must be 4-dimensional"));
    OP_REQUIRES(context, tensor_out.dims() == 4,
                errors::InvalidArgument("tensor_out must be 4-dimensional"));
    // For maxpooling, out_backprop should have 4 dimensions.
    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));

    TensorShape output_shape = tensor_in.shape();

    // Tensor index_tensor(context->allocator(), DT_INT32, output_shape);

    Tensor tensor_out_dup;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::v(),
                                          tensor_out.shape(), &tensor_out_dup));
    Tensor tensor_out_arg_max;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int64>::v(),
                                                   tensor_out.shape(),
                                                   &tensor_out_arg_max));

    PoolParameters params{context,  ksize_,      stride_,
                          padding_, FORMAT_NHWC, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    output->flat<T>().setZero();

    SpatialMaxPoolWithArgMaxHelper<CPUDevice, T>::Compute(
        &tensor_out_dup, &tensor_out_arg_max, tensor_in, params, padding_);
    auto out_backprop_flat = out_backprop.flat<T>();
    auto input_backprop_flat = output->flat<T>();
    auto out_arg_max_flat = tensor_out_arg_max.flat<int64>();
    int num_total_outputs = out_backprop.flat<T>().size();
    int num_total_inputs = input_backprop_flat.size();

    for (int index = 0; index < num_total_outputs; ++index) {
      int input_backprop_index = out_arg_max_flat(index);
      // Although this check is in the inner loop, it is worth its value
      // so we don't end up with memory corruptions. Our benchmark shows that
      // the performance impact is quite small
      CHECK(input_backprop_index >= 0 &&
            input_backprop_index < num_total_inputs)
          << "Invalid input backprop index: " << input_backprop_index << ", "
          << num_total_inputs;
      input_backprop_flat(input_backprop_index) += out_backprop_flat(index);
    }
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

REGISTER_KERNEL_BUILDER(Name("MaxPoolGrad").Device(DEVICE_CPU),
                        MaxPoolingGradOp<CPUDevice, float>);

#ifdef GOOGLE_CUDA

static void MaxPoolingBackwardCustomKernel(
    OpKernelContext* context, const std::vector<int32>& size,
    const std::vector<int32>& stride, Padding padding, const Tensor* tensor_in,
    const Tensor& out_backprop, const TensorShape& tensor_in_shape) {
  Tensor* output = nullptr;

  OP_REQUIRES_OK(context,
                 context->allocate_output(0, tensor_in_shape, &output));

  PoolParameters params{context, size,        stride,
                        padding, FORMAT_NHWC, tensor_in_shape};
  if (!context->status().ok()) {
    return;
  }

  MaxPoolBackwardNoMask(
      tensor_in->flat<float>().data(), params.tensor_in_batch,
      params.tensor_in_rows, params.tensor_in_cols, params.depth,
      params.out_height, params.out_width, params.window_rows,
      params.window_cols, params.row_stride, params.col_stride, params.pad_rows,
      params.pad_cols, out_backprop.flat<float>().data(),
      output->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}

template <class T>
class MaxPoolingGradOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit MaxPoolingGradOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    const int32 ksize_n = GetTensorDim(ksize_, data_format_, 'N');
    const int32 stride_n = GetTensorDim(stride_, data_format_, 'N');
    OP_REQUIRES(context, ksize_n == 1 && stride_n == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));

    use_dnn_ = CanUseCudnn();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& tensor_out = context->input(1);
    const Tensor& out_backprop = context->input(2);

    // For maxpooling, tensor_in should have 4 dimensions.
    OP_REQUIRES(context, tensor_in.dims() == 4,
                errors::InvalidArgument("tensor_in must be 4-dimensional 4"));
    OP_REQUIRES(context, tensor_out.dims() == 4,
                errors::InvalidArgument("tensor_out must be 4-dimensional"));
    // For maxpooling, out_backprop should have 4 dimensions.
    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));

    TensorShape output_shape = tensor_in.shape();

    if (use_dnn_) {
      DnnPoolingGradOp<T>::Compute(
          context, perftools::gputools::dnn::PoolingMode::kMaximum, ksize_,
          stride_, padding_, data_format_, &tensor_in, &tensor_out,
          out_backprop, output_shape);
    } else {
      CHECK(data_format_ == FORMAT_NHWC)
          << "Non-Cudnn MaxPoolGrad only supports NHWC format";
      MaxPoolingBackwardCustomKernel(context, ksize_, stride_, padding_,
                                     &tensor_in, out_backprop, output_shape);
    }
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool use_dnn_;
};

REGISTER_KERNEL_BUILDER(Name("MaxPoolGrad").Device(DEVICE_GPU),
                        MaxPoolingGradOp<Eigen::GpuDevice, float>);

#endif  // GOOGLE_CUDA

template <typename Device, typename T>
struct LaunchMaxPoolingNoMask;

template <typename Device, typename T>
class MaxPoolingNoMaskOp : public OpKernel {
 public:
  explicit MaxPoolingNoMaskOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, data_format_ == FORMAT_NHWC,
                errors::InvalidArgument(
                    "Default MaxPoolingNoMaskOp only supports NHWC."));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);

    PoolParameters params{context,  ksize_,       stride_,
                          padding_, data_format_, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    TensorShape out_shape({params.tensor_in_batch, params.out_height,
                           params.out_width, params.depth});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    LaunchMaxPoolingNoMask<Device, T>::launch(context, params, tensor_in,
                                              output);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

template <typename Device, typename T>
struct LaunchMaxPoolingWithArgmax;

template <typename Device, typename T>
class MaxPoolingWithArgmaxOp : public OpKernel {
 public:
  explicit MaxPoolingWithArgmaxOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);

    PoolParameters params{context,  ksize_,      stride_,
                          padding_, FORMAT_NHWC, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    TensorShape out_shape({params.tensor_in_batch, params.out_height,
                           params.out_width, params.depth});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    Tensor* argmax = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, out_shape, &argmax));

    LaunchMaxPoolingWithArgmax<Device, T>::launch(context, params, tensor_in,
                                                  output, argmax);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
};

template <typename Device, typename T>
struct LaunchMaxPoolingGradWithArgmax;

template <typename Device, typename T>
class MaxPoolingGradWithArgmaxOp : public OpKernel {
 public:
  explicit MaxPoolingGradWithArgmaxOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& grad_in = context->input(1);
    const Tensor& argmax = context->input(2);

    PoolParameters params{context,  ksize_,      stride_,
                          padding_, FORMAT_NHWC, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    TensorShape out_shape({params.tensor_in_batch, params.tensor_in_rows,
                           params.tensor_in_cols, params.depth});
    Tensor* grad_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &grad_out));

    LaunchMaxPoolingGradWithArgmax<Device, T>::launch(context, params, grad_in,
                                                      argmax, grad_out);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
};

#if GOOGLE_CUDA
template <typename T>
class MaxPoolingNoMaskOp<GPUDevice, T> : public OpKernel {
 public:
  typedef GPUDevice Device;
  explicit MaxPoolingNoMaskOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    const int32 ksize_n = GetTensorDim(ksize_, data_format_, 'N');
    const int32 stride_n = GetTensorDim(stride_, data_format_, 'N');
    OP_REQUIRES(context, ksize_n == 1 && stride_n == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    use_dnn_ = CanUseCudnn();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);

    PoolParameters params{context,  ksize_,       stride_,
                          padding_, data_format_, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    TensorShape out_shape =
        ShapeFromFormat(data_format_, params.tensor_in_batch, params.out_height,
                        params.out_width, params.depth);
    if (use_dnn_ && data_format_ == FORMAT_NCHW) {
      DnnPoolingOp<T>::Compute(
          context, perftools::gputools::dnn::PoolingMode::kMaximum, ksize_,
          stride_, padding_, data_format_, tensor_in, out_shape);
    } else {
      CHECK(data_format_ == FORMAT_NHWC)
          << "Non-Cudnn MaxPool only supports NHWC format";
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
      LaunchMaxPoolingNoMask<Device, T>::launch(context, params, tensor_in,
                                                output);
    }
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool use_dnn_;
};

template <typename T>
struct LaunchMaxPoolingNoMask<Eigen::GpuDevice, T> {
  static void launch(OpKernelContext* context, const PoolParameters& params,
                     const Tensor& input, Tensor* output) {
    bool status = MaxPoolForwardWithOptionalArgmax(
        input.flat<T>().data(), params.tensor_in_batch, params.tensor_in_rows,
        params.tensor_in_cols, params.depth, params.out_height,
        params.out_width, params.window_rows, params.window_cols,
        params.row_stride, params.col_stride, params.pad_rows, params.pad_cols,
        output->flat<T>().data(), nullptr, context->eigen_gpu_device());
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching MaxPoolForwardNoMask"));
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MaxPool").Device(DEVICE_GPU),
                        MaxPoolingNoMaskOp<Eigen::GpuDevice, float>);

template <typename T>
struct LaunchMaxPoolingWithArgmax<Eigen::GpuDevice, T> {
  static void launch(OpKernelContext* context, const PoolParameters& params,
                     const Tensor& input, Tensor* output, Tensor* argmax) {
    bool status = MaxPoolForwardWithOptionalArgmax(
        input.flat<T>().data(), params.tensor_in_batch, params.tensor_in_rows,
        params.tensor_in_cols, params.depth, params.out_height,
        params.out_width, params.window_rows, params.window_cols,
        params.row_stride, params.col_stride, params.pad_rows, params.pad_cols,
        output->flat<T>().data(),
        reinterpret_cast<int64*>(argmax->flat<int64>().data()),
        context->eigen_gpu_device());
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching MaxPoolForwardWithArgmax"));
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MaxPoolWithArgmax")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int64>("Targmax"),
                        MaxPoolingWithArgmaxOp<Eigen::GpuDevice, float>);

template <typename T>
struct LaunchMaxPoolingGradWithArgmax<Eigen::GpuDevice, T> {
  static void launch(OpKernelContext* context, const PoolParameters& params,
                     const Tensor& grad_in, const Tensor& argmax,
                     Tensor* grad_out) {
    const int input_size = params.tensor_in_batch * params.tensor_in_rows *
                           params.tensor_in_cols * params.depth;
    const int output_size = params.tensor_in_batch * params.out_height *
                            params.out_width * params.depth;
    const int top_offset = params.out_height * params.out_width * params.depth;
    const int bottom_offset =
        params.tensor_in_rows * params.tensor_in_cols * params.depth;
    bool status = MaxPoolBackwardWithArgmax(
        output_size, input_size, grad_in.flat<T>().data(),
        reinterpret_cast<const int64*>(argmax.flat<int64>().data()), top_offset,
        bottom_offset, grad_out->flat<T>().data(), context->eigen_gpu_device());
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching MaxPoolForwardWithArgmax"));
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MaxPoolGradWithArgmax")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int64>("Targmax"),
                        MaxPoolingGradWithArgmaxOp<Eigen::GpuDevice, float>);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
