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
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/pooling_ops_3d.h"

#include <array>

#include "Eigen/Core"  // from @eigen_archive
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/eigen_pooling.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/work_sharder.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/cudnn_pooling_gpu.h"
#include "tensorflow/core/kernels/pooling_ops_3d_gpu.h"
#endif


namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

Pool3dParameters::Pool3dParameters(OpKernelContext* context,
                                   const std::vector<int32>& ksize,
                                   const std::vector<int32>& stride,
                                   Padding padding, TensorFormat data_format,
                                   const TensorShape& tensor_in_shape) {
  // For maxpooling, tensor_in should have 4 dimensions.
  OP_REQUIRES(context, tensor_in_shape.dims() == 5,
              errors::InvalidArgument("tensor_in must be 4-dimensional"));

  this->data_format = data_format;
  depth = GetTensorDim(tensor_in_shape, data_format, 'C');
  tensor_in_planes = GetTensorDim(tensor_in_shape, data_format, '0');
  tensor_in_rows = GetTensorDim(tensor_in_shape, data_format, '1');
  tensor_in_cols = GetTensorDim(tensor_in_shape, data_format, '2');
  tensor_in_batch = GetTensorDim(tensor_in_shape, data_format, 'N');
  window_planes = GetTensorDim(ksize, data_format, '0');
  window_rows = GetTensorDim(ksize, data_format, '1');
  window_cols = GetTensorDim(ksize, data_format, '2');
  depth_window = GetTensorDim(ksize, data_format, 'C');
  plane_stride = GetTensorDim(stride, data_format, '0');
  row_stride = GetTensorDim(stride, data_format, '1');
  col_stride = GetTensorDim(stride, data_format, '2');
  depth_stride = GetTensorDim(stride, data_format, 'C');

  // We only support 3D pooling across plane/width/height. Depthwise
  // pooling is not supported.
  OP_REQUIRES(
      context, depth_window == 1 && depth_stride == 1,
      errors::Unimplemented(
          "Pooling3d only supports pooling across plane/width/height."));

  OP_REQUIRES_OK(
      context, GetWindowedOutputSize(tensor_in_planes, window_planes,
                                     /*dilation_rate=*/1, plane_stride, padding,
                                     &out_plane, &pad_planes));
  OP_REQUIRES_OK(context, GetWindowedOutputSize(
                              tensor_in_rows, window_rows, /*dilation_rate=*/1,
                              row_stride, padding, &out_height, &pad_rows));
  OP_REQUIRES_OK(context, GetWindowedOutputSize(
                              tensor_in_cols, window_cols, /*dilation_rate=*/1,
                              col_stride, padding, &out_width, &pad_cols));
}

absl::Status Pool3dParameters::forward_output_shape(TensorShape* shape) {
  return ShapeFromFormatWithStatus(data_format, tensor_in_batch,
                                   {{out_plane, out_height, out_width}}, depth,
                                   shape);
}

template <typename T>
struct LaunchPoolingOp<CPUDevice, T, AVG> {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Padding padding_type,
                     Tensor* output) {
    output->tensor<T, 5>().device(context->eigen_device<CPUDevice>()) =
        Eigen::CuboidAvgPooling(tensor_in.tensor<T, 5>(), window[0], window[1],
                                window[2], stride[0], stride[1], stride[2],
                                BrainPadding2EigenPadding(padding_type));
  }
};

template <typename T>
struct LaunchPoolingOp<CPUDevice, T, MAX> {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Padding padding_type,
                     Tensor* output) {
    output->tensor<T, 5>().device(context->eigen_device<CPUDevice>()) =
        Eigen::CuboidMaxPooling(tensor_in.tensor<T, 5>(), window[0], window[1],
                                window[2], stride[0], stride[1], stride[2],
                                BrainPadding2EigenPadding(padding_type));
  }
};

template <typename Device, typename T, PoolingType Type>
class Pooling3DOp : public UnaryOp<T> {
 public:
  explicit Pooling3DOp(OpKernelConstruction* context) : UnaryOp<T>(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    if (context->device_type() == DEVICE_CPU) {
      OP_REQUIRES(
          context, data_format_ == FORMAT_NHWC,
          errors::InvalidArgument("Default Pooling3DOp only supports NDHWC ",
                                  "on device type ",
                                  DeviceTypeString(context->device_type())));
    }
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 5,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 5 dimensions"));
    bool non_negative =
        std::all_of(ksize_.begin(), ksize_.end(), [](int k) { return k > 0; });
    OP_REQUIRES(context, non_negative,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "have non-negative dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context,
                (GetTensorDim(ksize_, data_format_, 'N') == 1 &&
                 GetTensorDim(stride_, data_format_, 'N') == 1),
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    OP_REQUIRES(context,
                (GetTensorDim(ksize_, data_format_, 'C') == 1 &&
                 GetTensorDim(stride_, data_format_, 'C') == 1),
                errors::Unimplemented(
                    "Pooling is not yet supported on the depth dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);

    OP_REQUIRES(context, tensor_in.dims() == 5,
                errors::InvalidArgument("tensor_in must be 5-dimensional"));
    const int64_t depth = GetTensorDim(tensor_in, data_format_, 'C');
    const int64_t in_batch = GetTensorDim(tensor_in, data_format_, 'N');

    // Dimension order for these arrays is: x, y, z.
    std::array<int64_t, 3> input_size{
        {GetTensorDim(tensor_in, data_format_, '2'),
         GetTensorDim(tensor_in, data_format_, '1'),
         GetTensorDim(tensor_in, data_format_, '0')}};
    std::array<int64_t, 3> window{{GetTensorDim(ksize_, data_format_, '2'),
                                   GetTensorDim(ksize_, data_format_, '1'),
                                   GetTensorDim(ksize_, data_format_, '0')}};
    std::array<int64_t, 3> stride{{GetTensorDim(stride_, data_format_, '2'),
                                   GetTensorDim(stride_, data_format_, '1'),
                                   GetTensorDim(stride_, data_format_, '0')}};
    std::array<int64_t, 3> padding, out;
    std::array<int64_t, 3> dilations{1, 1, 1};

    OP_REQUIRES_OK(context,
                   Get3dOutputSizeV2(input_size, window, dilations, stride,
                                     padding_, &out, &padding));

    TensorShape out_shape;
    OP_REQUIRES_OK(context, ShapeFromFormatWithStatus(
                                data_format_, in_batch,
                                {{out[2], out[1], out[0]}}, depth, &out_shape));
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    if (out_shape.num_elements() == 0) return;
    LaunchPoolingOp<Device, T, Type>::launch(context, tensor_in, window, stride,
                                             padding, data_format_, padding_,
                                             output);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

template <typename T>
struct LaunchMaxPooling3dGradOp<CPUDevice, T> {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const Tensor& tensor_out, const Tensor& out_backprop,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& out,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Tensor* output) {
    output->flat<T>().setZero();
    for (int64_t p = 0; p < out_backprop.dim_size(3); ++p) {
      // Calculate broadcast size for planes/rows/cols. For SAME padding,
      // current index could be in the padding area, and
      //   p * stride_planes + window_planes
      // could be beyond the input tensor's boundary. In such cases, change
      // the starting index and reduce the broadcast size.
      //
      // The same procedure is repeated for every spatial dimension in the
      // nested loops below.
      int pindex, psize;
      std::array<int64_t, 3> input_size{{tensor_in.dim_size(3),
                                         tensor_in.dim_size(2),
                                         tensor_in.dim_size(1)}};
      OP_REQUIRES_OK(context,
                     GetBroadcastSize(p, input_size[0], window[0], stride[0],
                                      padding[0], &pindex, &psize));
      for (int64_t r = 0; r < out_backprop.dim_size(2); ++r) {
        int rindex, rsize;
        OP_REQUIRES_OK(context,
                       GetBroadcastSize(r, input_size[1], window[1], stride[1],
                                        padding[1], &rindex, &rsize));
        for (int64_t c = 0; c < out_backprop.dim_size(1); ++c) {
          int cindex, csize;
          OP_REQUIRES_OK(
              context, GetBroadcastSize(c, input_size[2], window[2], stride[2],
                                        padding[2], &cindex, &csize));
          TensorSlice src{{0, -1}, {c, 1}, {r, 1}, {p, 1}, {0, -1}};
          TensorSlice dst{{0, -1},
                          {cindex, csize},
                          {rindex, rsize},
                          {pindex, psize},
                          {0, -1}};
          Eigen::DSizes<Eigen::DenseIndex, 5> src_indices;
          Eigen::DSizes<Eigen::DenseIndex, 5> src_sizes;
          Eigen::DSizes<Eigen::DenseIndex, 5> dst_indices;
          Eigen::DSizes<Eigen::DenseIndex, 5> dst_sizes;
          src.FillIndicesAndSizes<5>(out_backprop.shape(), &src_indices,
                                     &src_sizes);
          dst.FillIndicesAndSizes<5>(tensor_in.shape(), &dst_indices,
                                     &dst_sizes);

          Eigen::IndexList<Eigen::type2index<1>, int, int, int,
                           Eigen::type2index<1>>
              bcast;
          bcast.set(1, csize);
          bcast.set(2, rsize);
          bcast.set(3, psize);

          // Slice from tensor_in.
          Eigen::Tensor<T, 5, Eigen::RowMajor> tensor_in_slice(dst_sizes);
          tensor_in_slice.device(context->eigen_cpu_device()) =
              tensor_in.tensor<T, 5>().slice(dst_indices, dst_sizes);

          // Slice from tensor_out.
          Eigen::Tensor<T, 5, Eigen::RowMajor> tensor_out_slice(src_sizes);
          tensor_out_slice.device(context->eigen_cpu_device()) =
              tensor_out.tensor<T, 5>().slice(src_indices, src_sizes);

          // Backprop slice.
          Eigen::Tensor<T, 5, Eigen::RowMajor> out_backprop_slice(src_sizes);
          out_backprop_slice.device(context->eigen_cpu_device()) =
              out_backprop.tensor<T, 5>().slice(src_indices, src_sizes);

          // The true backprop slice: if an element is the max, choose
          // the backprop slice; otherwise set to 0.
          Eigen::Tensor<T, 5, Eigen::RowMajor> select_slice(dst_sizes);
          Eigen::Tensor<T, 5, Eigen::RowMajor> mat0(dst_sizes);
          mat0.setZero();
          select_slice =
              ((tensor_in_slice - tensor_out_slice.broadcast(bcast)).abs() <
               tensor_in_slice.constant(T(1e-5)))
                  .select(out_backprop_slice.broadcast(bcast), mat0);

          output->tensor<T, 5>()
              .slice(dst_indices, dst_sizes)
              .device(context->eigen_cpu_device()) += select_slice;
        }
      }
    }
  }
};

template <class Device, class T>
class MaxPooling3dGradOp : public OpKernel {
 public:
  explicit MaxPooling3dGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    if (context->device_type() == DEVICE_CPU) {
      OP_REQUIRES(
          context, data_format_ == FORMAT_NHWC,
          errors::InvalidArgument(
              "Default MaxPooling3dGradOp only supports NDHWC ",
              "on device type ", DeviceTypeString(context->device_type())));
    }
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 5,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context,
                (GetTensorDim(ksize_, data_format_, 'N') == 1 &&
                 GetTensorDim(stride_, data_format_, 'N') == 1),
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    OP_REQUIRES(context,
                (GetTensorDim(ksize_, data_format_, 'C') == 1 &&
                 GetTensorDim(stride_, data_format_, 'C') == 1),
                errors::Unimplemented(
                    "Pooling is not yet supported on the depth dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& tensor_out = context->input(1);
    const Tensor& out_backprop = context->input(2);
    OP_REQUIRES(context, tensor_in.dims() == 5,
                errors::InvalidArgument("tensor_in must be 5-dimensional"));
    OP_REQUIRES(context, tensor_out.dims() == 5,
                errors::InvalidArgument("tensor_out must be 5-dimensional"));
    OP_REQUIRES(context, out_backprop.dims() == 5,
                errors::InvalidArgument("out_backprop must be 5-dimensional"));

    const TensorShape& output_shape = tensor_in.shape();
    Tensor* input_backprop;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &input_backprop));
    std::array<int64_t, 3> input_size{
        {GetTensorDim(output_shape, data_format_, '2'),
         GetTensorDim(output_shape, data_format_, '1'),
         GetTensorDim(output_shape, data_format_, '0')}};
    std::array<int64_t, 3> window{{GetTensorDim(ksize_, data_format_, '2'),
                                   GetTensorDim(ksize_, data_format_, '1'),
                                   GetTensorDim(ksize_, data_format_, '0')}};
    std::array<int64_t, 3> stride{{GetTensorDim(stride_, data_format_, '2'),
                                   GetTensorDim(stride_, data_format_, '1'),
                                   GetTensorDim(stride_, data_format_, '0')}};
    std::array<int64_t, 3> out, padding;
    std::array<int64_t, 3> dilations{1, 1, 1};

    OP_REQUIRES_OK(context,
                   Get3dOutputSizeV2(input_size, window, dilations, stride,
                                     padding_, &out, &padding));

    const int64_t depth = GetTensorDim(tensor_in, data_format_, 'C');
    const int64_t in_batch = GetTensorDim(tensor_in, data_format_, 'N');
    TensorShape out_shape;
    OP_REQUIRES_OK(context, ShapeFromFormatWithStatus(
                                data_format_, in_batch,
                                {{out[2], out[1], out[0]}}, depth, &out_shape));
    OP_REQUIRES(
        context, tensor_out.shape() == out_shape,
        errors::InvalidArgument("Expected orig_output shape to be ", out_shape,
                                ", but got ", tensor_out.shape()));
    OP_REQUIRES(context, out_backprop.shape() == out_shape,
                errors::InvalidArgument("Expected grad shape to be ", out_shape,
                                        ", but got ", out_backprop.shape()));

    LaunchMaxPooling3dGradOp<Device, T>::launch(
        context, tensor_in, tensor_out, out_backprop, window, stride, out,
        padding, data_format_, input_backprop);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

template <typename T>
struct LaunchAvgPooling3dGradOp<CPUDevice, T> {
  static void launch(OpKernelContext* context,
                     const TensorShape& tensor_in_shape,
                     const Tensor& out_backprop,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& output_shape,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Tensor* output) {
    OP_REQUIRES(
        context, tensor_in_shape.dim_size(0) == out_backprop.dim_size(0),
        errors::InvalidArgument(
            "Expected first dimension of tensor_in_shape and "
            "out_backprop to match, got ",
            tensor_in_shape.dim_size(0), " and ", out_backprop.dim_size(0)));
    OP_REQUIRES(
        context, tensor_in_shape.dim_size(4) == out_backprop.dim_size(4),
        errors::InvalidArgument(
            "Expected last dimension of tensor_in_shape and "
            "out_backprop to match, got ",
            tensor_in_shape.dim_size(4), " and ", out_backprop.dim_size(4)));

    output->flat<T>().setZero();
    std::array<int64_t, 3> input_size = {{tensor_in_shape.dim_size(3),
                                          tensor_in_shape.dim_size(2),
                                          tensor_in_shape.dim_size(1)}};
    for (int64_t p = 0; p < out_backprop.dim_size(3); ++p) {
      // Calculate broadcast size for planes/rows/cols. For SAME padding,
      // current index could be in the padding area, and
      //   p * stride_planes + window_planes
      // could be beyond the input tensor's boundary. In such cases, change
      // the starting index and reduce the broadcast size.
      //
      // The same procedure is repeated for every spatial dimension in the
      // nested loops below.
      int pindex, psize;
      OP_REQUIRES_OK(context,
                     GetBroadcastSize(p, input_size[0], window[0], stride[0],
                                      padding[0], &pindex, &psize));
      for (int64_t r = 0; r < out_backprop.dim_size(2); ++r) {
        int rindex, rsize;
        OP_REQUIRES_OK(context,
                       GetBroadcastSize(r, input_size[1], window[1], stride[1],
                                        padding[1], &rindex, &rsize));
        for (int64_t c = 0; c < out_backprop.dim_size(1); ++c) {
          int cindex, csize;
          OP_REQUIRES_OK(
              context, GetBroadcastSize(c, input_size[2], window[2], stride[2],
                                        padding[2], &cindex, &csize));
          TensorSlice src{{0, -1}, {c, 1}, {r, 1}, {p, 1}, {0, -1}};
          TensorSlice dst{{0, -1},
                          {cindex, csize},
                          {rindex, rsize},
                          {pindex, psize},
                          {0, -1}};
          Eigen::DSizes<Eigen::DenseIndex, 5> src_indices;
          Eigen::DSizes<Eigen::DenseIndex, 5> src_sizes;
          Eigen::DSizes<Eigen::DenseIndex, 5> dst_indices;
          Eigen::DSizes<Eigen::DenseIndex, 5> dst_sizes;
          src.FillIndicesAndSizes<5>(out_backprop.shape(), &src_indices,
                                     &src_sizes);
          dst.FillIndicesAndSizes<5>(tensor_in_shape, &dst_indices, &dst_sizes);
          Eigen::IndexList<Eigen::type2index<1>, int, int, int,
                           Eigen::type2index<1>>
              bcast;
          bcast.set(1, csize);
          bcast.set(2, rsize);
          bcast.set(3, psize);
          Eigen::Tensor<T, 5, Eigen::RowMajor> slices(src_sizes);
          slices.device(context->eigen_cpu_device()) =
              out_backprop.tensor<T, 5>().slice(src_indices, src_sizes);
          // Divide by the size of the actual patch (psize * rsize * csize).
          float divide_size = rsize * csize * psize * 1.0f;
          slices *= slices.constant(T(1.0f / divide_size));

          output->tensor<T, 5>()
              .slice(dst_indices, dst_sizes)
              .device(context->eigen_cpu_device()) += slices.broadcast(bcast);
        }
      }
    }
  }
};

template <class Device, class T>
class AvgPooling3dGradOp : public OpKernel {
 public:
  explicit AvgPooling3dGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    if (context->device_type() == DEVICE_CPU) {
      OP_REQUIRES(
          context, data_format_ == FORMAT_NHWC,
          errors::InvalidArgument(
              "Default AvgPooling3dGradOp only supports NDHWC ",
              "on device type ", DeviceTypeString(context->device_type())));
    }
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 5,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 5 dimensions"));
    for (std::size_t i = 0; i < ksize_.size(); ++i) {
      OP_REQUIRES(
          context, ksize_[i] > 0,
          errors::InvalidArgument("ksize must be positive, got: ", ksize_[i]));
    }
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context,
                (GetTensorDim(ksize_, data_format_, 'N') == 1 &&
                 GetTensorDim(stride_, data_format_, 'N') == 1),
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    OP_REQUIRES(context,
                (GetTensorDim(ksize_, data_format_, 'C') == 1 &&
                 GetTensorDim(stride_, data_format_, 'C') == 1),
                errors::Unimplemented(
                    "Pooling is not yet supported on the depth dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in_shape = context->input(0);
    const Tensor& out_backprop = context->input(1);
    OP_REQUIRES(
        context,
        tensor_in_shape.dims() == 1 && tensor_in_shape.NumElements() == 5,
        errors::InvalidArgument("tensor_in must be 1-dimensional and 5 "
                                "elements"));
    OP_REQUIRES(context, out_backprop.dims() == 5,
                errors::InvalidArgument("out_backprop must be 5-dimensional"));

    TensorShape output_shape;
    auto shape_vec = tensor_in_shape.vec<int32>();
    for (int64_t i = 0; i < tensor_in_shape.NumElements(); ++i) {
      OP_REQUIRES_OK(context, output_shape.AddDimWithStatus(shape_vec(i)));
    }

    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    // Dimension order for these arrays is x, y, z.
    std::array<int64_t, 3> input_size{
        {GetTensorDim(output_shape, data_format_, '2'),
         GetTensorDim(output_shape, data_format_, '1'),
         GetTensorDim(output_shape, data_format_, '0')}};
    std::array<int64_t, 3> window{{GetTensorDim(ksize_, data_format_, '2'),
                                   GetTensorDim(ksize_, data_format_, '1'),
                                   GetTensorDim(ksize_, data_format_, '0')}};
    std::array<int64_t, 3> stride{{GetTensorDim(stride_, data_format_, '2'),
                                   GetTensorDim(stride_, data_format_, '1'),
                                   GetTensorDim(stride_, data_format_, '0')}};
    std::array<int64_t, 3> padding, out;
    std::array<int64_t, 3> dilations{1, 1, 1};

    OP_REQUIRES_OK(context,
                   Get3dOutputSizeV2(input_size, window, dilations, stride,
                                     padding_, &out, &padding));

    LaunchAvgPooling3dGradOp<Device, T>::launch(
        context, output_shape, out_backprop, window, stride, out, padding,
        data_format_, output);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

template <typename T>
struct LaunchMaxPooling3dGradGradOp<CPUDevice, T> {
  static void launch(OpKernelContext* context, const Pool3dParameters& params,
                     const Tensor& tensor_in, const Tensor& tensor_out,
                     const Tensor& tensor_top_diff,
                     Tensor* tensor_bottom_diff) {
    OP_REQUIRES(
        context, params.data_format == FORMAT_NHWC,
        errors::InvalidArgument("Default MaxPooling3dGradGradOp only supports",
                                "NDHWC on CPU device type"));

    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
        ConstEigenMatrixMap;
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
        EigenMatrixMap;

    ConstEigenMatrixMap in_mat(tensor_in.flat<T>().data(), params.depth,
                               params.tensor_in_planes * params.tensor_in_cols *
                                   params.tensor_in_rows *
                                   params.tensor_in_batch);
    ConstEigenMatrixMap out_mat(tensor_out.flat<T>().data(), params.depth,
                                params.out_plane * params.out_width *
                                    params.out_height * params.tensor_in_batch);
    ConstEigenMatrixMap top_diff_mat(
        tensor_top_diff.flat<T>().data(), params.depth,
        params.tensor_in_planes * params.tensor_in_cols *
            params.tensor_in_rows * params.tensor_in_batch);
    EigenMatrixMap bottom_diff_mat(
        tensor_bottom_diff->flat<T>().data(), params.depth,
        params.out_plane * params.out_width * params.out_height *
            params.tensor_in_batch);

    const DeviceBase::CpuWorkerThreads& worker_threads =
        *(context->device()->tensorflow_cpu_worker_threads());

    auto shard = [&params, &in_mat, &out_mat, &top_diff_mat, &bottom_diff_mat](
                     int64_t start, int64_t limit) {
      const int32_t depth = params.depth;
      const int32_t in_planes = params.tensor_in_planes;
      const int32_t in_rows = params.tensor_in_rows;
      const int32_t in_cols = params.tensor_in_cols;
      const int32_t pad_planes = params.pad_planes;
      const int32_t pad_rows = params.pad_rows;
      const int32_t pad_cols = params.pad_cols;
      const int32_t window_planes = params.window_planes;
      const int32_t window_rows = params.window_rows;
      const int32_t window_cols = params.window_cols;
      const int32_t plane_stride = params.plane_stride;
      const int32_t row_stride = params.row_stride;
      const int32_t col_stride = params.col_stride;
      const int32_t out_plane = params.out_plane;
      const int32_t out_height = params.out_height;
      const int32_t out_width = params.out_width;

      {
        // Initializes the output grad backprop tensor with 0.
        const int32_t output_image_size =
            out_plane * out_height * out_width * params.depth;
        EigenMatrixMap bottom_diff_shard(
            bottom_diff_mat.data() + start * output_image_size, 1,
            (limit - start) * output_image_size);
        bottom_diff_shard.setZero();
      }

      for (int b = start; b < limit; ++b) {
        for (int pp = 0; pp < out_plane; ++pp) {
          for (int ph = 0; ph < out_height; ++ph) {
            for (int pw = 0; pw < out_width; ++pw) {
              // (p_start, p_end) * (h_start, h_end) * (w_start, w_end) is the
              // range that the input vector projects to.
              int p_start = pp * plane_stride - pad_planes;
              const int p_end = std::min(p_start + window_planes, in_planes);
              int h_start = ph * row_stride - pad_rows;
              const int h_end = std::min(h_start + window_rows, in_rows);
              int w_start = pw * col_stride - pad_cols;
              const int w_end = std::min(w_start + window_cols, in_cols);
              p_start = std::max(p_start, 0);
              h_start = std::max(h_start, 0);
              w_start = std::max(w_start, 0);
              const int out_index =
                  ((b * out_plane + pp) * out_height + ph) * out_width + pw;
              // Find value corresponding to the input maximum in top_diff.
              for (int d = 0; d < depth; ++d) {
                const T& output_ref = out_mat.coeffRef(d, out_index);
                bool should_stop = false;
                for (int p = p_start; p < p_end && !should_stop; ++p) {
                  for (int h = h_start; h < h_end && !should_stop; ++h) {
                    for (int w = w_start; w < w_end && !should_stop; ++w) {
                      const int in_index =
                          ((b * in_planes + p) * in_rows + h) * in_cols + w;
                      const T& input_ref = in_mat.coeffRef(d, in_index);
                      if (output_ref == input_ref) {
                        T& bottom_diff_ref =
                            bottom_diff_mat.coeffRef(d, out_index);
                        bottom_diff_ref = top_diff_mat.coeffRef(d, in_index);
                        should_stop = true;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    };
    const int64_t shard_cost =
        params.out_plane * params.out_height * params.out_width * params.depth *
        params.window_planes * params.window_rows * params.window_cols;
    Shard(worker_threads.num_threads, worker_threads.workers,
          params.tensor_in_batch, shard_cost, shard);
  }
};

template <class Device, class T>
class MaxPooling3dGradGradOp : public OpKernel {
 public:
  explicit MaxPooling3dGradGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 5,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    const int32_t ksize_c = GetTensorDim(ksize_, data_format_, 'C');
    const int32_t stride_c = GetTensorDim(stride_, data_format_, 'C');
    OP_REQUIRES(context, ksize_c == 1 && stride_c == 1,
                errors::Unimplemented("MaxPooling3dGradGrad is not yet "
                                      "supported on the depth dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& tensor_out = context->input(1);
    const Tensor& out_grad_backprop = context->input(2);

    // For maxpooling3d, tensor_in should have 5 dimensions.
    OP_REQUIRES(context, tensor_in.dims() == 5,
                errors::InvalidArgument("tensor_in must be 5-dimensional"));
    OP_REQUIRES(context, tensor_out.dims() == 5,
                errors::InvalidArgument("tensor_out must be 5-dimensional"));
    // For maxpooling3d, out_grad_backprop should have 5 dimensions.
    OP_REQUIRES(
        context, out_grad_backprop.dims() == 5,
        errors::InvalidArgument("out_grad_backprop must be 5-dimensional"));

    Pool3dParameters params{context,  ksize_,       stride_,
                            padding_, data_format_, tensor_in.shape()};
    if (!context->status().ok()) return;  // params is invalid
    TensorShape params_forward_output_shape;
    OP_REQUIRES_OK(context,
                   params.forward_output_shape(&params_forward_output_shape));
    OP_REQUIRES(context, tensor_out.shape() == params_forward_output_shape,
                errors::InvalidArgument("Expected orig_output shape to be ",
                                        params_forward_output_shape,
                                        ", but got ", tensor_out.shape()));
    OP_REQUIRES(
        context, out_grad_backprop.shape() == tensor_in.shape(),
        errors::InvalidArgument("Expected grad shape to be ", tensor_in.shape(),
                                ", but got ", out_grad_backprop.shape()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {2}, 0, tensor_out.shape(), &output));

    // Given access patterns in LaunchMaxPooling3dGradGradOp, these tensors must
    // have elements.
    OP_REQUIRES(context, tensor_in.NumElements() > 0,
                errors::InvalidArgument("received empty tensor tensor_in: ",
                                        tensor_in.DebugString()));
    OP_REQUIRES(context, tensor_out.NumElements() > 0,
                errors::InvalidArgument("received empty tensor tensor_out: ",
                                        tensor_out.DebugString()));
    OP_REQUIRES(
        context, out_grad_backprop.NumElements() > 0,
        errors::InvalidArgument("received empty tensor out_grad_backprop: ",
                                out_grad_backprop.DebugString()));
    OP_REQUIRES(context,
                tensor_in.NumElements() == out_grad_backprop.NumElements(),
                errors::InvalidArgument("tensor_in and out_grad_backprop must "
                                        "have same number of elements, got <",
                                        tensor_in.DebugString(), "> and <",
                                        out_grad_backprop.DebugString(), ">"));
    OP_REQUIRES(
        context, tensor_out.NumElements() == output->NumElements(),
        errors::InvalidArgument(
            "tensor_out and output must have same number of elements, got <",
            tensor_out.DebugString(), "> and <", output->DebugString(), ">"));

    LaunchMaxPooling3dGradGradOp<Device, T>::launch(
        context, params, tensor_in, tensor_out, out_grad_backprop, output);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

#define REGISTER_KERNELS(D, T)                                             \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("MaxPool3D").Device(DEVICE_##D).TypeConstraint<T>("T"),         \
      Pooling3DOp<D##Device, T, MAX>);                                     \
  REGISTER_KERNEL_BUILDER(Name("MaxPool3DGrad")                            \
                              .Device(DEVICE_##D)                          \
                              .TypeConstraint<T>("T")                      \
                              .TypeConstraint<T>("TInput"),                \
                          MaxPooling3dGradOp<D##Device, T>);               \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("MaxPool3DGradGrad").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      MaxPooling3dGradGradOp<D##Device, T>);                               \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("AvgPool3D").Device(DEVICE_##D).TypeConstraint<T>("T"),         \
      Pooling3DOp<D##Device, T, AVG>);                                     \
  REGISTER_KERNEL_BUILDER(Name("AvgPool3DGrad")                            \
                              .Device(DEVICE_##D)                          \
                              .TypeConstraint<T>("T")                      \
                              .HostMemory("orig_input_shape"),             \
                          AvgPooling3dGradOp<D##Device, T>);

#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T)
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename T>
struct LaunchPoolingOp<GPUDevice, T, AVG> {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Padding padding_type,
                     Tensor* output) {
    DnnPooling3dOp<T>::Compute(context, se::dnn::PoolingMode::kAverage, window,
                               stride, padding, data_format, tensor_in, output);
  }
};

template <typename T>
struct LaunchPoolingOp<GPUDevice, T, MAX> {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Padding padding_type,
                     Tensor* output) {
    DnnPooling3dOp<T>::Compute(context, se::dnn::PoolingMode::kMaximum, window,
                               stride, padding, data_format, tensor_in, output);
  }
};

template <typename T>
struct LaunchMaxPooling3dGradOp<GPUDevice, T> {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const Tensor& tensor_out, const Tensor& out_backprop,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& out,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Tensor* input_backprop) {
    const TensorShape output_shape = tensor_in.shape();
    DnnPooling3dGradOp<T>::Compute(context, se::dnn::PoolingMode::kMaximum,
                                   window, stride, padding, out, data_format,
                                   out_backprop, output_shape, &tensor_in,
                                   &tensor_out, input_backprop);
  }
};

template <typename T>
struct LaunchAvgPooling3dGradOp<GPUDevice, T> {
  static void launch(OpKernelContext* context,
                     const TensorShape& tensor_in_shape,
                     const Tensor& out_backprop,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& out,
                     const std::array<int64, 3>& padding,
                     TensorFormat data_format, Tensor* output) {
    DnnPooling3dGradOp<T>::Compute(
        context, se::dnn::PoolingMode::kAverage, window, stride, padding, out,
        data_format, out_backprop, tensor_in_shape, nullptr, nullptr, output);
  }
};

template <typename T>
struct LaunchMaxPooling3dGradGradOp<GPUDevice, T> {
  static void launch(OpKernelContext* context, const Pool3dParameters& params,
                     const Tensor& tensor_in, const Tensor& tensor_out,
                     const Tensor& tensor_top_diff,
                     Tensor* tensor_bottom_diff) {
    bool status = functor::MaxPool3dGradBackward<T>()(
        params.data_format, tensor_in.flat<T>().data(),
        tensor_out.flat<T>().data(), params.tensor_in_batch, params.out_plane,
        params.out_height, params.out_width, params.depth,
        params.tensor_in_planes, params.tensor_in_rows, params.tensor_in_cols,
        params.window_planes, params.window_rows, params.window_cols,
        params.plane_stride, params.row_stride, params.col_stride,
        params.pad_planes, params.pad_rows, params.pad_cols,
        tensor_top_diff.flat<T>().data(), tensor_bottom_diff->flat<T>().data(),
        context->eigen_gpu_device());
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching MaxPool3dGradBackward"));
    }
  }
};

#define REGISTER_GPU_KERNELS(T) REGISTER_KERNELS(GPU, T)
TF_CALL_float(REGISTER_GPU_KERNELS) TF_CALL_half(REGISTER_GPU_KERNELS)
    TF_CALL_bfloat16(REGISTER_GPU_KERNELS)
#undef REGISTER_GPU_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM


#undef REGISTER_KERNELS

}  // namespace tensorflow
