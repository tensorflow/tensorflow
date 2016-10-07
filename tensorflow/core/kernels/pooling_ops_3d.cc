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

#include <array>

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/eigen_pooling.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/padding.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/cudnn_pooling_gpu.h"
#endif
namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

enum PoolingType { MAX, AVG };

template <typename Device, typename T, PoolingType Type>
struct LaunchPoolingOp;

template <typename T>
struct LaunchPoolingOp<CPUDevice, T, AVG> {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& padding, Padding padding_type,
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
                     const std::array<int64, 3>& padding, Padding padding_type,
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
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 5,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    OP_REQUIRES(context, ksize_[4] == 1 && stride_[4] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the depth dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);

    OP_REQUIRES(context, tensor_in.dims() == 5,
                errors::InvalidArgument("tensor_in must be 5-dimensional"));
    const int64 depth = tensor_in.dim_size(4);
    const int64 in_batch = tensor_in.dim_size(0);

    // Dimension order for these arrays is: x, y, z.
    std::array<int64, 3> input_size{
        {tensor_in.dim_size(3), tensor_in.dim_size(2), tensor_in.dim_size(1)}};
    std::array<int64, 3> window{{ksize_[3], ksize_[2], ksize_[1]}};
    std::array<int64, 3> stride{{stride_[3], stride_[2], stride_[1]}};
    std::array<int64, 3> padding, out;

    OP_REQUIRES_OK(context, Get3dOutputSize(input_size, window, stride,
                                            padding_, &out, &padding));

    TensorShape out_shape({in_batch, out[2], out[1], out[0], depth});
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    LaunchPoolingOp<Device, T, Type>::launch(context, tensor_in, window, stride,
                                             padding, padding_, output);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
};
REGISTER_KERNEL_BUILDER(
    Name("AvgPool3D").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    Pooling3DOp<CPUDevice, float, AVG>);
REGISTER_KERNEL_BUILDER(
    Name("MaxPool3D").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    Pooling3DOp<CPUDevice, float, MAX>);

template <typename Device, typename T>
struct LaunchMaxPooling3dGradOp;

template <typename T>
struct LaunchMaxPooling3dGradOp<CPUDevice, T> {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const Tensor& tensor_out, const Tensor& out_backprop,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& out,
                     const std::array<int64, 3>& padding, Tensor* output) {
    output->flat<T>().setZero();
    for (int64 p = 0; p < out_backprop.dim_size(3); ++p) {
      // Calculate broadcast size for planes/rows/cols. For SAME padding,
      // current index could be in the padding area, and
      //   p * stride_planes + window_planes
      // could be beyond the input tensor's boundary. In such cases, change
      // the starting index and reduce the broadcast size.
      //
      // The same procedure is repeated for every spatial dimension in the
      // nested loops below.
      int pindex, psize;
      std::array<int64, 3> input_size{{tensor_in.dim_size(3),
                                       tensor_in.dim_size(2),
                                       tensor_in.dim_size(1)}};
      OP_REQUIRES_OK(context,
                     GetBroadcastSize(p, input_size[0], window[0], stride[0],
                                      padding[0], &pindex, &psize));
      for (int64 r = 0; r < out_backprop.dim_size(2); ++r) {
        int rindex, rsize;
        OP_REQUIRES_OK(context,
                       GetBroadcastSize(r, input_size[1], window[1], stride[1],
                                        padding[1], &rindex, &rsize));
        for (int64 c = 0; c < out_backprop.dim_size(1); ++c) {
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

#if !defined(EIGEN_HAS_INDEX_LIST)
          Eigen::array<int, 5> bcast = {1, csize, rsize, psize, 1};
#else
          Eigen::IndexList<Eigen::type2index<1>, int, int, int,
                           Eigen::type2index<1> >
              bcast;
          bcast.set(1, csize);
          bcast.set(2, rsize);
          bcast.set(3, psize);
#endif

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
               tensor_in_slice.constant(1e-5))
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
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 5,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    OP_REQUIRES(context, ksize_[4] == 1 && stride_[4] == 1,
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

    std::array<int64, 3> input_size = {{output_shape.dim_size(3),
                                        output_shape.dim_size(2),
                                        output_shape.dim_size(1)}};
    std::array<int64, 3> window = {{ksize_[3], ksize_[2], ksize_[1]}};
    std::array<int64, 3> stride = {{stride_[3], stride_[2], stride_[1]}};
    std::array<int64, 3> out, padding;

    OP_REQUIRES_OK(context, Get3dOutputSize(input_size, window, stride,
                                            padding_, &out, &padding));
    LaunchMaxPooling3dGradOp<Device, T>::launch(context, tensor_in, tensor_out,
                                                out_backprop, window, stride,
                                                out, padding, input_backprop);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
};

REGISTER_KERNEL_BUILDER(
    Name("MaxPool3DGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    MaxPooling3dGradOp<CPUDevice, float>);

template <typename Device, typename T>
struct LaunchAvgPooling3dGradOp;

template <typename T>
struct LaunchAvgPooling3dGradOp<CPUDevice, T> {
  static void launch(OpKernelContext* context,
                     const TensorShape& tensor_in_shape,
                     const Tensor& out_backprop,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& output_shape,
                     const std::array<int64, 3>& padding, Tensor* output) {
    output->flat<T>().setZero();
    std::array<int64, 3> input_size = {{tensor_in_shape.dim_size(3),
                                        tensor_in_shape.dim_size(2),
                                        tensor_in_shape.dim_size(1)}};
    for (int64 p = 0; p < out_backprop.dim_size(3); ++p) {
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
      for (int64 r = 0; r < out_backprop.dim_size(2); ++r) {
        int rindex, rsize;
        OP_REQUIRES_OK(context,
                       GetBroadcastSize(r, input_size[1], window[1], stride[1],
                                        padding[1], &rindex, &rsize));
        for (int64 c = 0; c < out_backprop.dim_size(1); ++c) {
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
#if !defined(EIGEN_HAS_INDEX_LIST)
          Eigen::array<int, 5> bcast = {1, csize, rsize, psize, 1};
#else
          Eigen::IndexList<Eigen::type2index<1>, int, int, int,
                           Eigen::type2index<1> >
              bcast;
          bcast.set(1, csize);
          bcast.set(2, rsize);
          bcast.set(3, psize);
#endif
          Eigen::Tensor<T, 5, Eigen::RowMajor> slices(src_sizes);
          slices.device(context->eigen_cpu_device()) =
              out_backprop.tensor<T, 5>().slice(src_indices, src_sizes);
          // Divide by the size of the actual patch (psize * rsize * csize).
          float divide_size = rsize * csize * psize * 1.0f;
          slices *= slices.constant(1.0f / divide_size);

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
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 5,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    OP_REQUIRES(context, ksize_[4] == 1 && stride_[4] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the depth dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in_shape = context->input(0);
    const Tensor& out_backprop = context->input(1);
    OP_REQUIRES(context, tensor_in_shape.dims() == 1 &&
                             tensor_in_shape.NumElements() == 5,
                errors::InvalidArgument("tensor_in must be 1-dimensional and 5 "
                                        "elements"));
    OP_REQUIRES(context, out_backprop.dims() == 5,
                errors::InvalidArgument("out_backprop must be 5-dimensional"));

    TensorShape output_shape;
    auto shape_vec = tensor_in_shape.vec<int32>();
    for (int64 i = 0; i < tensor_in_shape.NumElements(); ++i) {
      output_shape.AddDim(shape_vec(i));
    }

    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    // Dimension order for these arrays is x, y, z.
    std::array<int64, 3> input_size = {{output_shape.dim_size(3),
                                        output_shape.dim_size(2),
                                        output_shape.dim_size(1)}};
    std::array<int64, 3> window = {{ksize_[3], ksize_[2], ksize_[1]}};
    std::array<int64, 3> stride = {{stride_[3], stride_[2], stride_[1]}};
    std::array<int64, 3> padding, out;

    OP_REQUIRES_OK(context, Get3dOutputSize(input_size, window, stride,
                                            padding_, &out, &padding));

    LaunchAvgPooling3dGradOp<Device, T>::launch(context, output_shape,
                                                out_backprop, window, stride,
                                                out, padding, output);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
};

REGISTER_KERNEL_BUILDER(Name("AvgPool3DGrad")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T")
                            .HostMemory("orig_input_shape"),
                        AvgPooling3dGradOp<CPUDevice, float>);

#if GOOGLE_CUDA

template <typename T>
struct LaunchPoolingOp<GPUDevice, T, AVG> {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& padding, Padding padding_type,
                     Tensor* output) {
    DnnPooling3dOp<T>::Compute(context,
                               perftools::gputools::dnn::PoolingMode::kAverage,
                               window, stride, padding, tensor_in, output);
  }
};

template <typename T>
struct LaunchPoolingOp<GPUDevice, T, MAX> {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& padding, Padding padding_type,
                     Tensor* output) {
    DnnPooling3dOp<T>::Compute(context,
                               perftools::gputools::dnn::PoolingMode::kMaximum,
                               window, stride, padding, tensor_in, output);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("AvgPool3D").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    Pooling3DOp<GPUDevice, float, AVG>);
REGISTER_KERNEL_BUILDER(
    Name("MaxPool3D").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    Pooling3DOp<GPUDevice, float, MAX>);

template <typename T>
struct LaunchMaxPooling3dGradOp<GPUDevice, T> {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const Tensor& tensor_out, const Tensor& out_backprop,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& out,
                     const std::array<int64, 3>& padding,
                     Tensor* input_backprop) {
    const TensorShape output_shape = tensor_in.shape();
    DnnPooling3dGradOp<T>::Compute(
        context, perftools::gputools::dnn::PoolingMode::kMaximum, window,
        stride, padding, out, out_backprop, output_shape, &tensor_in,
        &tensor_out, input_backprop);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("MaxPool3DGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    MaxPooling3dGradOp<GPUDevice, float>);

template <typename T>
struct LaunchAvgPooling3dGradOp<GPUDevice, T> {
  static void launch(OpKernelContext* context,
                     const TensorShape& tensor_in_shape,
                     const Tensor& out_backprop,
                     const std::array<int64, 3>& window,
                     const std::array<int64, 3>& stride,
                     const std::array<int64, 3>& out,
                     const std::array<int64, 3>& padding, Tensor* output) {
    DnnPooling3dGradOp<T>::Compute(
        context, perftools::gputools::dnn::PoolingMode::kAverage, window,
        stride, padding, out, out_backprop, tensor_in_shape, nullptr, nullptr,
        output);
  }
};
REGISTER_KERNEL_BUILDER(Name("AvgPool3DGrad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .HostMemory("orig_input_shape"),
                        AvgPooling3dGradOp<GPUDevice, float>);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
