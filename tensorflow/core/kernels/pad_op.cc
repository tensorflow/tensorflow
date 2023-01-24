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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/pad_op.h"

#include <memory>
#include <string>
#include <utility>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Tpadding>
class PadOp : public OpKernel {
 public:
  explicit PadOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& in0 = context->input(0);
    const Tensor& in1 = context->input(1);
    const int dims = in0.dims();
    static const int kMinDims = 0;
    static const int kMaxDims = 8;
    OP_REQUIRES(context, kMinDims <= dims && dims <= kMaxDims,
                errors::Unimplemented("inputs rank not in [", kMinDims, ",",
                                      kMaxDims, "]: ", dims));
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsMatrix(in1.shape()) && in1.dim_size(1) == 2,
        errors::InvalidArgument("paddings must be a matrix with 2 columns: ",
                                in1.shape().DebugString()));
    OP_REQUIRES(
        context, dims == in1.dim_size(0),
        errors::InvalidArgument(
            "The first dimension of paddings must be the rank of inputs",
            in1.shape().DebugString(), " ", in0.shape().DebugString()));

    T pad_value = T();
    if (context->num_inputs() == 3) {
      const Tensor& constant_values = context->input(2);
      OP_REQUIRES(
          context, TensorShapeUtils::IsScalar(constant_values.shape()),
          errors::InvalidArgument("constant_values must be a scalar. Found: ",
                                  constant_values.shape().DebugString()));
      pad_value = context->input(2).scalar<T>()();
    }

    // Compute the shape of the output tensor, and allocate it.
    TensorShape output_shape;
    typename TTypes<Tpadding>::ConstMatrix paddings = in1.matrix<Tpadding>();
    for (int d = 0; d < dims; ++d) {
      const Tpadding before_d =
          paddings(d, 0);                       // Pad before existing elements.
      const Tpadding after_d = paddings(d, 1);  // Pad after existing elements.
      OP_REQUIRES(context, before_d >= 0 && after_d >= 0,
                  errors::InvalidArgument("Paddings must be non-negative: ",
                                          before_d, " ", after_d));
      const int64_t size_d = in0.dim_size(d);
      OP_REQUIRES_OK(
          context, output_shape.AddDimWithStatus(before_d + size_d + after_d));
    }

    // If there is no padding to be done, forward the input to output.
    if (output_shape.num_elements() == in0.NumElements()) {
      // When num_elements == 0, shape may have changed.
      Tensor out;
      CHECK(out.CopyFrom(in0, output_shape));
      context->set_output(0, out);
      return;
    }

    TensorShape collapsed_input_shape;
    TensorShape collapsed_output_shape;
    Tensor collapsed_paddings;
    if (dims > 1 && CollapseAdjacentNonPaddedDimensions(
                        in0.shape(), in1, output_shape, &collapsed_input_shape,
                        &collapsed_paddings, &collapsed_output_shape)) {
      Tensor collapsed_input;
      CHECK(collapsed_input.CopyFrom(in0, collapsed_input_shape));
      Tensor collapsed_output;
      AllocatorAttributes alloc_attrs;
      alloc_attrs.set_on_host(context->input_memory_type(0) == HOST_MEMORY);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(collapsed_input.dtype(),
                                            collapsed_output_shape,
                                            &collapsed_output, alloc_attrs));
      const Tensor& collapsed_paddings_ref = collapsed_paddings;
      typename TTypes<Tpadding>::ConstMatrix collapsed_paddings_matrix =
          collapsed_paddings_ref.matrix<Tpadding>();

      OperateWithVariableRank(context, collapsed_input_shape.dims(),
                              collapsed_input, collapsed_paddings_matrix,
                              pad_value, &collapsed_output);

      Tensor output;
      CHECK(output.CopyFrom(collapsed_output, output_shape));
      context->set_output(0, output);
    } else {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, output_shape, &output));
      OperateWithVariableRank(context, dims, in0, paddings, pad_value, output);
    }
  }

 private:
  // Collapses adjacent dimensions that are not padded to one dimension for
  // speed. Returns true if any two dimensions are collapsed. For example,
  //
  //   Pad(input_shape=[8, 28, 28, 3],
  //       paddings=[[0, 0], [0, 0], [0, 0], [0, 1]]
  // is equivalent to
  //   Pad(input_shape=[6272, 3],
  //       paddings=[[0, 0], [0, 1]])
  //
  // input_shape: the original input shape.
  // paddings_as_tensor: the original paddings.
  // output_shape: the original output shape.
  // collapsed_input_shape: the input shape after collapsing.
  // collapsed_paddings_as_tensor: the paddings after collapsing.
  // collapsed_output_shape: the output shape after collapsing.
  static bool CollapseAdjacentNonPaddedDimensions(
      const TensorShape& input_shape, const Tensor& paddings_as_tensor,
      const TensorShape& output_shape, TensorShape* collapsed_input_shape,
      Tensor* collapsed_paddings_as_tensor,
      TensorShape* collapsed_output_shape) {
    bool collapsed = false;
    typename TTypes<Tpadding>::ConstMatrix paddings =
        paddings_as_tensor.matrix<Tpadding>();
    std::vector<std::pair<int, int>> collapsed_paddings;
    int i = 0;
    while (i < paddings.dimension(0)) {
      if (paddings(i, 0) != 0 || paddings(i, 1) != 0) {
        // If padded, copy the original dimension over.
        collapsed_input_shape->InsertDim(collapsed_input_shape->dims(),
                                         input_shape.dim_size(i));
        collapsed_output_shape->InsertDim(collapsed_output_shape->dims(),
                                          output_shape.dim_size(i));
        collapsed_paddings.push_back({paddings(i, 0), paddings(i, 1)});
        ++i;
      } else {
        // If not padded, find the next dimension that is padded and collapse
        // all dimensions in between to one dimension.
        int64_t collapsed_input_dim_size = input_shape.dim_size(i);
        int64_t collapsed_output_dim_size = output_shape.dim_size(i);
        ++i;
        while (i < paddings.dimension(0) && paddings(i, 0) == 0 &&
               paddings(i, 1) == 0) {
          collapsed = true;
          collapsed_input_dim_size *= input_shape.dim_size(i);
          collapsed_output_dim_size *= output_shape.dim_size(i);
          ++i;
        }
        collapsed_input_shape->InsertDim(collapsed_input_shape->dims(),
                                         collapsed_input_dim_size);
        collapsed_output_shape->InsertDim(collapsed_output_shape->dims(),
                                          collapsed_output_dim_size);
        collapsed_paddings.push_back({0, 0});
      }
    }

    // Copy collapsed_paddings to collapsed_paddings_as_tensor.
    *collapsed_paddings_as_tensor = Tensor(
        paddings_as_tensor.dtype(),
        TensorShape({static_cast<int64_t>(collapsed_paddings.size()), 2}));
    auto collapsed_paddings_as_matrix =
        collapsed_paddings_as_tensor->matrix<Tpadding>();
    for (size_t i = 0; i < collapsed_paddings.size(); ++i) {
      collapsed_paddings_as_matrix(i, 0) = collapsed_paddings[i].first;
      collapsed_paddings_as_matrix(i, 1) = collapsed_paddings[i].second;
    }
    return collapsed;
  }

  void OperateWithVariableRank(OpKernelContext* context, int fixed_dims,
                               const Tensor& input,
                               typename TTypes<Tpadding>::ConstMatrix paddings,
                               T pad_value, Tensor* output) {
    // Invoke the dims-specific implementation.
    switch (fixed_dims) {
      case 0:
        Operate<0>(context, input.tensor<T, 0>(), paddings, pad_value, output);
        break;
      case 1:
        // TODO(irving): Once Pad doesn't need a scalar special case,
        // change flat to tensor.  That is, once !allow_legacy_scalars().
        Operate<1>(context, input.flat<T>(), paddings, pad_value, output);
        break;
      case 2:
        Operate<2>(context, input.tensor<T, 2>(), paddings, pad_value, output);
        break;
      case 3:
        Operate<3>(context, input.tensor<T, 3>(), paddings, pad_value, output);
        break;
      case 4:
        Operate<4>(context, input.tensor<T, 4>(), paddings, pad_value, output);
        break;
      case 5:
        Operate<5>(context, input.tensor<T, 5>(), paddings, pad_value, output);
        break;
      case 6:
        Operate<6>(context, input.tensor<T, 6>(), paddings, pad_value, output);
        break;
      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument("Only ranks up to 6 supported: ",
                                            input.shape().DebugString()));
    }
  }

  template <int Dims>
  void Operate(OpKernelContext* context,
               typename TTypes<T, Dims>::ConstTensor input,
               typename TTypes<Tpadding>::ConstMatrix paddings, T pad_value,
               Tensor* output) {
    CHECK_EQ(Dims, paddings.dimension(0));
    CHECK_EQ(2, paddings.dimension(1));
    Eigen::array<Eigen::IndexPair<Tpadding>, Dims> paddings_array;
    for (int i = 0; i < Dims; ++i) {
      paddings_array[i] = {paddings(i, 0), paddings(i, 1)};
    }
    functor::Pad<Device, T, Tpadding, Dims> functor;
    functor(context->eigen_device<Device>(), output->tensor<T, Dims>(), input,
            paddings_array, pad_value);
  }
};

#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(Name("Pad")                               \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int32>("Tpaddings")   \
                              .HostMemory("paddings"),              \
                          PadOp<CPUDevice, type, int32>);           \
  REGISTER_KERNEL_BUILDER(Name("Pad")                               \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int64_t>("Tpaddings") \
                              .HostMemory("paddings"),              \
                          PadOp<CPUDevice, type, int64>);           \
  REGISTER_KERNEL_BUILDER(Name("PadV2")                             \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int32>("Tpaddings")   \
                              .HostMemory("paddings")               \
                              .HostMemory("constant_values"),       \
                          PadOp<CPUDevice, type, int32>);           \
  REGISTER_KERNEL_BUILDER(Name("PadV2")                             \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int64_t>("Tpaddings") \
                              .HostMemory("paddings")               \
                              .HostMemory("constant_values"),       \
                          PadOp<CPUDevice, type, int64>);

TF_CALL_POD_TYPES(REGISTER_KERNEL);
TF_CALL_QUANTIZED_TYPES(REGISTER_KERNEL);
TF_CALL_tstring(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, Dims)                                           \
  template <>                                                               \
  void Pad<GPUDevice, T, int32, Dims>::operator()(                          \
      const GPUDevice& d, typename TTypes<T, Dims>::Tensor output,          \
      typename TTypes<T, Dims>::ConstTensor input,                          \
      Eigen::array<Eigen::IndexPair<int32>, Dims> paddings, T pad_value);   \
  extern template struct Pad<GPUDevice, T, int32, Dims>;                    \
  template <>                                                               \
  void Pad<GPUDevice, T, int64_t, Dims>::operator()(                        \
      const GPUDevice& d, typename TTypes<T, Dims>::Tensor output,          \
      typename TTypes<T, Dims>::ConstTensor input,                          \
      Eigen::array<Eigen::IndexPair<int64_t>, Dims> paddings, T pad_value); \
  extern template struct Pad<GPUDevice, T, int64_t, Dims>;

#define DECLARE_GPU_SPECS(T) \
  DECLARE_GPU_SPEC(T, 0);    \
  DECLARE_GPU_SPEC(T, 1);    \
  DECLARE_GPU_SPEC(T, 2);    \
  DECLARE_GPU_SPEC(T, 3);    \
  DECLARE_GPU_SPEC(T, 4);    \
  DECLARE_GPU_SPEC(T, 5);    \
  DECLARE_GPU_SPEC(T, 6);

TF_CALL_GPU_ALL_TYPES(DECLARE_GPU_SPECS);
TF_CALL_int8(DECLARE_GPU_SPECS);
TF_CALL_uint8(DECLARE_GPU_SPECS);
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(T)                                      \
  REGISTER_KERNEL_BUILDER(Name("Pad")                               \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<int32>("Tpaddings")   \
                              .HostMemory("paddings"),              \
                          PadOp<GPUDevice, T, int32>);              \
  REGISTER_KERNEL_BUILDER(Name("Pad")                               \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<int64_t>("Tpaddings") \
                              .HostMemory("paddings"),              \
                          PadOp<GPUDevice, T, int64>);              \
  REGISTER_KERNEL_BUILDER(Name("PadV2")                             \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<int32>("Tpaddings")   \
                              .HostMemory("paddings")               \
                              .HostMemory("constant_values"),       \
                          PadOp<GPUDevice, T, int32>)               \
  REGISTER_KERNEL_BUILDER(Name("PadV2")                             \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<int64_t>("Tpaddings") \
                              .HostMemory("paddings")               \
                              .HostMemory("constant_values"),       \
                          PadOp<GPUDevice, T, int64>)

TF_CALL_GPU_ALL_TYPES(REGISTER_GPU_KERNEL);
TF_CALL_int8(REGISTER_GPU_KERNEL);
TF_CALL_uint8(REGISTER_GPU_KERNEL);

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Pad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tpaddings")
                            .HostMemory("input")
                            .HostMemory("paddings")
                            .HostMemory("output"),
                        PadOp<CPUDevice, int32, int32>);
REGISTER_KERNEL_BUILDER(Name("Pad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64_t>("Tpaddings")
                            .HostMemory("input")
                            .HostMemory("paddings")
                            .HostMemory("output"),
                        PadOp<CPUDevice, int32, int64>);
REGISTER_KERNEL_BUILDER(Name("PadV2")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tpaddings")
                            .HostMemory("input")
                            .HostMemory("paddings")
                            .HostMemory("constant_values")
                            .HostMemory("output"),
                        PadOp<CPUDevice, int32, int32>);
REGISTER_KERNEL_BUILDER(Name("PadV2")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64_t>("Tpaddings")
                            .HostMemory("input")
                            .HostMemory("paddings")
                            .HostMemory("constant_values")
                            .HostMemory("output"),
                        PadOp<CPUDevice, int32, int64>);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // end namespace tensorflow
