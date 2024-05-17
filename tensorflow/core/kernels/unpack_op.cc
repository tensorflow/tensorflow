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

// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/split_lib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class UnpackOp : public OpKernel {
 public:
  explicit UnpackOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis_));
  }

  void Compute(OpKernelContext* context) override {
    const int32_t num = num_outputs();
    const Tensor& input = context->input(0);
    const TensorShape& input_shape = input.shape();

    int axis = axis_;
    if (axis < 0) axis += input_shape.dims();

    OP_REQUIRES(context, 0 <= axis && axis < input_shape.dims(),
                errors::InvalidArgument("axis = ", axis_, " not in [",
                                        -input_shape.dims(), ", ",
                                        input_shape.dims(), ")"));

    OP_REQUIRES(
        context, input_shape.dims() > 0 && input_shape.dim_size(axis) == num,
        errors::InvalidArgument("Input shape axis ", axis, " must equal ", num,
                                ", got shape ", input_shape.DebugString()));

    auto output_shape = input_shape;
    output_shape.RemoveDim(axis);
    const int64_t output_size = output_shape.num_elements();
    OP_REQUIRES(
        context,
        FastBoundsCheck(output_size,
                        std::numeric_limits<Eigen::DenseIndex>::max()),
        errors::InvalidArgument("output size must fit in Eigen DenseIndex"));

    // Special case: Aligned, so we can share the underlying buffer.
    //
    // Apply this optimization conservatively: if input is aligned,
    // the resulting tensors must be aligned. It's conservative
    // because if the immediate consumer of the resulting tensors are
    // not using eigen for computation, its perfectly fine to avoid
    // the copying.
    if (axis == 0 &&
        (output_size == 0 || IsInnerDimsSizeAligned<T>(input_shape))) {
      for (int i = 0; i < num; ++i) {
        Tensor output;
        CHECK(output.CopyFrom(input.Slice(i, i + 1), output_shape));
        context->set_output(i, output);
      }
      return;
    }

    Eigen::DenseIndex before_dim = 1;
    for (int i = 0; i < axis; ++i) {
      before_dim *= input_shape.dim_size(i);
    }

    Eigen::DenseIndex after_dim = 1;
    for (int i = axis + 1; i < input_shape.dims(); ++i) {
      after_dim *= input_shape.dim_size(i);
    }
    const Eigen::DenseIndex axis_dim = input_shape.dim_size(axis);

    // Except for shape, unpack is a special case of split, so we reuse the
    // same computational kernels.
    auto input_reshaped =
        input.shaped<T, 2>({before_dim, axis_dim * after_dim});

    for (int i = 0; i < num; ++i) {
      if (!context->output_required(i)) continue;

      Tensor* output;
      OP_REQUIRES_OK(context,
                     context->allocate_output(i, output_shape, &output));

      if (output_shape.num_elements() > 0) {
        auto output_shaped = output->shaped<T, 2>({before_dim, after_dim});
        Eigen::DSizes<Eigen::DenseIndex, 2> indices{0, i * after_dim};
        Eigen::DSizes<Eigen::DenseIndex, 2> sizes{before_dim, after_dim};
        functor::Split<Device, T, 2>()(context->eigen_device<Device>(),
                                       output_shaped, input_reshaped, indices,
                                       sizes);
      }
    }
  }

 private:
  int axis_;
};

#define REGISTER_UNPACK(type)                                      \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Unpack").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      UnpackOp<CPUDevice, type>)

TF_CALL_ALL_TYPES(REGISTER_UNPACK);
TF_CALL_float8_e5m2(REGISTER_UNPACK);
TF_CALL_float8_e4m3fn(REGISTER_UNPACK);

#undef REGISTER_UNPACK

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU(type)                                         \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Unpack").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      UnpackOp<GPUDevice, type>)

TF_CALL_uint8(REGISTER_GPU);
TF_CALL_GPU_ALL_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// A special DEVICE_DEFAULT kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Unpack")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("value")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        UnpackOp<CPUDevice, int32>);
REGISTER_KERNEL_BUILDER(Name("Unpack")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("value")
                            .HostMemory("output")
                            .TypeConstraint<int64_t>("T"),
                        UnpackOp<CPUDevice, int64>);

}  // end namespace tensorflow
