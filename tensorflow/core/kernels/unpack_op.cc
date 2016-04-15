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

// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
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
  explicit UnpackOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* context) override {
    const int32 num = num_outputs();
    const Tensor& input = context->input(0);
    const TensorShape& input_shape = input.shape();

    OP_REQUIRES(context,
                input_shape.dims() > 0 && input_shape.dim_size(0) == num,
                errors::InvalidArgument("Input shape must start with ", num,
                                        ", got ", input_shape.DebugString()));

    auto output_shape = input_shape;
    output_shape.RemoveDim(0);
    const int64 output_size = output_shape.num_elements();
    OP_REQUIRES(
        context, FastBoundsCheck(output_size,
                                 std::numeric_limits<Eigen::DenseIndex>::max()),
        errors::InvalidArgument("output size must fit in Eigen DenseIndex"));

    // Special case: Aligned, so we can share the underlying buffer.
    //
    // Apply this optimization conservatively: if input is aligned,
    // the resulting tensors must be aligned. It's conservative
    // because if the immediate consumer of the resulting tensors are
    // not using eigen for computation, its perfectly fine to avoid
    // the copying.
    if (output_size == 0 || IsInnerDimsSizeAligned<T>(input_shape)) {
      for (int i = 0; i < num; ++i) {
        Tensor output;
        CHECK(output.CopyFrom(input.Slice(i, i + 1), output_shape));
        context->set_output(i, output);
      }
      return;
    }

    // Except for shape, unpack is a special case of split, so we reuse the
    // same computational kernels.
    auto input_reshaped = input.shaped<T, 3>({1, num, output_size});

    for (int i = 0; i < num; ++i) {
      Tensor* output;
      OP_REQUIRES_OK(context,
                     context->allocate_output(i, output_shape, &output));
      auto output_shaped = output->shaped<T, 3>({1, 1, output_size});

      Eigen::DSizes<Eigen::DenseIndex, 3> indices{0, i, 0};
      Eigen::DSizes<Eigen::DenseIndex, 3> sizes{1, 1, output_size};
      functor::Split<Device, T>()(context->eigen_device<Device>(),
                                  output_shaped, input_reshaped, indices,
                                  sizes);
    }
  }
};

#define REGISTER_UNPACK(type)                                      \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Unpack").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      UnpackOp<CPUDevice, type>)

TF_CALL_ALL_TYPES(REGISTER_UNPACK);

#undef REGISTER_UNPACK

#if GOOGLE_CUDA

#define REGISTER_GPU(type)                                         \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Unpack").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      UnpackOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Unpack")
                            .Device(DEVICE_GPU)
                            .HostMemory("value")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        UnpackOp<CPUDevice, int32>);

#endif  // GOOGLE_CUDA

}  // end namespace tensorflow
