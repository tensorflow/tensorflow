/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/broadcast_to_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class BroadcastToOp : public OpKernel {
 public:
  explicit BroadcastToOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_tensor = ctx->input(0);
    const TensorShape& input_shape = input_tensor.shape();

    const Tensor& shape_tensor = ctx->input(1);

    TensorShape output_shape;
    OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_tensor, &output_shape));

    // Handle copy.
    if (output_shape == input_shape) {
      ctx->set_output(0, input_tensor);
      return;
    }

    OP_REQUIRES(ctx, input_shape.dims() <= output_shape.dims(),
                errors::InvalidArgument(
                    "Rank of input (", input_shape.dims(),
                    ") must be no greater than rank of output shape (",
                    output_shape.dims(), ")."));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output_tensor));
    // Handle empty case.
    if (output_shape.num_elements() == 0) {
      return;
    }

    // Handle broadcast from Scalar.
    const Device& device = ctx->eigen_device<Device>();
    if (input_shape.dims() == 0) {
      functor::FillFunctor<Device, T>()(device, output_tensor->flat<T>(),
                                        input_tensor.scalar<T>());
      return;
    }

    BCast bcast(BCast::FromShape(input_shape), BCast::FromShape(output_shape),
                /*fewer_dims_optimization=*/true);
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: ", input_shape.DebugString(), " vs. ",
                    output_shape.DebugString()));
    OP_REQUIRES(ctx, BCast::ToShape(bcast.output_shape()) == output_shape,
                errors::InvalidArgument("Unable to broadcast tensor of shape ",
                                        input_shape, " to tensor of shape ",
                                        output_shape));

    functor::BroadcastTo<Device, T>()(device, ctx, *output_tensor, output_shape,
                                      input_tensor, input_shape, bcast);
  }
};

// As tensor::MakeShape is able to handle both DT_INT32 and DT_INT64,
// no need to have TypeConstraint for `Tidx`
#define REGISTER_KERNEL(type)                                           \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("BroadcastTo").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BroadcastToOp<CPUDevice, type>);

TF_CALL_ALL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

namespace functor {
#define DECLARE_GPU_TEMPLATE(Type)                               \
  template <>                                                    \
  void BroadcastTo<GPUDevice, Type>::operator()(                 \
      const GPUDevice& d, OpKernelContext* ctx, Tensor& output,  \
      const TensorShape& output_shape, const Tensor& input,      \
      const TensorShape& input_shape, const BCast& bcast) const; \
  extern template struct BroadcastTo<GPUDevice, Type>;

TF_CALL_GPU_ALL_TYPES(DECLARE_GPU_TEMPLATE);
TF_CALL_int64(DECLARE_GPU_TEMPLATE);
#undef DECLARE_GPU_KERNEL
}  // namespace functor

#define REGISTER_KERNEL(type)                            \
  REGISTER_KERNEL_BUILDER(Name("BroadcastTo")            \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("shape"),      \
                          BroadcastToOp<GPUDevice, type>);

TF_CALL_GPU_ALL_TYPES(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
#undef REGISTER_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("BroadcastTo")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("shape")
                            .HostMemory("output"),
                        BroadcastToOp<CPUDevice, int32>);
#endif

}  // namespace tensorflow
