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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/contrib/repeat/kernels/repeat_op.h"

// used in cpu implementation sharded mode
const int kCostPerUnit = 10000;

namespace tensorflow{

typedef Eigen::ThreadPoolDevice CPUDevice;
#if GOOGLE_CUDA
typedef Eigen::GpuDevice GPUDevice;
#endif // GOOGLE_CUDA

template <typename Device, typename T>
class RepeatOp : public OpKernel {
 public:
  explicit RepeatOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis_));
  }
  
  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& repeats = context->input(1);
    const int input_rank = input.dims()==0 ? 1 : input.dims();
    const int32 axis = axis_>=0 ? axis_ : axis_+input_rank;
    
    OP_REQUIRES(context, TensorShapeUtils::IsVector(repeats.shape()) ||
                         TensorShapeUtils::IsScalar(repeats.shape()),
                errors::InvalidArgument("`repeats` expects a scalar or a 1-D vector."));
    OP_REQUIRES(context, FastBoundsCheck(axis, input_rank),
                errors::InvalidArgument(
                    "Expected -", input_rank, " <= `axis` < ", input_rank));
    OP_REQUIRES(context, repeats.NumElements() == input.dim_size(axis) ||
                         repeats.NumElements() == 1,
                errors::InvalidArgument(
                    "Expected `repeats` argument to be a vector of length ",
                    input.dim_size(axis_), " or 1, but got length ",
                    repeats.NumElements()));
    
    auto repeats_flat = repeats.flat<int32>();
    TensorShape output_shape({1});
    int old_dim;
    if (input.dims() != 0) {
      output_shape = input.shape();
      old_dim = input.shape().dim_size(axis);
    } else {
      old_dim = 1;
    }
    int new_dim = 0;
    if (repeats.NumElements() == 1) {
      new_dim = repeats_flat(0) * old_dim;
    } else {
      const int N = repeats_flat.size();
      for (int i = 0; i < N; ++i) {
        new_dim += repeats_flat(i);
      }
    }
    output_shape.set_dim(axis, new_dim);
    
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    

#if GOOGLE_CUDA
    if (std::is_same<Device, GPUDevice>::value) {
      RepeatGPUImpl<T>(context->eigen_gpu_device(), input, repeats_flat, axis, output);
      return ;
    }
#endif // GOOGLE_CUDA

    RepeatCPUImplMultiThreaded<T>(context->device(), input, repeats_flat,
                       axis, kCostPerUnit, output); 
  }
  
 private:
  int32 axis_;
  
};


#define REGISTER_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(                       \
      Name("Repeat")                             \
      .Device(DEVICE_CPU)                        \
      .TypeConstraint<type>("T"),                \
      RepeatOp<CPUDevice, type>)

TF_CALL_ALL_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#if GOOGLE_CUDA

#define REGISTER_KERNEL_GPU(type)                \
  REGISTER_KERNEL_BUILDER(                       \
      Name("Repeat")                             \
      .Device(DEVICE_GPU)                        \
      .TypeConstraint<type>("T")                 \
      .HostMemory("repeats"),                    \
      RepeatOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL_GPU);

#undef REGISTER_KERNEL_GPU

#endif // GOOGLE_CUDA

} //end namespace tensorflow
