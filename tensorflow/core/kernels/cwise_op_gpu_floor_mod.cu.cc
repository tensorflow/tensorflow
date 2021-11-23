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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/cwise_ops_gpu_common.cu.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/work_sharder.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
// TODO(b/32239807) No GPU ops for mod yet.
}  // namespace functor

template <typename T>
__global__ void floor_mod_kernel(const T* input_ptr, const int64 total,
                                 const int32 divisor, T* output_ptr_data) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < total; i += blockDim.x * gridDim.x) {
    output_ptr_data[i] = (uint64(input_ptr[i]) % divisor);
  }
}

template <typename T>
struct FloorModOpGPULaunch {
  void Run(const Eigen::GpuDevice& d, const T* input, int total, int32 divisor,
           T* output_ptr_data);
};

template <typename T>
void FloorModOpGPULaunch<T>::Run(const Eigen::GpuDevice& gpu_device,
                                 const T* input_ptr, int total, int32 divisor,
                                 T* output_ptr_data) {
  auto config = GetGpuLaunchConfig(total, gpu_device);
  // performance crossover is less than using maximum available shared
  // memory on most processors possibly due to decreasing occupancy
  // 4096 inputs is a lot, most code will take the smem path
  TF_CHECK_OK(GpuLaunchKernel(floor_mod_kernel<T>, config.block_count,
                              config.thread_per_block, 0, gpu_device.stream(),
                              input_ptr, total, divisor, output_ptr_data));
}

template <typename Device, typename T>
class FloorModOpBase : public OpKernel {
 public:
  explicit FloorModOpBase(OpKernelConstruction* c) : OpKernel(c) {
    // OP_REQUIRES_OK(c, c->GetAttr("divisor", &divisor_));
  }
  T divisor_ = 0;
};

template <typename T>
class FloorModOpGPU : public FloorModOpBase<GPUDevice, T> {
 public:
  typedef FloorModOpBase<GPUDevice, T> Base;
  explicit FloorModOpGPU(OpKernelConstruction* c) : Base(c) {}
  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& divisor = context->input(1);
    Base::divisor_ = (divisor.flat<T>()(0));
    auto flat_input_size = input.flat<T>().dimension(0);
    const TensorShape& input_shape = input.shape();
    TensorShape output_shape(input_shape);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    T* flat_output = &(output->flat<T>()(0));

    auto* flat_input = &(input.flat<T>()(0));
    FloorModOpGPULaunch<T>().Run(context->eigen_device<GPUDevice>(),
                                 input.flat<T>().data(), flat_input_size,
                                 Base::divisor_, flat_output);
    OP_REQUIRES(context, context->op_device_context()->stream()->ok(),
                errors::Internal("Launch of gpu kernel for SplitVOp failed"));
  }
};

#define REGISTER_GPU(type)                                \
  REGISTER_KERNEL_BUILDER(Name("FloorMod")                \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("y")            \
                              .Priority(1)                \
                              .TypeConstraint<type>("T"), \
                          FloorModOpGPU<type>)

REGISTER_GPU(int32);
REGISTER_GPU(int64);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
