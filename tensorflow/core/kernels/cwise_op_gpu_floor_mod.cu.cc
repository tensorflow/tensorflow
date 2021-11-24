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

enum KernelStatus { Success = 0, ZeroDivisionError = 1 };

template <typename T>
__global__ void floor_mod_kernel(const T* input_ptr, const int64 total,
                                 const T* divisor, T* output_ptr_data,
                                 KernelStatus* status) {
  if (0 == *divisor) {
    *status = KernelStatus::ZeroDivisionError;
    return;
  }

  GPU_1D_KERNEL_LOOP(i, total) {
    int64_t d = input_ptr[i] / *divisor;
    int64_t floordiv = d * *divisor == input_ptr[i]
                           ? d
                           : d - ((input_ptr[i] < 0) ^ (*divisor < 0));
    output_ptr_data[i] = (input_ptr[i] - (floordiv * (*divisor)));
  }
}

template <typename T>
__global__ void matrix_floor_mod_kernel(const T* input_ptr, const int64 total,
                                        const T* divisor, T* output_ptr_data,
                                        KernelStatus* status) {
  GPU_1D_KERNEL_LOOP(i, total) {
    if (0 == divisor[i]) {
      *status = KernelStatus::ZeroDivisionError;
      return;
    }
    int64_t d = input_ptr[i] / divisor[i];
    int64_t floordiv = d * divisor[i] == input_ptr[i]
                           ? d
                           : d - ((input_ptr[i] < 0) ^ (divisor[i] < 0));
    output_ptr_data[i] = (input_ptr[i] - (floordiv * divisor[i]));
  }
}

template <typename T>
class FloorModOpGPU : public OpKernel {
 public:
  explicit FloorModOpGPU(OpKernelConstruction* c) : OpKernel(c) {}
  enum ModMode { SingleMod = 1, NormalMod = 2, Others = 3 };
  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& divisor_tensor = context->input(1);
    ModMode mode = Others;
    const T* divisor = nullptr;
    if (divisor_tensor.flat<T>().size() == 1) {
      mode = SingleMod;
    } else if (divisor_tensor.shape() == input.shape()) {
      mode = NormalMod;
    } else {
      context->SetStatus(errors::InvalidArgument(
          "Meituan floor_mod currently only supports the divisor with a shape "
          "of 1 or the same as the input."));
      return;
    }
    divisor = divisor_tensor.flat<T>().data();
    int64 flat_input_size = input.flat<T>().dimension(0);
    const TensorShape& input_shape = input.shape();
    TensorShape output_shape(input_shape);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    T* flat_output = &(output->flat<T>()(0));

    auto* flat_input = &(input.flat<T>()(0));
    bool* error_flag = nullptr;
    const Eigen::GpuDevice& d = context->eigen_gpu_device();
    auto config = GetGpuLaunchConfig(flat_input_size, d);

    KernelStatus* status = nullptr;
    cudaMallocHost(&status, sizeof(KernelStatus));
    cudaError_t err = cudaSuccess;
#define CHECK_CUDA_ERROR(tag)                                                 \
  if (err != cudaSuccess) {                                                   \
    Status stat = Status(error::INTERNAL,                                     \
                         (tag) + std::string(cudaGetErrorName(err)) + " - " + \
                             std::string(cudaGetErrorString(err)));           \
    OP_REQUIRES_OK(context, stat);                                            \
  }

    cudaStream_t stream = d.stream();
    cudaEvent_t event;
    err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    CHECK_CUDA_ERROR("Create cuda event error .");

    if (mode == SingleMod) {
      TF_CHECK_OK(GpuLaunchKernel(floor_mod_kernel<T>, config.block_count,
                                  config.thread_per_block, 0, d.stream(),
                                  flat_input, flat_input_size, divisor,
                                  flat_output, status));
    } else if (mode == NormalMod) {
      TF_CHECK_OK(GpuLaunchKernel(matrix_floor_mod_kernel<T>,
                                  config.block_count, config.thread_per_block,
                                  0, d.stream(), flat_input, flat_input_size,
                                  divisor, flat_output, status));
    }

    err = cudaEventRecord(event, stream);
    CHECK_CUDA_ERROR("Cuda event error .");
    err = cudaEventSynchronize(event);
    CHECK_CUDA_ERROR("Cuda synchronize error .");
    if (*status == KernelStatus::ZeroDivisionError) {
      context->SetStatus(errors::InvalidArgument("ZeroDivisionError"));
      return;
    }

    OP_REQUIRES(context, context->op_device_context()->stream()->ok(),
                errors::Internal("Launch of gpu kernel for FloorMod failed"));
  }
};

#define REGISTER_GPU(type)                                \
  REGISTER_KERNEL_BUILDER(Name("FloorMod")                \
                              .Device(DEVICE_GPU)         \
                              .Priority(1)                \
                              .TypeConstraint<type>("T"), \
                          FloorModOpGPU<type>)

REGISTER_GPU(int32);
REGISTER_GPU(int64);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
