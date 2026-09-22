/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_INTEGER_POW_GPU_H_
#define TENSORFLOW_CORE_KERNELS_INTEGER_POW_GPU_H_

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

// Instantiated in integer_pow_gpu.cu.cc for the signed GPU Pow types.
template <typename T>
struct CheckNegativeExponentGpu {
  absl::Status operator()(const Eigen::GpuDevice& device, const T* exponents,
                          int64_t size, int32_t* found_negative);
};

// Applies the same exponent validation to the Eigen and MLIR GPU kernels.
// The check runs before Pow can reuse an exponent buffer for its output. Only
// a single flag is copied back to the host, without blocking the caller.
template <typename T, typename Kernel>
class IntegerPowGpuOp : public AsyncOpKernel {
 public:
  explicit IntegerPowGpuOp(OpKernelConstruction* context)
      : AsyncOpKernel(context), kernel_(context) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    const BCast bcast(BCast::FromShape(context->input(0).shape()),
                      BCast::FromShape(context->input(1).shape()));
    if (!bcast.IsValid() ||
        BCast::ToShape(bcast.output_shape()).num_elements() == 0) {
      kernel_.Compute(context);
      done();
      return;
    }

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES_ASYNC(context, stream != nullptr,
                      absl::InternalError("No GPU stream available."), done);

    Tensor found_negative;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_temp(DT_INT32, TensorShape({}), &found_negative),
        done);
    stream_executor::DeviceAddressBase device_flag(
        found_negative.flat<int32_t>().data(), sizeof(int32_t));
    OP_REQUIRES_OK_ASYNC(
        context, stream->Memset32(&device_flag, 0, sizeof(int32_t)), done);
    const auto& exponents = context->input(1).flat<T>();
    OP_REQUIRES_OK_ASYNC(
        context,
        CheckNegativeExponentGpu<T>()(context->eigen_device<Eigen::GpuDevice>(),
                                      exponents.data(), exponents.size(),
                                      found_negative.flat<int32_t>().data()),
        done);

    AllocatorAttributes attributes;
    attributes.set_on_host(true);
    attributes.set_gpu_compatible(true);
    Tensor host_flag;
    OP_REQUIRES_OK_ASYNC(context,
                         context->allocate_temp(DT_INT32, TensorShape({}),
                                                &host_flag, attributes),
                         done);
    OP_REQUIRES_ASYNC(
        context,
        stream
            ->Memcpy(host_flag.flat<int32_t>().data(), device_flag,
                     sizeof(int32_t))
            .ok(),
        absl::InternalError("GPU memcpy from device to host failed"), done);

    kernel_.Compute(context);

    TensorReference flag_ref(found_negative);
    auto check = [flag_ref, host_flag, context, done]() {
      flag_ref.Unref();
      if (context->status().ok() && host_flag.scalar<int32_t>()() != 0) {
        context->SetStatus(absl::InvalidArgumentError(
            "Integers to negative integer powers are not allowed"));
      }
      done();
    };
    context->device()
        ->tensorflow_accelerator_device_info()
        ->event_mgr->ThenExecute(stream, std::move(check));
  }

 private:
  Kernel kernel_;
};

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#endif  // TENSORFLOW_CORE_KERNELS_INTEGER_POW_GPU_H_
