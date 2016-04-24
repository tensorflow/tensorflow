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

#include <math.h>
#include <algorithm>
#include <numeric>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA
namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#if GOOGLE_CUDA
template <typename T>
struct CheckNumericsLaunch {
  void Run(const GPUDevice& d, const T* data, int size,
           int abnormal_detected[2]);
};
#endif

namespace {

template <typename Device, typename T>
class CheckNumericsOp;

// Partial specialization for CPU
template <typename T>
class CheckNumericsOp<CPUDevice, T> : public OpKernel {
 public:
  explicit CheckNumericsOp(OpKernelConstruction* context) : OpKernel(context) {
    // message_ is used as the prefix for the assertion error message. For
    // instance, this can be the name of the input op that produced the tensor.
    OP_REQUIRES_OK(context, context->GetAttr("message", &message_));
  }

  void Compute(OpKernelContext* context) override {
    // pass along the input to the output
    context->set_output(0, context->input(0));

    auto in = context->input(0).flat<T>();
    const T* data = in.data();
    const int64 size = in.size();
    // Check to see if any element of the tensor is NaN or Inf.
    int fp_props =
        std::accumulate(data, data + size, 0, [](const int& x, const T& y) {
          int prop = std::fpclassify(y);
          int result = x;
          if (prop == FP_INFINITE) {
            result |= kInfBit;
          } else if (prop == FP_NAN) {
            result |= kNaNBit;
          }
          return result;
        });
    string status;
    if ((fp_props & kInfBit) && (fp_props & kNaNBit)) {
      status = "Inf and NaN";
    } else {
      if (fp_props & kInfBit) {
        status = "Inf";
      }
      if (fp_props & kNaNBit) {
        status = "NaN";
      }
    }
    if (!status.empty()) {
      context->SetStatus(errors::InvalidArgument(message_, " : Tensor had ",
                                                 status, " values"));
    }
  }

 private:
  string message_;
  static const int kInfBit = 0x01;
  static const int kNaNBit = 0x02;
};

#if GOOGLE_CUDA
// Partial specialization for GPU
template <typename T>
class CheckNumericsOp<GPUDevice, T> : public OpKernel {
 public:
  typedef GPUDevice Device;

  explicit CheckNumericsOp(OpKernelConstruction* context) : OpKernel(context) {
    // message_ is used as the prefix for the assertion error message. For
    // instance, this can be the name of the input op that produced the tensor.
    OP_REQUIRES_OK(context, context->GetAttr("message", &message_));
  }

  void Compute(OpKernelContext* context) override {
    // pass along the input to the output
    context->set_output(0, context->input(0));
    auto input = context->input(0).flat<T>();

    // Allocate and initialize the elements to hold the check results
    const int abnormal_detected_size = 2;
    Tensor abnormal_detected;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DT_INT32, TensorShape({abnormal_detected_size}),
                                &abnormal_detected));

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    perftools::gputools::DeviceMemoryBase abnormal_detected_ptr(
        abnormal_detected.flat<int>().data(),
        abnormal_detected.flat<int>().size());
    stream->ThenMemset32(&abnormal_detected_ptr, 0,
                         abnormal_detected.flat<int>().size() * sizeof(int));

    // Call the Cuda kernels for the numerical checks
    const Device& d = context->eigen_device<Device>();
    CheckNumericsLaunch<T>().Run(d, input.data(), input.size(),
                                 abnormal_detected.flat<int>().data());

    // Copy the results from device to host
    AllocatorAttributes attr;
    attr.set_on_host(true);
    attr.set_gpu_compatible(true);
    Tensor abnormal_detected_out;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DT_INT32, TensorShape({abnormal_detected_size}),
                                &abnormal_detected_out, attr));
    int* abnormal_detected_host = abnormal_detected_out.flat<int>().data();
    stream->ThenMemcpy(abnormal_detected_host, abnormal_detected_ptr,
                       abnormal_detected_size * sizeof(int));
    stream->BlockHostUntilDone();
    OP_REQUIRES(context, stream->ok(),
                errors::Internal("cudaMemcpy from device to host failed"));

    int is_nan = abnormal_detected_host[0];
    int is_inf = abnormal_detected_host[1];
    if (is_nan || is_inf) {
      string status;
      LOG(ERROR) << "abnormal_detected_host @" << abnormal_detected_host
                 << " = {" << is_nan << ", " << is_inf << "} " << message_;

      // Results should always be 1 or 0.  If we see anything else then
      // there has been some GPU memory corruption.
      CHECK_GE(is_nan, 0);
      CHECK_GE(is_inf, 0);
      CHECK_LE(is_nan, 1);
      CHECK_LE(is_inf, 1);

      if (is_nan && is_inf) {
        status = "Inf and NaN";
      } else if (is_nan) {
        status = "NaN";
      } else if (is_inf) {
        status = "Inf";
      }
      context->SetStatus(errors::InvalidArgument(message_, " : Tensor had ",
                                                 status, " values"));
    }
  }

 private:
  string message_;
};
#endif  // GOOGLE_CUDA

}  // namespace

REGISTER_KERNEL_BUILDER(Name("CheckNumerics")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        CheckNumericsOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("CheckNumerics")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("T"),
                        CheckNumericsOp<CPUDevice, double>);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("CheckNumerics")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T"),
                        CheckNumericsOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("CheckNumerics")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<double>("T"),
                        CheckNumericsOp<GPUDevice, double>);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
