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

// clang-format off
#include "tensorflow/core/platform/bfloat16.h"

#include <math.h>  // NOLINT
#include <algorithm>  // NOLINT
#include <numeric>  // NOLINT
// clang-format on

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/types.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if GOOGLE_CUDA
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#elif TENSORFLOW_USE_ROCM
#include "tensorflow/core/platform/rocm.h"
#endif
namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename T>
struct CheckNumericsLaunch {
  void Run(const GPUDevice& d, const T* data, int size,
           int abnormal_detected[2]);
};

extern template struct CheckNumericsLaunch<Eigen::half>;
extern template struct CheckNumericsLaunch<float>;
extern template struct CheckNumericsLaunch<double>;

template <typename T>
struct CheckNumericsLaunchV2 {
  void Run(const GPUDevice& d, const T* data, int size,
           int abnormal_detected[3]);
};

extern template struct CheckNumericsLaunchV2<Eigen::half>;
extern template struct CheckNumericsLaunchV2<float>;
extern template struct CheckNumericsLaunchV2<double>;
#endif

namespace {

const int kInfBit = 0x01;
const int kNaNBit = 0x02;
const int kNegativeInfBit = 0x04;
const int kPositiveInfBit = 0x08;

template <typename Device, typename T>
class CheckNumericsOp;

// Partial specialization for CPU
// TODO(jeff,rmlarsen): We should make this variant be an AsyncOpKernel, as
// was done for the GPU case below.
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
    const int64_t size = in.size();
    // Check to see if any element of the tensor is NaN or Inf.
    int fp_props = std::accumulate(
        data, data + size, 0,
        [this](const int x, const T& y) { return checkFloatingElement(x, y); });
    if (fp_props != 0) {
      const string& status = getErrorString(fp_props);
      if (!status.empty()) {
        context->SetStatus(errors::InvalidArgument(message_, " : Tensor had ",
                                                   status, " values"));
      }
    }
  }

 protected:
  virtual int checkFloatingElement(const int x, const T& y) {
    int result = x;
    if (TF_PREDICT_TRUE(Eigen::numext::isfinite(y))) {
      // Do nothing: common case.
    } else {
      if (Eigen::numext::isinf(y)) {
        result |= kInfBit;
      } else if (Eigen::numext::isnan(y)) {
        result |= kNaNBit;
      }
    }
    return result;
  }

  virtual const string getErrorString(const int fp_props) {
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
    return status;
  }

 private:
  string message_;
};

template <typename Device, typename T>
class CheckNumericsV2Op;

// Partial specialization for CPU: v2.
// The v2 op differs from the v1 in that it distinguishes -inf and +inf.
template <typename T>
class CheckNumericsV2Op<CPUDevice, T> : public CheckNumericsOp<CPUDevice, T> {
 public:
  explicit CheckNumericsV2Op(OpKernelConstruction* context)
      : CheckNumericsOp<CPUDevice, T>(context) {}

 protected:
  int checkFloatingElement(const int x, const T& y) override {
    int result = x;
    if (TF_PREDICT_TRUE(Eigen::numext::isfinite(y))) {
      // Do nothing: common case.
    } else {
      if (Eigen::numext::isinf(y)) {
        result |= y < static_cast<T>(0.) ? kNegativeInfBit : kPositiveInfBit;
      } else if (Eigen::numext::isnan(y)) {
        result |= kNaNBit;
      }
    }
    return result;
  }

  const string getErrorString(const int fp_props) override {
    std::vector<string> anomalies;
    if (fp_props & kNegativeInfBit) {
      anomalies.push_back("-Inf");
    }
    if (fp_props & kPositiveInfBit) {
      anomalies.push_back("+Inf");
    }
    if (fp_props & kNaNBit) {
      anomalies.push_back("NaN");
    }
    if (anomalies.size() == 3) {
      return strings::StrCat(anomalies[0], ", ", anomalies[1], ", and ",
                             anomalies[2]);
    } else if (anomalies.size() == 2) {
      return strings::StrCat(anomalies[0], " and ", anomalies[1]);
    } else {
      return anomalies[0];
    }
  }
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Partial specialization for GPU
template <typename T>
class CheckNumericsOp<GPUDevice, T> : public AsyncOpKernel {
 public:
  typedef GPUDevice Device;

  explicit CheckNumericsOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    // message_ is used as the prefix for the assertion error message. For
    // instance, this can be the name of the input op that produced the tensor.
    OP_REQUIRES_OK(context, context->GetAttr("message", &message_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    // pass along the input to the output
    context->set_output(0, context->input(0));
    if (context->input(0).NumElements() == 0) {
      done();
      return;
    }
    auto input = context->input(0).flat<T>();

    // Allocate and initialize the elements to hold the check results
    Tensor abnormal_detected;
    const int abnormal_detected_size = getAnomalyIndicatorSize();
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DT_INT32, TensorShape({abnormal_detected_size}),
                                &abnormal_detected));

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES_ASYNC(context, stream != nullptr,
                      errors::Internal("No GPU stream available."), done);

    se::DeviceMemoryBase abnormal_detected_ptr(
        abnormal_detected.flat<int>().data(),
        abnormal_detected.flat<int>().size());
    stream->ThenMemset32(&abnormal_detected_ptr, 0,
                         abnormal_detected.flat<int>().size() * sizeof(int));

    // Call the GPU kernels for the numerical checks
    const Device& d = context->eigen_device<Device>();
    RunKernel(d, input.data(), input.size(),
              abnormal_detected.flat<int>().data());

    // Copy the results from device to host
    AllocatorAttributes attr;
    attr.set_on_host(true);
    attr.set_gpu_compatible(true);
    Tensor abnormal_detected_host;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_temp(DT_INT32, TensorShape({abnormal_detected_size}),
                               &abnormal_detected_host, attr),
        done);
    OP_REQUIRES_ASYNC(
        context,
        stream
            ->ThenMemcpy(abnormal_detected_host.flat<int>().data(),
                         abnormal_detected_ptr,
                         abnormal_detected_size * sizeof(int))
            .ok(),
        errors::Internal("GPU memcpy from device to host failed"), done);

    // We have observed crashes on some network stacks when not holding
    // this tensor reference.
    TensorReference abnormal_detected_ref(abnormal_detected);
    auto check_cb = [this, stream, abnormal_detected_ref,
                     abnormal_detected_host, context, done]() {
#if GOOGLE_CUDA
      se::cuda::ScopedActivateExecutorContext scoped_activation{
          stream->parent()};
#elif TENSORFLOW_USE_ROCM
      se::rocm::ScopedActivateExecutorContext scoped_activation{
          stream->parent()};
#endif
      TTypes<const int>::Vec abnormal_detected_host_flat =
          abnormal_detected_host.flat<int>();
      abnormal_detected_ref.Unref();
      checkForAnomalies(context, abnormal_detected_host_flat);
      done();
    };
    context->device()
        ->tensorflow_accelerator_device_info()
        ->event_mgr->ThenExecute(stream, std::move(check_cb));
  }

 protected:
  virtual int getAnomalyIndicatorSize() { return 2; }

  virtual void RunKernel(const GPUDevice& d, const T* data, int size,
                         int* abnormal_detected) {
    CheckNumericsLaunch<T>().Run(d, data, size, abnormal_detected);
  }

  virtual void checkForAnomalies(
      OpKernelContext* context,
      const TTypes<const int>::Vec& abnormality_indicators) {
    const int is_nan = abnormality_indicators(0);
    const int is_inf = abnormality_indicators(1);
    if (is_nan || is_inf) {
      LOG(ERROR) << "abnormal_detected_host @" << abnormality_indicators.data()
                 << " = {" << is_nan << ", " << is_inf << "} " << message_;

      string anomalies;
      if (is_nan && is_inf) {
        anomalies = "Inf and NaN";
      } else if (is_nan) {
        anomalies = "NaN";
      } else if (is_inf) {
        anomalies = "Inf";
      }
      context->SetStatus(errors::InvalidArgument(message_, " : Tensor had ",
                                                 anomalies, " values"));
    }
  }

  string message_;
};

template <typename T>
class CheckNumericsV2Op<GPUDevice, T> : public CheckNumericsOp<GPUDevice, T> {
 public:
  CheckNumericsV2Op(OpKernelConstruction* context)
      : CheckNumericsOp<GPUDevice, T>(context) {}

 protected:
  int getAnomalyIndicatorSize() override { return 3; }

  void RunKernel(const GPUDevice& d, const T* data, int size,
                 int* abnormal_detected) override {
    CheckNumericsLaunchV2<T>().Run(d, data, size, abnormal_detected);
  }

  void checkForAnomalies(
      OpKernelContext* context,
      const TTypes<const int>::Vec& abnormality_indicators) override {
    const int is_nan = abnormality_indicators(0);
    const int is_negative_inf = abnormality_indicators(1);
    const int is_positive_inf = abnormality_indicators(2);
    if (is_negative_inf || is_positive_inf || is_nan) {
      std::vector<string> anomalies;
      if (is_negative_inf) {
        anomalies.push_back("-Inf");
      }
      if (is_positive_inf) {
        anomalies.push_back("+Inf");
      }
      if (is_nan) {
        anomalies.push_back("NaN");
      }
      string all_anomalies;
      if (anomalies.size() == 3) {
        all_anomalies = strings::StrCat(anomalies[0], ", ", anomalies[1],
                                        ", and ", anomalies[2]);
      } else if (anomalies.size() == 2) {
        all_anomalies = strings::StrCat(anomalies[0], " and ", anomalies[1]);
      } else {
        all_anomalies = anomalies[0];
      }
      context->SetStatus(errors::InvalidArgument(
          this->message_, " : Tensor had ", all_anomalies, " values"));
    }
  }

  static constexpr int abnormal_detected_size = 3;
};

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace

#define REGISTER_CPU_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("CheckNumerics").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      CheckNumericsOp<CPUDevice, T>);
TF_CALL_half(REGISTER_CPU_KERNEL);
TF_CALL_bfloat16(REGISTER_CPU_KERNEL);
TF_CALL_float(REGISTER_CPU_KERNEL);
TF_CALL_double(REGISTER_CPU_KERNEL);

#define REGISTER_V2_CPU_KERNEL(T)                                        \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("CheckNumericsV2").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      CheckNumericsV2Op<CPUDevice, T>);
TF_CALL_half(REGISTER_V2_CPU_KERNEL);
TF_CALL_bfloat16(REGISTER_V2_CPU_KERNEL);
TF_CALL_float(REGISTER_V2_CPU_KERNEL);
TF_CALL_double(REGISTER_V2_CPU_KERNEL);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
REGISTER_KERNEL_BUILDER(
    Name("CheckNumerics").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
    CheckNumericsOp<GPUDevice, Eigen::half>);
REGISTER_KERNEL_BUILDER(
    Name("CheckNumerics").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    CheckNumericsOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("CheckNumerics").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    CheckNumericsOp<GPUDevice, double>);

REGISTER_KERNEL_BUILDER(
    Name("CheckNumericsV2").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
    CheckNumericsV2Op<GPUDevice, Eigen::half>);
REGISTER_KERNEL_BUILDER(
    Name("CheckNumericsV2").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    CheckNumericsV2Op<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("CheckNumericsV2").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    CheckNumericsV2Op<GPUDevice, double>);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
