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

#ifndef TENSORFLOW_KERNELS_DEBUG_OP_H_
#define TENSORFLOW_KERNELS_DEBUG_OP_H_

#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#endif
#include "tensorflow/core/debug/debug_io_utils.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {

// Copy op for debugging.
// Performs CPU-to-CPU or GPU-to-GPU deep-copying of tensor, depending on the
// device on which the tensor is allocated.
class CopyOp : public OpKernel {
 public:
  explicit CopyOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("tensor_name", &tensor_name_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& src_tensor = context->input(0);

    if (src_tensor.IsInitialized() &&
        DataTypeCanUseMemcpy(src_tensor.dtype())) {
      // Source tensor is initialized and is mem-copyable. Make a copy.
      Tensor* copied_tensor;
      OP_REQUIRES_OK(context, context->allocate_output(0, src_tensor.shape(),
                                                       &copied_tensor));

#if GOOGLE_CUDA
      Device* device = static_cast<Device*>(context->device());
      // Determine if the input tensor is not on CPU (e.g., on GPU).
      bool off_host_input = device->device_type() == DEVICE_GPU &&
                            !context->input_alloc_attr(0).on_host();

      if (off_host_input) {
        DeviceContext* device_ctxt = context->op_device_context();
        // Input is not on host: deep-copy it from GPU to the same GPU.
        Notification done_copy;
        GPUUtil::CopyGPUTensorToSameGPU(
            device, device_ctxt, &src_tensor, copied_tensor,
            [&done_copy](const Status& s) { done_copy.Notify(); });
        done_copy.WaitForNotification();
      } else {
        // The input tensor is on the host (CPU): deep-copy from CPU to CPU.
        *copied_tensor = tensor::DeepCopy(src_tensor);
      }
#else
      *copied_tensor = tensor::DeepCopy(src_tensor);
#endif
    } else {
      // Source tensor is NOT initialized and/or is not mem-copyable: Forward
      // the Tensor object.
      context->set_output(0, src_tensor);
    }
  }

  bool IsExpensive() override { return false; }

 private:
  string tensor_name_;
};

// Identity op for debugging.
//   Output slot 0 carries the debug signal and is always allocated on the
//   host (CPU) as a non-Ref tensor. In the case of DebugIdentityOp,
//   the debug signal is equal to the input tensor.
class DebugIdentityOp : public OpKernel {
 public:
  explicit DebugIdentityOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("tensor_name", &tensor_name_));
    OP_REQUIRES_OK(context, context->GetAttr("debug_urls", &debug_urls_));
  }

  void Compute(OpKernelContext* context) override {
    if (!debug_urls_.empty()) {
      // TODO(b/32704451): Don't just ignore the ::tensorflow::Status object!
      DebugIO::PublishDebugTensor(tensor_name_, "DebugIdentity",
                                  context->input(0),
                                  Env::Default()->NowMicros(), debug_urls_)
          .IgnoreError();
    }

    context->set_output(0, context->input(0));
  }

  bool IsExpensive() override { return false; }

 private:
  string tensor_name_;
  std::vector<string> debug_urls_;
};

// NaN-counter op for debugging.
template <typename T>
class DebugNanCountOp : public OpKernel {
 public:
  explicit DebugNanCountOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("tensor_name", &tensor_name_));
    OP_REQUIRES_OK(context, context->GetAttr("debug_urls", &debug_urls_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    // Use DT_INT64/int64 to be consistent with TensorShape::num_elements().
    int64 nan_count = 0;

    // If the input is an uninitialized tensor, let nan_count be 0.
    if (input.IsInitialized()) {
      // Count NaNs.
      const TensorShape& input_shape = input.shape();
      const T* input_flat = input.template flat<T>().data();

      for (int64 i = 0; i < input_shape.num_elements(); ++i) {
        if (Eigen::numext::isnan(static_cast<double>(input_flat[i]))) {
          nan_count++;
        }
      }
    }

    TensorShape shape({1});

    Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output_tensor));
    output_tensor->vec<int64>()(0) = nan_count;

    if (!debug_urls_.empty()) {
      // TODO(b/32704451): Don't just ignore the ::tensorflow::Status object!
      DebugIO::PublishDebugTensor(tensor_name_, "DebugNanCount", *output_tensor,
                                  Env::Default()->NowMicros(), debug_urls_)
          .IgnoreError();
    }
  }

  bool IsExpensive() override { return false; }

 private:
  string tensor_name_;
  std::vector<string> debug_urls_;
};

// Numeric summary op for debugging.
template <typename T>
class DebugNumericSummaryOp : public OpKernel {
 public:
  explicit DebugNumericSummaryOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("tensor_name", &tensor_name_));
    OP_REQUIRES_OK(context, context->GetAttr("debug_urls", &debug_urls_));
    OP_REQUIRES_OK(context, context->GetAttr("lower_bound", &lower_bound_));
    OP_REQUIRES_OK(context, context->GetAttr("upper_bound", &upper_bound_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("mute_if_healthy", &mute_if_healthy_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    int64 is_initialized = 0;
    int64 element_count = 0;
    int64 negative_inf_count = 0;
    int64 negative_count = 0;
    int64 zero_count = 0;
    int64 positive_count = 0;
    int64 positive_inf_count = 0;
    int64 nan_count = 0;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    double sum = 0.0;
    double mean = std::numeric_limits<double>::quiet_NaN();
    double variance = std::numeric_limits<double>::quiet_NaN();

    // Equal to negative_count + zero_count + positive_count.
    int64 non_inf_nan_count = 0;

    if (input.IsInitialized()) {
      is_initialized = 1;
      const TensorShape& input_shape = input.shape();
      const T* input_flat = input.template flat<T>().data();

      element_count = input_shape.num_elements();
      const bool is_lower_bound_custom = !Eigen::numext::isinf(lower_bound_);
      const bool is_upper_bound_custom = !Eigen::numext::isinf(upper_bound_);

      for (int64 i = 0; i < element_count; ++i) {
        const double x = static_cast<double>(input_flat[i]);
        if (Eigen::numext::isnan(x)) {
          nan_count++;
        } else if (Eigen::numext::isinf(x)) {
          if (x < 0.0) {
            negative_inf_count++;
          } else {
            positive_inf_count++;
          }
        } else {
          if (is_lower_bound_custom && x <= lower_bound_) {
            negative_inf_count++;
          } else if (is_upper_bound_custom && x >= upper_bound_) {
            positive_inf_count++;
          } else if (x < 0.0) {
            negative_count++;
          } else if (x > 0.0) {
            positive_count++;
          } else {
            zero_count++;
          }

          if (x < min) {
            min = x;
          }
          if (x > max) {
            max = x;
          }

          non_inf_nan_count++;
          sum += x;
        }
      }

      if (non_inf_nan_count > 0) {
        mean = sum / non_inf_nan_count;

        // Do a second pass to compute variance.
        variance = 0.0;
        for (int64 i = 0; i < element_count; ++i) {
          const double x = static_cast<double>(input_flat[i]);
          if (!Eigen::numext::isnan(x) && !Eigen::numext::isinf(x)) {
            variance += (x - mean) * (x - mean);
          }
        }
        variance /= non_inf_nan_count;
      }
    }

    TensorShape shape({12});

    Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output_tensor));
    output_tensor->vec<double>()(0) = static_cast<double>(is_initialized);
    output_tensor->vec<double>()(1) = static_cast<double>(element_count);
    output_tensor->vec<double>()(2) = static_cast<double>(nan_count);
    output_tensor->vec<double>()(3) = static_cast<double>(negative_inf_count);
    output_tensor->vec<double>()(4) = static_cast<double>(negative_count);
    output_tensor->vec<double>()(5) = static_cast<double>(zero_count);
    output_tensor->vec<double>()(6) = static_cast<double>(positive_count);
    output_tensor->vec<double>()(7) = static_cast<double>(positive_inf_count);
    output_tensor->vec<double>()(8) = min;
    output_tensor->vec<double>()(9) = max;
    output_tensor->vec<double>()(10) = mean;
    output_tensor->vec<double>()(11) = variance;

    bool mute = mute_if_healthy_ && nan_count == 0 && negative_inf_count == 0 &&
                positive_inf_count == 0;
    if (!mute && !debug_urls_.empty()) {
      // TODO(b/32704451): Don't just ignore the ::tensorflow::Status object!
      DebugIO::PublishDebugTensor(tensor_name_, "DebugNumericSummary",
                                  *output_tensor, Env::Default()->NowMicros(),
                                  debug_urls_)
          .IgnoreError();
    }
  }

  bool IsExpensive() override { return false; }

 private:
  string tensor_name_;
  std::vector<string> debug_urls_;
  float lower_bound_;
  float upper_bound_;
  bool mute_if_healthy_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_DEBUG_OP_H_
