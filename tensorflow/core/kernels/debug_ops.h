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

#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#include "tensorflow/core/debug/debug_io_utils.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/core/notification.h"

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

    DeviceContext* device_ctxt = context->op_device_context();
    Device* device = static_cast<Device*>(context->device());

    // Determine if the input tensor is not on CPU (e.g., on GPU).
    bool off_host_input = device->device_type() == DEVICE_GPU &&
                          !context->input_alloc_attr(0).on_host();

    Tensor* copied_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, src_tensor.shape(),
                                                     &copied_tensor));
#if GOOGLE_CUDA
    if (off_host_input) {
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
#endif  // GOOGLE_CUDA
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
      DebugIO::PublishDebugTensor(tensor_name_, "DebugIdentity",
                                  context->input(0),
                                  Env::Default()->NowMicros(), debug_urls_);
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

    const TensorShape& input_shape = input.shape();
    const T* input_flat = input.template flat<T>().data();

    // Count NaNs.
    // Use DT_INT64/int64 to be consistent with TensorShape::num_elements().
    int64 nan_count = 0;
    for (int64 i = 0; i < input_shape.num_elements(); ++i) {
      if (Eigen::numext::isnan(input_flat[i])) {
        nan_count++;
      }
    }

    TensorShape shape({1});

    Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output_tensor));
    output_tensor->vec<int64>()(0) = nan_count;
  }

  bool IsExpensive() override { return false; }

 private:
  string tensor_name_;
  std::vector<string> debug_urls_;
};

// TODO(cais): Add DebugInfinityCount
// TODO(cais): Add DebugZeroCount

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_IDENTITY_OP_H_
