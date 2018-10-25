/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifdef TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include <string.h>
#include <map>
#include <memory>
#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"

#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/gpu_fusion_ops.h"

#include "tensorflow/core/util/activation_mode.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

//-------------------------------------------------------------------

template <typename Device, typename T>
class ROCmFusionKernelAddRelu : public OpKernel {
 public:
  explicit ROCmFusionKernelAddRelu(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in0.shape(), &out));

    auto in0_data = AsDeviceMemory(in0.template flat<T>().data(),
                                   in0.template flat<T>().size());
    auto in1_data = AsDeviceMemory(in1.template flat<T>().data(),
                                   in1.template flat<T>().size());
    auto out_data = AsDeviceMemory(out->template flat<T>().data(),
                                   out->template flat<T>().size());

    if (in0.IsSameSize(in1)) {
      rocm_kernels::FusionAddRelu(ctx, static_cast<const T*>(in0_data.opaque()),
                                  static_cast<const T*>(in1_data.opaque()),
                                  static_cast<T*>(out_data.opaque()),
                                  in0.NumElements());
    } else {
      int in0_dims = in0.dims();
      int in1_dims = in1.dims();
      if ((in1_dims == 1) &&
          (in0.dim_size(in0_dims - 1) == in1.dim_size(in1_dims - 1))) {
        // simple broadcast
        rocm_kernels::FusionAddReluBcast(
            ctx, static_cast<const T*>(in0_data.opaque()),
            static_cast<const T*>(in1_data.opaque()),
            static_cast<T*>(out_data.opaque()), in0.NumElements(),
            in1.NumElements());
      } else {
        // non-trivial broadcast...bail for now
        LOG(FATAL) << "AddRelu - broadcast not supported: "
                   << "\tin0.shape() = " << in0.shape()
                   << "\tin1.shape() = " << in1.shape();
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("_ROCmFusedAddRelu").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    ROCmFusionKernelAddRelu<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("_ROCmFusedAddRelu")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<Eigen::half>("T"),
                        ROCmFusionKernelAddRelu<GPUDevice, Eigen::half>);

//-------------------------------------------------------------------

template <typename Device, typename T>
class ROCmFusionKernelAddNReluGrad : public OpKernel {
 public:
  explicit ROCmFusionKernelAddNReluGrad(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);
    const Tensor& in2 = ctx->input(2);

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in2.shape(), &out));

    auto in0_data = AsDeviceMemory(in0.template flat<T>().data(),
                                   in0.template flat<T>().size());
    auto in1_data = AsDeviceMemory(in1.template flat<T>().data(),
                                   in1.template flat<T>().size());
    auto in2_data = AsDeviceMemory(in2.template flat<T>().data(),
                                   in2.template flat<T>().size());
    auto out_data = AsDeviceMemory(out->template flat<T>().data(),
                                   out->template flat<T>().size());

    if (in0.IsSameSize(in1) && in0.IsSameSize(in2)) {
      rocm_kernels::FusionAddNReluGrad(
          ctx, static_cast<const T*>(in0_data.opaque()),
          static_cast<const T*>(in1_data.opaque()),
          static_cast<const T*>(in2_data.opaque()),
          static_cast<T*>(out_data.opaque()), in0.NumElements());
    } else {
      LOG(FATAL) << "AddNReluGrad - shape mismatch: "
                 << "\tin0.shape() = " << in0.shape()
                 << "\tin1.shape() = " << in1.shape()
                 << "\tin2.shape() = " << in2.shape();
    }
  }

 private:
};

REGISTER_KERNEL_BUILDER(Name("_ROCmFusedAddNReluGrad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T"),
                        ROCmFusionKernelAddNReluGrad<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("_ROCmFusedAddNReluGrad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<Eigen::half>("T"),
                        ROCmFusionKernelAddNReluGrad<GPUDevice, Eigen::half>);
//-------------------------------------------------------------------

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_ROCM
