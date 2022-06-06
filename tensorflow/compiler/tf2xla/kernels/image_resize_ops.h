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
#ifndef TENSORFLOW_COMPILER_TF2XLA_KERNELS_IMAGE_RESIZE_OPS_H_
#define TENSORFLOW_COMPILER_TF2XLA_KERNELS_IMAGE_RESIZE_OPS_H_

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/xla/primitive_util.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/compiler/tf2xla/kernels/gpu_tf_kernel_custom_call.h"
#endif

namespace tensorflow {

class ResizeNearestNeighborOp : public XlaOpKernel {
 public:
  explicit ResizeNearestNeighborOp(OpKernelConstruction* ctx);
  void Compile(XlaOpKernelContext* ctx) override;

 protected:
  bool align_corners_ = true;
  bool half_pixel_centers_ = true;
  bool is_kernel_bilinear_ = false;
};

class ResizeBilinearOp : public XlaOpKernel {
 public:
  explicit ResizeBilinearOp(OpKernelConstruction* ctx);

  void Compile(XlaOpKernelContext* ctx) override;

 protected:
  bool align_corners_ = true;
  bool half_pixel_centers_ = true;
  bool is_kernel_bilinear_ = true;
};

class ResizeBilinearGradOp : public XlaOpKernel {
 public:
  explicit ResizeBilinearGradOp(OpKernelConstruction* ctx);

  void Compile(XlaOpKernelContext* ctx) override;

 protected:
  bool align_corners_;
  bool half_pixel_centers_ = true;
  xla::PrimitiveType output_type_;

 private:
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  // Fallback light outside compilation kernel for the option combination we do
  // not support.
  std::optional<CallTfKernelOp> fallback_tf_kernel_;
#endif
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_KERNELS_IMAGE_RESIZE_OPS_H_
