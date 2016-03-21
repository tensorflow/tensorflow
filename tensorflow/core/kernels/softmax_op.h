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

// See docs in ../ops/nn_ops.cc.

#ifndef TENSORFLOW_KERNELS_SOFTMAX_OP_H_
#define TENSORFLOW_KERNELS_SOFTMAX_OP_H_

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/softmax_op_functor.h"

namespace tensorflow {

template <typename Device, typename T>
class SoftmaxOp : public OpKernel {
 public:
  explicit SoftmaxOp(OpKernelConstruction* context) : OpKernel(context) {
    log_ = StringPiece(name()).starts_with("Log");
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& logits_in = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(logits_in.shape()),
                errors::InvalidArgument("logits must be 2-dimensional"));
    Tensor* softmax_out = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, logits_in.shape(), &softmax_out));
    if (logits_in.NumElements()) {
      functor::SoftmaxFunctor<Device, T> functor;
      functor(context->eigen_device<Device>(), logits_in.matrix<T>(),
              softmax_out->matrix<T>(), log_);
    }
  }

 private:
  bool log_;
};

}  // namespace tensorflow

#undef EIGEN_USE_THREADS

#endif  // TENSORFLOW_KERNELS_SOFTMAX_OP_H_
