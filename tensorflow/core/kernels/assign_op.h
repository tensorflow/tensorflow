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

#ifndef TENSORFLOW_CORE_KERNELS_ASSIGN_OP_H_
#define TENSORFLOW_CORE_KERNELS_ASSIGN_OP_H_

#include "absl/status/status.h"
#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/ref_var.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

// TODO(jeff): Get rid of use_exclusive_lock_ option

// Computes *input[0] = input[1]
class AssignOp : public OpKernel {
 public:
  explicit AssignOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("validate_shape", &validate_shape_));
    OP_REQUIRES(context, IsRefType(context->input_type(0)),
                absl::InvalidArgumentError("lhs input needs to be a ref type"));
    if (!context
             ->GetAttr("_grappler_relax_allocator_constraints",
                       &relax_constraints_)
             .ok()) {
      relax_constraints_ = false;
    }
  }

  void Compute(OpKernelContext* context) override {
    constexpr int input_ref_index = 0;
    constexpr int output_ref_index = 0;
    constexpr int value_index = 1;

    auto copy = [this](OpKernelContext* cc_ctx, Tensor* lhs,
                       const Tensor& rhs) { Copy(cc_ctx, lhs, rhs); };

    AssignRefVariable(context, input_ref_index, output_ref_index, value_index,
                      use_exclusive_lock_, validate_shape_, relax_constraints_,
                      copy);
  }

  virtual void Copy(OpKernelContext* context, Tensor* lhs,
                    const Tensor& rhs) = 0;

  bool use_exclusive_lock_;
  bool validate_shape_;
  bool relax_constraints_;
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_ASSIGN_OP_H_
