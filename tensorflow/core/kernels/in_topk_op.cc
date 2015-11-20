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

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

template <typename T, typename TARGET_T>
class InTopK : public OpKernel {
 public:
  explicit InTopK(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
  }

  void Compute(OpKernelContext* context) override {
    const auto& predictions_in = context->input(0);
    const auto& targets_in = context->input(1);
    OP_REQUIRES(context, predictions_in.dims() == 2,
                errors::InvalidArgument("predictions must be 2-dimensional"));
    OP_REQUIRES(context, targets_in.dims() == 1,
                errors::InvalidArgument("targets must be 1-dimensional"));
    OP_REQUIRES(context, predictions_in.dim_size(0) == targets_in.dim_size(0),
                errors::InvalidArgument("First dimension of predictions ",
                                        predictions_in.dim_size(0),
                                        " must match length of targets ",
                                        targets_in.dim_size(0)));
    const auto& predictions = predictions_in.matrix<T>();
    const auto& targets = targets_in.vec<TARGET_T>();

    Tensor* t_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({targets_in.dim_size(0)}), &t_out));
    auto out = t_out->vec<bool>();

    const auto size = targets.size();
    const auto num_classes = predictions.dimension(1);
    for (int b = 0; b < size; b++) {
      T target_prediction = predictions(b, targets(b));
      int more_probable_classes = 0;
      for (int i = 0; i < num_classes; ++i) {
        if (predictions(b, i) > target_prediction) ++more_probable_classes;
      }
      out(b) = more_probable_classes < k_;
    }
  }

 private:
  int k_;
};

REGISTER_KERNEL_BUILDER(Name("InTopK")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("T"),
                        InTopK<float, int32>);
REGISTER_KERNEL_BUILDER(Name("InTopK")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int64>("T"),
                        InTopK<float, int64>);

}  // namespace tensorflow
