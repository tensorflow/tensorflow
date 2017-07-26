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

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

template <typename T, typename TARGET_T>
class InTopK : public OpKernel {
 public:
  explicit InTopK(OpKernelConstruction* context) : OpKernel(context) {
    if (context->num_inputs() == 2) {
      OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
    }
  }

  void Compute(OpKernelContext* context) override {
    const auto& predictions_in = context->input(0);
    const auto& targets_in = context->input(1);
    int64 k_val = k_;
    if (context->num_inputs() == 3) {
      const auto& k_in = context->input(2);

      OP_REQUIRES(context, TensorShapeUtils::IsScalar(k_in.shape()),
                  errors::InvalidArgument("k must be 0-D, got shape ",
                                          k_in.shape().DebugString()));

      if (k_in.dtype() == DT_INT32) {
        k_val = k_in.scalar<int32>()();
      } else {
        k_val = k_in.scalar<int64>()();
      }
    }

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
      auto target = internal::SubtleMustCopy(targets(b));
      OP_REQUIRES(context, FastBoundsCheck(target, num_classes),
                  errors::InvalidArgument("targets[", b, "] is out of range"));
      T target_prediction = predictions(b, target);
      bool cannot_say = !std::isfinite(target_prediction);
      int more_probable_classes = 0;
      if (!cannot_say) {
        for (int i = 0; i < num_classes; ++i) {
          T pred = predictions(b, i);
          if (!std::isfinite(pred)) {
            cannot_say = true;
            break;
          } else if (pred > target_prediction) {
            ++more_probable_classes;
          }
        }
      }
      out(b) = cannot_say ? false : (more_probable_classes < k_val);
    }
  }

 private:
  int k_;
};

REGISTER_KERNEL_BUILDER(
    Name("InTopK").Device(DEVICE_CPU)
    .HostMemory("predictions")
    .HostMemory("targets")
    .HostMemory("precision")
    .TypeConstraint<int32>("T"),
    InTopK<float, int32>);
REGISTER_KERNEL_BUILDER(
    Name("InTopK").Device(DEVICE_CPU)
    .HostMemory("predictions")
    .HostMemory("targets")
    .HostMemory("precision")
    .TypeConstraint<int64>("T"),
    InTopK<float, int64>);

REGISTER_KERNEL_BUILDER(
    Name("InTopKV2").Device(DEVICE_CPU)
    .HostMemory("predictions")
    .HostMemory("targets")
    .HostMemory("k")
    .HostMemory("precision")
    .TypeConstraint<int32>("T"),
    InTopK<float, int32>);
REGISTER_KERNEL_BUILDER(
    Name("InTopKV2").Device(DEVICE_CPU)
    .HostMemory("predictions")
    .HostMemory("targets")
    .HostMemory("k")
    .HostMemory("precision")
    .TypeConstraint<int64>("T"),
    InTopK<float, int64>);

}  // namespace tensorflow
