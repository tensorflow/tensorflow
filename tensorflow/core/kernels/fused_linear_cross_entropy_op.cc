/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

template <typename T>
class FusedLinearCrossEntropyOp : public OpKernel {
 public:
  explicit FusedLinearCrossEntropyOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& features = ctx->input(0);
    const Tensor& weights = ctx->input(1);
    const Tensor& labels = ctx->input(2);

    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(features.shape()),
                errors::InvalidArgument("features must be 2-D"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(weights.shape()),
                errors::InvalidArgument("weights must be 2-D"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(labels.shape()),
                errors::InvalidArgument("labels must be 2-D"));
    OP_REQUIRES(ctx, features.dim_size(1) == weights.dim_size(0),
                errors::InvalidArgument("features inner dim must match weights"));
    OP_REQUIRES(ctx, features.dim_size(0) == labels.dim_size(0),
                errors::InvalidArgument("features batch size must match labels"));
    OP_REQUIRES(ctx, weights.dim_size(1) == labels.dim_size(1),
                errors::InvalidArgument("weights classes must match labels"));

    const int64_t batch_size = features.dim_size(0);
    const int64_t hidden_dim = features.dim_size(1);
    const int64_t num_classes = weights.dim_size(1);

    Tensor* loss_output = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({batch_size}), &loss_output));

    if (batch_size == 0) return;

    auto features_flat = features.matrix<T>();
    auto weights_flat = weights.matrix<T>();
    auto labels_flat = labels.matrix<T>();
    auto loss_flat = loss_output->flat<T>();

    using AccT = std::conditional_t<std::is_same_v<T, Eigen::bfloat16>, float, T>;

    // Hoisted outside the batch loop to prevent allocation thrashing
    std::vector<T> logits(num_classes);

    for (int64_t i = 0; i < batch_size; ++i) {
      T max_logit = -std::numeric_limits<T>::infinity();

      for (int64_t j = 0; j < num_classes; ++j) {
        T logit = static_cast<T>(0);
        for (int64_t k = 0; k < hidden_dim; ++k) {
          logit += features_flat(i, k) * weights_flat(k, j);
        }
        logits[j] = logit;
        if (logit > max_logit) max_logit = logit;
      }

      AccT sum_exp = static_cast<AccT>(0);
      for (int64_t j = 0; j < num_classes; ++j) {
        sum_exp += std::exp(static_cast<AccT>(logits[j] - max_logit));
      }

      AccT log_sum_exp = static_cast<AccT>(max_logit) + std::log(sum_exp);
      AccT sample_loss = static_cast<AccT>(0);

      for (int64_t j = 0; j < num_classes; ++j) {
        sample_loss -= static_cast<AccT>(labels_flat(i, j)) * (static_cast<AccT>(logits[j]) - log_sum_exp);
      }
      loss_flat(i) = static_cast<T>(sample_loss);
    }
  }
};

#define REGISTER_CPU_KERNEL(T)                                      \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("FusedLinearCrossEntropy").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      FusedLinearCrossEntropyOp<T>);

REGISTER_CPU_KERNEL(float);
REGISTER_CPU_KERNEL(double);
REGISTER_CPU_KERNEL(bfloat16);
#undef REGISTER_CPU_KERNEL

}  // namespace tensorflow
