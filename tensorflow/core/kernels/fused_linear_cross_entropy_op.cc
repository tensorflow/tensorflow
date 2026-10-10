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

    const int64_t batch_size = features.dim_size(0);
    const int64_t hidden_dim = features.dim_size(1);
    const int64_t num_classes = weights.dim_size(1);

    Tensor* loss_output = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({batch_size}), &loss_output));

    auto features_flat = features.matrix<T>();
    auto weights_flat = weights.matrix<T>();
    auto labels_flat = labels.matrix<T>();
    auto loss_flat = loss_output->flat<T>();

    // Fused MatMul + Softmax Cross Entropy without saving intermediate logits
    for (int64_t i = 0; i < batch_size; ++i) {
      T max_logit = -std::numeric_limits<T>::infinity();
      std::vector<T> logits(num_classes, static_cast<T>(0));

      for (int64_t j = 0; j < num_classes; ++j) {
        T logit = static_cast<T>(0);
        for (int64_t k = 0; k < hidden_dim; ++k) {
          logit += features_flat(i, k) * weights_flat(k, j);
        }
        if (ctx->num_inputs() > 3) {
          const Tensor& biases = ctx->input(3);
          logit += biases.flat<T>()(j);
        }
        logits[j] = logit;
        if (logit > max_logit) max_logit = logit;
      }

      T sum_exp = static_cast<T>(0);
      for (int64_t j = 0; j < num_classes; ++j) {
        sum_exp += std::exp(logits[j] - max_logit);
      }

      T log_sum_exp = max_logit + std::log(sum_exp);
      T sample_loss = static_cast<T>(0);

      for (int64_t j = 0; j < num_classes; ++j) {
        sample_loss -= labels_flat(i, j) * (logits[j] - log_sum_exp);
      }
      loss_flat(i) = sample_loss;
    }
  }
};

#define REGISTER_CPU_KERNEL(T)                                    \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("FusedLinearCrossEntropy").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      FusedLinearCrossEntropyOp<T>);

REGISTER_CPU_KERNEL(float);
REGISTER_CPU_KERNEL(double);
#undef REGISTER_CPU_KERNEL

}  // namespace tensorflow
