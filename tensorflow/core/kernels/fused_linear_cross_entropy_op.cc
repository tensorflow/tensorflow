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
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({batch_size}), &loss_output));

  auto features_flat = features.matrix<float>();
  auto weights_flat = weights.matrix<float>();
  auto labels_flat = labels.matrix<float>();
  auto loss_flat = loss_output->flat<float>();

  // Fused MatMul + Softmax Cross Entropy without saving intermediate logits tensor
  for (int64_t i = 0; i < batch_size; ++i) {
    float max_logit = -std::numeric_limits<float>::infinity();
    std::vector<float> logits(num_classes, 0.0f);

    for (int64_t j = 0; j < num_classes; ++j) {
      float logit = 0.0f;
      for (int64_t k = 0; k < hidden_dim; ++k) {
        logit += features_flat(i, k) * weights_flat(k, j);
      }
      if (ctx->num_inputs() > 3) {
        const Tensor& biases = ctx->input(3);
        logit += biases.flat<float>()(j);
      }
      logits[j] = logit;
      if (logit > max_logit) max_logit = logit;
    }

    float sum_exp = 0.0f;
    for (int64_t j = 0; j < num_classes; ++j) {
      sum_exp += std::exp(logits[j] - max_logit);
    }

    float log_sum_exp = max_logit + std::log(sum_exp);
    float sample_loss = 0.0f;

    for (int64_t j = 0; j < num_classes; ++j) {
      sample_loss -= labels_flat(i, j) * (logits[j] - log_sum_exp);
    }
    loss_flat(i) = sample_loss;
  }
}
