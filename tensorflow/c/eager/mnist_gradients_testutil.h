/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/eager/gradients_internal.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/experimental/ops/nn_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/status.h"

// ========================== Tape Ops ==============================

namespace tensorflow {
namespace gradients {
namespace internal {
// Computes `inputs[0] + inputs[1]` and records it on the tape.
Status Add(AbstractContext* ctx, Tape* tape,
           absl::Span<AbstractTensorHandle* const> inputs,
           absl::Span<AbstractTensorHandle*> outputs,
           const GradientRegistry& registry);

// Computes `inputs[0] * inputs[1]` for matrices and records it on the tape.
Status MatMul(AbstractContext* ctx, Tape* tape,
              absl::Span<AbstractTensorHandle* const> inputs,
              absl::Span<AbstractTensorHandle*> outputs, const char* name,
              bool transpose_a, bool transpose_b,
              const GradientRegistry& registry);

// Computes `inputs[0] * inputs[1]` and records it on the tape.
Status Mul(AbstractContext* ctx, Tape* tape,
           absl::Span<AbstractTensorHandle* const> inputs,
           absl::Span<AbstractTensorHandle*> outputs, const char* name,
           const GradientRegistry& registry);

// Computes `Relu(inputs[0])` and records it on the tape.
Status Relu(AbstractContext* ctx, Tape* tape,
            absl::Span<AbstractTensorHandle* const> inputs,
            absl::Span<AbstractTensorHandle*> outputs, const char* name,
            const GradientRegistry& registry);

// Computes `SoftmaxLoss(scores, labels)` for matrices and records it on the
// tape.
Status SparseSoftmaxCrossEntropyLoss(
    AbstractContext* ctx, Tape* tape,
    absl::Span<AbstractTensorHandle* const> inputs,
    absl::Span<AbstractTensorHandle*> outputs, const char* name,
    const GradientRegistry& registry);

// ====================== End Tape Ops ============================

// Computes
// y = inputs[0] + inputs[1]
// return grad(y, {inputs[0], inputs[1]})
Status AddGradModel(AbstractContext* ctx,
                    absl::Span<AbstractTensorHandle* const> inputs,
                    absl::Span<AbstractTensorHandle*> outputs,
                    const GradientRegistry& registry);

// Computes
// y = inputs[0] * inputs[1]
// return grad(y, {inputs[0], inputs[1]})
Status MatMulGradModel(AbstractContext* ctx,
                       absl::Span<AbstractTensorHandle* const> inputs,
                       absl::Span<AbstractTensorHandle*> outputs,
                       const GradientRegistry& registry);

// Computes 2-layer Neural Network with Softmax Loss.
Status MNISTForwardModel(AbstractContext* ctx,
                         absl::Span<AbstractTensorHandle* const> inputs,
                         absl::Span<AbstractTensorHandle*> outputs,
                         const GradientRegistry& registry);

// Computes MatMul with first matrix tranposed.
Status MatMulTransposeModel(AbstractContext* ctx,
                            absl::Span<AbstractTensorHandle* const> inputs,
                            absl::Span<AbstractTensorHandle*> outputs,
                            const GradientRegistry& registry);

// Test Model to verify ReluGrad functionality
Status ReluGradModel(AbstractContext* ctx,
                     absl::Span<AbstractTensorHandle* const> inputs,
                     absl::Span<AbstractTensorHandle*> outputs,
                     const GradientRegistry& registry);

// Test Model to verify SoftmaxGrad functionality
Status SoftmaxLossGradModel(AbstractContext* ctx,
                            absl::Span<AbstractTensorHandle* const> inputs,
                            absl::Span<AbstractTensorHandle*> outputs,
                            const GradientRegistry& registry);

// Test Model to verify Multi-grad functionality for MNIST
Status MNISTGradModel(AbstractContext* ctx,
                      absl::Span<AbstractTensorHandle* const> inputs,
                      absl::Span<AbstractTensorHandle*> outputs,
                      const GradientRegistry& registry);

// Test Model to verify scalar-tensor multiplication Op
Status ScalarMulModel(AbstractContext* ctx,
                      absl::Span<AbstractTensorHandle* const> inputs,
                      absl::Span<AbstractTensorHandle*> outputs,
                      const GradientRegistry& registry);

Status MatMulModel(AbstractContext* ctx,
                            absl::Span<AbstractTensorHandle* const> inputs,
                            absl::Span<AbstractTensorHandle*> outputs,
                            const GradientRegistry& registry);

Status MulModel(AbstractContext* ctx,
                            absl::Span<AbstractTensorHandle* const> inputs,
                            absl::Span<AbstractTensorHandle*> outputs,
                            const GradientRegistry& registry);

Status SoftmaxModel(AbstractContext* ctx,
                            absl::Span<AbstractTensorHandle* const> inputs,
                            absl::Span<AbstractTensorHandle*> outputs,
                            const GradientRegistry& registry);

// Updates the weights for a neural network given incoming grads and learning
// rate
Status UpdateWeights(AbstractContext* ctx,
                     std::vector<AbstractTensorHandle*>& grads,
                     std::vector<AbstractTensorHandle*>& weights,
                     AbstractTensorHandle* learning_rate);

AbstractContext* BuildFunction(const char* fn_name);

Status CreateParamsForInputs(AbstractContext* ctx,
                             absl::Span<AbstractTensorHandle* const> inputs,
                             std::vector<AbstractTensorHandle*>* params);

using Model = std::function<Status(
    AbstractContext*, absl::Span<AbstractTensorHandle* const>,
    absl::Span<AbstractTensorHandle*>, const GradientRegistry&)>;

Status RunModel(Model model, AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs, bool use_function,
                const GradientRegistry& registry);

Status BuildImmediateExecutionContext(bool use_tfrt, AbstractContext** ctx);

}  // namespace internal
}  // namespace gradients
}  // namespace tensorflow
