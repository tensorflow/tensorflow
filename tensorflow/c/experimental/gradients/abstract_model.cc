/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
#include "tensorflow/c/experimental/gradients/abstract_model.h"

#include "tensorflow/c/experimental/gradients/tape/tape_context.h"
#include "tensorflow/c/experimental/ops/math_ops.h"

namespace tensorflow {
namespace gradients {
namespace {

Status MeanN(AbstractContext* ctx,
             absl::Span<AbstractTensorHandle* const> inputs,
             AbstractTensorHandle** output) {
  std::vector<AbstractTensorHandle*> temp_outputs(1);
  TF_RETURN_IF_ERROR(ops::AddN(ctx, inputs, absl::MakeSpan(temp_outputs)));
  auto temp_input = temp_outputs[0];
  AbstractTensorHandlePtr size;
  {
    AbstractTensorHandle* size_raw = nullptr;
    TF_RETURN_IF_ERROR(
        TestScalarTensorHandle<float, TF_FLOAT>(ctx, inputs.size(), &size_raw));
    size.reset(size_raw);
  }
  TF_RETURN_IF_ERROR(ops::DivNoNan(ctx, {temp_input, size.get()},
                                   absl::MakeSpan(temp_outputs),
                                   "AbstractModel_MeanN_Div"));
  temp_input->Unref();
  *output = temp_outputs[0];
  return Status::OK();
}

class TapeHolder {
 public:
  TapeHolder(Tape** tape) : tape_(tape) {}
  ~TapeHolder() { *tape_ = nullptr; }

 private:
  Tape** tape_;
};

}  // namespace

AbstractModel::AbstractModel(AbstractContext* ctx, Model loss_fn,
                             Model metric_fn, Model optimizer_fn,
                             GradientRegistry registry, bool use_function)
    : ctx_(ctx),
      loss_fn_(std::move(loss_fn)),
      metric_fn_(std::move(metric_fn)),
      optimizer_fn_(std::move(optimizer_fn)),
      registry_(std::move(registry)),
      use_function_(use_function) {}

AbstractModel::~AbstractModel() {
  for (auto weight : weights_) {
    weight->Unref();
  }
}

Status AbstractModel::Fit(absl::Span<AbstractTensorHandle* const> x_train,
                          absl::Span<AbstractTensorHandle* const> y_train,
                          size_t epoch) {
  static Model loss_func = [this](
                               AbstractContext* ctx,
                               absl::Span<AbstractTensorHandle* const> inputs,
                               absl::Span<AbstractTensorHandle*> outputs) {
    std::vector<AbstractTensorHandle*> temp_outputs(1);
    TF_RETURN_IF_ERROR(this->Compute(ctx, inputs[0], &temp_outputs[0]));
    TF_RETURN_IF_ERROR(
        this->loss_fn_(ctx, {temp_outputs[0], inputs[1]}, outputs));
    temp_outputs[0]->Unref();
    return Status::OK();
  };
  TapeHolder tape_holder(&this->tape_);

  for (size_t e{}; e < epoch; ++e) {
    for (size_t i{}; i < x_train.size(); ++i) {
      Tape tape(/*persistent=*/false);
      tape_ = &tape;
      for (size_t j{}; j < weights_.size(); ++j) {
        tape.Watch(weights_[j]);
      }

      AbstractContextPtr tape_ctx(new TapeContext(ctx_, &tape, registry_));
      std::vector<AbstractTensorHandle*> temp_outputs(1);

      // Compute model loss.
      TF_RETURN_IF_ERROR(RunModel(loss_func, tape_ctx.get(),
                                  {x_train[i], y_train[i]},
                                  absl::MakeSpan(temp_outputs), use_function_));
      auto temp_input = temp_outputs[0];

      // Compute gradients.
      temp_outputs.resize(weights_.size());
      TF_RETURN_IF_ERROR(tape.ComputeGradient(ctx_,
                                              /*targets=loss*/ {temp_input},
                                              /*sources=*/weights_,
                                              /*output_gradients=*/{},
                                              absl::MakeSpan(temp_outputs)));
      temp_input->Unref();

      // Update `weights` by optimizer.
      auto updated_weights = weights_;
      TF_RETURN_IF_ERROR(
          optimizer_fn_(ctx_, temp_outputs, absl::MakeSpan(updated_weights)));
      for (size_t j{}; j < weights_.size(); ++j) {
        weights_[j]->Unref();
        weights_[j] = updated_weights[j];
      }
    }
  }

  return Status::OK();
}

Status AbstractModel::Predict(absl::Span<AbstractTensorHandle* const> inputs,
                              absl::Span<AbstractTensorHandle*> outputs) {
  for (size_t i{}; i < inputs.size(); ++i) {
    TF_RETURN_IF_ERROR((*this)(ctx_, inputs[i], &outputs[i]));
  }
  return Status::OK();
}

Status AbstractModel::Evaluate(absl::Span<AbstractTensorHandle* const> x_test,
                               absl::Span<AbstractTensorHandle* const> y_test,
                               absl::Span<AbstractTensorHandle*> outputs) {
  std::vector<AbstractTensorHandle*> accuracies(x_test.size());
  auto acc_span = absl::MakeSpan(accuracies);

  for (size_t i{}; i < x_test.size(); ++i) {
    AbstractTensorHandle* temp_output;
    TF_RETURN_IF_ERROR((*this)(ctx_, x_test[i], &temp_output));
    TF_RETURN_IF_ERROR(
        metric_fn_(ctx_, {temp_output, y_test[i]}, acc_span.subspan(i)));
    temp_output->Unref();
  }

  TF_RETURN_IF_ERROR(MeanN(ctx_, accuracies, &outputs[0]));
  for (auto acc : accuracies) {
    acc->Unref();
  }
  return Status::OK();
}

}  // namespace gradients
}  // namespace tensorflow
