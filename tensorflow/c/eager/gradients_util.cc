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
#include "tensorflow/c/eager/gradients_util.h"

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
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace gradients {

using namespace std;

Status ScalarTensorHandleHelper(TFE_Context* ctx, float value,
                                TFE_TensorHandle** result) {
  float data[] = {value};
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_Tensor* t =
      TFE_AllocateHostTensor(ctx, TF_FLOAT, nullptr, 0, status.get());
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status.get());
  *result = th;
  TF_DeleteTensor(t);
  return StatusFromTF_Status(status.get());
}

Status TensorHandleWithDimsFloatHelper(TFE_Context* ctx, float data[],
                                       int64_t dims[], int num_dims,
                                       TFE_TensorHandle** result) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_Tensor* t =
      TFE_AllocateHostTensor(ctx, TF_FLOAT, &dims[0], num_dims, status.get());
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status.get());
  *result = th;
  TF_DeleteTensor(t);
  return StatusFromTF_Status(status.get());
}

Status TensorHandleWithDimsIntHelper(TFE_Context* ctx, int data[],
                                     int64_t dims[], int num_dims,
                                     TFE_TensorHandle** result) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_Tensor* t =
      TFE_AllocateHostTensor(ctx, TF_INT32, &dims[0], num_dims, status.get());
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status.get());
  *result = th;
  TF_DeleteTensor(t);
  return StatusFromTF_Status(status.get());
}

// Get a scalar TensorHandle with given value
Status ScalarTensorHandle(AbstractContext* ctx, float value,
                          AbstractTensorHandle** tensor) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_Context* eager_ctx =
      TF_ExecutionContextGetTFEContext(wrap(ctx), status.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status.get()));
  TFE_TensorHandle* input_eager;
  TF_RETURN_IF_ERROR(ScalarTensorHandleHelper(eager_ctx, value, &input_eager));
  *tensor =
      unwrap(TF_CreateAbstractTensorFromEagerTensor(input_eager, status.get()));
  return StatusFromTF_Status(status.get());
}

// Get a TensorHandle with given float values and dimensions
Status TensorHandleWithDimsFloat(AbstractContext* ctx, float data[],
                                 int64_t dims[], int num_dims,
                                 AbstractTensorHandle** tensor) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_Context* eager_ctx =
      TF_ExecutionContextGetTFEContext(wrap(ctx), status.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status.get()));
  TFE_TensorHandle* input_eager;
  TF_RETURN_IF_ERROR(TensorHandleWithDimsFloatHelper(eager_ctx, data, dims,
                                                     num_dims, &input_eager));
  *tensor =
      unwrap(TF_CreateAbstractTensorFromEagerTensor(input_eager, status.get()));
  return StatusFromTF_Status(status.get());
}

// Get a TensorHandle with given int values and dimensions
Status TensorHandleWithDimsInt(AbstractContext* ctx, int data[], int64_t dims[],
                               int num_dims, AbstractTensorHandle** tensor) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_Context* eager_ctx =
      TF_ExecutionContextGetTFEContext(wrap(ctx), status.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status.get()));
  TFE_TensorHandle* input_eager;
  TF_RETURN_IF_ERROR(TensorHandleWithDimsIntHelper(eager_ctx, data, dims,
                                                   num_dims, &input_eager));
  *tensor =
      unwrap(TF_CreateAbstractTensorFromEagerTensor(input_eager, status.get()));
  return StatusFromTF_Status(status.get());
}

Status GetValue(AbstractTensorHandle* t, TF_Tensor** result_tensor) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_TensorHandle* result_t =
      TF_AbstractTensorGetEagerTensor(wrap(t), status.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status.get()));
  *result_tensor = TFE_TensorHandleResolve(result_t, status.get());
  return StatusFromTF_Status(status.get());
}

AbstractTensorHandlePtr GetTensorHandleUtilFloat(AbstractContext* ctx,
                                                 float vals[], int64_t dims[],
                                                 int num_dims) {
  AbstractTensorHandlePtr A;
  AbstractTensorHandle* a_raw = nullptr;
  Status s = TensorHandleWithDimsFloat(ctx, vals, dims, num_dims, &a_raw);
  if (s.ok()) {
    A.reset(a_raw);
  }
  return A;
}

AbstractTensorHandlePtr GetTensorHandleUtilInt(AbstractContext* ctx, int vals[],
                                               int64_t dims[], int num_dims) {
  AbstractTensorHandlePtr A;
  AbstractTensorHandle* a_raw = nullptr;
  Status s = TensorHandleWithDimsInt(ctx, vals, dims, num_dims, &a_raw);
  if (s.ok()) {
    A.reset(a_raw);
  }
  return A;
}

AbstractTensorHandlePtr GetScalarTensorHandleUtil(AbstractContext* ctx,
                                                  float val) {
  AbstractTensorHandlePtr y;
  AbstractTensorHandle* y_raw = nullptr;
  Status s = ScalarTensorHandle(ctx, val, &y_raw);
  if (s.ok()) {
    y.reset(y_raw);
  }
  return y;
}

Status UpdateWeights(AbstractContext* ctx, vector<AbstractTensorHandle*>& grads,
                     vector<AbstractTensorHandle*>& weights,
                     AbstractTensorHandle* learning_rate) {
  /* Update weights one by one using gradient update rule:
   *
   *    w -= lr*grad[w]
   *
   *  NOTE: assuming learning rate is positive
   */

  int num_grads = grads.size();
  vector<AbstractTensorHandle*> temp_outputs(1);
  std::string update_str;

  // Negate learning rate for gradient descent
  TF_RETURN_IF_ERROR(ops::Neg(ctx, {learning_rate},
                              absl::MakeSpan(temp_outputs),
                              "neg_lr"));  // Compute -lr
  learning_rate = temp_outputs[0];

  for (int i = 0; i < num_grads; i++) {
    // Compute dW = -lr * grad(w[i])
    update_str = "update_mul_" + std::to_string(i);
    TF_RETURN_IF_ERROR(ops::Mul(ctx, {learning_rate, grads[i]},
                                absl::MakeSpan(temp_outputs),
                                update_str.c_str()));

    AbstractTensorHandle* dW = temp_outputs[0];

    // Compute temp = weights[i] + dW
    update_str = "update_add_" + std::to_string(i);
    TF_RETURN_IF_ERROR(ops::Add(ctx, {weights[i], dW},
                                absl::MakeSpan(temp_outputs),
                                update_str.c_str()));

    // Update the weights
    weights[i] = temp_outputs[0];
  }

  return Status::OK();
}

AbstractContext* BuildFunction(const char* fn_name) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_ExecutionContext* graph_ctx = TF_CreateFunction(fn_name, status.get());
  return unwrap(graph_ctx);
}

Status CreateParamsForInputs(AbstractContext* ctx,
                             absl::Span<AbstractTensorHandle* const> inputs,
                             vector<AbstractTensorHandle*>* params) {
  tracing::TracingTensorHandle* handle = nullptr;
  for (auto input : inputs) {
    PartialTensorShape shape;
    TF_RETURN_IF_ERROR(input->Shape(&shape));
    TF_RETURN_IF_ERROR(dyn_cast<tracing::TracingContext>(ctx)->AddParameter(
        input->DataType(), shape, &handle));
    params->emplace_back(handle);
  }
  return Status::OK();
}

Status RunModel(Model model, AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs, bool use_function,
                const GradientRegistry& registry) {
  if (use_function) {
    const char* fn_name = "test_fn";
    std::unique_ptr<AbstractFunction> scoped_func;
    // Returning null tensors from a tf.function is not supported, so we keep
    // track of indices in the model's outputs are nullptr in this set.
    // The FunctionDef only outputs the non-null tensors. We later pad the
    // function op outputs to have nullptrs at the `null_indices`.
    absl::flat_hash_set<int> null_indices;
    {
      AbstractContextPtr func_ctx(BuildFunction(fn_name));
      vector<AbstractTensorHandle*> func_inputs;
      func_inputs.reserve(inputs.size());
      TF_RETURN_IF_ERROR(
          CreateParamsForInputs(func_ctx.get(), inputs, &func_inputs));
      vector<AbstractTensorHandle*> model_outputs;
      model_outputs.resize(outputs.size());
      TF_RETURN_IF_ERROR(model(func_ctx.get(), absl::MakeSpan(func_inputs),
                               absl::MakeSpan(model_outputs), registry));
      for (auto func_input : func_inputs) {
        func_input->Unref();
      }
      AbstractFunction* func = nullptr;
      OutputList output_list;
      output_list.expected_num_outputs = 0;
      output_list.outputs.reserve(outputs.size());
      for (int i = 0; i < model_outputs.size(); i++) {
        if (model_outputs[i]) {
          output_list.outputs.emplace_back(model_outputs[i]);
          output_list.expected_num_outputs += 1;
        } else {
          null_indices.insert(i);
        }
      }
      TF_RETURN_IF_ERROR(dyn_cast<tracing::TracingContext>(func_ctx.get())
                             ->Finalize(&output_list, &func));
      scoped_func.reset(func);
      for (auto output : output_list.outputs) {
        output->Unref();
      }
      TF_RETURN_IF_ERROR(ctx->RegisterFunction(func));
    }

    AbstractOperationPtr fn_op(ctx->CreateOperation());
    TF_RETURN_IF_ERROR(fn_op->Reset(fn_name, /*raw_device_name=*/nullptr));
    for (auto input : inputs) {
      TF_RETURN_IF_ERROR(fn_op->AddInput(input));
    }
    int retvals = outputs.size() - null_indices.size();
    vector<AbstractTensorHandle*> fn_outputs(retvals);
    TF_RETURN_IF_ERROR(fn_op->Execute(
        absl::Span<AbstractTensorHandle*>(fn_outputs.data(), fn_outputs.size()),
        &retvals));
    int skipped_indices = 0;
    for (int i = 0; i < outputs.size(); i++) {
      if (!null_indices.contains(i)) {
        outputs[i] = fn_outputs[i - skipped_indices];
      } else {
        skipped_indices += 1;
      }
    }
    TF_RETURN_IF_ERROR(ctx->RemoveFunction(fn_name));
    return Status::OK();
  } else {
    return model(ctx, inputs, outputs, registry);
  }
}

Status BuildImmediateExecutionContext(bool use_tfrt, AbstractContext** ctx) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetTfrt(opts, use_tfrt);
  *ctx = unwrap(TF_NewEagerExecutionContext(opts, status.get()));
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status.get()));
  TFE_DeleteContextOptions(opts);
  return Status::OK();
}

}  // namespace gradients
}  // namespace tensorflow
