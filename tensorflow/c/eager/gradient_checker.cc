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
#include "tensorflow/c/eager/gradient_checker.h"

#include <memory>

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/tf_tensor.h"

namespace tensorflow {
namespace gradients {

using namespace std;

// ================== Helper functions =================

// Fills data with values [start,end) with given step size.
void Range(vector<int32_t>* data, int32_t start, int32_t end,
           int32_t step = 1) {
  for (int32_t i = start; i < end; i += step) {
    (*data)[i] = i;
  }
}

// Fills out_dims with the dimensions of the given tensor.
void GetDims(const TF_Tensor* t, int64_t* out_dims) {
  int num_dims = TF_NumDims(t);
  for (int i = 0; i < num_dims; i++) {
    out_dims[i] = TF_Dim(t, i);
  }
}

// Runs model as is if output is a scalar,
// else sums the output tensor before returning.
Status RunAndMaybeSum(AbstractContext* ctx, Model forward,
                      absl::Span<AbstractTensorHandle* const> inputs,
                      absl::Span<AbstractTensorHandle*> outputs,
                      bool use_function) {
  AbstractTensorHandle* model_outputs[1];

  // Run the model.
  TF_RETURN_IF_ERROR(
      RunModel(forward, ctx, inputs, model_outputs, use_function));
  AbstractTensorHandlePtr model_out(model_outputs[0]);

  TF_Tensor* model_out_tensor;
  TF_RETURN_IF_ERROR(GetValue(model_out.get(), &model_out_tensor));
  int num_dims_out = TF_NumDims(model_out_tensor);
  TF_DeleteTensor(model_out_tensor);

  // If the output is a scalar, then return the scalar output
  if (num_dims_out == 0) {
    outputs[0] = model_out.release();
    return Status::OK();
  }

  // Else, reduce sum the output to get a scalar

  // Will sum all dimensions, so get a Tensor containing [0,...,num_dims_out-1].
  AbstractTensorHandlePtr sum_dims;
  {
    vector<int32_t> vals(num_dims_out);
    int64_t vals_shape[] = {num_dims_out};
    Range(&vals, 0, num_dims_out);
    AbstractTensorHandle* sum_dims_raw = nullptr;
    TF_RETURN_IF_ERROR(TestTensorHandleWithDims<int32_t, TF_INT32>(
        ctx, vals.data(), vals_shape, 1, &sum_dims_raw));
    sum_dims.reset(sum_dims_raw);
  }

  // Reduce sum the output on all dimensions.
  TF_RETURN_IF_ERROR(ops::Sum(ctx, model_out.get(), sum_dims.get(), &outputs[0],
                              /*keep_dims=*/false, "sum_output"));
  return Status::OK();
}
// ========================= End Helper Functions==============================

Status CalcNumericalGrad(AbstractContext* ctx, Model forward,
                         absl::Span<AbstractTensorHandle* const> inputs,
                         int input_index, bool use_function,
                         AbstractTensorHandle** numerical_grad) {
  vector<AbstractTensorHandle*> theta_inputs(inputs.size());
  for (int i{}; i < inputs.size(); ++i) {
    theta_inputs[i] = inputs[i];
  }

  AbstractTensorHandle* theta =
      theta_inputs[input_index];  // parameter we are grad checking

  // Convert from AbstractTensor to TF_Tensor.
  TF_Tensor* theta_tensor;
  TF_RETURN_IF_ERROR(GetValue(theta, &theta_tensor));

  // Get number of elements and fill data.
  int num_elems = TF_TensorElementCount(theta_tensor);
  vector<float> theta_data(num_elems);
  memcpy(theta_data.data(), TF_TensorData(theta_tensor),
         TF_TensorByteSize(theta_tensor));

  // Initialize space for the numerical gradient.
  vector<float> dtheta_approx(num_elems);

  // Get theta shape and store in theta_dims.
  int num_dims = TF_NumDims(theta_tensor);
  vector<int64_t> theta_dims(num_dims);
  GetDims(theta_tensor, theta_dims.data());

  // Initialize auxilary data structures.
  vector<float> thetaPlus_data(num_elems);
  vector<float> thetaMinus_data(num_elems);
  AbstractTensorHandle* f_outputs[1];

  // Numerical Grad Check
  for (int i = 0; i < num_elems; i++) {
    // Get relative epsilon value
    float epsilon = theta_data[i] == 0 ? 1e-4 : std::abs(theta_data[i] * 1e-4);
    AbstractTensorHandlePtr two_eps;
    {
      AbstractTensorHandle* two_eps_raw = nullptr;
      TF_RETURN_IF_ERROR(TestScalarTensorHandle<float, TF_FLOAT>(
          ctx, 2 * epsilon, &two_eps_raw));
      two_eps.reset(two_eps_raw);
    }

    // Initialize theta[i] + epsilon.
    memcpy(thetaPlus_data.data(), TF_TensorData(theta_tensor),
           TF_TensorByteSize(theta_tensor));
    thetaPlus_data[i] += epsilon;
    AbstractTensorHandlePtr thetaPlus;
    {
      AbstractTensorHandle* thetaPlus_raw = nullptr;
      TF_RETURN_IF_ERROR(TestTensorHandleWithDims<float, TF_FLOAT>(
          ctx, thetaPlus_data.data(), theta_dims.data(), num_dims,
          &thetaPlus_raw));
      thetaPlus.reset(thetaPlus_raw);
    }

    // Initialize theta[i] - epsilon.
    memcpy(&thetaMinus_data[0], TF_TensorData(theta_tensor),
           TF_TensorByteSize(theta_tensor));
    thetaMinus_data[i] -= epsilon;
    AbstractTensorHandlePtr thetaMinus;
    {
      AbstractTensorHandle* thetaMinus_raw = nullptr;
      TF_RETURN_IF_ERROR(TestTensorHandleWithDims<float, TF_FLOAT>(
          ctx, thetaMinus_data.data(), theta_dims.data(), num_dims,
          &thetaMinus_raw));
      thetaMinus.reset(thetaMinus_raw);
    }

    // Get f(theta + eps):
    theta_inputs[input_index] = thetaPlus.get();
    TF_RETURN_IF_ERROR(
        RunAndMaybeSum(ctx, forward, theta_inputs, f_outputs, use_function));
    AbstractTensorHandlePtr fPlus(f_outputs[0]);

    // Get f(theta - eps):
    theta_inputs[input_index] = thetaMinus.get();
    TF_RETURN_IF_ERROR(
        RunAndMaybeSum(ctx, forward, theta_inputs, f_outputs, use_function));
    AbstractTensorHandlePtr fMinus(f_outputs[0]);

    // Take Difference of both estimates: (f(theta + eps) - f(theta - eps)).
    TF_RETURN_IF_ERROR(
        ops::Sub(ctx, fPlus.get(), fMinus.get(), f_outputs, "sub_top"));
    AbstractTensorHandlePtr fDiff(f_outputs[0]);

    // Calculate using the difference quotient definition:
    // (f(theta + eps) - f(theta - eps)) / (2 * eps).
    TF_RETURN_IF_ERROR(
        ops::Div(ctx, fDiff.get(), two_eps.get(), f_outputs, "diff_quotient"));
    AbstractTensorHandlePtr diff_quotient(f_outputs[0]);

    TF_Tensor* grad_tensor;
    TF_RETURN_IF_ERROR(GetValue(diff_quotient.get(), &grad_tensor));
    float grad_data[1];
    memcpy(&grad_data[0], TF_TensorData(grad_tensor),
           TF_TensorByteSize(grad_tensor));
    TF_DeleteTensor(grad_tensor);
    dtheta_approx[i] = grad_data[0];
  }

  // Populate *numerical_grad with the data from dtheta_approx.
  TF_RETURN_IF_ERROR(TestTensorHandleWithDims<float, TF_FLOAT>(
      ctx, dtheta_approx.data(), theta_dims.data(), num_dims, numerical_grad));
  TF_DeleteTensor(theta_tensor);
  return Status::OK();
}

}  // namespace gradients
}  // namespace tensorflow
