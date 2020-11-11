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
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/eager/gradients_internal.h"
#include "tensorflow/c/experimental/gradients/math_grad.h"
#include "tensorflow/c/experimental/gradients/nn_grad.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace gradients {

using namespace std;

// ================== Helper functions =================

// Fills data with values [start,end) with given step size.
void Range(vector<int>* data, int start, int end, int step = 1) {
  for (int i = start; i < end; i += step) {
    (*data)[i] = i;
  }
}

// Returns AbstractTensorHandlePtr containing [0, ..., n-1].
AbstractTensorHandlePtr GetRangeTensorHandleUtil(AbstractContext* ctx, int n) {
  vector<int> vals(n);
  int64_t vals_shape[] = {n};
  Range(&vals, 0, n);
  AbstractTensorHandlePtr r =
      GetTensorHandleUtilInt(ctx, vals.data(), vals_shape, 1);
  return r;
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
                      absl::Span<AbstractTensorHandle*> inputs,
                      absl::Span<AbstractTensorHandle*> outputs,
                      bool use_function) {
  GradientRegistry registry;
  std::vector<AbstractTensorHandle*> model_outputs(1);

  // Run the model.
  TF_RETURN_IF_ERROR(RunModel(forward, ctx, inputs,
                              absl::MakeSpan(model_outputs), use_function,
                              registry));
  AbstractTensorHandle* model_out = model_outputs[0];

  TF_Tensor* model_out_tensor;
  TF_RETURN_IF_ERROR(GetValue(model_out, &model_out_tensor));
  int num_dims_out = TF_NumDims(model_out_tensor);

  // If the output is a scalar, then return the scalar output
  if (num_dims_out == 0) {
    outputs[0] = model_out;
    return Status::OK();
  }

  // Else, reduce sum the output to get a scalar

  // Will sum all dimensions, so get a Tensor containing [0,...,num_dims_out-1].
  AbstractTensorHandlePtr sum_dims =
      GetRangeTensorHandleUtil(ctx, num_dims_out);

  // Reduce sum the output on all dimensions.
  std::vector<AbstractTensorHandle*> sum_inputs(2);
  sum_inputs[0] = model_out;
  sum_inputs[1] = sum_dims.get();

  TF_RETURN_IF_ERROR(ops::Sum(ctx, absl::MakeSpan(sum_inputs),
                              absl::MakeSpan(model_outputs), "sum_output"));
  outputs[0] = model_outputs[0];
  return Status::OK();
}
// ========================= End Helper Functions==============================

Status CalcNumericalGrad(AbstractContext* ctx, Model forward,
                         absl::Span<AbstractTensorHandle*> inputs,
                         int input_index, bool use_function,
                         AbstractTensorHandle** numerical_grad) {
  AbstractTensorHandle* theta =
      inputs[input_index];  // parameter we are grad checking

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
  std::vector<AbstractTensorHandle*> f_outputs(1);

  // Numerical Grad Check
  for (int i = 0; i < num_elems; i++) {
    // Get relative epsilon value
    float epsilon =
        std::abs(theta_data[i] * 1e-4 + 1e-4);  // add 1e-4 to prevent div by 0
    AbstractTensorHandlePtr two_eps =
        GetScalarTensorHandleUtil(ctx, 2 * epsilon);

    // Initialize theta[i] + epsilon.
    memcpy(thetaPlus_data.data(), TF_TensorData(theta_tensor),
           TF_TensorByteSize(theta_tensor));
    thetaPlus_data[i] += epsilon;
    AbstractTensorHandlePtr thetaPlus = GetTensorHandleUtilFloat(
        ctx, thetaPlus_data.data(), theta_dims.data(), num_dims);

    // Initialize theta[i] - epsilon.
    memcpy(&thetaMinus_data[0], TF_TensorData(theta_tensor),
           TF_TensorByteSize(theta_tensor));
    thetaMinus_data[i] -= epsilon;
    AbstractTensorHandlePtr thetaMinus = GetTensorHandleUtilFloat(
        ctx, thetaMinus_data.data(), theta_dims.data(), num_dims);

    // Get f(theta + eps):
    inputs[input_index] = thetaPlus.get();
    TF_RETURN_IF_ERROR(RunAndMaybeSum(ctx, forward, inputs,
                                      absl::MakeSpan(f_outputs), use_function));
    AbstractTensorHandle* fPlus = f_outputs[0];

    // Get f(theta - eps):
    inputs[input_index] = thetaMinus.get();
    TF_RETURN_IF_ERROR(RunAndMaybeSum(ctx, forward, inputs,
                                      absl::MakeSpan(f_outputs), use_function));
    AbstractTensorHandle* fMinus = f_outputs[0];

    // Take Difference of both estimates: (f(theta + eps) - f(theta - eps)).
    TF_RETURN_IF_ERROR(
        ops::Sub(ctx, {fPlus, fMinus}, absl::MakeSpan(f_outputs), "sub_top"));
    AbstractTensorHandle* fDiff = f_outputs[0];

    // Calculate using the difference quotient definition:
    // (f(theta + eps) - f(theta - eps)) / (2 * eps).
    TF_RETURN_IF_ERROR(ops::DivNoNan(ctx, {fDiff, two_eps.get()},
                                     absl::MakeSpan(f_outputs),
                                     "diff_quotient"));
    AbstractTensorHandle* diff_quotient = f_outputs[0];

    TF_Tensor* grad_tensor;
    TF_RETURN_IF_ERROR(GetValue(diff_quotient, &grad_tensor));
    float grad_data[1];
    memcpy(&grad_data[0], TF_TensorData(grad_tensor),
           TF_TensorByteSize(grad_tensor));

    dtheta_approx[i] = grad_data[0];
  }

  // Populate *numerical_grad with the data from dtheta_approx.
  TF_RETURN_IF_ERROR(TensorHandleWithDimsFloat(
      ctx, dtheta_approx.data(), theta_dims.data(), num_dims, numerical_grad));
  return Status::OK();
}

}  // namespace gradients
}  // namespace tensorflow
