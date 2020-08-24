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
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/eager/gradients_internal.h"
#include "tensorflow/c/eager/mnist_gradients_testutil.h"
#include "tensorflow/c/experimental/gradients/math_grad.h"
#include "tensorflow/c/experimental/gradients/nn_grad.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

using namespace std;

// ================== TensorHandle generating functions =================

// Get a scalar TensorHandle with given value
Status TestScalarTensorHandle(AbstractContext* ctx, float value,
                              AbstractTensorHandle** tensor) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_Context* eager_ctx =
      TF_ExecutionContextGetTFEContext(wrap(ctx), status.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status.get()));
  TFE_TensorHandle* input_eager = TestScalarTensorHandle(eager_ctx, value);
  *tensor =
      unwrap(TF_CreateAbstractTensorFromEagerTensor(input_eager, status.get()));
  return Status::OK();
}

// Get a TensorHandle with given float values and dimensions
Status TestTensorHandleWithDimsFloat(AbstractContext* ctx, float data[],
                                     int64_t dims[], int num_dims,
                                     AbstractTensorHandle** tensor) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_Context* eager_ctx =
      TF_ExecutionContextGetTFEContext(wrap(ctx), status.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status.get()));
  TFE_TensorHandle* input_eager =
      TestTensorHandleWithDimsFloat(eager_ctx, data, dims, num_dims);
  *tensor =
      unwrap(TF_CreateAbstractTensorFromEagerTensor(input_eager, status.get()));
  return Status::OK();
}

// Get a TensorHandle with given int values and dimensions
Status TestTensorHandleWithDimsInt(AbstractContext* ctx, int data[],
                                   int64_t dims[], int num_dims,
                                   AbstractTensorHandle** tensor) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_Context* eager_ctx =
      TF_ExecutionContextGetTFEContext(wrap(ctx), status.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status.get()));
  TFE_TensorHandle* input_eager =
      TestTensorHandleWithDimsInt(eager_ctx, data, dims, num_dims);
  *tensor =
      unwrap(TF_CreateAbstractTensorFromEagerTensor(input_eager, status.get()));
  return Status::OK();
}

Status GetValue(AbstractTensorHandle* t, TF_Tensor** result_tensor) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_TensorHandle* result_t =
      TF_AbstractTensorGetEagerTensor(wrap(t), status.get());
  TF_RETURN_IF_ERROR(StatusFromTF_Status(status.get()));
  *result_tensor = TFE_TensorHandleResolve(result_t, status.get());
  return Status::OK();
}

AbstractTensorHandlePtr GetTensorHandleUtilFloat(AbstractContext* ctx,
                                                 float vals[], int64_t dims[],
                                                 int num_dims) {
  AbstractTensorHandlePtr A;
  AbstractTensorHandle* a_raw = nullptr;
  Status s = TestTensorHandleWithDimsFloat(ctx, vals, dims, num_dims, &a_raw);
  A.reset(a_raw);
  return A;
}

AbstractTensorHandlePtr GetTensorHandleUtilInt(AbstractContext* ctx, int vals[],
                                               int64_t dims[], int num_dims) {
  AbstractTensorHandlePtr A;
  AbstractTensorHandle* a_raw = nullptr;
  Status s = TestTensorHandleWithDimsInt(ctx, vals, dims, num_dims, &a_raw);
  A.reset(a_raw);
  return A;
}

// Fills data with values [start,end) with given step size.
void Range(int data[], int start, int end, int step = 1) {
  for (int i = start; i < end; i += step) {
    data[i] = i;
  }
}

// Returns AbstractTensorHandlePtr containing [0, ..., n-1].
AbstractTensorHandlePtr GetRangeTensorHandleUtil(AbstractContext* ctx, int n) {
  int vals[n];
  int64_t vals_shape[] = {n};
  Range(vals, 0, n);
  AbstractTensorHandlePtr r = GetTensorHandleUtilInt(ctx, vals, vals_shape, 1);
  return r;
}

// Fills out_dims with the dimensions of the given tensor.
void GetDims(const TF_Tensor* t, int64_t* out_dims) {
  int num_dims = TF_NumDims(t);
  for (int i = 0; i < num_dims; i++) {
    out_dims[i] = TF_Dim(t, i);
  }
}

// Runs Model and reduce_sums the output.
Status RunModelAndSum(AbstractContext* ctx, Model forward,
                      absl::Span<AbstractTensorHandle*> inputs,
                      absl::Span<AbstractTensorHandle*> outputs,
                      bool use_function) {
  GradientRegistry registry;
  std::vector<AbstractTensorHandle*> model_outputs(1);

  // Run the model.
  Status s = RunModel(forward, ctx, inputs, absl::MakeSpan(model_outputs),
                      use_function, registry);
  AbstractTensorHandle* f_toSum = model_outputs[0];

  TF_Tensor* model_out_tensor;
  s = GetValue(f_toSum, &model_out_tensor);
  int num_dims_out = TF_NumDims(model_out_tensor);

  // Will sum all dimensions, so get a Tensor containing [0,...,num_dims_out-1].
  AbstractTensorHandlePtr sum_dims =
      GetRangeTensorHandleUtil(ctx, num_dims_out);

  // Reduce sum the output on all dimensions.
  std::vector<AbstractTensorHandle*> sum_inputs(2);
  sum_inputs[0] = f_toSum;
  sum_inputs[1] = sum_dims.get();

  s = ops::Sum(ctx, absl::MakeSpan(sum_inputs), absl::MakeSpan(model_outputs),
               "sum_output");
  outputs[0] = model_outputs[0];
  return Status::OK();
}

// Runs model as is if output is a scalar,
// else sums the output tensor before returning.
Status RunAndMaybeSum(AbstractContext* ctx, Model forward,
                      absl::Span<AbstractTensorHandle*> inputs,
                      absl::Span<AbstractTensorHandle*> outputs,
                      bool use_function, bool is_scalar_out) {
  Status s;
  if (is_scalar_out) {
    GradientRegistry registry;
    s = RunModel(forward, ctx, inputs, outputs, use_function, registry);
  } else {
    s = RunModelAndSum(ctx, forward, inputs, outputs, use_function);
  }
  return Status::OK();
}
// ========================= End Util Functions==============================

Status GradientCheck(AbstractContext* ctx, Model forward,
                     std::vector<AbstractTensorHandle*> inputs,
                     float* dtheta_approx, int gradIndex, bool use_function,
                     bool is_scalar_out) {
  Status s;
  AbstractTensorHandle* theta =
      inputs[gradIndex];  // parameter we are grad checking

  // Convert from AbstractTensor to TF_Tensor.
  TF_Tensor* theta_tensor;
  s = GetValue(theta, &theta_tensor);

  // Get number of elements and fill data.
  int num_elems = TF_TensorElementCount(theta_tensor);
  float theta_data[num_elems] = {0};
  memcpy(&theta_data[0], TF_TensorData(theta_tensor),
         TF_TensorByteSize(theta_tensor));

  // Get theta shape and store in theta_dims
  int num_dims = TF_NumDims(theta_tensor);
  int64_t theta_dims[num_dims];
  GetDims(theta_tensor, theta_dims);

  // Initialize data structures
  float thetaPlus_data[num_elems];
  float thetaMinus_data[num_elems];
  std::vector<AbstractTensorHandle*> f_outputs(1);

  // Numerical Grad Check
  for (int i = 0; i < num_elems; i++) {
    // Get relative epsilon value
    float epsilon =
        std::abs(theta_data[i] * 1e-4 + 1e-4);  // add 1e-4 to prevent div by 0

    // Initialize theta[i] + epsilon.
    memcpy(&thetaPlus_data[0], TF_TensorData(theta_tensor),
           TF_TensorByteSize(theta_tensor));
    thetaPlus_data[i] += epsilon;
    AbstractTensorHandlePtr thetaPlus =
        GetTensorHandleUtilFloat(ctx, thetaPlus_data, theta_dims, num_dims);

    // Initialize theta[i] - epsilon.
    memcpy(&thetaMinus_data[0], TF_TensorData(theta_tensor),
           TF_TensorByteSize(theta_tensor));
    thetaMinus_data[i] -= epsilon;
    AbstractTensorHandlePtr thetaMinus =
        GetTensorHandleUtilFloat(ctx, thetaMinus_data, theta_dims, num_dims);

    // Get f(theta + eps):
    inputs[gradIndex] = thetaPlus.get();
    s = RunAndMaybeSum(ctx, forward, absl::MakeSpan(inputs),
                       absl::MakeSpan(f_outputs), use_function, is_scalar_out);
    AbstractTensorHandle* fPlus = f_outputs[0];

    // Get f(theta - eps):
    inputs[gradIndex] = thetaMinus.get();
    s = RunAndMaybeSum(ctx, forward, absl::MakeSpan(inputs),
                       absl::MakeSpan(f_outputs), use_function, is_scalar_out);
    AbstractTensorHandle* fMinus = f_outputs[0];

    // Take Difference of both estimates: (f(x + eps) - f(x - eps)).
    s = ops::Sub(ctx, {fPlus, fMinus}, absl::MakeSpan(f_outputs), "sub_top");
    AbstractTensorHandle* fDiff = f_outputs[0];

    // Get difference value for calculation.
    TF_Tensor* fDiff_tensor;
    s = GetValue(fDiff, &fDiff_tensor);
    float fDiff_data[1];
    memcpy(&fDiff_data[0], TF_TensorData(fDiff_tensor),
           TF_TensorByteSize(fDiff_tensor));

    // Calculate using the difference quotient definition:
    // (f(x + eps) - f(x - eps)) / (2 * eps).
    float grad_approx = fDiff_data[0] / (2.0 * epsilon);
    dtheta_approx[i] = grad_approx;
  }

  return Status::OK();
}
