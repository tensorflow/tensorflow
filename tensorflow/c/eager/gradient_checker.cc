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
#include "tensorflow/c/eager/mnist_gradients_util.h"

#include <memory>

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/eager/gradients_internal.h"
#include "tensorflow/c/eager/mnist_gradients_util.h"
#include "tensorflow/c/experimental/gradients/math_grad.h"
#include "tensorflow/c/experimental/gradients/nn_grad.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"


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

// Get a Matrix TensorHandle with given float values and dimensions
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

// Get a Matrix TensorHandle with given int values and dimensions
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

void printArr(auto data [], int n) {
    std::cout.precision(17);
    std::cout<<"[";
    for (int i = 0; i < n-1; i++) {
        std::cout << std::fixed << data[i] << ", ";
    }
    std::cout << std::fixed << data[n-1] << "]"<<std::endl;
}

// Fills out_dims with the dimensions of the given tensor
void GetDims(const TF_Tensor* t, int64_t* out_dims) {

    int num_dims = TF_NumDims(t);
    for (int i = 0; i < num_dims; i++) {
        out_dims[i] = TF_Dim(t, i); 
    }
}

// Fills data with values [start,end) with given step size
void range(int data[], int start, int end, int step = 1) {
    for(int i = start; i < end; i += step) {
        data[i] = i;
    }
}

// ====================================================================


Status GradientCheck(AbstractContext* ctx, Model forward, 
                     std::vector<AbstractTensorHandle*> inputs,
                     int gradIndex,
                     AbstractTensorHandle* dtheta){
    
    float epsilon = 1e-4;
    GradientRegistry registry;

    Status s;
    AbstractTensorHandle* theta = inputs[gradIndex]; // parameter we are grad checking
    
    // Convert from AbstractTensor to TF_Tensor
    TF_Tensor* theta_tensor;
    s = GetValue(theta, &theta_tensor);

    // Get number of elements
    int num_elems = TF_TensorElementCount(theta_tensor);
    
    // Get theta shape
    int num_dims = TF_NumDims(theta_tensor);
    int64_t theta_dims [num_dims];
    GetDims(theta_tensor, theta_dims);
   
    // Initialize data structures
    float thetaPlus_data [num_elems];
    float thetaMinus_data [num_elems];
    float dtheta_approx[num_elems];

    std::vector<AbstractTensorHandle*> sum_inputs(2);
    std::vector<AbstractTensorHandle*> sum_outputs(1);
    std::vector<AbstractTensorHandle*> model_outputs(1);


    // make this a helper function
    int dims_to_sum [num_dims];
    int64_t dims_shape[] = {num_dims};
    range(dims_to_sum, 0, num_dims);
    //printArr(dims_to_sum, num_dims);
    AbstractTensorHandlePtr sum_dims = 
      GetTensorHandleUtilInt(ctx, dims_to_sum, dims_shape, 1);
    
    for (int i = 0; i < num_elems; i++) {
        
        // initialize theta[i] + epsilon
        memcpy(&thetaPlus_data[0], TF_TensorData(theta_tensor),
               TF_TensorByteSize(theta_tensor)); 
        thetaPlus_data[i] += epsilon;
        // std::cout << "thetaP: " <<std::endl;
        // printArr(thetaPlus_data, num_elems);

        AbstractTensorHandlePtr thetaPlus =
          GetTensorHandleUtilFloat(ctx, thetaPlus_data, theta_dims, num_dims);
        
        // initialize theta[i] - epsilon
        memcpy(&thetaMinus_data[0], TF_TensorData(theta_tensor),
               TF_TensorByteSize(theta_tensor)); 
        thetaMinus_data[i] -= epsilon;
        // std::cout << "thetaM: " << std::endl;
        // printArr(thetaMinus_data, num_elems);

        AbstractTensorHandlePtr thetaMinus =
          GetTensorHandleUtilFloat(ctx, thetaMinus_data, theta_dims, num_dims);
                  
        // Get f(theta + eps)
        inputs[gradIndex] = thetaPlus.get();
        
        s = RunModel(forward, ctx, absl::MakeSpan(inputs),
                     absl::MakeSpan(model_outputs),
               /*use_function=*/false, registry);
        
        AbstractTensorHandle* fPlus_toSum = model_outputs[0];
        sum_inputs[0] = fPlus_toSum;
        sum_inputs[1] = sum_dims.get();

        s = ops::Sum(ctx, absl::MakeSpan(sum_inputs), absl::MakeSpan(sum_outputs), "sum_output");

        AbstractTensorHandle* fPlus = sum_outputs[0];     

        // Get f(theta - eps)
        inputs[gradIndex] = thetaMinus.get();
        
        s = RunModel(forward, ctx, absl::MakeSpan(inputs),
                     absl::MakeSpan(model_outputs),
               /*use_function=*/false, registry);
        
        AbstractTensorHandle* fMinus_toSum = model_outputs[0];
        sum_inputs[0] = fMinus_toSum;
        sum_inputs[1] = sum_dims.get();

        s = ops::Sum(ctx, absl::MakeSpan(sum_inputs), absl::MakeSpan(sum_outputs), "sum_output");

        AbstractTensorHandle* fMinus = sum_outputs[0]; 

        // Difference Quotient
        sum_inputs[0] = fPlus;
        sum_inputs[1] = fMinus;

        s = ops::Sub(ctx, absl::MakeSpan(sum_inputs), absl::MakeSpan(sum_outputs), "sub_top");
        AbstractTensorHandle* fDiff = sum_outputs[0];

        TF_Tensor* fDiff_tensor;
        s = GetValue(fDiff, &fDiff_tensor);
       // ASSERT_EQ(errors::OK, s.code()) << s.error_message();

        float fDiff_data[1];
        memcpy(&fDiff_data[0], TF_TensorData(fDiff_tensor),
                TF_TensorByteSize(fDiff_tensor));
        
        float diff = fDiff_data[0];
        float grad_approx = diff / (2.0*epsilon);

        dtheta_approx[i] = grad_approx;
        
    }

    printArr(dtheta_approx, num_elems);

    return Status::OK();
}