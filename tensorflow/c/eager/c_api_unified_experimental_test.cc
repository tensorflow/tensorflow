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

#include "tensorflow/c/eager/c_api_unified_experimental.h"

#include <memory>

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/platform/test.h"

using tensorflow::string;

namespace tensorflow {
namespace {

class UnifiedCAPI : public ::testing::TestWithParam<const char*> {
 protected:
  void SetUp() override { TF_SetTracingImplementation(GetParam()); }
};

TEST_P(UnifiedCAPI, TestBasicEager) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TF_ExecutionContext* ctx = TF_NewEagerExecutionContext(opts, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TFE_DeleteContextOptions(opts);

  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build an abstract input tensor.
  TFE_Context* eager_ctx = TF_ExecutionContextGetTFEContext(ctx);
  TFE_TensorHandle* t = TestScalarTensorHandle(eager_ctx, 2.0f);
  TF_AbstractTensor* at =
      TF_CreateAbstractTensorFromEagerTensor(t, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build an abstract operation.
  auto* op = TF_NewAbstractOp(ctx);
  TF_AbstractOpSetOpType(op, "Add", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build inputs and outputs.
  TF_AbstractTensor* inputs[2] = {at, at};
  TF_OutputList* o = TF_NewOutputList();
  TF_OutputListSetNumOutputs(o, 1, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Execute.
  TF_ExecuteOperation(op, 2, inputs, o, ctx, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Clean up operation and inputs.
  TF_DeleteAbstractOp(op);
  TF_DeleteAbstractTensor(at);

  // Verify the results.
  ASSERT_EQ(1, TF_OutputListNumOutputs(o));
  TF_AbstractTensor* result = TF_OutputListGet(o, 0);
  TFE_TensorHandle* result_t =
      TF_AbstractTensorGetEagerTensor(result, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_Tensor* result_tensor = TFE_TensorHandleResolve(result_t, status.get());
  float* result_value = static_cast<float*>(TF_TensorData(result_tensor));
  EXPECT_EQ(*result_value, 4.0);

  TF_DeleteTensor(result_tensor);
  TF_DeleteAbstractTensor(result);
  TF_DeleteOutputList(o);
  TF_DeleteExecutionContext(ctx);
}


// MatMul Test
TEST_P(UnifiedCAPI, TestBasicEagerMatMul) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TF_ExecutionContext* ctx = TF_NewEagerExecutionContext(opts, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TFE_DeleteContextOptions(opts);

  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  /* Want to test simple MatMul example: 

    [ [0,0] ,   *   [ [0,0] ,   =   [ [0,0],
      [0,0] ]         [0,0] ]         [0,0] ]

  */

  // Build an abstract input tensor.
  int64_t dims [] = {2,2}; // Matrices will be 2 x 2
  int num_dims = sizeof(dims)/sizeof(dims[0]); 

  float vals [] = {0.0f,0.0f,0.0f,0.0f};
  TFE_Context* eager_ctx = TF_ExecutionContextGetTFEContext(ctx);
  TFE_TensorHandle* t = TestMatrixTensorHandleWithInput(eager_ctx, vals, dims,num_dims); //, dims[0],dims[1]);
  
  TF_AbstractTensor* at =
      TF_CreateAbstractTensorFromEagerTensor(t, status.get()); // get abstract tensor
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build an abstract operation.
  auto* op = TF_NewAbstractOp(ctx);
  TF_AbstractOpSetOpType(op, "MatMul", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build inputs and outputs.
  TF_AbstractTensor* inputs[2] = {at, at};
  TF_OutputList* o = TF_NewOutputList();
  TF_OutputListSetNumOutputs(o, 1, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Execute.
  TF_ExecuteOperation(op, 2, inputs, o, ctx, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Clean up operation and inputs.
  TF_DeleteAbstractOp(op);
  TF_DeleteAbstractTensor(at);

  // Verify the results.
  ASSERT_EQ(1, TF_OutputListNumOutputs(o));
  TF_AbstractTensor* result = TF_OutputListGet(o, 0);
  TFE_TensorHandle* result_t =
      TF_AbstractTensorGetEagerTensor(result, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_Tensor* result_tensor = TFE_TensorHandleResolve(result_t, status.get());
  
  // Copy Tensor data into an array. 
  float result_data[4] = {0};
  memcpy(&result_data[0], TF_TensorData(result_tensor), TF_TensorByteSize(result_tensor));

  int data_len = 4; // length of result_data
  for(int i = 0; i < data_len; i++){
      EXPECT_EQ(result_data[i], 0);
  }

  TF_DeleteTensor(result_tensor);
  TF_DeleteAbstractTensor(result);
  TF_DeleteOutputList(o);
  TF_DeleteExecutionContext(ctx);
}


// MatMul Test 2 
TEST_P(UnifiedCAPI, TestBasicEagerMatMul2) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TF_ExecutionContext* ctx = TF_NewEagerExecutionContext(opts, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TFE_DeleteContextOptions(opts);

  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  /* Want to test simple MatMul example with abstract tensors: 

    [ [1,2] ,   *   [ [5,6] ,   =   [ [19,22],
      [3,4] ]         [7,8] ]         [43,50] ]

  */

  // Build 1st Matrix.
  int64_t dims [] = {2,2}; // Matrices will be 2 x 2   
  int num_dims = sizeof(dims)/sizeof(dims[0]);

  float vals1 [] = {1.0f,2.0f,3.0f,4.0f};
  TFE_Context* eager_ctx = TF_ExecutionContextGetTFEContext(ctx);
  TFE_TensorHandle* t1 = TestMatrixTensorHandleWithInput(eager_ctx, vals1, dims, num_dims);
  
  TF_AbstractTensor* at1 =
      TF_CreateAbstractTensorFromEagerTensor(t1, status.get()); // get abstract tensor
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build 2nd Matrix.
  float vals2 [] = {5.0f,6.0f,7.0f,8.0f};
  TFE_TensorHandle* t2 = TestMatrixTensorHandleWithInput(eager_ctx, vals2, dims, num_dims);
  
  TF_AbstractTensor* at2 =
      TF_CreateAbstractTensorFromEagerTensor(t2, status.get()); // get abstract tensor
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build an abstract operation.
  auto* op = TF_NewAbstractOp(ctx);
  TF_AbstractOpSetOpType(op, "MatMul", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build inputs and outputs.
  TF_AbstractTensor* inputs[2] = {at1, at2};
  TF_OutputList* o = TF_NewOutputList();
  TF_OutputListSetNumOutputs(o, 1, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Execute.
  TF_ExecuteOperation(op, 2, inputs, o, ctx, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Clean up operation and inputs.
  TF_DeleteAbstractOp(op);
  TF_DeleteAbstractTensor(at1);
  TF_DeleteAbstractTensor(at2);

  // Verify the results.
  ASSERT_EQ(1, TF_OutputListNumOutputs(o));
  TF_AbstractTensor* result = TF_OutputListGet(o, 0);
  TFE_TensorHandle* result_t =
      TF_AbstractTensorGetEagerTensor(result, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  
  TF_Tensor* result_tensor = TFE_TensorHandleResolve(result_t, status.get());

  // Copy Tensor data into array.
  float result_data[4] = {0};
  memcpy(&result_data[0], TF_TensorData(result_tensor), TF_TensorByteSize(result_tensor));

  // Build expected result & verify.
  float e_vals [] = {19.0f,22.0f,43.0f,50.0f};

  int data_len = 4; // length of e_vals
  for(int i = 0; i < data_len; i++){ 
    EXPECT_EQ(result_data[i], e_vals[i]);
  }

  TF_DeleteTensor(result_tensor);
  TF_DeleteAbstractTensor(result);
  TF_DeleteOutputList(o);
  TF_DeleteExecutionContext(ctx);
}


TEST_P(UnifiedCAPI, TestBasicGraph) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  
  // Start a new function / execution context.
  string fn_name = "double";
  TF_ExecutionContext* graph_ctx =
      TF_CreateFunction(fn_name.c_str(), status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  auto* placeholder_t =
      TF_AddFunctionParameter(graph_ctx, TF_FLOAT, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build an abstract operation.
  auto* add_op = TF_NewAbstractOp(graph_ctx);
  TF_AbstractOpSetOpType(add_op, "Add", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_AbstractOpSetOpName(add_op, "my_add", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build inputs and outputs.
  TF_AbstractTensor* inputs[2] = {placeholder_t, placeholder_t};
  TF_OutputList* add_outputs = TF_NewOutputList();

  // Execute.
  TF_ExecuteOperation(add_op, 2, inputs, add_outputs, graph_ctx, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Clean up operation and inputs.
  TF_DeleteAbstractOp(add_op);

  TF_AbstractFunction* func =
      TF_FinalizeFunction(graph_ctx, add_outputs, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build eager context.
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TF_ExecutionContext* eager_execution_ctx =
      TF_NewEagerExecutionContext(opts, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TFE_DeleteContextOptions(opts);

  TF_ExecutionContextRegisterFunction(eager_execution_ctx, func, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  
  // Build the abstract op to run the function.
  TF_AbstractOp* fn_op = TF_NewAbstractOp(eager_execution_ctx);
  TF_AbstractOpSetOpType(fn_op, fn_name.c_str(), status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build an abstract input tensor.
  TFE_Context* eager_ctx =
      TF_ExecutionContextGetTFEContext(eager_execution_ctx);
  TFE_TensorHandle* input_eager = TestScalarTensorHandle(eager_ctx, 2.0f);
  TF_AbstractTensor* input_t =
      TF_CreateAbstractTensorFromEagerTensor(input_eager, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  TF_OutputListSetNumOutputs(add_outputs, 1, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_ExecuteOperation(fn_op, 1, &input_t, add_outputs, eager_execution_ctx,
                      status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  ASSERT_EQ(1, TF_OutputListNumOutputs(add_outputs));
  TF_AbstractTensor* final_result = TF_OutputListGet(add_outputs, 0);
  TFE_TensorHandle* final =
      TF_AbstractTensorGetEagerTensor(final_result, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_Tensor* f_t = TFE_TensorHandleResolve(final, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  float* f_value = static_cast<float*>(TF_TensorData(f_t));
  ASSERT_EQ(*f_value, 4.0);

  TF_DeleteOutputList(add_outputs);
  TF_DeleteAbstractOp(fn_op);
  TF_DeleteAbstractTensor(input_t);
  TF_DeleteAbstractTensor(final_result);
  TF_DeleteTensor(f_t);
  TF_DeleteAbstractFunction(func);

  TF_DeleteExecutionContext(eager_execution_ctx);
}

// Graph Tracing for MatMul
TEST_P(UnifiedCAPI, TestBasicGraphMatMul) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  
  // Start a new function / execution context.
  string fn_name = "matrix_multiply";
  TF_ExecutionContext* graph_ctx =
      TF_CreateFunction(fn_name.c_str(), status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  auto* placeholder_t =
      TF_AddFunctionParameter(graph_ctx, TF_FLOAT, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build an abstract operation.
  auto* matmul_op = TF_NewAbstractOp(graph_ctx);
  TF_AbstractOpSetOpType(matmul_op, "MatMul", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_AbstractOpSetOpName(matmul_op, "my_matmul", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build inputs and outputs.
  TF_AbstractTensor* inputs[2] = {placeholder_t, placeholder_t};
  TF_OutputList* mm_outputs = TF_NewOutputList();

  // Execute.
  TF_ExecuteOperation(matmul_op, 2, inputs, mm_outputs, graph_ctx, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Clean up operation and inputs.
  TF_DeleteAbstractOp(matmul_op);

  TF_AbstractFunction* func =
      TF_FinalizeFunction(graph_ctx, mm_outputs, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  /* Now that the graph is built, test graph implementation on matmul example:

    [ [1,1] ,   *   [ [1,1] ,   =   [ [2,2],
      [1,1] ]         [1,1] ]         [2,2] ]

  */

  // Build eager context.
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TF_ExecutionContext* eager_execution_ctx =
      TF_NewEagerExecutionContext(opts, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TFE_DeleteContextOptions(opts);

  TF_ExecutionContextRegisterFunction(eager_execution_ctx, func, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  
  // Build the abstract op to run the function.
  TF_AbstractOp* fn_op = TF_NewAbstractOp(eager_execution_ctx);
  TF_AbstractOpSetOpType(fn_op, fn_name.c_str(), status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());


  // Build an abstract input tensor.
  TFE_Context* eager_ctx =
      TF_ExecutionContextGetTFEContext(eager_execution_ctx);

  float vals [] = {1.0f,1.0f,1.0f,1.0f};
  int64_t dims [] = {2,2}; // Matrices will be 2 x 2   
  int num_dims = sizeof(dims)/sizeof(dims[0]);

  TFE_TensorHandle* input_eager = TestMatrixTensorHandleWithInput(eager_ctx, vals, dims, num_dims);
  TF_AbstractTensor* input_t =
      TF_CreateAbstractTensorFromEagerTensor(input_eager, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  TF_OutputListSetNumOutputs(mm_outputs, 1, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_ExecuteOperation(fn_op, 1, &input_t, mm_outputs, eager_execution_ctx,
                      status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  ASSERT_EQ(1, TF_OutputListNumOutputs(mm_outputs));
  TF_AbstractTensor* final_result = TF_OutputListGet(mm_outputs, 0);
  TFE_TensorHandle* final =
      TF_AbstractTensorGetEagerTensor(final_result, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_Tensor* f_t = TFE_TensorHandleResolve(final, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  float result_data[4] = {0};
  memcpy(&result_data[0], TF_TensorData(f_t), TF_TensorByteSize(f_t));

  int data_len = 4;
  for(int i = 0; i < data_len; i++){
    ASSERT_EQ(result_data[i], 2.0f);
  }

  TF_DeleteOutputList(mm_outputs);
  TF_DeleteAbstractOp(fn_op);
  TF_DeleteAbstractTensor(input_t);
  TF_DeleteAbstractTensor(final_result);
  TF_DeleteTensor(f_t);
  TF_DeleteAbstractFunction(func);

  TF_DeleteExecutionContext(eager_execution_ctx);
}


TEST_P(UnifiedCAPI, TestMultiOutputGraph) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_Status* s = status.get();

  // Start a new function / execution context.
  string fn_name = "two_adds";
  TF_ExecutionContext* graph_ctx = TF_CreateFunction(fn_name.c_str(), s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  auto* arg0 = TF_AddFunctionParameter(graph_ctx, TF_FLOAT, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  auto* arg1 = TF_AddFunctionParameter(graph_ctx, TF_FLOAT, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Create a first "Add" computing `arg0 + arg1`.
  TF_AbstractTensor* add_output1;
  {
    // Build an abstract operation, inputs and output.
    auto* add_op = TF_NewAbstractOp(graph_ctx);
    TF_AbstractOpSetOpType(add_op, "Add", s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_AbstractOpSetOpName(add_op, "my_add1", s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_AbstractTensor* inputs[2] = {arg0, arg1};
    TF_OutputList* add_outputs = TF_NewOutputList();
    // Trace the operation now (create a node in the graph).
    TF_ExecuteOperation(add_op, 2, inputs, add_outputs, graph_ctx, s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_DeleteAbstractOp(add_op);
    // Extract the resulting tensor.
    add_output1 = TF_OutputListGet(add_outputs, 0);
    TF_DeleteOutputList(add_outputs);
  }

  // Same with a second "Add" computing `arg1 + arg1`.
  TF_AbstractTensor* add_output2;
  {
    // Build an abstract operation, inputs and output.
    auto* add_op = TF_NewAbstractOp(graph_ctx);
    TF_AbstractOpSetOpType(add_op, "Add", s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_AbstractOpSetOpName(add_op, "my_add2", s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_AbstractTensor* inputs[2] = {arg1, arg1};
    TF_OutputList* add_outputs = TF_NewOutputList();
    // Trace the operation now (create a node in the graph).
    TF_ExecuteOperation(add_op, 2, inputs, add_outputs, graph_ctx, s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_DeleteAbstractOp(add_op);
    // Extract the resulting tensor.
    add_output2 = TF_OutputListGet(add_outputs, 0);
    TF_DeleteOutputList(add_outputs);
  }

  // Finalize the function by providing the returned values.
  TF_AbstractFunction* func;
  {
    // We want to return the output of both add operations, create a new list
    // and populate it.
    TF_OutputList* func_outputs = TF_NewOutputList();
    TF_OutputListPushBack(func_outputs, add_output1, s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_OutputListPushBack(func_outputs, add_output2, s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    func = TF_FinalizeFunction(graph_ctx, func_outputs, s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_DeleteOutputList(func_outputs);
  }

  /**
   * We traced so far this function:
   *
   *   def two_adds(a, b):
   *     my_add1 = a + b
   *     my_add2 = b + b
   *     return my_add1, my_add2
   *
   * Now we will execute this function with an eager context:
   *
   *   output1, output2 = two_adds(2.0, 3.0)
   *
   * and check that we got 5.0 and 6.0 as results.
   */

  // Build eager context.
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TF_ExecutionContext* eager_execution_ctx =
      TF_NewEagerExecutionContext(opts, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TFE_DeleteContextOptions(opts);

  TF_ExecutionContextRegisterFunction(eager_execution_ctx, func, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Build the abstract op to run the function.
  TF_AbstractOp* fn_op = TF_NewAbstractOp(eager_execution_ctx);
  TF_AbstractOpSetOpType(fn_op, fn_name.c_str(), s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Build two abstract input tensors as function arguments.
  std::vector<TF_AbstractTensor*> func_args;
  {
    TFE_Context* eager_ctx =
        TF_ExecutionContextGetTFEContext(eager_execution_ctx);
    TFE_TensorHandle* input_eager = TestScalarTensorHandle(eager_ctx, 2.0f);
    func_args.push_back(TF_CreateAbstractTensorFromEagerTensor(input_eager, s));
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    input_eager = TestScalarTensorHandle(eager_ctx, 3.0f);
    func_args.push_back(TF_CreateAbstractTensorFromEagerTensor(input_eager, s));
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  }

  TF_OutputList* func_outputs = TF_NewOutputList();
  TF_OutputListSetNumOutputs(func_outputs, 2, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_ExecuteOperation(fn_op, func_args.size(), func_args.data(), func_outputs,
                      eager_execution_ctx, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_DeleteAbstractOp(fn_op);
  for (TF_AbstractTensor* t : func_args) TF_DeleteAbstractTensor(t);

  ASSERT_EQ(2, TF_OutputListNumOutputs(func_outputs));
  float results[2];
  for (int idx = 0; idx < 2; ++idx) {
    TF_AbstractTensor* result = TF_OutputListGet(func_outputs, idx);
    TFE_TensorHandle* handle = TF_AbstractTensorGetEagerTensor(result, s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_Tensor* f_t = TFE_TensorHandleResolve(handle, s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    results[idx] = *static_cast<float*>(TF_TensorData(f_t));
    TF_DeleteTensor(f_t);
  }
  ASSERT_EQ(results[0], 5.0);
  ASSERT_EQ(results[1], 6.0);

  for (int idx = 0; idx < 2; ++idx) {
    TF_AbstractTensor* result = TF_OutputListGet(func_outputs, idx);
    TF_DeleteAbstractTensor(result);
  }
  TF_DeleteOutputList(func_outputs);
  TF_DeleteExecutionContext(eager_execution_ctx);
  TF_DeleteAbstractFunction(func);
}

TEST(UnifiedCAPI, TF_ExecutionContextToFunctionWithEagerContextRaises) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TF_ExecutionContext* ctx = TF_NewEagerExecutionContext(opts, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TFE_DeleteContextOptions(opts);

  TF_AbstractFunction* func = TF_FinalizeFunction(ctx, nullptr, status.get());
  ASSERT_EQ(nullptr, func);
  ASSERT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(status.get()));
}

TEST_P(UnifiedCAPI, TF_CallingSetOpTypeAfterFinishingOpBuildingRaises) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_ExecutionContext* graph_ctx = TF_CreateFunction("some_func", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Add a placeholder to the graph.
  auto* placeholder_op = TF_NewAbstractOp(graph_ctx);
  TF_AbstractOpSetOpType(placeholder_op, "Placeholder", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_AbstractOpSetOpName(placeholder_op, "my_ph", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // This should fail.
  TF_AbstractOpSetOpType(placeholder_op, "Placeholder", status.get());
  ASSERT_EQ(TF_FAILED_PRECONDITION, TF_GetCode(status.get()));

  TF_DeleteAbstractOp(placeholder_op);
  TF_DeleteExecutionContext(graph_ctx);
}

TEST_P(UnifiedCAPI, TF_CallingSetOpNameAfterFinishingOpBuildingRaises) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_ExecutionContext* graph_ctx = TF_CreateFunction("some_func", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Add a placeholder to the graph.
  auto* placeholder_op = TF_NewAbstractOp(graph_ctx);
  TF_AbstractOpSetOpType(placeholder_op, "Placeholder", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_AbstractOpSetOpName(placeholder_op, "my_ph", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // This should fail.
  TF_AbstractOpSetOpName(placeholder_op, "my_ph", status.get());
  ASSERT_EQ(TF_FAILED_PRECONDITION, TF_GetCode(status.get()));

  TF_DeleteAbstractOp(placeholder_op);
  TF_DeleteExecutionContext(graph_ctx);
}

TEST_P(UnifiedCAPI, TestExecutingEagerOpInGraphModeRaises) {
  // Build an Eager context.
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TF_ExecutionContext* ctx = TF_NewEagerExecutionContext(opts, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TFE_DeleteContextOptions(opts);

  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build an Eager operation.
  auto* op = TF_NewAbstractOp(ctx);
  TF_AbstractOpSetOpType(op, "Add", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build an abstract input tensor.
  TFE_Context* eager_ctx = TF_ExecutionContextGetTFEContext(ctx);
  TFE_TensorHandle* t = TestScalarTensorHandle(eager_ctx, 2.0f);
  TF_AbstractTensor* at =
      TF_CreateAbstractTensorFromEagerTensor(t, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build inputs and outputs.
  TF_AbstractTensor* inputs[2] = {at, at};
  TF_OutputList* o = TF_NewOutputList();
  TF_OutputListSetNumOutputs(o, 1, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build a Graph context.
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_ExecutionContext* graph_ctx = TF_CreateFunction("some_func", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Execute eager op using graph context.
  TF_ExecuteOperation(op, 2, inputs, o, graph_ctx, status.get());
  ASSERT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(status.get()));

  // Clean up operation and inputs.
  TF_DeleteAbstractOp(op);
  TF_DeleteAbstractTensor(at);

  TF_DeleteOutputList(o);
  TF_DeleteExecutionContext(ctx);
  TF_DeleteExecutionContext(graph_ctx);
}

TEST_P(UnifiedCAPI, TestExecutingGraphOpInEagerModeRaises) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_ExecutionContext* graph_ctx = TF_CreateFunction("some_func", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Add a placeholder to the graph.
  auto* placeholder_op = TF_NewAbstractOp(graph_ctx);
  TF_AbstractOpSetOpType(placeholder_op, "Placeholder", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_AbstractOpSetOpName(placeholder_op, "my_ph", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_AbstractOpSetAttrType(placeholder_op, "dtype", TF_FLOAT, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build inputs and outputs.
  TF_OutputList* placeholder_outputs = TF_NewOutputList();

  // Execute.
  TF_ExecuteOperation(placeholder_op, 0, nullptr, placeholder_outputs,
                      graph_ctx, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  ASSERT_EQ(1, TF_OutputListNumOutputs(placeholder_outputs));
  TF_AbstractTensor* placeholder_t = TF_OutputListGet(placeholder_outputs, 0);

  // Delete placeholder op.
  TF_DeleteAbstractOp(placeholder_op);

  // Build an abstract operation.
  auto* add_op = TF_NewAbstractOp(graph_ctx);
  TF_AbstractOpSetOpType(add_op, "Add", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_AbstractOpSetOpName(add_op, "my_add", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build inputs and outputs.
  TF_AbstractTensor* inputs[2] = {placeholder_t, placeholder_t};
  TF_OutputList* add_outputs = TF_NewOutputList();

  // Build eager context.
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TF_ExecutionContext* eager_execution_ctx =
      TF_NewEagerExecutionContext(opts, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TFE_DeleteContextOptions(opts);

  // Execute.
  TF_ExecuteOperation(add_op, 2, inputs, add_outputs, eager_execution_ctx,
                      status.get());
  ASSERT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(status.get()));

  // Clean up operation and inputs.
  TF_DeleteAbstractTensor(placeholder_t);
  TF_DeleteAbstractOp(add_op);
  TF_DeleteOutputList(add_outputs);
  TF_DeleteOutputList(placeholder_outputs);
  TF_DeleteExecutionContext(graph_ctx);
  TF_DeleteExecutionContext(eager_execution_ctx);
}

INSTANTIATE_TEST_SUITE_P(Tracing, UnifiedCAPI,
                         ::testing::Values("graphdef", "mlir"));

}  // namespace
}  // namespace tensorflow
