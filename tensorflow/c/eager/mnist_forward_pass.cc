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
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/platform/test.h"

using tensorflow::string;

namespace tensorflow {
namespace {

class UnifiedCAPI
    : public ::testing::TestWithParam<std::tuple<const char*, bool>> {
 protected:
  void SetUp() override {
    TF_SetTracingImplementation(std::get<0>(GetParam()));
  }
};

// MNIST
TEST_P(UnifiedCAPI, MNIST) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_Status* s = status.get();

  // Start a new function / execution context.
  string fn_name = "MNIST forward pass";
  TF_ExecutionContext* graph_ctx = TF_CreateFunction(fn_name.c_str(), s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  auto* X_abstract = TF_AddFunctionParameter(graph_ctx, TF_FLOAT, s); // X = data
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  auto* W1_abstract = TF_AddFunctionParameter(graph_ctx, TF_FLOAT, s); // W1 = first FC layer
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  auto* W2_abstract = TF_AddFunctionParameter(graph_ctx, TF_FLOAT, s); // W2  = second FC layer
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  auto* y_abstract = TF_AddFunctionParameter(graph_ctx, TF_FLOAT, s); // W2  = second FC layer
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Create a first "matrix multiply" computing `matmul(X, W)`.
  TF_AbstractTensor* mm_out_1;
  {
    // Build an abstract operation, inputs and output.
    auto* mm_op1 = TF_NewAbstractOp(graph_ctx);
    TF_AbstractOpSetOpType(mm_op1, "MatMul", s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_AbstractOpSetOpName(mm_op1, "fc_1", s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_AbstractTensor* inputs[2] = {X_abstract, W1_abstract};
    TF_OutputList* mm_outputs = TF_NewOutputList();
    TF_OutputListSetNumOutputs(mm_outputs, 1, status.get());
    ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
    // Trace the operation now (create a node in the graph).
    TF_ExecuteOperation(mm_op1, 2, inputs, mm_outputs, s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_DeleteAbstractOp(mm_op1);
    // Extract the resulting tensor.
    mm_out_1 = TF_OutputListGet(mm_outputs, 0);
    TF_DeleteOutputList(mm_outputs);
  }
  

  // Compute ReLu(X*W) (or do we have to do a `max` operation with 0?)
  TF_AbstractTensor* hidden_layer;
  {
    // Build an abstract operation, inputs and output.
    auto* relu_op = TF_NewAbstractOp(graph_ctx);
    TF_AbstractOpSetOpType(relu_op, "ReLu", s); // Not sure about capitalization here... is there a 
                                             // place to check what the names of the abstract ops are?

    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s); 
    TF_AbstractOpSetOpName(relu_op, "relu", s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_AbstractTensor* inputs[1] = {mm_out_1};
    TF_OutputList* relu_outputs = TF_NewOutputList();
    TF_OutputListSetNumOutputs(relu_outputs, 1, status.get());
    ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
    // Trace the operation now (create a node in the graph).
    TF_ExecuteOperation(relu_op, 1, inputs, relu_outputs, s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_DeleteAbstractOp(relu_op);
    // Extract the resulting tensor.
    hidden_layer = TF_OutputListGet(relu_outputs, 0);
    TF_DeleteOutputList(relu_outputs);
  }

  // Get class scores by computing scores = ReLu(X*W1)*W2
  TF_AbstractTensor* scores; 
  {
    // Build an abstract operation, inputs and output.
    auto* mm_op2 = TF_NewAbstractOp(graph_ctx);
    TF_AbstractOpSetOpType(mm_op2, "MatMul", s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_AbstractOpSetOpName(mm_op2, "fc_2", s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_AbstractTensor* inputs[2] = {hidden_layer, W2_abstract};
    TF_OutputList* mm_outputs = TF_NewOutputList();
    TF_OutputListSetNumOutputs(mm_outputs, 1, status.get());
    ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
    // Trace the operation now (create a node in the graph).
    TF_ExecuteOperation(mm_op2, 2, inputs, mm_outputs, s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_DeleteAbstractOp(mm_op2);
    // Extract the resulting tensor.
    scores = TF_OutputListGet(mm_outputs, 0);
    TF_DeleteOutputList(mm_outputs);
  }

  // Softmax... Is this abstract op implemented?
  TF_AbstractTensor* softmax; 
  {
    // Build an abstract operation, inputs and output.
    auto* sm = TF_NewAbstractOp(graph_ctx);
    TF_AbstractOpSetOpType(sm, "Softmax", s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_AbstractOpSetOpName(sm, "softmax", s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_AbstractTensor* inputs[2] = {scores,y_abstract}; // correct order for Softmax? 
    TF_OutputList* softmax_outputs = TF_NewOutputList();
    TF_OutputListSetNumOutputs(softmax_outputs, 2, status.get());
    ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
    
    // Trace the operation now (create a node in the graph).
    TF_ExecuteOperation(sm, 2, inputs, softmax_outputs, s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_DeleteAbstractOp(sm);
    
    // Extract the resulting tensor.
    softmax = TF_OutputListGet(softmax_outputs, 0);
    TF_DeleteOutputList(softmax_outputs);
  }

  TF_DeleteAbstractTensor(X_abstract);
  TF_DeleteAbstractTensor(W1_abstract);
  TF_DeleteAbstractTensor(W2_abstract);
  TF_DeleteAbstractTensor(y_abstract)

  // Finalize the function by providing the returned values.
  TF_AbstractFunction* func;
  {
    // We want to return the output of the forward
    TF_OutputList* func_outputs = TF_NewOutputList();
    TF_OutputListPushBack(func_outputs, scores, s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_OutputListPushBack(func_outputs, softmax, s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    func = TF_FinalizeFunction(graph_ctx, func_outputs, s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_DeleteAbstractTensor(mm_out_1);
    TF_DeleteAbstractTensor(hidden_layer);
    TF_DeleteAbstractTensor(scores);
    TF_DeleteAbstractTensor(softmax);
    TF_DeleteOutputList(func_outputs);
  }

  /**
   * We traced so far this function:
   *
   *   def mnist_forward(X, W1, W2):
   *     mm_out_1 = tf.matmul(X,W1)
   *     hidden_layer = tf.ReLu(mm_out_1)
   *     scores = tf.matmul(hidden_layer,W2)
   *     softmax = tf.softmax(scores)
   *     return scores, softmax 
   *
   * Now we will execute this function with an eager context:
   *
   *   output1, output2 = mnist_forward(X, W1, W2)
   * 
   */

  // eager execution...


  /* Code from previous test:

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
        TF_ExecutionContextGetTFEContext(eager_execution_ctx,s);

    // 1st Arg
    float vals1 [] = {0.0f,0.0f,0.0f,0.0f};
    int64_t dims [] = {2,2}; // Matrices will be 2 x 2   
    int num_dims = sizeof(dims)/sizeof(dims[0]);

    TFE_TensorHandle* input_eager = TestMatrixTensorHandleWithInput(eager_ctx, vals1, dims, num_dims);
    func_args.push_back(TF_CreateAbstractTensorFromEagerTensor(input_eager, s));
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

    // 2nd Arg
    float vals2 [] = {1.0f,0.0f,0.0f,1.0f};
    input_eager = TestMatrixTensorHandleWithInput(eager_ctx, vals2, dims, num_dims);
    func_args.push_back(TF_CreateAbstractTensorFromEagerTensor(input_eager, s));
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  }

  TF_OutputList* func_outputs = TF_NewOutputList();
  TF_OutputListSetNumOutputs(func_outputs, 3, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_ExecuteOperation(fn_op, func_args.size(), func_args.data(), func_outputs, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_DeleteAbstractOp(fn_op);
  for (TF_AbstractTensor* t : func_args) TF_DeleteAbstractTensor(t);

  ASSERT_EQ(3, TF_OutputListNumOutputs(func_outputs));
  
  float expected_outputs [3][4] = {{1.0f,0.0f,0.0f,1.0f}, 
                                   {2.0f,0.0f,0.0f,2.0f},
                                   {2.0f,0.0f,0.0f,2.0f}};

  float result_data[4];
  for (int idx = 0; idx < 3; ++idx) {
    TF_AbstractTensor* result = TF_OutputListGet(func_outputs, idx);
    TFE_TensorHandle* handle = TF_AbstractTensorGetEagerTensor(result, s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_Tensor* f_t = TFE_TensorHandleResolve(handle, s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

    memcpy(&result_data[0], TF_TensorData(f_t), TF_TensorByteSize(f_t));
    
    // Verify results for each output 
    for(int j = 0; j < 4; j++){
      ASSERT_EQ(result_data[j], expected_outputs[idx][j]);
    }
    
    TF_DeleteTensor(f_t);
  }

  // Free memory associated with add and MatMul outputs
  for (int idx = 0; idx < 3; ++idx) {
    TF_AbstractTensor* result = TF_OutputListGet(func_outputs, idx);
    TF_DeleteAbstractTensor(result);
  }
  
  TF_DeleteOutputList(func_outputs);
  TF_DeleteExecutionContext(eager_execution_ctx);
  TF_DeleteAbstractFunction(func);
  */
}


// end MNIST


#ifdef PLATFORM_GOOGLE
INSTANTIATE_TEST_SUITE_P(Tracing, UnifiedCAPI,
                         ::testing::Combine(::testing::Values("graphdef",
                                                              "mlir"),
                                            ::testing::Values(true, false)));
#else
INSTANTIATE_TEST_SUITE_P(Tracing, UnifiedCAPI,
                         ::testing::Combine(::testing::Values("graphdef",
                                                              "mlir"),
                                            ::testing::Values(false)));
#endif

}  // namespace
}  // namespace tensorflow
