#include "tensorflow/c/eager/mnist_util.h"

#include <memory>
#include <cstdlib>

// #include "testing/base/public/gmock.h"
// #include "testing/base/public/gunit.h"
// #include "third_party/absl/memory/memory.h"

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

class UnifiedCAPI_MNIST
    : public ::testing::TestWithParam<std::tuple<const char*, bool>> {
 protected:
  void SetUp() override {
    TF_SetTracingImplementation(std::get<0>(GetParam()));
  }
};


// Graph Tracing with mnist_util helper functions
TEST_P(UnifiedCAPI_MNIST, TestBasicGraphAddandMatMul) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_Status* s = status.get();

  // Start a new function / execution context.
  string fn_name = "two_adds_and_mm";
  TF_ExecutionContext* graph_ctx = TF_CreateFunction(fn_name.c_str(), s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  auto* arg0 = TF_AddFunctionParameter(graph_ctx, TF_FLOAT, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  auto* arg1 = TF_AddFunctionParameter(graph_ctx, TF_FLOAT, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Create a first "Add" computing `arg0 + arg1`
  TF_AbstractTensor* add1 = AbstractAdd(arg0, arg1, "add1", graph_ctx, s);

  // Same with a second "Add" computing `arg1 + arg1`.
  TF_AbstractTensor* add2 = AbstractAdd(arg1, arg1, "add2", graph_ctx, s);

  // Create a MatMul computing add1 * add2
  TF_AbstractTensor* mm = AbstractMatMul(add1, add2, "matmul", graph_ctx, s);
  
  TF_DeleteAbstractTensor(arg0);
  TF_DeleteAbstractTensor(arg1);

  TF_OutputList* f_outputs = TF_NewOutputList();
  TF_OutputListPushBack(f_outputs, add1, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_OutputListPushBack(f_outputs, add2, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_OutputListPushBack(f_outputs, mm, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Finalize the function by providing the returned values.
  TF_AbstractFunction* func = AbstractFinalizeFunction(f_outputs, graph_ctx, s);
  
 /**
   * We traced so far this function:
   *
   *   def two_adds_and_mm(A, B):
   *     my_add1 = A + B
   *     my_add2 = B + B
   *     mm = tf.MatMul(my_add1,my_add2)
   *     return my_add1, my_add2, mm
   *
   * Now we will execute this function with an eager context:
   *
   *   A = [[0, 1], [1, 0]] 
   *   B = [[1, 0], [0, 1]]   
   *
   *   output1, output2, output3 = two_adds_and_mm(A, B)
   *
   * We expect outputs: 
   *    
   *   output1 =  [[1, 1], [1, 1]]  
   *   output2 =  [[2, 0], [0, 2]]  
   *   output3 =  [[2, 2], [2, 2]]  
   *
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
        TF_ExecutionContextGetTFEContext(eager_execution_ctx,s);

    // 1st Arg
    float vals1 [] = {0.0f,1.0f,1.0f,0.0f};
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
  
  float expected_outputs [3][4] = {{1.0f,1.0f,1.0f,1.0f}, 
                                   {2.0f,0.0f,0.0f,2.0f},
                                   {2.0f,2.0f,2.0f,2.0f}};

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

  //Free memory associated with add and MatMul outputs
  for (int idx = 0; idx < 3; ++idx) {
    TF_AbstractTensor* result = TF_OutputListGet(func_outputs, idx);
    TF_DeleteAbstractTensor(result);
  }
  
   TF_DeleteOutputList(func_outputs);
   TF_DeleteExecutionContext(eager_execution_ctx);
   TF_DeleteAbstractFunction(func);
  
}

TEST_P(UnifiedCAPI_MNIST, TestBasicGraphRelu) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_Status* s = status.get();
  // Start a new function / execution context.
  string fn_name = "relu_fn";
  TF_ExecutionContext* graph_ctx =
      TF_CreateFunction(fn_name.c_str(), status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  
  auto* arg0 =
      TF_AddFunctionParameter(graph_ctx, TF_FLOAT, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  
  // Build an abstract operation.
  TF_AbstractTensor* relu_out = AbstractRelu(arg0, "relu", graph_ctx, s);
  TF_OutputList* f_outputs = TF_NewOutputList();
  TF_OutputListPushBack(f_outputs, relu_out, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  

  TF_AbstractFunction* func = AbstractFinalizeFunction(f_outputs, graph_ctx, s);

  /* Now that the graph is built, test graph implementation on relu example:

     Relu (   [ [3,  -1] ,       =  [ [3,0],
                [-10, 5] ]   )        [0,5] ]   

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
      TF_ExecutionContextGetTFEContext(eager_execution_ctx,status.get());

  float vals [] = {3.0f,-1.0f,-10.0f,5.0f};
  int64_t dims [] = {2,2}; // Matrices will be 2 x 2   
  int num_dims = sizeof(dims)/sizeof(dims[0]);

  TFE_TensorHandle* input_eager = TestMatrixTensorHandleWithInput(eager_ctx, vals, dims, num_dims);
  TF_AbstractTensor* input_t =
      TF_CreateAbstractTensorFromEagerTensor(input_eager, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  TF_OutputList* func_outputs = TF_NewOutputList();
  TF_OutputListSetNumOutputs(func_outputs, 1, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_ExecuteOperation(fn_op, 1, &input_t, func_outputs, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  
  ASSERT_EQ(1, TF_OutputListNumOutputs(func_outputs));

  TF_AbstractTensor* final_result = TF_OutputListGet(func_outputs, 0);
  TFE_TensorHandle* final =
      TF_AbstractTensorGetEagerTensor(final_result, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_Tensor* f_t = TFE_TensorHandleResolve(final, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  float result_data [4] = {0};
  memcpy(&result_data[0], TF_TensorData(f_t), TF_TensorByteSize(f_t));

  float expected_vals [] = {3.0f,0.0f,0.0f,5.0f};
  int data_len = 4;
  for(int i = 0; i < data_len; i++){
    ASSERT_EQ(result_data[i], expected_vals[i]);
  }


  TF_DeleteAbstractTensor(final_result);
  TF_DeleteOutputList(func_outputs);
  TF_DeleteAbstractTensor(arg0);
  TF_DeleteAbstractOp(fn_op);
  TF_DeleteAbstractTensor(input_t);
  TF_DeleteTensor(f_t);
  TF_DeleteAbstractFunction(func);
  TF_DeleteExecutionContext(eager_execution_ctx);
  
}

TEST_P(UnifiedCAPI_MNIST, TestBasicGraphSoftmaxLoss) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_Status* s = status.get();

  // Start a new function / execution context.
  string fn_name = "softmax_loss_fn";
  TF_ExecutionContext* graph_ctx = TF_CreateFunction(fn_name.c_str(), s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  auto* scores_abstract = TF_AddFunctionParameter(graph_ctx, TF_FLOAT, s);
  auto* y_labels_abstract = TF_AddFunctionParameter(graph_ctx, TF_FLOAT, s);
 
  // Create a softmax loss, passing in scores and labels 
  TF_AbstractTensor* softmax_loss = AbstractSparseSoftmaxCrossEntropyLoss(scores_abstract, y_labels_abstract, "sm_loss", graph_ctx, s);

  TF_DeleteAbstractTensor(scores_abstract);
  TF_DeleteAbstractTensor(y_labels_abstract);

  TF_OutputList* f_outputs = TF_NewOutputList();
  TF_OutputListPushBack(f_outputs, softmax_loss, s);

  ASSERT_EQ(1, TF_OutputListNumOutputs(f_outputs));

  // Finalize the function by providing the returned values.
  TF_AbstractFunction* func = AbstractFinalizeFunction(f_outputs, graph_ctx, s);  // **TEST FAILS HERE
  
  /* Now that the graph is built, test graph implementation on relu example:
   *
   * scores = [ [1,2,5 ], 
   *            [2,1,-3] ]
   *  
   * labels =  [0,2]
   * SoftmaxLoss(scores, labels) = [4.06588, 5.31817]
   *
   */
   

  //Build eager context.
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
    float scores [] = {1.0f, 2.0f, 5.0f, 2.0f, 1.0f, -3.0f};
    int64_t dims_scores [] = {3,2}; // Input matrix will be 3x2  
    int num_dims_scores = sizeof(dims_scores)/sizeof(dims_scores[0]);

    TFE_TensorHandle* input_eager = TestMatrixTensorHandleWithInput(eager_ctx, scores, dims_scores, num_dims_scores);
    func_args.push_back(TF_CreateAbstractTensorFromEagerTensor(input_eager, s));
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

    // 2nd Arg
    float labels [] = {0,2};
    int64_t dims_labels [] = {1,2}; 
    int num_dims_labels = sizeof(dims_labels)/sizeof(dims_labels[0]);
    input_eager = TestMatrixTensorHandleWithInput(eager_ctx, labels, dims_labels, num_dims_labels);
    func_args.push_back(TF_CreateAbstractTensorFromEagerTensor(input_eager, s));
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  }

  TF_OutputList* func_outputs = TF_NewOutputList();
  TF_OutputListSetNumOutputs(func_outputs, 1, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_ExecuteOperation(fn_op, func_args.size(), func_args.data(), func_outputs, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_DeleteAbstractOp(fn_op);
  for (TF_AbstractTensor* t : func_args) TF_DeleteAbstractTensor(t);

  ASSERT_EQ(1, TF_OutputListNumOutputs(func_outputs));
  
  float expected_outputs [2] = {4.06588f, 5.31817f};

  float result_data[2];
  
  TF_AbstractTensor* result = TF_OutputListGet(func_outputs, 0);
  TFE_TensorHandle* handle = TF_AbstractTensorGetEagerTensor(result, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_Tensor* f_t = TFE_TensorHandleResolve(handle, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  memcpy(&result_data[0], TF_TensorData(f_t), TF_TensorByteSize(f_t));
    
  // Verify results for each output 

  float threshold = 1e-3;

  for(int j = 0; j < 2; j++){
      float diff = result_data[j] - expected_outputs[j];
      bool belowThreshold = diff < threshold;
      ASSERT_EQ(true, belowThreshold);
  }
    
  TF_DeleteTensor(f_t);
  

  //Free memory associated with outputs
  TF_DeleteAbstractTensor(result);

  
  TF_DeleteOutputList(func_outputs);
  TF_DeleteExecutionContext(eager_execution_ctx);
  TF_DeleteAbstractFunction(func);
  
}

#ifdef PLATFORM_GOOGLE
INSTANTIATE_TEST_SUITE_P(Tracing, UnifiedCAPI_MNIST,
                         ::testing::Combine(::testing::Values("graphdef",
                                                              "mlir"),
                                            ::testing::Values(true, false)));
#else
INSTANTIATE_TEST_SUITE_P(Tracing, UnifiedCAPI_MNIST,
                         ::testing::Combine(::testing::Values("graphdef",
                                                              "mlir"),
                                            ::testing::Values(false)));
#endif

}  // namespace
}  // namespace tensorflow
