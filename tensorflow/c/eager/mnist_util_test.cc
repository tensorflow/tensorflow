#include "tensorflow/c/eager/mnist_util.h"

#include <memory>

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

TEST_P(UnifiedCAPI_MNIST, TestBasicEager) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetTfrt(opts, std::get<1>(GetParam()));
  TF_ExecutionContext* ctx = TF_NewEagerExecutionContext(opts, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TFE_DeleteContextOptions(opts);

  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build an abstract input tensor.
  TFE_Context* eager_ctx = TF_ExecutionContextGetTFEContext(ctx, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
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
  TF_ExecuteOperation(op, 2, inputs, o, status.get());
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
TEST_P(UnifiedCAPI_MNIST, TestBasicEagerMatMul) {
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
  TFE_Context* eager_ctx = TF_ExecutionContextGetTFEContext(ctx,status.get());
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
  TF_ExecuteOperation(op, 2, inputs, o, status.get());
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

// Graph Tracing for MatMul
TEST_P(UnifiedCAPI_MNIST, TestBasicGraphAdd) {
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

  TF_AbstractTensor* mm = AbstractMatMul(add1,add2, "matmul", graph_ctx, s);
  
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
