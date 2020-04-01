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

#include <string.h>

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/cc/profiler/profiler.h"
#include "tensorflow/core/lib/monitoring/collection_registry.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

using tensorflow::string;

namespace tensorflow {
namespace {

TEST(UnifedCAPI, TestBasicEager) {
  TF_ExecutionContext* ctx = TF_NewExecutionContext();
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* eager_ctx = TFE_NewContext(opts, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TFE_DeleteContextOptions(opts);

  // Enter the eager context.
  TF_ExecutionContextSetEagerContext(ctx, eager_ctx, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build an abstract input tensor.
  TFE_TensorHandle* t = TestScalarTensorHandle(2.0f);
  TF_AbstractTensor* at = TF_NewAbstractTensor();
  TF_AbstractTensorSetEagerTensor(at, t, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build an abstract operation.
  auto* op = TF_NewAbstractOp();
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
  TFE_DeleteTensorHandle(t);

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
  TFE_DeleteTensorHandle(result_t);
  TF_DeleteOutputList(o);
  TFE_DeleteContext(eager_ctx);
  TF_DeleteExecutionContext(ctx);
}

TEST(UnifedCAPI, TestBasicGraph) {
  TF_ExecutionContext* ctx = TF_NewExecutionContext();
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);

  // Enter a graph context.
  TF_Graph* g = TF_NewGraph();
  TF_GraphContext* graph_context = TF_NewGraphContext(g);
  TF_ExecutionContextSetGraphContext(ctx, graph_context, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Add a placeholder to the graph.
  auto* placeholder_op = TF_NewOperation(g, "Placeholder", "Placeholder");
  TF_SetAttrType(placeholder_op, "dtype", TF_FLOAT);
  auto* operation = TF_FinishOperation(placeholder_op, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_Output placeholder_t = {operation, 0};
  TF_GraphTensor* graph_t =
      TF_NewGraphTensor(graph_context, placeholder_t, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_AbstractTensor* t = TF_NewAbstractTensor();
  TF_AbstractTensorSetGraphTensor(t, graph_t, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build an abstract operation.
  auto* op = TF_NewAbstractOp();
  TF_AbstractOpSetOpType(op, "Add", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_AbstractOpSetOpName(op, "my_add", status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build inputs and outputs.
  TF_AbstractTensor* inputs[2] = {t, t};
  TF_OutputList* o = TF_NewOutputList();

  // Execute.
  TF_ExecuteOperation(op, 2, inputs, o, ctx, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Clean up operation and inputs.
  TF_DeleteAbstractOp(op);
  TF_DeleteAbstractTensor(t);
  TF_DeleteGraphTensor(graph_t);

  TF_AbstractTensor* result = TF_OutputListGet(o, 0);
  TF_GraphTensor* result_graph_tensor =
      TF_AbstractTensorGetGraphTensor(result, status.get());
  TF_DeleteAbstractTensor(result);
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_Output result_output =
      TF_GraphTensorToOutput(result_graph_tensor, status.get());
  TF_DeleteGraphTensor(result_graph_tensor);
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  string fn_name = "double";
  TF_Function* f = TF_GraphToFunction(
      g, fn_name.c_str(), 0, -1, nullptr, 1, &placeholder_t, 1, &result_output,
      nullptr, nullptr, fn_name.c_str(), status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build an eager context to run the function.
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* eager_ctx = TFE_NewContext(opts, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TFE_DeleteContextOptions(opts);

  // Build the abstract op to run the function.
  TFE_ContextAddFunction(eager_ctx, f, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_AbstractOp* fn_op = TF_NewAbstractOp();
  TF_AbstractOpSetOpType(fn_op, fn_name.c_str(), status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build an abstract input tensor.
  TFE_TensorHandle* input_eager = TestScalarTensorHandle(2.0f);
  TF_AbstractTensor* input_t = TF_NewAbstractTensor();
  TF_AbstractTensorSetEagerTensor(input_t, input_eager, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Enter the eager context.
  TF_ExecutionContextSetEagerContext(ctx, eager_ctx, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_OutputListSetNumOutputs(o, 1, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_ExecuteOperation(fn_op, 1, &input_t, o, ctx, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  ASSERT_EQ(1, TF_OutputListNumOutputs(o));
  TF_AbstractTensor* final_result = TF_OutputListGet(o, 0);
  TFE_TensorHandle* final =
      TF_AbstractTensorGetEagerTensor(final_result, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_Tensor* f_t = TFE_TensorHandleResolve(final, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  float* f_value = static_cast<float*>(TF_TensorData(f_t));
  ASSERT_EQ(*f_value, 4.0);

  TF_DeleteOutputList(o);
  TF_DeleteAbstractOp(fn_op);
  TF_DeleteAbstractTensor(input_t);
  TFE_DeleteTensorHandle(input_eager);
  TF_DeleteAbstractTensor(final_result);
  TFE_DeleteTensorHandle(final);
  TF_DeleteTensor(f_t);
  TF_DeleteFunction(f);

  TF_DeleteGraphContext(graph_context);
  TF_DeleteGraph(g);
  TFE_DeleteContext(eager_ctx);
  TF_DeleteExecutionContext(ctx);
}

}  // namespace
}  // namespace tensorflow
