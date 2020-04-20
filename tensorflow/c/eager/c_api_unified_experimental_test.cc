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
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TF_ExecutionContextOptions* options = TF_NewEagerContextOptions(opts);
  TF_ExecutionContext* ctx = TF_NewExecutionContext(options, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TFE_DeleteContextOptions(opts);

  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build an abstract input tensor.
  TFE_Context* eager_ctx = TF_ExecutionContextGetTFEContext(ctx);
  TFE_TensorHandle* t = TestScalarTensorHandle(eager_ctx, 2.0f);
  TF_AbstractTensor* at = TF_NewAbstractTensor();
  TF_AbstractTensorSetEagerTensor(at, t, status.get());
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
  TF_DeleteExecutionContextOptions(options);
}

TEST(UnifedCAPI, TestBasicGraph) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_ExecutionContextOptions* options = TF_NewGraphContextOptions();
  TF_ExecutionContext* graph_ctx =
      TF_NewExecutionContext(options, status.get());
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

  // Execute.
  TF_ExecuteOperation(add_op, 2, inputs, add_outputs, graph_ctx, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TF_AbstractTensor* output_t = TF_OutputListGet(add_outputs, 0);

  // Clean up operation and inputs.
  TF_DeleteAbstractOp(add_op);

  string fn_name = "double";
  TF_AbstractFunction* func = TF_ExecutionContextToFunction(
      graph_ctx, fn_name.c_str(), 1, placeholder_t, 1, output_t, status.get());
  TF_DeleteAbstractTensor(placeholder_t);
  TF_DeleteAbstractTensor(output_t);

  // Build eager context.
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TF_ExecutionContextOptions* eager_ctx_options =
      TF_NewEagerContextOptions(opts);
  TF_ExecutionContext* eager_execution_ctx =
      TF_NewExecutionContext(eager_ctx_options, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TFE_DeleteContextOptions(opts);

  TF_ExecutionContextRegisterFunction(eager_execution_ctx, func, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  // Build the abstract op to run the function.
  TF_AbstractOp* fn_op = TF_NewAbstractOp(eager_execution_ctx);
  TF_AbstractOpSetOpType(fn_op, fn_name.c_str(), status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build an abstract input tensor.
  TF_AbstractTensor* input_t = TF_NewAbstractTensor();
  TFE_Context* eager_ctx =
      TF_ExecutionContextGetTFEContext(eager_execution_ctx);
  TFE_TensorHandle* input_eager = TestScalarTensorHandle(eager_ctx, 2.0f);
  TF_AbstractTensorSetEagerTensor(input_t, input_eager, status.get());
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
  TF_DeleteOutputList(placeholder_outputs);
  TF_DeleteAbstractOp(fn_op);
  TF_DeleteAbstractTensor(input_t);
  TF_DeleteAbstractTensor(final_result);
  TF_DeleteTensor(f_t);
  TF_DeleteAbstractFunction(func);

  TF_DeleteExecutionContext(graph_ctx);
  TF_DeleteExecutionContext(eager_execution_ctx);
  TF_DeleteExecutionContextOptions(eager_ctx_options);
  TF_DeleteExecutionContextOptions(options);
}

TEST(UnifedCAPI, TF_ExecutionContextToFunctionWithEagerContextRaises) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TF_ExecutionContextOptions* options = TF_NewEagerContextOptions(opts);
  TF_ExecutionContext* ctx = TF_NewExecutionContext(options, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TFE_DeleteContextOptions(opts);

  TF_AbstractFunction* func = TF_ExecutionContextToFunction(
      ctx, nullptr, 0, nullptr, 0, nullptr, status.get());
  ASSERT_EQ(nullptr, func);
  ASSERT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(status.get()));

  TF_DeleteExecutionContext(ctx);
  TF_DeleteExecutionContextOptions(options);
}

TEST(UnifedCAPI, TF_CallingSetOpTypeAfterFinishingOpBuildingRaises) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_ExecutionContextOptions* options = TF_NewGraphContextOptions();
  TF_ExecutionContext* graph_ctx =
      TF_NewExecutionContext(options, status.get());
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
  TF_DeleteExecutionContextOptions(options);
}

TEST(UnifedCAPI, TF_CallingSetOpNameAfterFinishingOpBuildingRaises) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_ExecutionContextOptions* options = TF_NewGraphContextOptions();
  TF_ExecutionContext* graph_ctx =
      TF_NewExecutionContext(options, status.get());
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
  TF_DeleteExecutionContextOptions(options);
}

TEST(UnifedCAPI, TestExecutingEagerOpInGraphModeRaises) {
  // Build an Eager context.
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TF_ExecutionContextOptions* options = TF_NewEagerContextOptions(opts);
  TF_ExecutionContext* ctx = TF_NewExecutionContext(options, status.get());
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
  TF_AbstractTensor* at = TF_NewAbstractTensor();
  TF_AbstractTensorSetEagerTensor(at, t, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build inputs and outputs.
  TF_AbstractTensor* inputs[2] = {at, at};
  TF_OutputList* o = TF_NewOutputList();
  TF_OutputListSetNumOutputs(o, 1, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Build a Graph context.
  TF_ExecutionContextOptions* graph_options = TF_NewGraphContextOptions();
  TF_ExecutionContext* graph_ctx =
      TF_NewExecutionContext(graph_options, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Execute eager op using graph context.
  TF_ExecuteOperation(op, 2, inputs, o, graph_ctx, status.get());
  ASSERT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(status.get()));

  // Clean up operation and inputs.
  TF_DeleteAbstractOp(op);
  TF_DeleteAbstractTensor(at);

  TF_DeleteOutputList(o);
  TF_DeleteExecutionContext(ctx);
  TF_DeleteExecutionContextOptions(options);
  TF_DeleteExecutionContext(graph_ctx);
  TF_DeleteExecutionContextOptions(graph_options);
}

TEST(UnifedCAPI, TestExecutingGraphOpInEagerModeRaises) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_ExecutionContextOptions* options = TF_NewGraphContextOptions();
  TF_ExecutionContext* graph_ctx =
      TF_NewExecutionContext(options, status.get());
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
  TF_ExecutionContextOptions* eager_ctx_options =
      TF_NewEagerContextOptions(opts);
  TF_ExecutionContext* eager_execution_ctx =
      TF_NewExecutionContext(eager_ctx_options, status.get());
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
  TF_DeleteExecutionContextOptions(eager_ctx_options);
  TF_DeleteExecutionContextOptions(options);
}

}  // namespace
}  // namespace tensorflow
