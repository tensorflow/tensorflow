/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <stddef.h>

#include <cstring>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_graph.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

namespace {

struct OpData {
  int then_subgraph_index;
  int else_subgraph_index;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  const auto* params =
      reinterpret_cast<const TfLiteIfParams*>(node->builtin_data);
  op_data->then_subgraph_index = params->then_subgraph_index;
  op_data->else_subgraph_index = params->else_subgraph_index;

  TF_LITE_ENSURE(context, node->inputs->size > 0);

  // The first input is the condition.
  const TfLiteTensor* cond;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &cond));
  TF_LITE_ENSURE_EQ(context, cond->type, kTfLiteBool);
  TF_LITE_ENSURE_EQ(context, NumElements(cond), 1);

  // The first input of the node is the condition. The rest of inputs are
  // passed to the branch subgraphs. Therefore, the number of subgraph inputs
  // will be the number of node inputs - 1.
  size_t num_inputs = node->inputs->size - 1;
  size_t num_outputs = node->outputs->size;

  // Casting to TfliteIntArray is required since we are re-using
  // GetExecutionPlan from TfLiteContext. On TFLM this method returns a
  // MicroGraph.
  // TODO(b/188226309): Design a cleaner way to get a graph from kernel context.
  MicroGraph* graph_info;
  context->GetExecutionPlan(context,
                            reinterpret_cast<TfLiteIntArray**>(&graph_info));

  TF_LITE_ENSURE(context,
                 op_data->then_subgraph_index < graph_info->NumSubgraphs());
  TF_LITE_ENSURE(context,
                 op_data->else_subgraph_index < graph_info->NumSubgraphs());

  TF_LITE_ENSURE_EQ(
      context, num_inputs,
      graph_info->NumSubgraphInputs(op_data->then_subgraph_index));
  TF_LITE_ENSURE_EQ(
      context, num_outputs,
      graph_info->NumSubgraphOutputs(op_data->then_subgraph_index));

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* cond;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &cond));
  bool cond_value = cond->data.b[0];

  // Casting to TfliteIntArray is required since we are re-using
  // GetExecutionPlan from TfLiteContext. On TFLM this method returns a
  // MicroGraph.
  // TODO(b/188226309): Design a cleaner way to get a graph from kernel context.
  MicroGraph* graph_info;
  context->GetExecutionPlan(context,
                            reinterpret_cast<TfLiteIntArray**>(&graph_info));

  // Currently we copy the input / output between the subgraphs. This isn't
  // optimized yet.
  int active_branch_subgraph_index =
      cond_value ? op_data->then_subgraph_index : op_data->else_subgraph_index;

  for (size_t i = 0;
       i < graph_info->NumSubgraphInputs(active_branch_subgraph_index); ++i) {
    const TfLiteEvalTensor* input =
        tflite::micro::GetEvalInput(context, node, i + 1);

    TfLiteEvalTensor* subgraph_input =
        graph_info->GetSubgraphInput(active_branch_subgraph_index, i);

    // These checks must occur in Eval since TfLiteEvalTensors are not available
    // during Prepare.
    size_t input_bytes;
    size_t subgraph_input_bytes;
    TF_LITE_ENSURE_OK(context, TfLiteEvalTensorByteLength(input, &input_bytes));
    TF_LITE_ENSURE_OK(context, TfLiteEvalTensorByteLength(
                                   subgraph_input, &subgraph_input_bytes));
    TF_LITE_ENSURE_TYPES_EQ(context, input->type, subgraph_input->type);
    TF_LITE_ENSURE_EQ(context, input_bytes, subgraph_input_bytes);
    memcpy(subgraph_input->data.raw, input->data.raw, input_bytes);
  }

  TF_LITE_ENSURE_OK(context,
                    graph_info->InvokeSubgraph(active_branch_subgraph_index));

  for (size_t i = 0;
       i < graph_info->NumSubgraphOutputs(active_branch_subgraph_index); ++i) {
    const TfLiteEvalTensor* output =
        tflite::micro::GetEvalOutput(context, node, i);

    TfLiteEvalTensor* subgraph_output =
        graph_info->GetSubgraphOutput(active_branch_subgraph_index, i);

    // These checks must occur in Eval since TfLiteEvalTensors are not available
    // during Prepare.
    size_t output_bytes;
    size_t subgraph_output_bytes;
    TF_LITE_ENSURE_OK(context,
                      TfLiteEvalTensorByteLength(output, &output_bytes));
    TF_LITE_ENSURE_OK(context, TfLiteEvalTensorByteLength(
                                   subgraph_output, &subgraph_output_bytes));
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, subgraph_output->type);
    TF_LITE_ENSURE_EQ(context, output_bytes, subgraph_output_bytes);
    memcpy(output->data.raw, subgraph_output->data.raw, output_bytes);
  }
  return kTfLiteOk;
}

}  // namespace.

TfLiteRegistration Register_IF() {
  return {/*init=*/Init,
          /*free=*/nullptr,
          /*prepare=*/Prepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
