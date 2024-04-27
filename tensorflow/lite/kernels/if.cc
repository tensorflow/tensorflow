/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/control_flow_common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace if_kernel {

struct OpData {
  int then_subgraph_index;
  int else_subgraph_index;
  bool subgraph_has_dynamic_output_tensors;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData;
  const auto* params = reinterpret_cast<const TfLiteIfParams*>(buffer);
  op_data->then_subgraph_index = params->then_subgraph_index;
  op_data->else_subgraph_index = params->else_subgraph_index;
  op_data->subgraph_has_dynamic_output_tensors = false;
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE(context, node->inputs->size > 0);

  // The first input is the condition.
  const TfLiteTensor* cond;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &cond));
  // Currently only bool is supported.
  // TODO(ycling): Support other types since TensorFlow also support
  // non-bool types as condition.
  TF_LITE_ENSURE_EQ(context, cond->type, kTfLiteBool);
  TF_LITE_ENSURE_EQ(context, NumElements(cond), 1);

  // The first input of the node is the condition. The rest of inputs are
  // passed to the branch subgraphs. Therefore, the number of subgraph inputs
  // will be the number of node inputs - 1.
  int num_inputs = node->inputs->size - 1;
  int num_outputs = node->outputs->size;

  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();
  TF_LITE_ENSURE(context, op_data->then_subgraph_index < subgraphs->size());
  TF_LITE_ENSURE(context, op_data->else_subgraph_index < subgraphs->size());

  Subgraph* then_subgraph = (*subgraphs)[op_data->then_subgraph_index].get();
  Subgraph* else_subgraph = (*subgraphs)[op_data->else_subgraph_index].get();

  for (auto* subgraph : {then_subgraph, else_subgraph}) {
    TF_LITE_ENSURE_EQ(context, num_inputs, subgraph->inputs().size());
    TF_LITE_ENSURE_EQ(context, num_outputs, subgraph->outputs().size());
  }

  // Remove unused inputs of both subgraphs to skip copying unnecessary
  // inputs.
  then_subgraph->RemoveUnusedInputs();
  else_subgraph->RemoveUnusedInputs();

  const int* const start = node->inputs->data + 1;
  std::vector<int> node_inputs(start, start + num_inputs);
  // Prepare and check the subgraphs.
  for (auto* subgraph : {then_subgraph, else_subgraph}) {
    TF_LITE_ENSURE_OK(
        context, CopyTensorsShapeAndType(context, this_subgraph, node_inputs,
                                         subgraph, subgraph->inputs(), true));
  }

  for (auto* subgraph : {then_subgraph, else_subgraph}) {
    for (int i = 0; i < num_inputs; ++i) {
      int input_idx = subgraph->inputs()[i];
      if (input_idx == kTfLiteOptionalTensor) continue;
      TfLiteTensor* subgraph_input = subgraph->tensor(input_idx);
      if (!IsResourceOrVariant(subgraph_input)) {
        // Set the allocation type to custom to prevent memory allocation.
        subgraph_input->allocation_type = kTfLiteCustom;
      }
    }
    TF_LITE_ENSURE_OK(context, subgraph->AllocateTensors());
    op_data->subgraph_has_dynamic_output_tensors |=
        subgraph->HasDynamicTensors();
  }

  if (!op_data->subgraph_has_dynamic_output_tensors) {
    for (int i = 0; i < num_outputs; ++i) {
      TfLiteTensor* then_output =
          then_subgraph->tensor(then_subgraph->outputs()[i]);
      TfLiteTensor* else_output =
          else_subgraph->tensor(else_subgraph->outputs()[i]);
      // If the 2 subgraphs have static but different output shapes, the output
      // tensors of the IF op have dynamic sizes.
      if (!TfLiteIntArrayEqual(then_output->dims, else_output->dims)) {
        op_data->subgraph_has_dynamic_output_tensors = true;
        break;
      }
    }
  }

  for (int i = 0; i < num_outputs; ++i) {
    if (node->outputs->data[i] == kTfLiteOptionalTensor) continue;
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &output));
    if (op_data->subgraph_has_dynamic_output_tensors) {
      SetTensorToDynamic(output);
    } else {
      TfLiteTensor* then_output =
          then_subgraph->tensor(then_subgraph->outputs()[i]);
      TfLiteIntArray* output_size = TfLiteIntArrayCopy(then_output->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, output, output_size));
    }
  }
  return kTfLiteOk;
}

// Returns the subgraph input tensor index if the given output is also an input.
int output_is_input(int output_idx, const std::vector<int>& subgraph_inputs) {
  auto e =
      std::find(subgraph_inputs.begin(), subgraph_inputs.end(), output_idx);
  return (e != subgraph_inputs.end()) ? (e - subgraph_inputs.begin()) : -1;
}

// Evaluate IF op when subgraphs have dynamic outputs.
TfLiteStatus Eval_dynamic(TfLiteContext* context, TfLiteNode* node,
                          Subgraph* active_branch_subgraph) {
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);

  TF_LITE_ENSURE_OK(context, active_branch_subgraph->AllocateTensors());
  const int num_inputs = node->inputs->size - 1;
  const int num_outputs = node->outputs->size;
  const int* const start = node->inputs->data + 1;
  std::vector<int> node_inputs(start, start + num_inputs);
  // node->inputs -> subgraph->inputs
  TF_LITE_ENSURE_OK(
      context, DeepOrShallowCopyTensorsShapeTypeData(
                   context, node, this_subgraph, node_inputs,
                   active_branch_subgraph, active_branch_subgraph->inputs()));

  // Invoke active_branch_subgraph subgraph
  TF_LITE_ENSURE_OK(context, active_branch_subgraph->Invoke());
  for (int tensor_index : active_branch_subgraph->outputs()) {
    active_branch_subgraph->EnsureTensorDataIsReadable(tensor_index);
  }

  // subgraph->outputs -> node->outputs
  TF_LITE_ENSURE_OK(context,
                    DeepCopyTensorsShapeTypeData(
                        context, node, active_branch_subgraph,
                        active_branch_subgraph->outputs(), this_subgraph,
                        TfLiteIntArrayView(node->outputs), true));

  for (int i = 0; i < num_outputs; ++i) {
    const int input_pos = output_is_input(active_branch_subgraph->outputs()[i],
                                          active_branch_subgraph->inputs());
    if (input_pos != -1) {
      TfLiteTensor* this_input =
          this_subgraph->tensor(node->inputs->data[input_pos + 1]);
      TfLiteTensor* this_output = this_subgraph->tensor(node->outputs->data[i]);
      TfLiteTensorCopy(this_input, this_output);
    }
  }
  return kTfLiteOk;
}

// Evaluate IF op when subgraphs has static outputs.
TfLiteStatus Eval_static(TfLiteContext* context, TfLiteNode* node,
                         Subgraph* active_branch_subgraph) {
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);

  const int num_inputs = node->inputs->size - 1;
  const int num_outputs = node->outputs->size;
  const int* const start = node->inputs->data + 1;
  std::vector<int> node_inputs(start, start + num_inputs);
  for (int i = 0; i < num_outputs; ++i) {
    int output_idx = active_branch_subgraph->outputs()[i];
    if (output_idx == kTfLiteOptionalTensor) continue;
    TfLiteTensor* subgraph_output = active_branch_subgraph->tensor(output_idx);
    if (!IsResourceOrVariant(subgraph_output) &&
        !IsConstantTensor(subgraph_output)) {
      subgraph_output->allocation_type = kTfLiteCustom;
    }
  }
  // node->inputs -> subgraph->inputs
  TF_LITE_ENSURE_OK(
      context, DeepOrShallowCopyTensorsShapeTypeData(
                   context, node, this_subgraph, node_inputs,
                   active_branch_subgraph, active_branch_subgraph->inputs()));

  TF_LITE_ENSURE_OK(
      context,
      CopyTensorsShapeAndType(context, active_branch_subgraph,
                              active_branch_subgraph->outputs(), this_subgraph,
                              TfLiteIntArrayView(node->outputs), false));
  for (int i = 0; i < num_outputs; ++i) {
    TfLiteTensor* this_output = this_subgraph->tensor(node->outputs->data[i]);
    TfLiteTensor* subgraph_output =
        active_branch_subgraph->tensor(active_branch_subgraph->outputs()[i]);
    if (active_branch_subgraph->outputs()[i] == kTfLiteOptionalTensor) {
      TfLiteTensor* this_input =
          this_subgraph->tensor(node->inputs->data[i + 1]);
      TfLiteTensorResizeMaybeCopy(this_input->bytes, this_output, false);
      TfLiteTensorCopy(this_input, this_output);
    } else {
      const int input_pos =
          output_is_input(active_branch_subgraph->outputs()[i],
                          active_branch_subgraph->inputs());
      if (input_pos != -1) {
        TfLiteTensor* this_input =
            this_subgraph->tensor(node->inputs->data[input_pos + 1]);
        TfLiteTensorResizeMaybeCopy(this_input->bytes, this_output, false);
        TfLiteTensorCopy(this_input, this_output);
      } else if (IsConstantTensor(subgraph_output)) {
        TfLiteTensorCopy(subgraph_output, this_output);
      } else {
        subgraph_output->data = this_output->data;
      }
    }
  }

  // Invoke subgraph
  TF_LITE_ENSURE_OK(context, active_branch_subgraph->Invoke());
  for (int tensor_index : active_branch_subgraph->outputs()) {
    active_branch_subgraph->EnsureTensorDataIsReadable(tensor_index);
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();
  Subgraph* then_subgraph = (*subgraphs)[op_data->then_subgraph_index].get();
  Subgraph* else_subgraph = (*subgraphs)[op_data->else_subgraph_index].get();

  const TfLiteTensor* cond;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &cond));
  bool cond_value = cond->data.b[0];

  Subgraph* active_branch_subgraph;
  if (cond_value) {
    active_branch_subgraph = then_subgraph;
  } else {
    active_branch_subgraph = else_subgraph;
  }

  if (op_data->subgraph_has_dynamic_output_tensors) {
    TF_LITE_ENSURE_OK(context,
                      Eval_dynamic(context, node, active_branch_subgraph));
  } else {
    TF_LITE_ENSURE_OK(context,
                      Eval_static(context, node, active_branch_subgraph));
  }

  if (!this_subgraph->ShouldPreserveAllTensors()) {
    TF_LITE_ENSURE_OK(context, active_branch_subgraph->ReleaseMemory());
  }

  return kTfLiteOk;
}
}  // namespace if_kernel

TfLiteRegistration* Register_IF() {
  static TfLiteRegistration r = {if_kernel::Init, if_kernel::Free,
                                 if_kernel::Prepare, if_kernel::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
