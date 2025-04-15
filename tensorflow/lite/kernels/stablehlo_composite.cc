/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/control_flow_common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_composite {

struct State {
  int32_t subgraph_index;
  bool subgraph_has_dynamic_output_tensors = false;
};

void* Init(TfLiteContext* context, const char* options, size_t options_len) {
  auto data = std::make_unique<State>();
  const TfLiteStablehloCompositeParams* params =
      reinterpret_cast<const TfLiteStablehloCompositeParams*>(options);
  data->subgraph_index = params->subgraph_index;
  return data.release();
}

void Free(TfLiteContext* context, void* node_data) {
  delete static_cast<State*>(node_data);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  State* op_state = reinterpret_cast<State*>(node->user_data);

  TF_LITE_ENSURE(context, node->inputs->size > 0);

  const int num_inputs = node->inputs->size;
  const int num_outputs = node->outputs->size;

  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  const auto* subgraphs = this_subgraph->GetSubgraphs();
  TF_LITE_ENSURE(context, op_state->subgraph_index < subgraphs->size());

  Subgraph* decomposition_subgraph =
      (*subgraphs)[op_state->subgraph_index].get();

  TF_LITE_ENSURE_EQ(context, num_inputs,
                    decomposition_subgraph->inputs().size());
  TF_LITE_ENSURE_EQ(context, num_outputs,
                    decomposition_subgraph->outputs().size());

  // Remove unused inputs of subgraph to skip copying unnecessary inputs.
  decomposition_subgraph->RemoveUnusedInputs();

  std::vector<int> node_inputs(node->inputs->data,
                               node->inputs->data + num_inputs);

  // Prepare and check the subgraphs.
  TF_LITE_ENSURE_OK(context,
                    CopyTensorsShapeAndType(context, this_subgraph, node_inputs,
                                            decomposition_subgraph,
                                            decomposition_subgraph->inputs(),
                                            /*resize_subgraph_inputs=*/true));

  // Handle resource input tensors.
  for (int i = 0; i < num_inputs; ++i) {
    int input_idx = decomposition_subgraph->inputs()[i];
    if (input_idx == kTfLiteOptionalTensor) {
      continue;
    }
    TfLiteTensor* subgraph_input = decomposition_subgraph->tensor(input_idx);
    if (!IsResourceOrVariant(subgraph_input)) {
      // Set the allocation type to custom to prevent memory allocation.
      subgraph_input->allocation_type = kTfLiteCustom;
    }
  }

  // Allocate the memory for the subgraph.
  TF_LITE_ENSURE_OK(context, decomposition_subgraph->AllocateTensors());
  op_state->subgraph_has_dynamic_output_tensors |=
      decomposition_subgraph->HasDynamicTensors();

  for (int i = 0; i < num_outputs; ++i) {
    if (node->outputs->data[i] == kTfLiteOptionalTensor) {
      continue;
    }
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &output));
    if (op_state->subgraph_has_dynamic_output_tensors) {
      SetTensorToDynamic(output);
    } else {
      TfLiteTensor* subgraph_output =
          decomposition_subgraph->tensor(decomposition_subgraph->outputs()[i]);
      TfLiteIntArray* output_size = TfLiteIntArrayCopy(subgraph_output->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, output, output_size));
    }
  }
  return kTfLiteOk;
}

// Evaluate the COMPOSITE op when the subgraph has dynamic outputs.
TfLiteStatus Eval_dynamic(TfLiteContext* context, TfLiteNode* node,
                          Subgraph* this_subgraph,
                          Subgraph* decomposition_subgraph) {
  TF_LITE_ENSURE_OK(context, decomposition_subgraph->AllocateTensors());
  const int num_inputs = node->inputs->size;
  const int num_outputs = node->outputs->size;
  const int* const start = node->inputs->data;
  std::vector<int> node_inputs(start, start + num_inputs);
  // node->inputs -> subgraph->inputs
  TF_LITE_ENSURE_OK(
      context, DeepOrShallowCopyTensorsShapeTypeData(
                   context, node, this_subgraph, node_inputs,
                   decomposition_subgraph, decomposition_subgraph->inputs()));

  // Invoke decomposition_subgraph subgraph
  TF_LITE_ENSURE_OK(context, decomposition_subgraph->Invoke());
  for (int tensor_index : decomposition_subgraph->outputs()) {
    decomposition_subgraph->EnsureTensorDataIsReadable(tensor_index);
  }

  // subgraph->outputs -> node->outputs
  TF_LITE_ENSURE_OK(context,
                    DeepCopyTensorsShapeTypeData(
                        context, node, decomposition_subgraph,
                        decomposition_subgraph->outputs(), this_subgraph,
                        TfLiteIntArrayView(node->outputs), true));

  for (int i = 0; i < num_outputs; ++i) {
    const int input_pos = OutputIsInput(decomposition_subgraph->outputs()[i],
                                        decomposition_subgraph->inputs());
    if (input_pos != -1) {
      TfLiteTensor* this_input =
          this_subgraph->tensor(node->inputs->data[input_pos]);
      TfLiteTensor* this_output = this_subgraph->tensor(node->outputs->data[i]);
      TfLiteTensorCopy(this_input, this_output);
    }
  }
  return kTfLiteOk;
}

// Evaluate the COMPOSITE op when the subgraph has static outputs.
TfLiteStatus Eval_static(TfLiteContext* context, TfLiteNode* node,
                         Subgraph* this_subgraph,
                         Subgraph* decomposition_subgraph) {
  const int num_inputs = node->inputs->size;
  const int num_outputs = node->outputs->size;
  const int* const start = node->inputs->data;
  std::vector<int> node_inputs(start, start + num_inputs);
  for (int i = 0; i < num_outputs; ++i) {
    int output_idx = decomposition_subgraph->outputs()[i];
    if (output_idx == kTfLiteOptionalTensor) continue;
    TfLiteTensor* subgraph_output = decomposition_subgraph->tensor(output_idx);
    if (!IsResourceOrVariant(subgraph_output) &&
        !IsConstantTensor(subgraph_output)) {
      subgraph_output->allocation_type = kTfLiteCustom;
    }
  }
  // node->inputs -> subgraph->inputs
  TF_LITE_ENSURE_OK(
      context, DeepOrShallowCopyTensorsShapeTypeData(
                   context, node, this_subgraph, node_inputs,
                   decomposition_subgraph, decomposition_subgraph->inputs()));

  TF_LITE_ENSURE_OK(
      context,
      CopyTensorsShapeAndType(context, decomposition_subgraph,
                              decomposition_subgraph->outputs(), this_subgraph,
                              TfLiteIntArrayView(node->outputs), false));
  for (int i = 0; i < num_outputs; ++i) {
    TfLiteTensor* this_output = this_subgraph->tensor(node->outputs->data[i]);
    TfLiteTensor* subgraph_output =
        decomposition_subgraph->tensor(decomposition_subgraph->outputs()[i]);
    if (decomposition_subgraph->outputs()[i] == kTfLiteOptionalTensor) {
      TfLiteTensor* this_input = this_subgraph->tensor(node->inputs->data[i]);
      TfLiteTensorResizeMaybeCopy(this_input->bytes, this_output, false);
      TfLiteTensorCopy(this_input, this_output);
    } else {
      const int input_pos = OutputIsInput(decomposition_subgraph->outputs()[i],
                                          decomposition_subgraph->inputs());
      if (input_pos != -1) {
        TfLiteTensor* this_input =
            this_subgraph->tensor(node->inputs->data[input_pos]);
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
  TF_LITE_ENSURE_OK(context, decomposition_subgraph->Invoke());
  for (int tensor_index : decomposition_subgraph->outputs()) {
    decomposition_subgraph->EnsureTensorDataIsReadable(tensor_index);
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  State* op_state = reinterpret_cast<State*>(node->user_data);
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();
  Subgraph* decomposition_subgraph =
      (*subgraphs)[op_state->subgraph_index].get();

  if (op_state->subgraph_has_dynamic_output_tensors) {
    TF_LITE_ENSURE_OK(context, Eval_dynamic(context, node, this_subgraph,
                                            decomposition_subgraph));
  } else {
    TF_LITE_ENSURE_OK(context, Eval_static(context, node, this_subgraph,
                                           decomposition_subgraph));
  }

  if (!this_subgraph->ShouldPreserveAllTensors()) {
    TF_LITE_ENSURE_OK(context, decomposition_subgraph->ReleaseMemory());
  }

  return kTfLiteOk;
}

}  // namespace stablehlo_composite

TfLiteRegistration* Register_STABLEHLO_COMPOSITE() {
  static TfLiteRegistration r = {/*.init=*/stablehlo_composite::Init,
                                 /*.free=*/stablehlo_composite::Free,
                                 /*.prepare=*/stablehlo_composite::Prepare,
                                 /*.invoke=*/stablehlo_composite::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
