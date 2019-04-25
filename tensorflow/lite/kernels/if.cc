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
#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace if_kernel {

struct OpData {
  int then_subgraph_index;
  int else_subgraph_index;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData;
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  op_data->then_subgraph_index = m["then_subgraph_index"].AsInt32();
  op_data->else_subgraph_index = m["else_subgraph_index"].AsInt32();
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE(context, node->inputs->size > 0);

  // The first input is the condition.
  const TfLiteTensor* cond = GetInput(context, node, 0);
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

  bool has_dynamic_output_tensors = false;
  for (auto* subgraph : {then_subgraph, else_subgraph}) {
    for (int i = 0; i < num_inputs; ++i) {
      // The first input of the node is the condition. The indices of the inputs
      // passed to the subgraphs are offset by 1.
      const TfLiteTensor* input = GetInput(context, node, i + 1);
      std::vector<int> dims(input->dims->data,
                            input->dims->data + input->dims->size);
      subgraph->ResizeInputTensor(i, dims);
      TfLiteTensor* subgraph_input = subgraph->tensor(subgraph->inputs()[i]);
      TF_LITE_ENSURE_EQ(context, input->type, subgraph_input->type);
    }
    // Note: The `Prepare` function is responsible to run `AllocateTensors` on
    // both subgraphs. It's intentionally not to break out of the loop when
    // finding a dynamic output tensor.
    TF_LITE_ENSURE_OK(context, subgraph->AllocateTensors());
    has_dynamic_output_tensors |= subgraph->HasDynamicTensors();
  }

  if (!has_dynamic_output_tensors) {
    for (int i = 0; i < num_outputs; ++i) {
      TfLiteTensor* then_output =
          then_subgraph->tensor(then_subgraph->outputs()[i]);
      TfLiteTensor* else_output =
          else_subgraph->tensor(else_subgraph->outputs()[i]);
      // If the 2 subgraphs have static but different output shapes, the output
      // tensors of the IF op have dynamic sizes.
      if (!TfLiteIntArrayEqual(then_output->dims, else_output->dims)) {
        has_dynamic_output_tensors = true;
        break;
      }
    }
  }

  for (int i = 0; i < num_outputs; ++i) {
    TfLiteTensor* output = GetOutput(context, node, i);
    if (has_dynamic_output_tensors) {
      SetTensorToDynamic(output);
    } else {
      // When there's no dynamic output tensors, the 2 subgraph has exactly
      // the same static sized outputs.
      TfLiteTensor* then_output =
          then_subgraph->tensor(then_subgraph->outputs()[i]);
      TfLiteIntArray* output_size = TfLiteIntArrayCopy(then_output->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, output, output_size));
    }
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* cond = GetInput(context, node, 0);
  bool cond_value = cond->data.b[0];

  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();

  // Currently we copy the input / output between the subgraphs. This isn't
  // optimized yet.
  // TODO(b/120234921): Optimize and avoid copying tensors between subgraphs.
  int active_branch_subgraph_index =
      cond_value ? op_data->then_subgraph_index : op_data->else_subgraph_index;
  Subgraph& active_branch_subgraph =
      *(*subgraphs)[active_branch_subgraph_index];
  for (int i = 0; i < active_branch_subgraph.inputs().size(); ++i) {
    const TfLiteTensor* input = GetInput(context, node, i + 1);
    TfLiteTensor* subgraph_input =
        active_branch_subgraph.tensor(active_branch_subgraph.inputs()[i]);
    TF_LITE_ENSURE_EQ(context, input->bytes, subgraph_input->bytes);
    memcpy(subgraph_input->data.raw, input->data.raw, input->bytes);
  }

  // Note: It's guaranteed that the subgraphs' `AllocateTensors` are called
  // in `Prepare`, so we don't need to do it here again.
  TF_LITE_ENSURE_OK(context, active_branch_subgraph.Invoke());

  for (int tensor_index : active_branch_subgraph.outputs()) {
    active_branch_subgraph.EnsureTensorDataIsReadable(tensor_index);
  }

  bool has_dynamic_output_tensors = false;
  for (int i = 0; i < node->outputs->size; ++i) {
    TfLiteTensor* output = GetOutput(context, node, i);
    if (IsDynamicTensor(output)) {
      has_dynamic_output_tensors = true;
      break;
    }
  }

  if (has_dynamic_output_tensors) {
    for (int i = 0; i < node->outputs->size; ++i) {
      TfLiteTensor* output = GetOutput(context, node, i);
      TfLiteTensor* subgraph_output =
          active_branch_subgraph.tensor(active_branch_subgraph.outputs()[i]);
      TfLiteIntArray* output_size = TfLiteIntArrayCopy(subgraph_output->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, output, output_size));
    }
  }

  for (int i = 0; i < active_branch_subgraph.outputs().size(); ++i) {
    const TfLiteTensor* subgraph_output =
        active_branch_subgraph.tensor(active_branch_subgraph.outputs()[i]);
    TfLiteTensor* output = GetOutput(context, node, i);
    TF_LITE_ENSURE_EQ(context, output->bytes, subgraph_output->bytes);
    memcpy(output->data.raw, subgraph_output->data.raw, output->bytes);
  }
  return kTfLiteOk;
}

}  // namespace if_kernel

TfLiteRegistration* Register_IF() {
  static TfLiteRegistration r = {if_kernel::Init, if_kernel::Free,
                                 if_kernel::Prepare, if_kernel::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
