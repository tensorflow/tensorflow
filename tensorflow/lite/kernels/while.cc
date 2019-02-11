/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace while_kernel {

namespace {

TfLiteStatus ResizeSubgraphInputs(TfLiteContext* context, TfLiteNode* node,
                                  Subgraph* subgraph) {
  int num_inputs = node->inputs->size;
  for (int i = 0; i < num_inputs; ++i) {
    const TfLiteTensor* input = GetInput(context, node, i);
    std::vector<int> dims(input->dims->data,
                          input->dims->data + input->dims->size);
    subgraph->ResizeInputTensor(i, dims);
    TfLiteTensor* subgraph_input = subgraph->tensor(subgraph->inputs()[i]);
    TF_LITE_ENSURE_EQ(context, input->type, subgraph_input->type);
  }
  return kTfLiteOk;
}

template <typename SrcVector, typename DstVector>
TfLiteStatus CopyTensors(TfLiteContext* context, Subgraph* src_subgraph,
                         const SrcVector& src_tensor_indices,
                         Subgraph* dst_subgraph,
                         const DstVector& dst_tensor_indices) {
  TF_LITE_ENSURE_EQ(context, src_tensor_indices.size(),
                    dst_tensor_indices.size());
  for (int i = 0; i < src_tensor_indices.size(); ++i) {
    const TfLiteTensor* src_tensor =
        src_subgraph->tensor(src_tensor_indices[i]);
    TfLiteTensor* dst_tensor = dst_subgraph->tensor(dst_tensor_indices[i]);
    TF_LITE_ENSURE_EQ(context, src_tensor->bytes, dst_tensor->bytes);
    memcpy(dst_tensor->data.raw, src_tensor->data.raw, src_tensor->bytes);
  }
  return kTfLiteOk;
}

}  // namespace

struct OpData {
  int cond_subgraph_index;
  int body_subgraph_index;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData;
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  op_data->cond_subgraph_index = m["cond_subgraph_index"].AsInt32();
  op_data->body_subgraph_index = m["body_subgraph_index"].AsInt32();
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  int num_inputs = node->inputs->size;
  // The number of outputs should be the same as number of inputs.
  TF_LITE_ENSURE_EQ(context, node->outputs->size, num_inputs);

  // Check subgraph indices and get subgraphs.
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();
  TF_LITE_ENSURE(context, op_data->cond_subgraph_index < subgraphs->size());
  TF_LITE_ENSURE(context, op_data->body_subgraph_index < subgraphs->size());

  Subgraph* cond_subgraph = (*subgraphs)[op_data->cond_subgraph_index].get();
  Subgraph* body_subgraph = (*subgraphs)[op_data->body_subgraph_index].get();

  // Check input & output count of the condition subgraph.
  TF_LITE_ENSURE_EQ(context, cond_subgraph->inputs().size(), num_inputs);
  TF_LITE_ENSURE_EQ(context, cond_subgraph->outputs().size(), 1);

  // Check input & output count of the body subgraph.
  TF_LITE_ENSURE_EQ(context, body_subgraph->inputs().size(), num_inputs);
  TF_LITE_ENSURE_EQ(context, body_subgraph->outputs().size(), num_inputs);

  // Prepare and check the condition subgraph.
  ResizeSubgraphInputs(context, node, cond_subgraph);
  TF_LITE_ENSURE_OK(context, cond_subgraph->AllocateTensors());
  TfLiteTensor* cond_output =
      cond_subgraph->tensor(cond_subgraph->outputs()[0]);
  // The condition output must be a single boolean value.
  TF_LITE_ENSURE_EQ(context, cond_output->type, kTfLiteBool);
  TF_LITE_ENSURE_EQ(context, cond_output->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, cond_output->dims->data[0], 1);

  // TODO(ycling): Handle the case where condition graph has dynamic
  // sized tensors.

  // Prepare and check the body subgraph.
  ResizeSubgraphInputs(context, node, body_subgraph);
  TF_LITE_ENSURE_OK(context, body_subgraph->AllocateTensors());
  for (int i = 0; i < num_inputs; ++i) {
    TfLiteTensor* body_input =
        body_subgraph->tensor(body_subgraph->inputs()[i]);
    TfLiteTensor* body_output =
        body_subgraph->tensor(body_subgraph->outputs()[i]);
    TF_LITE_ENSURE_EQ(context, body_input->type, body_output->type);

    // TODO(ycling): Support dynamic sized body subgraph.
    TF_LITE_ENSURE(context, !IsDynamicTensor(body_output));
    TF_LITE_ENSURE(context,
                   TfLiteIntArrayEqual(body_input->dims, body_output->dims));

    TfLiteTensor* output = GetOutput(context, node, i);
    TfLiteIntArray* output_size = TfLiteIntArrayCopy(body_output->dims);
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, output, output_size));
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();
  Subgraph* cond_subgraph = (*subgraphs)[op_data->cond_subgraph_index].get();
  Subgraph* body_subgraph = (*subgraphs)[op_data->body_subgraph_index].get();

  // Currently we copy the input / output between the subgraphs. This isn't
  // optimized yet.
  // TODO(b/120234921): Optimize and avoid copying tensors between subgraphs.
  TF_LITE_ENSURE_OK(
      context,
      CopyTensors(context, this_subgraph, TfLiteIntArrayView(node->inputs),
                  cond_subgraph, cond_subgraph->inputs()));
  TF_LITE_ENSURE_OK(
      context,
      CopyTensors(context, this_subgraph, TfLiteIntArrayView(node->inputs),
                  body_subgraph, body_subgraph->inputs()));

  while (true) {
    TF_LITE_ENSURE_OK(context, cond_subgraph->Invoke());
    TfLiteTensor* cond_output =
        cond_subgraph->tensor(cond_subgraph->outputs()[0]);
    if (!cond_output->data.b[0]) {
      break;
    }
    TF_LITE_ENSURE_OK(context, body_subgraph->Invoke());

    TF_LITE_ENSURE_OK(
        context, CopyTensors(context, body_subgraph, body_subgraph->outputs(),
                             body_subgraph, body_subgraph->inputs()));
    TF_LITE_ENSURE_OK(
        context, CopyTensors(context, body_subgraph, body_subgraph->outputs(),
                             cond_subgraph, cond_subgraph->inputs()));
  }

  // Note that copying from body's output will fail if body is never invoked.
  // TODO(b/120234921): Optimize and avoid copying tensors between subgraphs.
  TF_LITE_ENSURE_OK(
      context, CopyTensors(context, body_subgraph, body_subgraph->inputs(),
                           this_subgraph, TfLiteIntArrayView(node->outputs)));
  return kTfLiteOk;
}

}  // namespace while_kernel

TfLiteRegistration* Register_WHILE() {
  static TfLiteRegistration r = {while_kernel::Init, while_kernel::Free,
                                 while_kernel::Prepare, while_kernel::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
