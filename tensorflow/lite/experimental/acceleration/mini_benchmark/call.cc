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
#include <stddef.h>

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/call_register.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace acceleration {
namespace ops {
namespace call_kernel {

namespace {

bool MatchDimensionsExceptBatchSize(TfLiteTensor* a, TfLiteTensor* b) {
  if (a->dims->size != b->dims->size) {
    return false;
  }
  // First dimension is the batch size.
  for (int i = 1; i < a->dims->size; ++i) {
    if (a->dims->data[i] != b->dims->data[i]) {
      return false;
    }
  }
  return true;
}

// Verifies that number, shape and type of inputs match for subgraph and CALL
// node. If shape of inputs is unset for subgraph, it is inferred.
TfLiteStatus ValidateAndResizeInputsIfNeeded(TfLiteContext* context,
                                             TfLiteNode* node,
                                             Subgraph* subgraph,
                                             int loop_count) {
  // Match number of inputs for subgraph and CALL node.
  TF_LITE_ENSURE_EQ(context, subgraph->inputs().size(), node->inputs->size);

  for (int i = 0; i < node->inputs->size; ++i) {
    TfLiteTensor* node_input = context->tensors + node->inputs->data[i];
    TfLiteTensor* subgraph_input = subgraph->tensor(subgraph->inputs()[i]);
    // Match input types.
    TF_LITE_ENSURE_TYPES_EQ(context, node_input->type, subgraph_input->type);
    TF_LITE_ENSURE_MSG(
        context, node_input->dims->size > 0,
        "Dimensions of all of call node's inputs should be non-zero.");
    // Ensure batch size of CALL node's input is same as loop size.
    TF_LITE_ENSURE_EQ(context, node_input->dims->data[0], loop_count);
    if (!subgraph_input->dims->size) {
      // Subgraph input dimensions unset and will be inferred.
      std::vector<int> new_dims;
      new_dims.reserve(node_input->dims->size);
      new_dims.push_back(1);  // Batch size is fixed as 1 for subgraph.
      new_dims.insert(new_dims.end(), node_input->dims->data + 1,
                      node_input->dims->data + node_input->dims->size);
      subgraph->ResizeInputTensor(subgraph->inputs()[i], new_dims);
    } else {
      // Dimensions already set for subgraph, match input dimensions.
      if (!MatchDimensionsExceptBatchSize(node_input, subgraph_input)) {
        std::stringstream node_input_dims, subgraph_input_dims;
        for (int i = 0; i < node_input->dims->size; i++) {
          node_input_dims << node_input->dims->data[i] << " ";
          subgraph_input_dims << subgraph_input->dims->data[i] << " ";
        }
        TF_LITE_KERNEL_LOG(
            context,
            "%s:%d: All dimensions except the batch size should match for call "
            "node and the subgraph to invoke (input tensor %s[ %s], subgraph "
            "tensor %s[ %s])",
            __FILE__, __LINE__, node_input->name, node_input_dims.str().c_str(),
            subgraph_input->name, subgraph_input_dims.str().c_str());
        return kTfLiteError;
      }
      // Batch size of subgraph's input should be 1.
      TF_LITE_ENSURE_EQ(context, subgraph_input->dims->data[0], 1);
    }
  }

  return kTfLiteOk;
}

// Verifies that number of outputs match for subgraph and CALL node. Infer the
// shape and type of outputs for CALL node and resize accordingly.
TfLiteStatus ValidateAndResizeOutputs(TfLiteContext* context, TfLiteNode* node,
                                      Subgraph* subgraph, int loop_count) {
  // Match number of outputs for subgraph and CALL node.
  TF_LITE_ENSURE_EQ(context, subgraph->outputs().size(), node->outputs->size);

  // Infer output shape for the CALL node.
  for (int i = 0; i < node->outputs->size; ++i) {
    const TfLiteTensor* subgraph_output =
        subgraph->tensor(subgraph->outputs()[i]);
    TfLiteTensor* node_output = context->tensors + node->outputs->data[i];

    TF_LITE_ASSERT(subgraph_output->dims->size > 0);
    TfLiteIntArray* new_dims_array = TfLiteIntArrayCopy(subgraph_output->dims);
    // Batch size is fixed as `loop_count` for CALL node.
    new_dims_array->data[0] = loop_count;
    TF_LITE_ENSURE_OK(
        context, context->ResizeTensor(context, node_output, new_dims_array));
    node_output->type = subgraph_output->type;
  }
  return kTfLiteOk;
}

// Copy input tensor data from CALL node's inputs to subgraph's inputs.
TfLiteStatus CopyInputTensorsData(TfLiteContext* context, TfLiteNode* node,
                                  Subgraph* dst_subgraph, int loop_index,
                                  int loop_count) {
  const std::vector<int>& dst_tensor_indices = dst_subgraph->inputs();
  TF_LITE_ENSURE_EQ(context, node->inputs->size, dst_tensor_indices.size());
  for (int i = 0; i < dst_tensor_indices.size(); ++i) {
    TfLiteTensor* src_tensor = context->tensors + node->inputs->data[i];
    TfLiteTensor* dst_tensor = dst_subgraph->tensor(dst_tensor_indices[i]);
    size_t offset = src_tensor->bytes / loop_count * loop_index;
    TF_LITE_ENSURE_EQ(context, src_tensor->bytes / loop_count,
                      dst_tensor->bytes);
    memcpy(dst_tensor->data.raw, src_tensor->data.raw + offset,
           src_tensor->bytes / loop_count);
  }
  return kTfLiteOk;
}

// Copy tensor data from subgraph's outputs to CALL node's outputs.
TfLiteStatus CopyOutputTensorsData(TfLiteContext* context,
                                   Subgraph* src_subgraph, TfLiteNode* node,
                                   int loop_index, int loop_count) {
  const std::vector<int>& src_tensor_indices = src_subgraph->outputs();
  TF_LITE_ENSURE_EQ(context, src_tensor_indices.size(), node->outputs->size);
  for (int i = 0; i < src_tensor_indices.size(); ++i) {
    const TfLiteTensor* src_tensor =
        src_subgraph->tensor(src_tensor_indices[i]);
    TfLiteTensor* dst_tensor = context->tensors + node->outputs->data[i];
    size_t offset = dst_tensor->bytes / loop_count * loop_index;
    TF_LITE_ENSURE_EQ(context, src_tensor->bytes,
                      dst_tensor->bytes / loop_count);
    memcpy(dst_tensor->data.raw + offset, src_tensor->data.raw,
           src_tensor->bytes);
  }
  return kTfLiteOk;
}

}  // namespace

struct OpData {
  // Index of the subgraph that needs to be invoked.
  // Subgraph should have batch size 1.
  int subgraph_index;
  // The number of times the CALL op should call the subgraph.
  // The inputs to the call op are expected to have this value as their batch
  // size.
  int loop_count;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  if (!buffer) {
    return nullptr;
  }
  auto* op_data = new OpData;
  const uint8_t* buffer_fixed_width = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& map =
      flexbuffers::GetRoot(buffer_fixed_width, length).AsMap();
  // Note: The values below will be set as 0 if the parsing fails or if the
  // values have been unset.
  op_data->subgraph_index = map["subgraph_index"].AsInt32();
  op_data->loop_count = map["loop_count"].AsInt32();
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, op_data);

  // Check subgraph index and get subgraph.
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();
  TF_LITE_ENSURE_MSG(context,
                     (op_data->subgraph_index < subgraphs->size()) &&
                         (op_data->subgraph_index >= 0),
                     "Index of subgraph to be invoked is invalid.");
  Subgraph* subgraph = (*subgraphs)[op_data->subgraph_index].get();
  TF_LITE_ENSURE_MSG(
      context, subgraph != this_subgraph,
      "Subgraph to invoke must be different from the invoking graph.");
  int loop_count = op_data->loop_count;
  TF_LITE_ENSURE_MSG(context, loop_count >= 0, "Loop count must be positive. ");

  // Check if the inputs and outputs of the CALL node and the subgraph have
  // proper shapes and types.
  TF_LITE_ENSURE_OK(context, ValidateAndResizeInputsIfNeeded(
                                 context, node, subgraph, loop_count));
  TF_LITE_ENSURE_OK(context, subgraph->AllocateTensors());
  TF_LITE_ENSURE_OK(
      context, ValidateAndResizeOutputs(context, node, subgraph, loop_count));

  // Since delegates don't support acceleration of models with dynamic outputs,
  // this op doesn't support it either.
  TF_LITE_ENSURE(context, !subgraph->HasDynamicTensors());

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();
  Subgraph* subgraph = (*subgraphs)[op_data->subgraph_index].get();

  // The following graph illustrates the current implementation.
  //
  // This Subgraph        Subgraph to invoke
  // +-----------+   (1)   +------------+
  // |   CALL    |-------->|  SUBGRAPH  |
  // |   INPUT   |         |   INPUT    |
  // +-----------+         +------------+
  //                            |
  //               (3)          | (2)
  //                            v
  // +-----------+        +------------+
  // |   CALL    |<-------|  SUBGRAPH  |
  // |   OUTPUT  |        |   OUTPUT   |
  // +-----------+        +------------+
  // For every ith loop iteration:
  // (1) Copy the ith input of CALL op to the inputs of subgraph.
  // (2) Invoke subgraph.
  // (3) Copy the outputs of subgraph to the ith output of CALL op.
  //
  // Requires the subgraph to have a batch size of 1.
  // Requires the CALL node's inputs and outputs to have a batch size equal to
  // `loop_count`.
  //
  //
  // TODO(b/120234921): Optimize and avoid copying tensors between subgraphs.

  for (int loop_index = 0; loop_index < op_data->loop_count; loop_index++) {
    // Copy inputs needed for this iteration.
    TF_LITE_ENSURE_OK(context,
                      CopyInputTensorsData(context, node, subgraph, loop_index,
                                           op_data->loop_count));
    // Invoke subgraph for this iteration.
    TF_LITE_ENSURE_OK(context, subgraph->Invoke());

    for (int tensor_index : subgraph->outputs()) {
      subgraph->EnsureTensorDataIsReadable(tensor_index);
    }

    TF_LITE_ENSURE_OK(context,
                      CopyOutputTensorsData(context, subgraph, node, loop_index,
                                            op_data->loop_count));
  }
  return kTfLiteOk;
}

}  // namespace call_kernel

TfLiteRegistration* Register_CALL() {
  static TfLiteRegistration r = {call_kernel::Init, call_kernel::Free,
                                 call_kernel::Prepare, call_kernel::Eval};
  return &r;
}

}  // namespace ops
}  // namespace acceleration
}  // namespace tflite
