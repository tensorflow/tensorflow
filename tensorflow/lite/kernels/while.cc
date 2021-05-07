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

#include <stddef.h>

#include <cstring>
#include <vector>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace while_kernel {

namespace {

// Propagate tensor shapes and types from `src_tensor_indices` in `src_subgraph`
// to `dst_tensor_indices` in `dst_subgraph`.
//
// When `resize_subgraph_inputs` is true, the function calls subgraphs's
// `ResizeInputTensor` function, and it may trigger the memory planner to
// reallocate memory.
// When `resize_subgraph_inputs` is false, it implies `context` belongs to
// `dst_subgraph`. The function calls `context->ResizeTensor`. This happens
// when resizing `While` op's outputs.
template <typename SrcVector, typename DstVector>
TfLiteStatus CopyTensorsShapeAndType(TfLiteContext* context,
                                     Subgraph* src_subgraph,
                                     const SrcVector& src_tensor_indices,
                                     Subgraph* dst_subgraph,
                                     const DstVector& dst_tensor_indices,
                                     bool resize_subgraph_inputs) {
  TF_LITE_ENSURE_EQ(context, src_tensor_indices.size(),
                    dst_tensor_indices.size());
  for (int i = 0; i < src_tensor_indices.size(); ++i) {
    const TfLiteTensor* src_tensor =
        src_subgraph->tensor(src_tensor_indices[i]);

    TfLiteTensor* dst_tensor = dst_subgraph->tensor(dst_tensor_indices[i]);
    if (resize_subgraph_inputs) {
      std::vector<int> dims(src_tensor->dims->data,
                            src_tensor->dims->data + src_tensor->dims->size);
      dst_subgraph->ResizeInputTensor(dst_tensor_indices[i], dims);
    } else {
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, dst_tensor,
                                         TfLiteIntArrayCopy(src_tensor->dims)));
    }
    dst_tensor->type = src_tensor->type;
  }
  return kTfLiteOk;
}

// Copy the tensors data from tensors `src_tensor_indices` in `src_subgraph`
// to `dst_tensor_indices` in `dst_subgraph`.
template <typename SrcVector, typename DstVector>
TfLiteStatus CopyTensorsData(TfLiteContext* context, Subgraph* src_subgraph,
                             const SrcVector& src_tensor_indices,
                             Subgraph* dst_subgraph,
                             const DstVector& dst_tensor_indices) {
  TF_LITE_ENSURE_EQ(context, src_tensor_indices.size(),
                    dst_tensor_indices.size());
  for (int i = 0; i < src_tensor_indices.size(); ++i) {
    const TfLiteTensor* src_tensor =
        src_subgraph->tensor(src_tensor_indices[i]);
    TfLiteTensor* dst_tensor = dst_subgraph->tensor(dst_tensor_indices[i]);
    if (IsDynamicTensor(dst_tensor)) {
      TfLiteTensorRealloc(src_tensor->bytes, dst_tensor);
    }
    TF_LITE_ENSURE_EQ(context, src_tensor->bytes, dst_tensor->bytes);
    memcpy(dst_tensor->data.raw, src_tensor->data.raw, src_tensor->bytes);
  }
  return kTfLiteOk;
}

TfLiteStatus CheckCondOutput(TfLiteContext* context,
                             const TfLiteTensor* cond_output) {
  // The condition output must be a single boolean value.
  TF_LITE_ENSURE_TYPES_EQ(context, cond_output->type, kTfLiteBool);
  if (cond_output->dims->size == 0) {
    // It's okay if it's a 0D scalar.
    return kTfLiteOk;
  }
  // Otherwise it must be 1D with shape [1].
  TF_LITE_ENSURE_EQ(context, cond_output->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, cond_output->dims->data[0], 1);
  return kTfLiteOk;
}

}  // namespace

struct OpData {
  int cond_subgraph_index;
  int body_subgraph_index;
  bool cond_has_dynamic_output_tensors;
  bool body_has_dynamic_output_tensors;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData;
  const auto* params = reinterpret_cast<const TfLiteWhileParams*>(buffer);
  op_data->cond_subgraph_index = params->cond_subgraph_index;
  op_data->body_subgraph_index = params->body_subgraph_index;
  op_data->cond_has_dynamic_output_tensors = false;
  op_data->body_has_dynamic_output_tensors = false;
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
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
  TF_LITE_ENSURE_OK(
      context, CopyTensorsShapeAndType(
                   context, this_subgraph, TfLiteIntArrayView(node->inputs),
                   cond_subgraph, cond_subgraph->inputs(), true));
  TF_LITE_ENSURE_OK(context, cond_subgraph->AllocateTensors());
  TfLiteTensor* cond_output =
      cond_subgraph->tensor(cond_subgraph->outputs()[0]);
  // This should rarely happens. In most cases the output is static with shape
  // [1]. However theoretically intermediate tensors in the cond subgraph
  // can be dynamic.
  if (IsDynamicTensor(cond_output)) {
    op_data->cond_has_dynamic_output_tensors = true;
  } else {
    TF_LITE_ENSURE_STATUS(CheckCondOutput(context, cond_output));
  }

  // Prepare and check the body subgraph.
  TF_LITE_ENSURE_OK(
      context, CopyTensorsShapeAndType(
                   context, this_subgraph, TfLiteIntArrayView(node->inputs),
                   body_subgraph, body_subgraph->inputs(), true));
  TF_LITE_ENSURE_OK(context, body_subgraph->AllocateTensors());
  if (body_subgraph->HasDynamicTensors()) {
    op_data->body_has_dynamic_output_tensors = true;
  } else {
    for (int i = 0; i < num_inputs; ++i) {
      TfLiteTensor* body_input =
          body_subgraph->tensor(body_subgraph->inputs()[i]);
      TfLiteTensor* body_output =
          body_subgraph->tensor(body_subgraph->outputs()[i]);
      TF_LITE_ENSURE_TYPES_EQ(context, body_input->type, body_output->type);

      TF_LITE_ENSURE(context, !IsDynamicTensor(body_output));
      if (!TfLiteIntArrayEqual(body_input->dims, body_output->dims)) {
        // If the output shape of the body subgraph is static w.r.t. a fixed
        // input size, but it's different from input size, it's still considered
        // dynamic. For example: If a subgraph keeps padding its input with a
        // fixed padding, the output shape is static w.r.t the input shape and
        // padding, but running it in a loop will keep bloating the tensor.
        op_data->body_has_dynamic_output_tensors = true;
        break;
      }
    }
  }
  for (int i = 0; i < num_inputs; ++i) {
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &output));
    if (op_data->body_has_dynamic_output_tensors) {
      SetTensorToDynamic(output);
    } else {
      TfLiteTensor* body_output =
          body_subgraph->tensor(body_subgraph->outputs()[i]);
      TfLiteIntArray* output_size = TfLiteIntArrayCopy(body_output->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, output, output_size));
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();
  Subgraph* cond_subgraph = (*subgraphs)[op_data->cond_subgraph_index].get();
  Subgraph* body_subgraph = (*subgraphs)[op_data->body_subgraph_index].get();

  // The follow graph illustrates the current implementation.
  //
  // This Subgraph          Cond Subgraph         Body Subgraph
  // +-----------+   (1)   +------------+   (3)   +------------+
  // |   WHILE   |-------->|  SUBGRAPH  |-------->|  SUBGRAPH  |
  // |   INPUT   |        /|   INPUT    |<-----   |   INPUT    |
  // +-----------+       / +------------+      \  +------------+
  //                    /        |              \       |
  //               (6) /         | (2)       (5) \      | (4)
  //                  /          v                \     v
  // +-----------+   /     +------------+         +------------+
  // |   WHILE   |<--      |  SUBGRAPH  |         |  SUBGRAPH  |
  // |   OUTPUT  |         |   OUTPUT   |         |   OUTPUT   |
  // +-----------+         +------------+         +------------+
  //
  // (1) Copy the inputs of WHILE op to the inputs of condition subgraph.
  // (2) Invoke condition subgraph.
  //     Jump to step 5 if result is false.
  // (3) Copy the inputs of condition subgraph to the inputs of body subgraph.
  // (4) Invoke body subgraph.
  // (5) Copy the outputs of body subgraph to the inputs condition subgraph.
  //     Jump back to step 2!
  // (6) Copy the inputs of condition subgraph to the outputs of WHILE op.
  //
  // If the body subgraph has dynamic sized outputs, it's required to resize the
  // tensor before copying in step 1, 3, 4 and 6.
  //
  // Note the flow is carefully designed to handle the dynamic sized output
  // case. The loop invariant is: The newest value is in the inputs of condition
  // subgraph. This is always true before step 2.
  //
  // This is the best we can do without sharing tensor buffer across subgraph
  // boundary. Currently we copy the input / output between the subgraphs. This
  // isn't optimized yet and a lot of redundant copies are made.
  // TODO(b/120234921): Optimize and avoid copying tensors between subgraphs.

  if (op_data->body_has_dynamic_output_tensors) {
    // If body subgraph has dynamic outputs, the input of condition subgraph may
    // be changed in the last invocation and may need resizing.
    TF_LITE_ENSURE_OK(
        context, CopyTensorsShapeAndType(
                     context, this_subgraph, TfLiteIntArrayView(node->inputs),
                     cond_subgraph, cond_subgraph->inputs(), true));
    TF_LITE_ENSURE_OK(context, cond_subgraph->AllocateTensors());
  }
  TF_LITE_ENSURE_OK(
      context,
      CopyTensorsData(context, this_subgraph, TfLiteIntArrayView(node->inputs),
                      cond_subgraph, cond_subgraph->inputs()));

  while (true) {
    TF_LITE_ENSURE_OK(context, cond_subgraph->Invoke());
    int cond_subgraph_output_index = cond_subgraph->outputs()[0];
    cond_subgraph->EnsureTensorDataIsReadable(cond_subgraph_output_index);
    TfLiteTensor* cond_output =
        cond_subgraph->tensor(cond_subgraph_output_index);
    if (op_data->cond_has_dynamic_output_tensors) {
      TF_LITE_ENSURE_STATUS(CheckCondOutput(context, cond_output));
    }

    if (!cond_output->data.b[0]) {
      break;
    }
    if (op_data->body_has_dynamic_output_tensors) {
      TF_LITE_ENSURE_OK(context,
                        CopyTensorsShapeAndType(
                            context, cond_subgraph, cond_subgraph->inputs(),
                            body_subgraph, body_subgraph->inputs(), true));
      TF_LITE_ENSURE_OK(context, body_subgraph->AllocateTensors());
    }

    TF_LITE_ENSURE_OK(
        context,
        CopyTensorsData(context, cond_subgraph, cond_subgraph->inputs(),
                        body_subgraph, body_subgraph->inputs()));

    TF_LITE_ENSURE_OK(context, body_subgraph->Invoke());

    for (int tensor_index : body_subgraph->outputs()) {
      body_subgraph->EnsureTensorDataIsReadable(tensor_index);
    }

    if (op_data->body_has_dynamic_output_tensors) {
      TF_LITE_ENSURE_OK(context,
                        CopyTensorsShapeAndType(
                            context, body_subgraph, body_subgraph->outputs(),
                            cond_subgraph, cond_subgraph->inputs(), true));
      TF_LITE_ENSURE_OK(context, cond_subgraph->AllocateTensors());
    }

    TF_LITE_ENSURE_OK(
        context,
        CopyTensorsData(context, body_subgraph, body_subgraph->outputs(),
                        cond_subgraph, cond_subgraph->inputs()));
  }

  // Note that copying from body's output will fail if body is never invoked.
  // TODO(b/120234921): Optimize and avoid copying tensors between subgraphs.
  if (op_data->body_has_dynamic_output_tensors) {
    TF_LITE_ENSURE_OK(
        context, CopyTensorsShapeAndType(
                     context, cond_subgraph, cond_subgraph->inputs(),
                     this_subgraph, TfLiteIntArrayView(node->outputs), false));
  }

  TF_LITE_ENSURE_OK(
      context,
      CopyTensorsData(context, cond_subgraph, cond_subgraph->inputs(),
                      this_subgraph, TfLiteIntArrayView(node->outputs)));
  return kTfLiteOk;
}

}  // namespace while_kernel

TfLiteRegistration* Register_WHILE() {
  static TfLiteRegistration r = {while_kernel::Init, while_kernel::Free,
                                 while_kernel::Prepare, while_kernel::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
