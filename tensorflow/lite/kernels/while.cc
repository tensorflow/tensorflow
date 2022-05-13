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

struct OpData {
  int cond_subgraph_index;
  int body_subgraph_index;
  bool cond_has_dynamic_output_tensors;
  bool body_has_dynamic_output_tensors;
  bool body_use_shallow_copy;
  bool subgraphs_allocated;
  // set when Prepare_impl() is called.
  bool subgraphs_prepared;
};

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
    // Skip copying unused destination tensors.
    if (dst_tensor_indices[i] == kTfLiteOptionalTensor) continue;

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
    // Skip copying unused destination tensors.
    if (dst_tensor_indices[i] == kTfLiteOptionalTensor) continue;

    const TfLiteTensor* src_tensor =
        src_subgraph->tensor(src_tensor_indices[i]);
    TfLiteTensor* dst_tensor = dst_subgraph->tensor(dst_tensor_indices[i]);
    if (IsDynamicTensor(dst_tensor)) {
      TfLiteTensorRealloc(src_tensor->bytes, dst_tensor);
    }
    TF_LITE_ENSURE_EQ(context, src_tensor->bytes, dst_tensor->bytes);
    TfLiteTensorCopy(src_tensor, dst_tensor);
  }
  return kTfLiteOk;
}

// Propagate tensor shapes and types from `src_tensor_indices` in `src_subgraph`
// to `dst_tensor_indices` in `dst_subgraph` and copy data deeply.
template <typename SrcVector, typename DstVector>
TfLiteStatus DeepCopyTensorsShapeTypeData(TfLiteContext* context,
                                          TfLiteNode* node,
                                          Subgraph* src_subgraph,
                                          const SrcVector& src_tensor_indices,
                                          Subgraph* dst_subgraph,
                                          const DstVector& dst_tensor_indices) {
  const OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  if (op_data->body_has_dynamic_output_tensors) {
    Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
    bool resize_subgraph_inputs = (dst_subgraph != this_subgraph);
    TF_LITE_ENSURE_OK(
        context, CopyTensorsShapeAndType(
                     context, src_subgraph, src_tensor_indices, dst_subgraph,
                     dst_tensor_indices, resize_subgraph_inputs));
    if (resize_subgraph_inputs) {
      TF_LITE_ENSURE_OK(context, dst_subgraph->AllocateTensors());
    }
  }
  TF_LITE_ENSURE_OK(context,
                    CopyTensorsData(context, src_subgraph, src_tensor_indices,
                                    dst_subgraph, dst_tensor_indices));
  return kTfLiteOk;
}

// Propagate tensor shapes and types from `src_tensor_indices` in `src_subgraph`
// to `dst_tensor_indices` in `dst_subgraph` and copy data shallowly.
template <typename SrcVector, typename DstVector>
TfLiteStatus ShallowCopyTensorsShapeTypeData(
    TfLiteContext* context, TfLiteNode* node, Subgraph* src_subgraph,
    const SrcVector& src_tensor_indices, Subgraph* dst_subgraph,
    const DstVector& dst_tensor_indices) {
  const OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  TF_LITE_ENSURE_EQ(context, op_data->body_has_dynamic_output_tensors, true);
  // Only allow shallow copy from main node input.
  TF_LITE_ENSURE_EQ(context, src_subgraph, this_subgraph);

  TF_LITE_ENSURE_EQ(context, src_tensor_indices.size(),
                    dst_tensor_indices.size());
  bool reallocation_needed = false;
  for (int i = 0; i < src_tensor_indices.size(); ++i) {
    // Skip copying unused destination tensors.
    if (dst_tensor_indices[i] == kTfLiteOptionalTensor) continue;

    const TfLiteTensor* src_tensor =
        src_subgraph->tensor(src_tensor_indices[i]);
    TfLiteTensor* dst_tensor = dst_subgraph->tensor(dst_tensor_indices[i]);

    if (!TfLiteIntArrayEqual(src_tensor->dims, dst_tensor->dims)) {
      reallocation_needed = true;
      TfLiteIntArrayFree(dst_tensor->dims);
      dst_tensor->dims = TfLiteIntArrayCopy(src_tensor->dims);
    }
    dst_tensor->type = src_tensor->type;
    dst_tensor->bytes = 0;  // Don't allocate memory with AllocateTensors().
    dst_tensor->data.raw = nullptr;
  }

  if (reallocation_needed && dst_subgraph != this_subgraph) {
    TF_LITE_ENSURE_OK(context, dst_subgraph->AllocateTensors());
  }

  for (int i = 0; i < src_tensor_indices.size(); ++i) {
    // Skip copying unused destination tensors.
    if (dst_tensor_indices[i] == kTfLiteOptionalTensor) continue;

    const TfLiteTensor* src_tensor =
        src_subgraph->tensor(src_tensor_indices[i]);
    TfLiteTensor* dst_tensor = dst_subgraph->tensor(dst_tensor_indices[i]);

    dst_tensor->bytes = src_tensor->bytes;
    dst_tensor->data.raw = src_tensor->data.raw;
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

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData;
  const auto* params = reinterpret_cast<const TfLiteWhileParams*>(buffer);
  op_data->cond_subgraph_index = params->cond_subgraph_index;
  op_data->body_subgraph_index = params->body_subgraph_index;
  op_data->cond_has_dynamic_output_tensors = false;
  op_data->body_has_dynamic_output_tensors = false;
  op_data->body_use_shallow_copy = false;
  op_data->subgraphs_allocated = false;
  op_data->subgraphs_prepared = false;
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare_impl(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  int num_inputs = node->inputs->size;
  // The number of outputs should be the same as number of inputs.
  TF_LITE_ENSURE_EQ(context, node->outputs->size, num_inputs);

  // Check subgraph indices and get subgraphs.
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();
  TF_LITE_ENSURE(context, op_data->cond_subgraph_index < subgraphs->size());
  TF_LITE_ENSURE(context, op_data->body_subgraph_index < subgraphs->size());
  TF_LITE_ENSURE(context,
                 op_data->cond_subgraph_index != op_data->body_subgraph_index);

  Subgraph* cond_subgraph = (*subgraphs)[op_data->cond_subgraph_index].get();
  Subgraph* body_subgraph = (*subgraphs)[op_data->body_subgraph_index].get();

  // Check input & output count of the condition subgraph.
  TF_LITE_ENSURE_EQ(context, cond_subgraph->inputs().size(), num_inputs);
  TF_LITE_ENSURE_EQ(context, cond_subgraph->outputs().size(), 1);

  // Check input & output count of the body subgraph.
  TF_LITE_ENSURE_EQ(context, body_subgraph->inputs().size(), num_inputs);
  TF_LITE_ENSURE_EQ(context, body_subgraph->outputs().size(), num_inputs);

  // Remove unused inputs of the condition subgraph to skip copying unnecessary
  // inputs.
  cond_subgraph->RemoveUnusedInputs();

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

  if (this_subgraph->IsMemoryOptimizationForLargeTensorsEnabled()) {
    // The current shallow copy requires to use dynamic tensors which introduces
    // additional overheads. Therefore, use the method only if dynamic
    // allocation is enabled.
    op_data->body_use_shallow_copy = true;
    op_data->body_has_dynamic_output_tensors = true;
    // Make body inputs dynamic to use shallow copy with Eval_dynamic().
    for (int i = 0; i < num_inputs; ++i) {
      TfLiteTensor* body_input =
          body_subgraph->tensor(body_subgraph->inputs()[i]);
      SetTensorToDynamic(body_input);
      body_input->bytes = 0;
    }
  }

  TF_LITE_ENSURE_OK(context, body_subgraph->AllocateTensors());
  op_data->subgraphs_allocated = true;
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
  op_data->subgraphs_prepared = true;
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  if (this_subgraph->IsMemoryOptimizationForLargeTensorsEnabled()) {
    // Apply lazy initialization of WHILE kernel.
    // Just make node output tensors dynamic.
    int num_outputs = node->outputs->size;
    for (int i = 0; i < num_outputs; ++i) {
      TfLiteTensor* output;
      TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &output));
      SetTensorToDynamic(output);
    }
    return kTfLiteOk;
  }
  return Prepare_impl(context, node);
}

TfLiteStatus Prepare_lazy(TfLiteContext* context, TfLiteNode* node) {
  return Prepare_impl(context, node);
}

// Evaluate cond subgraph and set the result.
TfLiteStatus Eval_cond_subgraph(TfLiteContext* context, Subgraph* cond_subgraph,
                                bool cond_has_dynamic_output_tensors,
                                bool* cond_subgraph_output) {
  TF_LITE_ENSURE_OK(context, cond_subgraph->Invoke());
  int cond_subgraph_output_index = cond_subgraph->outputs()[0];
  cond_subgraph->EnsureTensorDataIsReadable(cond_subgraph_output_index);
  TfLiteTensor* cond_output = cond_subgraph->tensor(cond_subgraph_output_index);
  if (cond_has_dynamic_output_tensors) {
    TF_LITE_ENSURE_STATUS(CheckCondOutput(context, cond_output));
  }

  *cond_subgraph_output = (cond_output->data.b[0]);
  return kTfLiteOk;
}

// Evaluate WHILE op when body subgraph has dynamic outputs.
TfLiteStatus Eval_dynamic(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();
  Subgraph* cond_subgraph = (*subgraphs)[op_data->cond_subgraph_index].get();
  Subgraph* body_subgraph = (*subgraphs)[op_data->body_subgraph_index].get();

  // The follow graph illustrates the current implementation.
  //
  // This Subgraph          Cond Subgraph         Body Subgraph
  // +-----------+   (1)   +------------+         +------------+
  // |   WHILE   |-------->|  SUBGRAPH  |         |  SUBGRAPH  |
  // |   INPUT   |         |   INPUT    |         |   INPUT    |
  // |           |         |     ---------------->|            |
  // |           |         |   /        | <----   |            |
  // +-----------+         +--/---------+      \  +------------+
  //      |                 /    |              \       |
  //      | (2)       (4) /      | (3)       (6) \      | (5)
  //      v             /        v                \     v
  // +-----------+    /    +------------+         +------------+
  // |   WHILE   |--/      |  SUBGRAPH  |         |  SUBGRAPH  |
  // |   OUTPUT  |    (7)  |   OUTPUT   |         |   OUTPUT   |
  // |           |<-------------------------------|            |
  // +-----------+         +------------+         +------------+
  //
  // (1) Copy the inputs of WHILE op to the inputs of condition subgraph.
  // (2) Copy the inputs of WHILE op to the outputs of WHILE op
  // (3) Invoke condition subgraph.
  //     Exit the loop if the result is false.
  // (4) Copy the outputs of WHILE op to the inputs of body subgraph.
  // (5) Invoke body subgraph.
  // (6) Copy the outputs of body subgraph to the inputs condition subgraph.
  // (7) Copy the outputs of body subgraph to the outputs of WHILE op.
  //     Jump back to step 3!
  //
  // If the body subgraph has dynamic sized outputs, it's required to resize the
  // tensor before copying in step 1, 2, 4, 6 and 7.
  //
  // Note the flow is carefully designed to handle the dynamic sized output
  // case. The loop invariant is: The newest value is in the inputs of condition
  // subgraph. This is always true before step 3.

  // Step 1. node->inputs -> cond->inputs (fast)
  TF_LITE_ENSURE_OK(context, DeepCopyTensorsShapeTypeData(
                                 context, node, this_subgraph,
                                 TfLiteIntArrayView(node->inputs),
                                 cond_subgraph, cond_subgraph->inputs()));

  // Step 2. node->inputs -> node->outputs
  TF_LITE_ENSURE_OK(
      context, DeepCopyTensorsShapeTypeData(context, node, this_subgraph,
                                            TfLiteIntArrayView(node->inputs),
                                            this_subgraph,
                                            TfLiteIntArrayView(node->outputs)));

  while (true) {
    // Step 3. Eval cond subgraph
    bool cond_subgraph_output;
    TF_LITE_ENSURE_OK(
        context, Eval_cond_subgraph(context, cond_subgraph,
                                    op_data->cond_has_dynamic_output_tensors,
                                    &cond_subgraph_output));
    if (!cond_subgraph_output) {
      break;
    }

    // Step 4. node->outputs -> body->inputs
    if (op_data->body_use_shallow_copy) {
      TF_LITE_ENSURE_OK(context, ShallowCopyTensorsShapeTypeData(
                                     context, node, this_subgraph,
                                     TfLiteIntArrayView(node->outputs),
                                     body_subgraph, body_subgraph->inputs()));
    } else {
      TF_LITE_ENSURE_OK(context, DeepCopyTensorsShapeTypeData(
                                     context, node, this_subgraph,
                                     TfLiteIntArrayView(node->outputs),
                                     body_subgraph, body_subgraph->inputs()));
    }

    // Step 5. Invoke body subgraph
    TF_LITE_ENSURE_OK(context, body_subgraph->Invoke());
    for (int tensor_index : body_subgraph->outputs()) {
      body_subgraph->EnsureTensorDataIsReadable(tensor_index);
    }

    // Step 6. body->outputs -> cond->inputs (fast)
    TF_LITE_ENSURE_OK(
        context, DeepCopyTensorsShapeTypeData(
                     context, node, body_subgraph, body_subgraph->outputs(),
                     cond_subgraph, cond_subgraph->inputs()));

    // Step 7. body->outputs -> node->outputs
    TF_LITE_ENSURE_OK(
        context, DeepCopyTensorsShapeTypeData(
                     context, node, body_subgraph, body_subgraph->outputs(),
                     this_subgraph, TfLiteIntArrayView(node->outputs)));
  }

  if (op_data->body_use_shallow_copy) {
    // Clean up shallow copied pointer of body inputs.
    for (int i = 0; i < body_subgraph->inputs().size(); ++i) {
      TfLiteTensor* body_input =
          body_subgraph->tensor(body_subgraph->inputs()[i]);
      body_input->data.raw = nullptr;
    }
  }

  return kTfLiteOk;
}

// Evaluate WHILE op when body subgraph has static outputs.
TfLiteStatus Eval_static(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();
  Subgraph* cond_subgraph = (*subgraphs)[op_data->cond_subgraph_index].get();
  Subgraph* body_subgraph = (*subgraphs)[op_data->body_subgraph_index].get();

  // The follow graph illustrates the current implementation.
  //
  // This Subgraph          Cond Subgraph         Body Subgraph
  // +-----------+   (1)   +------------+         +------------+
  // |   WHILE   |-------->|  SUBGRAPH  |         |  SUBGRAPH  |
  // |   INPUT   |  (3-1) /|   INPUT    |         |   INPUT    |
  // |           |------------------------------->|            |
  // |           |         |            | <----   |            |
  // +-----------+         +------------+      \  +------------+
  //                             |              \       |     ^
  //                             | (2)       (5) \      | (4) | (3-2)
  //                             v                \     v     |
  // +-----------+         +------------+         +------------+
  // |   WHILE   |         |  SUBGRAPH  |         |  SUBGRAPH  |
  // |   OUTPUT  |    (6)  |   OUTPUT   |         |   OUTPUT   |
  // |           |<-------------------------------|            |
  // +-----------+         +------------+         +------------+
  //
  // (1) Copy the inputs of WHILE op to the inputs of condition subgraph.
  // (2) Invoke condition subgraph.
  //     Jump to step 6 if the result is false.
  // (3) If body is never invoked, run the step 3-1, else run the step 3-2.
  // (3-1) Copy the inputs of WHILE op to the inputs of body subgraph.
  // (3-2) Copy the outputs of body subgraph to the inputs of body subgraph.
  // (4) Invoke body subgraph.
  // (5) Copy the outputs of body subgraph to the inputs condition subgraph.
  //     Jump back to step 2!
  // (6) Copy the outputs of body subgraph to the outputs of WHILE op.
  //
  // The body subgraph shouldn't have dynamic sized outputs.

  // Step 1. node->inputs -> cond->inputs (fast)
  TF_LITE_ENSURE_OK(
      context,
      CopyTensorsData(context, this_subgraph, TfLiteIntArrayView(node->inputs),
                      cond_subgraph, cond_subgraph->inputs()));

  bool body_invoked = false;
  while (true) {
    // Step 2. Eval cond subgraph
    bool cond_subgraph_output;
    TF_LITE_ENSURE_OK(
        context, Eval_cond_subgraph(context, cond_subgraph,
                                    op_data->cond_has_dynamic_output_tensors,
                                    &cond_subgraph_output));
    if (!cond_subgraph_output) {
      break;
    }

    if (body_invoked) {
      // Step 3-2. body->output -> body->inputs
      TF_LITE_ENSURE_OK(
          context,
          CopyTensorsData(context, body_subgraph, body_subgraph->outputs(),
                          body_subgraph, body_subgraph->inputs()));
    } else {
      // Step 3-1. node->inputs -> body->inputs
      TF_LITE_ENSURE_OK(
          context, CopyTensorsData(context, this_subgraph,
                                   TfLiteIntArrayView(node->inputs),
                                   body_subgraph, body_subgraph->inputs()));
    }

    // Step 4. Invoke body subgraph
    TF_LITE_ENSURE_OK(context, body_subgraph->Invoke());
    body_invoked = true;
    for (int tensor_index : body_subgraph->outputs()) {
      body_subgraph->EnsureTensorDataIsReadable(tensor_index);
    }

    // Step 5. body->output -> cond->inputs (fast)
    TF_LITE_ENSURE_OK(
        context,
        CopyTensorsData(context, body_subgraph, body_subgraph->outputs(),
                        cond_subgraph, cond_subgraph->inputs()));
  }

  if (body_invoked) {
    // Step 6. Copy body->output -> node->outputs
    TF_LITE_ENSURE_OK(
        context,
        CopyTensorsData(context, body_subgraph, body_subgraph->outputs(),
                        this_subgraph, TfLiteIntArrayView(node->outputs)));
  } else {
    // Copy node->inputs if body is never invoked.
    TF_LITE_ENSURE_OK(
        context, CopyTensorsData(
                     context, this_subgraph, TfLiteIntArrayView(node->inputs),
                     this_subgraph, TfLiteIntArrayView(node->outputs)));
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();
  Subgraph* cond_subgraph = (*subgraphs)[op_data->cond_subgraph_index].get();
  Subgraph* body_subgraph = (*subgraphs)[op_data->body_subgraph_index].get();

  if (op_data->subgraphs_prepared == false) {
    TF_LITE_ENSURE_OK(context, Prepare_lazy(context, node));
  } else if (op_data->subgraphs_allocated == false) {
    TF_LITE_ENSURE_OK(context, cond_subgraph->AllocateTensors());
    TF_LITE_ENSURE_OK(context, body_subgraph->AllocateTensors());
  }

  if (op_data->body_has_dynamic_output_tensors) {
    TF_LITE_ENSURE_OK(context, Eval_dynamic(context, node));
  } else {
    TF_LITE_ENSURE_OK(context, Eval_static(context, node));
  }

  TF_LITE_ENSURE_OK(context, cond_subgraph->ReleaseNonPersistentMemory());
  TF_LITE_ENSURE_OK(context, body_subgraph->ReleaseNonPersistentMemory());
  op_data->subgraphs_allocated = false;

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
