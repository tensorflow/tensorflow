/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_CONTROL_FLOW_COMMON_H_
#define TENSORFLOW_LITE_KERNELS_CONTROL_FLOW_COMMON_H_

#include <vector>

#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
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
    TF_LITE_ENSURE_OK(context, TfLiteTensorCopy(src_tensor, dst_tensor));
  }
  return kTfLiteOk;
}

// Propagate tensor shapes and types from `src_tensor_indices` in `src_subgraph`
// to `dst_tensor_indices` in `dst_subgraph` and copy data deeply.
template <typename SrcVector, typename DstVector>
TfLiteStatus DeepCopyTensorsShapeTypeData(
    TfLiteContext* context, TfLiteNode* node, Subgraph* src_subgraph,
    const SrcVector& src_tensor_indices, Subgraph* dst_subgraph,
    const DstVector& dst_tensor_indices, bool body_has_dynamic_output_tensors) {
  if (body_has_dynamic_output_tensors) {
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

template <typename SrcVector, typename DstVector>
TfLiteStatus DeepOrShallowCopyTensorsShapeTypeData(
    TfLiteContext* context, TfLiteNode* node, Subgraph* src_subgraph,
    const SrcVector& src_tensor_indices, Subgraph* dst_subgraph,
    const DstVector& dst_tensor_indices) {
  // Resize the destination subgraph inputs.
  for (int i = 0; i < src_tensor_indices.size(); ++i) {
    // Skip copying unused destination tensors.
    if (dst_tensor_indices[i] == kTfLiteOptionalTensor) continue;
    if (src_tensor_indices[i] == kTfLiteOptionalTensor) continue;

    const TfLiteTensor* src_tensor =
        src_subgraph->tensor(src_tensor_indices[i]);
    TfLiteTensor* dst_tensor = dst_subgraph->tensor(dst_tensor_indices[i]);
    std::vector<int> dims(src_tensor->dims->data,
                          src_tensor->dims->data + src_tensor->dims->size);
    dst_subgraph->ResizeInputTensor(dst_tensor_indices[i], dims);
    dst_tensor->type = src_tensor->type;
    if (!IsResourceOrVariant(src_tensor)) {
      dst_tensor->bytes = 0;  // Don't allocate memory with AllocateTensors().
      dst_tensor->data.raw = nullptr;
    }
  }
  TF_LITE_ENSURE_OK(context, dst_subgraph->AllocateTensors());
  // Deep or shallow copy the data from src subgraph to dst.
  for (int i = 0; i < src_tensor_indices.size(); ++i) {
    // Skip copying unused destination tensors.
    if (dst_tensor_indices[i] == kTfLiteOptionalTensor) continue;
    if (src_tensor_indices[i] == kTfLiteOptionalTensor) continue;

    const TfLiteTensor* src_tensor =
        src_subgraph->tensor(src_tensor_indices[i]);
    TfLiteTensor* dst_tensor = dst_subgraph->tensor(dst_tensor_indices[i]);
    if (IsResourceOrVariant(src_tensor)) {
      TfLiteTensorRealloc(src_tensor->bytes, dst_tensor);
      TF_LITE_ENSURE_OK(context, TfLiteTensorCopy(src_tensor, dst_tensor));
    } else {
      // Make a shallow copy of the data. This is only safe because the caller
      // is expected to have previously set dst_tensor->allocation_type to
      // kTfLiteCustom, to ensure the buffer is never double-freed later on.
      TF_LITE_ENSURE_EQ(context, dst_tensor->allocation_type, kTfLiteCustom);
      dst_tensor->bytes = src_tensor->bytes;
      dst_tensor->data.raw = src_tensor->data.raw;
    }
  }
  return kTfLiteOk;
}
}  // namespace builtin
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CONTROL_FLOW_COMMON_H_
