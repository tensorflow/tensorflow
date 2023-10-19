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
#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>

#include "tensorflow/lite/array.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/portable_tensor.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/variants/list_ops_util.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace variants {
namespace ops {
namespace add_n {
namespace {

using ::tflite::variants::TensorArray;

constexpr int kInputTensor1 = 0;
constexpr int kOutputTensor = 0;

struct OpData {
  // The index of the temporary tensor where temporary accumulations are kept.
  int scratch_tensor_index;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData();
  context->AddTensors(context, 1, &op_data->scratch_tensor_index);
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, NumInputs(node) >= 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(1);
  node->temporaries->data[0] = op_data->scratch_tensor_index;
  TfLiteTensor* scratch_tensor;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/0, &scratch_tensor));
  scratch_tensor->type = kTfLiteNoType;
  scratch_tensor->allocation_type = kTfLiteDynamic;

  for (int i = kInputTensor1 + 1; i < NumInputs(node); ++i) {
    const TfLiteTensor* input;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i, &input));
    TF_LITE_ENSURE_EQ(context, input->type, kTfLiteVariant);
  }
  output->type = kTfLiteVariant;
  output->allocation_type = kTfLiteVariantObject;
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  ///
  // Prepare Inputs/Outputs
  // ---------------------------------------------------------------------------

  // Fetch backend cpu context.
  CpuBackendContext* cpu_backend_context =
      CpuBackendContext::GetFromContext(context);

  // Parse Tensors.
  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TfLiteTensor* scratch_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, 0, &scratch_tensor));

  // Collect vector of input list pointers, validate num_elements and type,
  // compute merged shape.
  const TensorArray* const arr =
      reinterpret_cast<const TensorArray*>(input1->data.data);
  const int num_elements = arr->NumElements();
  const TfLiteType t = arr->ElementType();
  const int num_inputs = NumInputs(node);
  IntArrayUniquePtr merged_shape = BuildTfLiteArray(*arr->ElementShape());
  std::vector<const TensorArray*> input_arrs;
  input_arrs.reserve(num_inputs);
  input_arrs.push_back(arr);
  for (int i = 1; i < num_inputs; ++i) {
    const TfLiteTensor* input;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i, &input));
    const TensorArray* const arr_i =
        reinterpret_cast<const TensorArray*>(input->data.data);
    TF_LITE_ENSURE_EQ(context, num_elements, arr_i->NumElements());
    TF_LITE_ENSURE_EQ(context, t, arr_i->ElementType());
    merged_shape = variants::MergeShapesOrNull(
        std::move(merged_shape), BuildTfLiteArray(*arr_i->ElementShape()));
    TF_LITE_ENSURE(context, merged_shape != nullptr);
    input_arrs.push_back(arr_i);
  }

  // Allocate output list with same length and type as all inputs.
  TF_LITE_ENSURE_OK(context, TfLiteTensorVariantRealloc<TensorArray>(
                                 output, t, BuildTfLiteArray(0)));
  TensorArray* const output_arr =
      reinterpret_cast<TensorArray*>(output->data.data);
  output_arr->Resize(num_elements);

  ///
  // Compute out_list[i] = Sum(in_list_0[i] + ...) for 0 < i < num_elements
  // ---------------------------------------------------------------------------

  for (int i = 0; i < num_elements; ++i) {
    TfLiteIntArray* row_shape = nullptr;
    std::vector<TfLiteTensor*> row_tensors;
    // Compute resolved shape for this row. Save tensor* for all present
    // elements at list_j`i`.
    for (const auto* array : input_arrs) {
      const TfLiteTensor* at = array->At(i);
      if (!at) continue;
      if (!row_shape)
        row_shape = at->dims;
      else
        TF_LITE_ENSURE(context, TfLiteIntArrayEqual(row_shape, at->dims));
      // We need const_cast to comply with optimized_ops::AddN type.
      row_tensors.push_back(const_cast<TfLiteTensor*>(at));
    }
    if (row_shape == nullptr) {
      // There exists no set elements in any list at position `i`. We can
      // set output[i] to be a zeroed tensor with shape from input
      // lists `ElementShape`.
      TF_LITE_ENSURE(context,
                     variants::IsShapeFullyDefined(*merged_shape.get()));
      TensorUniquePtr row_output = BuildTfLiteTensor(
          t, BuildTfLiteArray(*merged_shape.get()), kTfLiteDynamic);
      memset(row_output->data.data, 0, row_output->bytes);
      output_arr->Set(i, std::move(row_output));
      continue;
    }
    // Allocate tensor for the sum of this row.
    TensorUniquePtr row_output =
        BuildTfLiteTensor(t, BuildTfLiteArray(*row_shape), kTfLiteDynamic);
    if (row_tensors.size() < 2) {
      // There is only one set item in all `input_j[i]`, so just use that.
      TfLiteTensorCopy(row_tensors[0], row_output.get());
      output_arr->Set(i, std::move(row_output));
      continue;
    }
    // Resize scratch tensor so it can be used for each row.
    const int num_inputs_for_row = static_cast<int>(row_tensors.size());
    const int thread_count =
        std::min(std::max(1, static_cast<int>(num_inputs_for_row) / 2),
                 cpu_backend_context->max_num_threads());
    IntArrayUniquePtr scratch_shape = BuildTfLiteArray(
        {thread_count * static_cast<int>(NumElements(row_tensors[0]))});
    scratch_tensor->type = t;
    TF_LITE_ENSURE_OK(
        context, context->ResizeTensor(context, scratch_tensor,
                                       BuildTfLiteArray(*row_shape).release()));
    const RuntimeShape row_runtime_shape(row_shape->size, row_shape->data);

    // Compute sum of row.
    if (t == kTfLiteInt32) {
      VectorOfTensors<int> tensors(row_tensors);
      optimized_ops::AddN<int>(row_runtime_shape, num_inputs, tensors.data(),
                               GetTensorData<int>(row_output.get()),
                               GetTensorData<int>(scratch_tensor),
                               cpu_backend_context);
    } else if (t == kTfLiteFloat32) {
      VectorOfTensors<float> tensors(row_tensors);
      optimized_ops::AddN<float>(row_runtime_shape, num_inputs, tensors.data(),
                                 GetTensorData<float>(row_output.get()),
                                 GetTensorData<float>(scratch_tensor),
                                 cpu_backend_context);
    } else {
      TF_LITE_KERNEL_LOG(context, "Subtype is not supported for variant addn.");
      return kTfLiteError;
    }
    // Write row sum to appropriate element in output list.
    TF_LITE_ENSURE(context, output_arr->Set(i, std::move(row_output)));
  }

  return kTfLiteOk;
}
}  // namespace
}  // namespace add_n

TfLiteRegistration* Register_VARIANT_ADD_N() {
  static TfLiteRegistration r = {add_n::Init, add_n::Free, add_n::Prepare,
                                 add_n::Eval};
  return &r;
}

}  // namespace ops
}  // namespace variants
}  // namespace tflite
