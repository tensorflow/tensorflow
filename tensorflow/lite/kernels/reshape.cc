/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <cstring>
#include <memory>

#include "tensorflow/lite/array.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace reshape {

constexpr int kInputTensor = 0;
constexpr int kShapeTensor = 1;
constexpr int kOutputTensor = 0;

struct OpData {
  // Store the output pointer here if the output was written during 'Prepare'.
  // This is to prevent incorrect results when mischievous users overwrite
  // output pointers with their own.
  const void* output_ptr;
};

TfLiteIntArray* GetOutputShape(TfLiteContext*, TfLiteNode*);

TfLiteStatus ResizeOutput(TfLiteContext* context, TfLiteNode* node) {
  TfLiteIntArray* output_shape = GetOutputShape(context, node);
  IntArrayUniquePtr scoped_output_shape(output_shape);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // Tensorflow's Reshape allows one of the shape components to have the
  // special -1 value, meaning it will be calculated automatically based on the
  // input. Here we calculate what that dimension should be so that the number
  // of output elements is the same as the number of input elements.
  int64_t non_zero_num_input_elements = 1, num_input_elements = 1;
  const RuntimeShape& input_shape = GetTensorShape(input);
  for (int i = 0; i < input_shape.DimensionsCount(); ++i) {
    const int value = input_shape.Dims(i);
    num_input_elements *= value;
    if (value != 0) {
      non_zero_num_input_elements *= value;
    }
  }

  int64_t non_zero_num_output_elements = 1, num_output_elements = 1;
  int stretch_dim = -1;
  for (int i = 0; i < output_shape->size; ++i) {
    const int value = output_shape->data[i];
    if (value == -1) {
      TF_LITE_ENSURE_EQ(context, stretch_dim, -1);
      stretch_dim = i;
      continue;
    } else if (value != 0) {
      non_zero_num_output_elements *= value;
    }
    num_output_elements *= value;
  }

  if (stretch_dim != -1) {
    if (num_input_elements == 0 && num_output_elements != 0) {
      output_shape->data[stretch_dim] = 0;
    } else {
      output_shape->data[stretch_dim] =
          non_zero_num_input_elements / non_zero_num_output_elements;
    }
    num_output_elements *= output_shape->data[stretch_dim];
  }

  TF_LITE_ENSURE_EQ(context, num_input_elements, num_output_elements);
  return context->ResizeTensor(context, output, scoped_output_shape.release());
}

inline TfLiteIntArray* GetOutputShapeFromTensor(TfLiteContext* context,
                                                TfLiteNode* node) {
  const TfLiteTensor* shape = GetInput(context, node, kShapeTensor);
  if (shape == nullptr) return nullptr;

  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(shape->dims->data[0]);
  for (int i = 0; i < output_shape->size; ++i) {
    output_shape->data[i] = shape->data.i32[i];
  }

  return output_shape;
}

inline TfLiteIntArray* GetOutputShapeFromParam(TfLiteContext* context,
                                               TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteReshapeParams*>(node->builtin_data);

  // The function is returned above this line if the shape tensor is usable.
  // Now fallback to the shape parameter in `TfLiteReshapeParams`.
  int num_dimensions = params->num_dimensions;
  if (num_dimensions == 1 && params->shape[0] == 0) {
    // Legacy tflite models use a shape parameter of [0] to indicate scalars,
    // so adjust accordingly. TODO(b/111614235): Allow zero-sized buffers during
    // toco conversion.
    num_dimensions = 0;
  }
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(num_dimensions);
  for (int i = 0; i < num_dimensions; ++i) {
    output_shape->data[i] = params->shape[i];
  }

  return output_shape;
}

// Check if the shape tensor is valid. Shapes should be int32 vectors.
inline bool ShapeIsVector(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* shape = GetInput(context, node, kShapeTensor);
  return (shape != nullptr && shape->dims->size == 1 &&
          shape->type == kTfLiteInt32);
}

TfLiteIntArray* GetOutputShape(TfLiteContext* context, TfLiteNode* node) {
  if (NumInputs(node) == 2 && ShapeIsVector(context, node)) {
    return GetOutputShapeFromTensor(context, node);
  } else {
    return GetOutputShapeFromParam(context, node);
  }
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, NumInputs(node) == 1 || NumInputs(node) == 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  op_data->output_ptr = nullptr;

  // Always postpone sizing string tensors, even if we could in principle
  // calculate their shapes now. String tensors don't benefit from having their
  // shapes precalculated because the actual memory can only be allocated after
  // we know all the content.
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  if (output->type != kTfLiteString) {
    const TfLiteTensor* input = GetInput(context, node, kInputTensor);
    const TfLiteTensor* shape = GetInput(context, node, kShapeTensor);
    if (NumInputs(node) == 1 || IsConstantOrPersistentTensor(shape)) {
      if (IsConstantOrPersistentTensor(input)) {
        SetTensorToPersistentRo(output);
        TF_LITE_ENSURE_OK(context, ResizeOutput(context, node));
        op_data->output_ptr = output->data.data;
        memcpy(output->data.data, input->data.data, input->bytes);
        return kTfLiteOk;
      } else {
        TF_LITE_ENSURE_OK(context, ResizeOutput(context, node));
      }
    } else {
      SetTensorToDynamic(output);
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // There are two ways in which the 'output' can be made dynamic: it could be
  // a string tensor, or its shape cannot be calculated during Prepare(). In
  // either case, we now have all the information to calculate its shape.
  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context, ResizeOutput(context, node));
  }

  // Note that string tensors are always "dynamic" in the sense that their size
  // is not known until we have all the content. This applies even when their
  // shape is known ahead of time. As a result, a string tensor is never given
  // any memory by ResizeOutput(), and we need to do it manually here. Since
  // reshape doesn't change the data, the output tensor needs exactly as many
  // bytes as the input tensor.
  if (output->type == kTfLiteString) {
    auto bytes_required = input->bytes;
    TfLiteTensorRealloc(bytes_required, output);
    output->bytes = bytes_required;
  }

  if (op_data->output_ptr == output->data.data) {
    return kTfLiteOk;
  }
  // Only copy data if input and output do not share a buffer.
  if (output->data.data != input->data.data) {
    memcpy(output->data.data, input->data.data, input->bytes);
  }

  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return new OpData;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

}  // namespace reshape

TfLiteRegistration* Register_RESHAPE() {
  static TfLiteRegistration r = {
      reshape::Init,
      reshape::Free,
      reshape::Prepare,
      reshape::Eval,
      /*profiling_string=*/nullptr,
      /*builtin_code=*/0,
      /*custom_name=*/nullptr,
      /*version=*/0,
      /*registration_external=*/nullptr,
      /*async_kernel=*/nullptr,
      kTfLiteInplaceOpInput0Shared | kTfLiteInplaceOpDataUnmodified};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
