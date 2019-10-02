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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace concatenation {

namespace {
constexpr int kOutputTensor = 0;
}  // namespace

template <typename Scalar>
inline void Concatenation(const ConcatenationParams& params,
                          const RuntimeShape* const* input_shapes,
                          const Scalar* const* input_data,
                          const RuntimeShape& output_shape,
                          Scalar* output_data) {
  int axis = params.axis;
  int inputs_count = params.inputs_count;
  const int concat_dimensions = output_shape.DimensionsCount();
  TFLITE_DCHECK_LT(axis, concat_dimensions);

  int64_t concat_size = 0;
  for (int i = 0; i < inputs_count; i++) {
    TFLITE_DCHECK_EQ(input_shapes[i]->DimensionsCount(), concat_dimensions);
    for (int j = 0; j < concat_dimensions; j++) {
      if (j != axis) {
        MatchingDim(*input_shapes[i], j, output_shape, j);
      }
    }
    concat_size += input_shapes[i]->Dims(axis);
  }
  TFLITE_DCHECK_EQ(concat_size, output_shape.Dims(axis));
  int64_t outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= output_shape.Dims(i);
  }

  int64_t base_inner_size = 1;
  for (int i = axis + 1; i < concat_dimensions; ++i) {
    base_inner_size *= output_shape.Dims(i);
  }

  Scalar* output_ptr = output_data;
  for (int k = 0; k < outer_size; k++) {
    for (int i = 0; i < inputs_count; ++i) {
      const int copy_size = input_shapes[i]->Dims(axis) * base_inner_size;
      memcpy(output_ptr, input_data[i] + k * copy_size,
             copy_size * sizeof(Scalar));
      output_ptr += copy_size;
    }
  }
}

void ConcatenationEvalInt8(TfLiteContext* context, TfLiteNode* node,
                           TfLiteConcatenationParams* params, int axis,
                           TfLiteTensor* output) {
  VectorOfTensors<int8_t> all_inputs(*context, *node->inputs);
  tflite::ConcatenationParams op_params;
  op_params.axis = axis;
  op_params.inputs_count = node->inputs->size;

  Concatenation<int8_t>(op_params, all_inputs.shapes(), all_inputs.data(),
                        GetTensorShape(output), GetTensorData<int8_t>(output));
}

void ConcatenationEvalUInt8(TfLiteContext* context, TfLiteNode* node,
                            TfLiteConcatenationParams* params, int axis,
                            TfLiteTensor* output) {
  VectorOfQuantizedTensors all_inputs(*context, *node->inputs);
  tflite::ConcatenationParams op_params;
  op_params.axis = axis;
  op_params.input_zeropoint = all_inputs.zero_point();
  op_params.input_scale = all_inputs.scale();
  op_params.inputs_count = node->inputs->size;
  op_params.output_zeropoint = output->params.zero_point;
  op_params.output_scale = output->params.scale;

  Concatenation<uint8_t>(op_params, all_inputs.shapes(), all_inputs.data(),
                         GetTensorShape(output),
                         GetTensorData<uint8_t>(output));
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteConcatenationParams*>(node->builtin_data);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  int axis = params->axis;
  if (axis < 0) axis += output->dims->size;

  tflite::ConcatenationParams op_params;
  op_params.axis = axis;
  op_params.inputs_count = node->inputs->size;

  // Get the data type from the first input tensor to concatenate
  TfLiteType type = GetInput(context, node, 0)->type;

  // All the inputs and output share the same data type
  switch (type) {
    case kTfLiteUInt8:
      ConcatenationEvalUInt8(context, node, params, axis, output);
      break;
    case kTfLiteInt8:
      ConcatenationEvalInt8(context, node, params, axis, output);
      break;
    default:
      context->ReportError(context, "Input type %s is not currently supported",
                           TfLiteTypeGetName(type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}
}  // namespace concatenation

TfLiteRegistration* Register_CONCATENATION() {
  static TfLiteRegistration r = {
      concatenation::Init,
      concatenation::Free,
      concatenation::Prepare,
      concatenation::Eval,
  };
  return &r;
}
}  // namespace micro
}  // namespace ops
}  // namespace tflite
