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
#include "tensorflow/lite/kernels/internal/reference/concatenation.h"
#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace concatenation {

constexpr int kMaxInputNum = 10;  // Maximum number of input tensors
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // This function only checks the types. Additional shape validations are
  // performed in the reference implementation called during Eval().
  const TfLiteConcatenationParams* params =
      reinterpret_cast<TfLiteConcatenationParams*>(node->builtin_data);

  TfLiteType input_type = GetInput(context, node, 0)->type;
  TfLiteType output_type = GetOutput(context, node, kOutputTensor)->type;

  // Check activation and input type
  TF_LITE_ENSURE_EQ(context, params->activation, kTfLiteActNone);
  TF_LITE_ENSURE(context,
                 input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8 ||
                     input_type == kTfLiteInt8 || input_type == kTfLiteInt32 ||
                     input_type == kTfLiteInt64);

  // Output type must match input type
  TF_LITE_ENSURE_EQ(context, output_type, input_type);

  // This implementation does not support large number of input tensors
  const int num_inputs = NumInputs(node);
  TF_LITE_ENSURE(context, num_inputs <= kMaxInputNum);

  // Shapes with dimensions >4 are not yet supported with static allocation.
  for (int i = 0; i < num_inputs; ++i) {
    const TfLiteTensor* input = GetInput(context, node, i);
    int num_dimensions = NumDimensions(input);

    if (num_dimensions > 4) {
      context->ReportError(
          context,
          "Op Concatenation does not currently support num dimensions >4 "
          "Tensor '%s' has %d dimensions.",
          input->name, num_dimensions);
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

// Handles negative axis index, coerces to positive index value.
inline int CalculatePositiveAxis(int axis, const TfLiteTensor* output_tensor) {
  if (axis >= 0) {
    return axis;
  } else {
    return NumDimensions(output_tensor) + axis;
  }
}

// The following functions are helpers to get tensor data in the format that the
// reference op implementation expects. They provide the same functionality as
// class VectorOfTensors and class VectorOfQuantizedTensors in TFLite.

// Gets shapes from a list of tensors.
inline void GetAllTensorShapes(const TfLiteContext& context,
                               const TfLiteIntArray& tensor_list,
                               RuntimeShape all_shapes[kMaxInputNum]) {
  for (int i = 0; i < tensor_list.size; ++i) {
    const TfLiteTensor* t = &context.tensors[tensor_list.data[i]];
    RuntimeShape shape = GetTensorShape(t);
    all_shapes[i].ReplaceWith(shape.DimensionsCount(), shape.DimsData());
  }
}

// Get shape pointers from a list of shapes.
inline void GetShapesPointers(const RuntimeShape* shapes, size_t num,
                              const RuntimeShape* pointers[]) {
  for (size_t i = 0; i < num; ++i) {
    pointers[i] = &shapes[i];
  }
}

// Gets data pointers from a list of tensors.
template <typename T>
inline void GetAllTensorData(const TfLiteContext& context,
                             const TfLiteIntArray& tensor_list,
                             T* all_data[kMaxInputNum]) {
  for (int i = 0; i < tensor_list.size; ++i) {
    const TfLiteTensor* t = &context.tensors[tensor_list.data[i]];
    all_data[i] = GetTensorData<T>(t);
  }
}

// Gets scale and zero point from a list of tensors
inline void GetAllQuantizationParam(const TfLiteContext& context,
                                    const TfLiteIntArray& tensor_list,
                                    float scales[kMaxInputNum],
                                    int32 zero_points[kMaxInputNum]) {
  for (int i = 0; i < tensor_list.size; ++i) {
    const TfLiteTensor* t = &context.tensors[tensor_list.data[i]];
    scales[i] = t->params.scale;
    zero_points[i] = t->params.zero_point;
  }
}

template <typename data_type>
void EvalUnquantized(TfLiteContext* context, TfLiteNode* node) {
  // Collect the shapes and data pointer of input tensors
  RuntimeShape inputs_shape[kMaxInputNum];
  const RuntimeShape* inputs_shape_ptr[kMaxInputNum];
  const data_type* inputs_data[kMaxInputNum];
  GetAllTensorShapes(*context, *node->inputs, inputs_shape);
  GetShapesPointers(inputs_shape, node->inputs->size, inputs_shape_ptr);
  GetAllTensorData(*context, *node->inputs, inputs_data);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  const TfLiteConcatenationParams* params =
      reinterpret_cast<TfLiteConcatenationParams*>(node->builtin_data);

  ConcatenationParams op_params;
  op_params.axis = CalculatePositiveAxis(params->axis, output);
  op_params.inputs_count = NumInputs(node);

  reference_ops::Concatenation(op_params, inputs_shape_ptr, inputs_data,
                               GetTensorShape(output),
                               GetTensorData<data_type>(output));
}

void EvalQuantizedUInt8(TfLiteContext* context, TfLiteNode* node) {
  // Collect the shapes and data pointer of input tensors
  RuntimeShape inputs_shape[kMaxInputNum];
  const RuntimeShape* inputs_shape_ptr[kMaxInputNum];
  const uint8_t* inputs_data[kMaxInputNum];
  float inputs_scale[kMaxInputNum];
  int32 inputs_zero_point[kMaxInputNum];
  GetAllTensorShapes(*context, *node->inputs, inputs_shape);
  GetShapesPointers(inputs_shape, node->inputs->size, inputs_shape_ptr);
  GetAllTensorData(*context, *node->inputs, inputs_data);
  GetAllQuantizationParam(*context, *node->inputs, inputs_scale,
                          inputs_zero_point);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  const TfLiteConcatenationParams* params =
      reinterpret_cast<TfLiteConcatenationParams*>(node->builtin_data);

  ConcatenationParams op_params;
  op_params.axis = CalculatePositiveAxis(params->axis, output);
  op_params.inputs_count = NumInputs(node);
  op_params.input_zeropoint = inputs_zero_point;
  op_params.input_scale = inputs_scale;
  op_params.output_zeropoint = output->params.zero_point;
  op_params.output_scale = output->params.scale;

  reference_ops::ConcatenationWithScaling(
      op_params, inputs_shape_ptr, inputs_data, GetTensorShape(output),
      GetTensorData<uint8>(output));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteType output_type = GetOutput(context, node, kOutputTensor)->type;

  switch (output_type) {  // Already know in/outtypes are same.
    case kTfLiteFloat32:
      EvalUnquantized<float>(context, node);
      break;
    case kTfLiteInt32:
      EvalUnquantized<int32_t>(context, node);
      break;
    case kTfLiteUInt8:
      EvalQuantizedUInt8(context, node);
      break;
    case kTfLiteInt8:
      EvalUnquantized<int8_t>(context, node);
      break;
    case kTfLiteInt64:
      EvalUnquantized<int64_t>(context, node);
      break;

    default:
      context->ReportError(
          context, "Op Concatenation does not currently support Type '%s'.",
          TfLiteTypeGetName(output_type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace concatenation

TfLiteRegistration* Register_CONCATENATION() {
  static TfLiteRegistration r = {/* init */ nullptr,
                                 /* free */ nullptr, concatenation::Prepare,
                                 concatenation::Eval};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
