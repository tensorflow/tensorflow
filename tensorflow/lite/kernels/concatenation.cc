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
#include "tensorflow/lite/kernels/internal/reference/concatenation.h"

#include <stdint.h>

#include <cstddef>
#include <cstring>
#include <limits>

#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace concatenation {

// This file has two implementation of Concatenation.
enum KernelType {
  kReference,
  kGenericOptimized,
};

template <KernelType kernel_type>
TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node, int axis,
                      TfLiteTensor* output) {
// TODO(ahentz): Creating 'all_inputs' below is not very efficient. We should
// allocate and populate these during Prepare().
// TODO(ycling): Activation function parameter is ignored. For now we don't have
// a model with a Concatenation with fused activation function.
#define TF_LITE_CONCATENATION(scalar)                                         \
  {                                                                           \
    VectorOfTensors<scalar> all_inputs(*context, *node->inputs);              \
    tflite::ConcatenationParams op_params;                                    \
    op_params.axis = axis;                                                    \
    op_params.inputs_count = node->inputs->size;                              \
    if (kernel_type == kReference) {                                          \
      reference_ops::Concatenation(op_params, all_inputs.shapes(),            \
                                   all_inputs.data(), GetTensorShape(output), \
                                   GetTensorData<scalar>(output));            \
    } else {                                                                  \
      optimized_ops::Concatenation(op_params, all_inputs.shapes(),            \
                                   all_inputs.data(), GetTensorShape(output), \
                                   GetTensorData<scalar>(output));            \
    }                                                                         \
  }

#define TF_LITE_CONCATENATION_QUANTIZED()                         \
  {                                                               \
    VectorOfQuantizedTensors all_inputs(*context, *node->inputs); \
    tflite::ConcatenationParams op_params;                        \
    op_params.axis = axis;                                        \
    op_params.input_zeropoint = all_inputs.zero_point();          \
    op_params.input_scale = all_inputs.scale();                   \
    op_params.inputs_count = node->inputs->size;                  \
    op_params.output_zeropoint = output->params.zero_point;       \
    op_params.output_scale = output->params.scale;                \
    if (kernel_type == kReference) {                              \
      reference_ops::ConcatenationWithScaling(                    \
          op_params, all_inputs.shapes(), all_inputs.data(),      \
          GetTensorShape(output), GetTensorData<uint8>(output));  \
    } else {                                                      \
      optimized_ops::ConcatenationWithScaling(                    \
          op_params, all_inputs.shapes(), all_inputs.data(),      \
          GetTensorShape(output), GetTensorData<uint8>(output));  \
    }                                                             \
  }

  switch (output->type) {  // Already know in/outtypes are same.
    case kTfLiteFloat32:
      TF_LITE_CONCATENATION(float);
      break;
    case kTfLiteFloat16:
      TF_LITE_CONCATENATION(Eigen::half);
      break;
    case kTfLiteBFloat16:
      TF_LITE_CONCATENATION(Eigen::bfloat16);
      break;
    case kTfLiteInt32:
      TF_LITE_CONCATENATION(int32);
      break;
    case kTfLiteUInt32:
      TF_LITE_CONCATENATION(uint32_t);
      break;
    case kTfLiteUInt8:
      TF_LITE_CONCATENATION_QUANTIZED();
      break;
    case kTfLiteInt8:
      TF_LITE_CONCATENATION(int8_t);
      break;
    case kTfLiteInt64:
      TF_LITE_CONCATENATION(int64_t);
      break;
    case kTfLiteInt16:
      TF_LITE_CONCATENATION(int16_t);
      break;
    case kTfLiteBool:
      TF_LITE_CONCATENATION(bool);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported currently.",
                         TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }

#undef TF_LITE_CONCATENATION_QUANTIZED
#undef TF_LITE_CONCATENATION

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteConcatenationParams*>(node->builtin_data);
  int axis = params->axis;
  int num_inputs = node->inputs->size;

  // The number of dimensions of the input tensors must match, and all
  // dimensions except 'axis' must be equal.
  const TfLiteTensor* t0;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &t0));
  TfLiteType input_type = t0->type;
  if (axis < 0) axis += t0->dims->size;
  TF_LITE_ENSURE(context, axis >= 0);
  TF_LITE_ENSURE(context,
                 axis < t0->dims->size || (t0->dims->size == 0 && axis == 0));

  TF_LITE_ENSURE_EQ(context, params->activation, kTfLiteActNone);
  TF_LITE_ENSURE(context,
                 input_type == kTfLiteFloat32 || input_type == kTfLiteFloat16 ||
                     input_type == kTfLiteBFloat16 ||
                     input_type == kTfLiteUInt8 || input_type == kTfLiteInt8 ||
                     input_type == kTfLiteInt16 || input_type == kTfLiteInt32 ||
                     input_type == kTfLiteInt64 || input_type == kTfLiteBool ||
                     input_type == kTfLiteUInt32);

  // Check to see if we can calculate the output now.
  bool all_inputs_at_prepare = true;
  for (int i = 0; i < num_inputs; ++i) {
    const TfLiteTensor* t;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i, &t));
    if (!IsConstantOrPersistentTensor(t)) {
      all_inputs_at_prepare = false;
      break;
    }
  }
  // Output dimensions will match input dimensions, except 'axis', which
  // will be the sum of inputs
  int sum_axis = t0->dims->size > 0 ? t0->dims->data[axis] : 1;
  // Check if we are concatenating constant scalars.
  if (all_inputs_at_prepare && t0->dims->size == 0 && axis == 0) {
    for (int i = 1; i < num_inputs; ++i) {
      const TfLiteTensor* t;
      TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i, &t));
      TF_LITE_ENSURE_EQ(context, t->dims->size, t0->dims->size);
    }
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
    TfLiteIntArray* output_size = TfLiteIntArrayCreate(1);
    output_size->data[0] = num_inputs;
    SetTensorToPersistentRo(output);
    context->ResizeTensor(context, output, output_size);
    size_t input_type_size;
    TF_LITE_ENSURE_STATUS(GetSizeOfType(context, t0->type, &input_type_size));
    void* o_data = output->data.data;
    for (int i = 0; i < num_inputs; ++i) {
      const TfLiteTensor* t;
      TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i, &t));
      const void* i_data = t->data.data;
      memcpy(o_data, i_data, input_type_size);
      o_data = (void*)((uintptr_t)o_data + input_type_size);
    }
    return kTfLiteOk;
  } else {
    for (int i = 1; i < num_inputs; ++i) {
      const TfLiteTensor* t;
      TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i, &t));
      TF_LITE_ENSURE_EQ(context, t->dims->size, t0->dims->size);
      TF_LITE_ENSURE_EQ(context, t->type, input_type);
      for (int d = 0; d < t0->dims->size; ++d) {
        if (d == axis) {
          // Avoid integer overflow in sum_axis below
          TF_LITE_ENSURE(context, t->dims->data[axis] >= 0);
          TF_LITE_ENSURE(context,
                         t->dims->data[axis] <=
                             std::numeric_limits<int>::max() - sum_axis);
          sum_axis += t->dims->data[axis];
        } else {
          TF_LITE_ENSURE_EQ(context, t->dims->data[d], t0->dims->data[d]);
        }
      }
    }
  }

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(t0->dims->size);
  for (int d = 0; d < t0->dims->size; ++d) {
    output_size->data[d] = (d == axis) ? sum_axis : t0->dims->data[d];
  }

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, input_type);

  if (input_type == kTfLiteInt8) {
    // Make sure there is no re-scaling needed for Int8 quantized kernel. This
    // is a restriction we introduced to Int8 kernels.
    VectorOfTensors<int8_t> all_inputs(*context, *node->inputs);
    for (int i = 0; i < node->inputs->size; ++i) {
      const TfLiteTensor* t;
      TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i, &t));
      TF_LITE_ENSURE_EQ(context, t->params.scale, output->params.scale);
      TF_LITE_ENSURE_EQ(context, t->params.zero_point,
                        output->params.zero_point);
    }
  }

  if (input_type == kTfLiteInt16) {
    // Make sure that all Int16 inputs have a null zero-point.
    for (int i = 0; i < node->inputs->size; ++i) {
      const TfLiteTensor* t = GetInput(context, node, i);
      TF_LITE_ENSURE_EQ(context, t->params.zero_point, 0);
    }
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
  }

  if (all_inputs_at_prepare) {
    SetTensorToPersistentRo(output);
    context->ResizeTensor(context, output, output_size);
    return EvalImpl<kReference>(context, node, axis, output);
  }
  return context->ResizeTensor(context, output, output_size);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteConcatenationParams*>(node->builtin_data);
  int axis = params->axis;
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  if (IsConstantOrPersistentTensor(output)) {
    // Output is computed in Prepare.
    return kTfLiteOk;
  }
  if (axis < 0) axis += output->dims->size;

  return EvalImpl<kernel_type>(context, node, axis, output);
}

#undef TF_LITE_MACRO_DISPATCH

}  // namespace concatenation

TfLiteRegistration* Register_CONCATENATION_REF() {
  static TfLiteRegistration r = {
      nullptr, nullptr, concatenation::Prepare,
      concatenation::Eval<concatenation::kReference>};
  return &r;
}

TfLiteRegistration* Register_CONCATENATION_GENERIC_OPT() {
  static TfLiteRegistration r = {
      nullptr, nullptr, concatenation::Prepare,
      concatenation::Eval<concatenation::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_CONCATENATION() {
  // TODO(ahentz): It turns out the two versions of Concatenation are almost
  // identical, so we should consider removing one.
  return Register_CONCATENATION_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
