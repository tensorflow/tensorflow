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
#include <unistd.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace concatenation {

// This file has two implementation of Concatenation.
enum KernelType {
  kReference,
  kGenericOptimized,
};

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteConcatenationParams*>(node->builtin_data);
  int axis = params->axis;
  int num_inputs = node->inputs->size;

  // The number of dimensions of the input tensors must match, and all
  // dimensions except 'axis' must be equal.
  TfLiteTensor* t0 = &context->tensors[node->inputs->data[0]];
  TfLiteType input_type = t0->type;
  TF_LITE_ENSURE(context, axis >= 0);
  TF_LITE_ENSURE(context, axis < t0->dims->size);

  // TODO(ahentz): These are limitations of our implementation that could be
  // removed with a bit of effort.
  TF_LITE_ENSURE(context, t0->dims->size <= 4);
  TF_LITE_ENSURE_EQ(context, params->activation, kTfLiteActNone);
  TF_LITE_ENSURE(context,
                 input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8);

  // Output dimensions will match input dimensions, except 'axis', which
  // will be the sum of inputs
  int sum_axis = t0->dims->data[axis];
  for (int i = 1; i < num_inputs; ++i) {
    TfLiteTensor* t = &context->tensors[node->inputs->data[i]];
    TF_LITE_ENSURE_EQ(context, t->dims->size, t0->dims->size);
    TF_LITE_ENSURE_EQ(context, t->type, input_type);
    if (input_type == kTfLiteUInt8) {
      TF_LITE_ENSURE_EQ(context, t->params.zero_point, t0->params.zero_point);
      TF_LITE_ENSURE_EQ(context, t->params.scale, t0->params.scale);
    }
    for (int d = 0; d < t0->dims->size; ++d) {
      if (d == axis) {
        sum_axis += t->dims->data[axis];
      } else {
        TF_LITE_ENSURE_EQ(context, t->dims->data[d], t0->dims->data[d]);
      }
    }
  }

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(t0->dims->size);
  for (int d = 0; d < t0->dims->size; ++d) {
    output_size->data[d] = (d == axis) ? sum_axis : t0->dims->data[d];
  }

  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
  TF_LITE_ENSURE_EQ(context, output->type, input_type);
  if (input_type == kTfLiteUInt8) {
    TF_LITE_ENSURE_EQ(context, output->params.zero_point,
                      t0->params.zero_point);
    TF_LITE_ENSURE_EQ(context, output->params.scale, t0->params.scale);
  }

  return context->ResizeTensor(context, output, output_size);
}

template <typename T>
class VectorOfInputs {
 public:
  VectorOfInputs(const TfLiteContext& context, const TfLiteIntArray& inputs) {
    int num_inputs = inputs.size;

    all_data_.reserve(num_inputs);
    all_dims_.reserve(num_inputs);
    all_dims_ptr_.reserve(num_inputs);

    for (int i = 0; i < num_inputs; ++i) {
      TfLiteTensor* input = &context.tensors[inputs.data[i]];
      all_data_.push_back(GetTensorData<T>(input));
      all_dims_.push_back(GetTensorDims(input));
    }

    // Taking the pointer from inside a std::vector is only OK if the vector is
    // never modified, so we populate all_dims in the previous loop and then we
    // are free to grab iterators here.
    for (int i = 0; i < num_inputs; ++i) {
      all_dims_ptr_.push_back(&all_dims_[i]);
    }
  }
  const T* const* data() const { return all_data_.data(); }
  const Dims<4>* const* dims() const { return all_dims_ptr_.data(); }

 private:
  std::vector<T*> all_data_;
  std::vector<Dims<4>> all_dims_;
  std::vector<Dims<4>*> all_dims_ptr_;
};

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteConcatenationParams*>(node->builtin_data);

  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];

// TODO(ahentz): Creating 'all_inputs' below is not very efficient. We should
// allocate and populate these during Prepare().
// TODO(ycling): Activation function parameter is ignored. For now we dont have
// a model with a Concatenation with fused activation function.
#define TF_LITE_CONCATENATION(type, scalar)                                 \
  VectorOfInputs<scalar> all_inputs(*context, *node->inputs);               \
  type::Concatenation<FusedActivationFunctionType::kNone, scalar>(          \
      RemapDim(NumDimensions(output), params->axis), all_inputs.data(),     \
      all_inputs.dims(), node->inputs->size, GetTensorData<scalar>(output), \
      GetTensorDims(output))

  switch (output->type) {  // Already know in/outtypes are same.
    case kTfLiteFloat32:
      if (kernel_type == kReference) {
        TF_LITE_CONCATENATION(reference_ops, float);
      } else {
        TF_LITE_CONCATENATION(optimized_ops, float);
      }
      break;
    case kTfLiteUInt8:
      if (kernel_type == kReference) {
        TF_LITE_CONCATENATION(reference_ops, uint8_t);
      } else {
        TF_LITE_CONCATENATION(optimized_ops, uint8_t);
      }
      break;
    default:
      context->ReportError(context,
                           "Only float32 and uint8 are currently supported.");
      return kTfLiteError;
  }

#undef TF_LITE_CONCATENATION

  return kTfLiteOk;
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
