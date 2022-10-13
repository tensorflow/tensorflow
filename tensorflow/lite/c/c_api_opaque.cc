/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/c/c_api_opaque.h"

#include <unordered_map>

#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_opaque_internal.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"

TfLiteType TfLiteOpaqueTensorType(const TfLiteOpaqueTensor* opaque_tensor) {
  return TfLiteTensorType(reinterpret_cast<const TfLiteTensor*>(opaque_tensor));
}

int32_t TfLiteOpaqueTensorNumDims(const TfLiteOpaqueTensor* opaque_tensor) {
  return TfLiteTensorNumDims(
      reinterpret_cast<const TfLiteTensor*>(opaque_tensor));
}

int32_t TfLiteOpaqueTensorDim(const TfLiteOpaqueTensor* opaque_tensor,
                              int32_t dim_index) {
  return TfLiteTensorDim(reinterpret_cast<const TfLiteTensor*>(opaque_tensor),
                         dim_index);
}

size_t TfLiteOpaqueTensorByteSize(const TfLiteOpaqueTensor* opaque_tensor) {
  return TfLiteTensorByteSize(
      reinterpret_cast<const TfLiteTensor*>(opaque_tensor));
}

void* TfLiteOpaqueTensorData(const TfLiteOpaqueTensor* opaque_tensor) {
  return TfLiteTensorData(reinterpret_cast<const TfLiteTensor*>(opaque_tensor));
}

const char* TfLiteOpaqueTensorName(const TfLiteOpaqueTensor* opaque_tensor) {
  return TfLiteTensorName(reinterpret_cast<const TfLiteTensor*>(opaque_tensor));
}

TfLiteStatus TfLiteOpaqueTensorCopyFromBuffer(TfLiteOpaqueTensor* opaque_tensor,
                                              const void* input_data,
                                              size_t input_data_size) {
  return TfLiteTensorCopyFromBuffer(
      reinterpret_cast<TfLiteTensor*>(opaque_tensor), input_data,
      input_data_size);
}

TfLiteStatus TfLiteOpaqueTensorCopyToBuffer(
    const TfLiteOpaqueTensor* opaque_tensor, void* output_data,
    size_t output_data_size) {
  return TfLiteTensorCopyToBuffer(
      reinterpret_cast<const TfLiteTensor*>(opaque_tensor), output_data,
      output_data_size);
}

const TfLiteOpaqueTensor* TfLiteOpaqueNodeGetInput(
    TfLiteOpaqueContext* opaque_context, const TfLiteOpaqueNode* opaque_node,
    int index) {
  const TfLiteTensor* tensor =
      tflite::GetInput(reinterpret_cast<TfLiteContext*>(opaque_context),
                       reinterpret_cast<const TfLiteNode*>(opaque_node), index);
  return reinterpret_cast<const TfLiteOpaqueTensor*>(tensor);
}

TfLiteOpaqueTensor* TfLiteOpaqueNodeGetOutput(
    TfLiteOpaqueContext* opaque_context, const TfLiteOpaqueNode* opaque_node,
    int index) {
  TfLiteTensor* tensor = tflite::GetOutput(
      reinterpret_cast<TfLiteContext*>(opaque_context),
      reinterpret_cast<const TfLiteNode*>(opaque_node), index);
  return reinterpret_cast<TfLiteOpaqueTensor*>(tensor);
}

int TfLiteOpaqueNodeNumberOfInputs(const TfLiteOpaqueNode* opaque_node) {
  return reinterpret_cast<const TfLiteNode*>(opaque_node)->inputs->size;
}

int TfLiteOpaqueNodeNumberOfOutputs(const TfLiteOpaqueNode* opaque_node) {
  return reinterpret_cast<const TfLiteNode*>(opaque_node)->outputs->size;
}

void* TfLiteOpaqueNodeGetUserData(const TfLiteOpaqueNode* opaque_node) {
  return reinterpret_cast<const TfLiteNode*>(opaque_node)->user_data;
}

TfLiteStatus TfLiteOpaqueContextGetExecutionPlan(
    TfLiteOpaqueContext* opaque_context, TfLiteIntArray** execution_plan) {
  // The following casts are safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // TfLiteOpaqueContext and TfLiteContext being equivalent.
  auto context = reinterpret_cast<TfLiteContext*>(opaque_context);
  return context->GetExecutionPlan(context, execution_plan);
}

TfLiteStatus TfLiteOpaqueContextGetNodeAndRegistration(
    struct TfLiteOpaqueContext* opaque_context, int node_index,
    TfLiteOpaqueNode** node,
    TfLiteRegistrationExternal** registration_external) {
  // The following casts are safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // TfLiteOpaqueContext and TfLiteContext being equivalent, or on
  // TfLiteOpaqueNode and TfLiteNode being equivalent.
  TfLiteContext* context = reinterpret_cast<TfLiteContext*>(opaque_context);
  TfLiteNode* local_node;
  TfLiteRegistration* registration;

  TfLiteStatus status = context->GetNodeAndRegistration(
      context, node_index, &local_node, &registration);
  if (status != kTfLiteOk) return status;

  // When the 'registration' object obtained via 'GetNodeAndRegistration'
  // has its 'registration_external' field set then we can load that into the
  // caller's 'registration_external' pointer and return early.
  *node = reinterpret_cast<TfLiteOpaqueNode*>(local_node);
  if (registration->registration_external) {
    *registration_external = registration->registration_external;
    return kTfLiteOk;
  }

  // When the 'registration' object obtained via 'GetNodeAndRegistration'
  // does *not* have its 'registration_external' field set then we need to
  // create a TfLiteRegistrationExternal on the fly, and set its field according
  // to the 'TfLiteRegistration' object.
  auto derived_registration =
      tflite::internal::CommonOpaqueConversionUtil::ObtainRegistrationExternal(
          context, registration);

  if (derived_registration == nullptr) return kTfLiteError;

  *registration_external = derived_registration;
  return kTfLiteOk;
}

TfLiteStatus TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
    struct TfLiteOpaqueContext* opaque_context,
    TfLiteRegistrationExternal* registration_external,
    const TfLiteIntArray* nodes_to_replace,
    struct TfLiteOpaqueDelegateStruct* opaque_delegate) {
  // The following casts are safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // TfLiteOpaqueContext and TfLiteContext being equivalent, or on
  // TfLiteOpaqueNode and TfLiteNode being equivalent.

  TfLiteContext* context = reinterpret_cast<TfLiteContext*>(opaque_context);
  TfLiteDelegate* delegate = reinterpret_cast<TfLiteDelegate*>(opaque_delegate);

  TfLiteRegistration registration{};
  registration.registration_external = registration_external;

  TfLiteStatus status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, registration, nodes_to_replace, delegate);
  return status;
}
