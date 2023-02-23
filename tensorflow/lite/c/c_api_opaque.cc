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

#include <cstdio>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_opaque_internal.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace {

const TfLiteTensor* Convert(const TfLiteOpaqueTensor* opaque_tensor) {
  // The following cast is safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // TfLiteOpaqueTensor and TfLiteTensor being equivalent.
  return reinterpret_cast<const TfLiteTensor*>(opaque_tensor);
}

const TfLiteNode* Convert(const TfLiteOpaqueNode* opaque_node) {
  // The following cast is safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // TfLiteOpaqueNode and TfLiteNode being equivalent.
  return reinterpret_cast<const TfLiteNode*>(opaque_node);
}

const TfLiteContext* Convert(const TfLiteOpaqueContext* opaque_context) {
  // The following cast is safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // TfLiteOpaqueContext and TfLiteContext being equivalent.
  return reinterpret_cast<const TfLiteContext*>(opaque_context);
}

const ::tflite::Subgraph* GetSubgraph(
    const TfLiteOpaqueContext* opaque_context) {
  // The following cast is safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // TfLiteContext::impl_ having type ::tflite::Subgraph*.
  return reinterpret_cast<const ::tflite::Subgraph*>(
      Convert(opaque_context)->impl_);
}

}  // namespace

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

TfLiteStatus TfLiteOpaqueTensorGetNumDimsSignature(
    const TfLiteOpaqueTensor* opaque_tensor, int32_t* num_dims) {
  const TfLiteTensor* tensor = Convert(opaque_tensor);
  if (!tensor->dims_signature) {
    *num_dims = -1;
    return kTfLiteOk;
  }
  *num_dims = tensor->dims_signature->size;
  return kTfLiteOk;
}

TfLiteStatus TfLiteOpaqueTensorGetDimSignature(
    const TfLiteOpaqueTensor* opaque_tensor, int32_t dim_index,
    int32_t* dim_length) {
  const TfLiteTensor* tensor = Convert(opaque_tensor);
  // `dims_signature` is not defined when no unknown dimensions are present.
  if (tensor->dims_signature != nullptr && tensor->dims_signature->size != 0) {
    *dim_length = tensor->dims_signature->data[dim_index];
  } else {
    *dim_length = tensor->dims->data[dim_index];
  }
  return kTfLiteOk;
}

int TfLiteOpaqueTensorIsVariable(const TfLiteOpaqueTensor* opaque_tensor) {
  return Convert(opaque_tensor)->is_variable ? 1 : 0;
}

size_t TfLiteOpaqueTensorByteSize(const TfLiteOpaqueTensor* opaque_tensor) {
  return TfLiteTensorByteSize(
      reinterpret_cast<const TfLiteTensor*>(opaque_tensor));
}

void* TfLiteOpaqueTensorData(const TfLiteOpaqueTensor* opaque_tensor) {
  return TfLiteTensorData(reinterpret_cast<const TfLiteTensor*>(opaque_tensor));
}

TfLiteAllocationType TfLiteOpaqueTensorGetAllocationType(
    const TfLiteOpaqueTensor* opaque_tensor) {
  return Convert(opaque_tensor)->allocation_type;
}

const char* TfLiteOpaqueTensorName(const TfLiteOpaqueTensor* opaque_tensor) {
  return TfLiteTensorName(reinterpret_cast<const TfLiteTensor*>(opaque_tensor));
}

TfLiteQuantization TfLiteOpaqueTensorGetQuantization(
    const TfLiteOpaqueTensor* opaque_tensor) {
  return Convert(opaque_tensor)->quantization;
}

TfLiteQuantizationParams TfLiteOpaqueTensorGetQuantizationParams(
    const TfLiteOpaqueTensor* opaque_tensor) {
  return Convert(opaque_tensor)->params;
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
    const TfLiteOpaqueContext* opaque_context,
    const TfLiteOpaqueNode* opaque_node, int index) {
  const TfLiteTensor* tensor =
      tflite::GetInput(reinterpret_cast<const TfLiteContext*>(opaque_context),
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

void* TfLiteOpaqueNodeGetBuiltinData(const TfLiteOpaqueNode* opaque_node) {
  return Convert(opaque_node)->builtin_data;
}

TfLiteStatus TfLiteOpaqueNodeGetCustomInitialData(
    const TfLiteOpaqueNode* opaque_node, const void** init_data, int* size) {
  *init_data = Convert(opaque_node)->custom_initial_data;
  *size = Convert(opaque_node)->custom_initial_data_size;
  return kTfLiteOk;
}

TfLiteStatus TfLiteOpaqueNodeInputs(const TfLiteOpaqueNode* opaque_node,
                                    const int** inputs, int* num_inputs) {
  const TfLiteNode* node = Convert(opaque_node);
  *inputs = node->inputs->data;
  *num_inputs = node->inputs->size;
  return kTfLiteOk;
}

TfLiteStatus TfLiteOpaqueNodeOutputs(const TfLiteOpaqueNode* opaque_node,
                                     const int** outputs, int* num_outputs) {
  const TfLiteNode* node = Convert(opaque_node);
  *outputs = node->outputs->data;
  *num_outputs = node->outputs->size;
  return kTfLiteOk;
}

TfLiteStatus TfLiteOpaqueNodeTemporaries(const TfLiteOpaqueNode* opaque_node,
                                         const int** temporaries,
                                         int* num_temporaries) {
  const TfLiteNode* node = Convert(opaque_node);
  *temporaries = node->temporaries->data;
  *num_temporaries = node->temporaries->size;
  return kTfLiteOk;
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
          context, registration, node_index);

  if (derived_registration == nullptr) return kTfLiteError;

  *registration_external = derived_registration;
  return kTfLiteOk;
}

TfLiteStatus TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
    struct TfLiteOpaqueContext* opaque_context,
    TfLiteRegistrationExternal* registration_external,
    const TfLiteIntArray* nodes_to_replace,
    TfLiteOpaqueDelegate* opaque_delegate) {
  // The following casts are safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // TfLiteOpaqueContext and TfLiteContext being equivalent, or on
  // TfLiteOpaqueNode and TfLiteNode being equivalent.

  TfLiteContext* context = reinterpret_cast<TfLiteContext*>(opaque_context);
  TfLiteDelegate* delegate = reinterpret_cast<TfLiteDelegate*>(opaque_delegate);

  // Wrap the provided 'registration_external' as a regular 'TfLiteRegistration'
  // object to reduce the places in the TF Lite runtime that need to be aware
  // of 'TfLiteRegistrationExternal's.  Note that it is important to
  // brace-initialize the 'TfLiteRegistration' so that we pass a registration to
  // 'ReplaceNodeSubsetsWithDelegateKernels' that has all of its fields set to
  // null, except the 'registration_external' one.
  TfLiteRegistration registration{};
  registration.registration_external = registration_external;

  TfLiteStatus status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, registration, nodes_to_replace, delegate);
  return status;
}

TfLiteOpaqueTensor* TfLiteOpaqueContextGetOpaqueTensor(
    const TfLiteOpaqueContext* opaque_context, int index) {
  // The following casts are safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // TfLiteOpaqueContext and TfLiteContext being equivalent, or on
  // TfLiteTensor and TfLiteOpaqueTensor being equivalent.
  auto context = reinterpret_cast<const TfLiteContext*>(opaque_context);
  return reinterpret_cast<TfLiteOpaqueTensor*>(&context->tensors[index]);
}

TfLiteStatus TfLiteOpaqueContextGetInputs(
    const struct TfLiteOpaqueContext* opaque_context, const int** inputs,
    int* num_inputs) {
  auto* subgraph = GetSubgraph(opaque_context);
  const std::vector<int>& subgraph_inputs = subgraph->inputs();

  *inputs = subgraph_inputs.data();
  *num_inputs = subgraph_inputs.size();
  return kTfLiteOk;
}

TfLiteStatus TfLiteOpaqueContextGetOutputs(
    const struct TfLiteOpaqueContext* opaque_context, const int** outputs,
    int* num_outputs) {
  auto* subgraph = GetSubgraph(opaque_context);
  const std::vector<int>& subgraph_outputs = subgraph->outputs();
  *outputs = subgraph_outputs.data();
  *num_outputs = subgraph_outputs.size();
  return kTfLiteOk;
}

TfLiteStatus TfLiteOpaqueContextGetVariables(
    const struct TfLiteOpaqueContext* opaque_context, const int** variables,
    int* num_variables) {
  auto* subgraph = GetSubgraph(opaque_context);

  const std::vector<int>& subgraph_variables = subgraph->variables();
  *variables = subgraph_variables.data();
  *num_variables = subgraph_variables.size();
  return kTfLiteOk;
}

size_t TfLiteOpaqueContextGetNumNodes(
    const struct TfLiteOpaqueContext* opaque_context) {
  auto* subgraph = GetSubgraph(opaque_context);
  return subgraph->nodes_size();
}

size_t TfLiteOpaqueContextGetNumTensors(
    const struct TfLiteOpaqueContext* opaque_context) {
  auto* subgraph = GetSubgraph(opaque_context);
  return subgraph->tensors_size();
}

const char* TfLiteOpaqueContextGetName(
    const struct TfLiteOpaqueContext* opaque_context) {
  auto* subgraph = GetSubgraph(opaque_context);
  return subgraph->GetName().c_str();
}

TfLiteStatus TfLiteOpaqueContextResizeTensor(TfLiteOpaqueContext* context,
                                             TfLiteOpaqueTensor* tensor,
                                             TfLiteIntArray* new_size) {
  // The following casts are safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // TfLiteOpaqueContext and TfLiteContext being equivalent, or on
  // TfLiteOpaqueTensor and TfLiteTensor being equivalent.
  TfLiteContext* tflite_context = reinterpret_cast<TfLiteContext*>(context);
  return tflite_context->ResizeTensor(
      tflite_context, reinterpret_cast<TfLiteTensor*>(tensor), new_size);
}

void TfLiteOpaqueContextReportError(struct TfLiteOpaqueContext* opaque_context,
                                    const char* format, ...) {
  va_list vlist;
  va_start(vlist, format);
  TfLiteOpaqueContextReportErrorVa(opaque_context, format, vlist);
  va_end(vlist);
}
void TfLiteOpaqueContextReportErrorVa(
    struct TfLiteOpaqueContext* opaque_context, const char* format,
    va_list vlist) {
  // Determine the length of the resulting error message.
  va_list copy;
  va_copy(copy, vlist);
  int n = vsnprintf(nullptr, 0, format, copy);
  if (n < 0) {
    return;
  }
  size_t size = (size_t)n + 1;  // +1 for '\0'.
  char* buffer = new char[size];
  n = vsnprintf(buffer, size, format, vlist);
  if (n < 0) {
    delete[] buffer;
    return;
  }
  // The following cast is safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // TfLiteOpaqueContext and TfLiteContext being equivalent.
  auto* context = reinterpret_cast<TfLiteContext*>(opaque_context);
  TF_LITE_KERNEL_LOG(context, "%s", buffer);
  delete[] buffer;
}
