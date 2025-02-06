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

#include "tensorflow/lite/core/c/c_api_opaque.h"

#include <stdarg.h>
#include <stdint.h>

#include <cstdio>
#include <vector>

#include "tensorflow/lite/c/c_api_opaque_internal.h"
#include "tensorflow/lite/core/c/c_api.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/util.h"

namespace {

const TfLiteTensor* Convert(const TfLiteOpaqueTensor* opaque_tensor) {
  // The following cast is safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // TfLiteOpaqueTensor and TfLiteTensor being equivalent.
  return reinterpret_cast<const TfLiteTensor*>(opaque_tensor);
}

TfLiteTensor* Convert(TfLiteOpaqueTensor* opaque_tensor) {
  // The following cast is safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // TfLiteOpaqueTensor and TfLiteTensor being equivalent.
  return reinterpret_cast<TfLiteTensor*>(opaque_tensor);
}

TfLiteNode* Convert(TfLiteOpaqueNode* opaque_node) {
  // The following cast is safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // TfLiteOpaqueNode and TfLiteNode being equivalent.
  return reinterpret_cast<TfLiteNode*>(opaque_node);
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

TfLiteContext* Convert(TfLiteOpaqueContext* opaque_context) {
  // The following cast is safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // TfLiteOpaqueContext and TfLiteContext being equivalent.
  return reinterpret_cast<TfLiteContext*>(opaque_context);
}

TfLiteOpaqueContext* Convert(TfLiteContext* tflite_context) {
  // The following cast is safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // TfLiteOpaqueContext and TfLiteContext being equivalent.
  return reinterpret_cast<TfLiteOpaqueContext*>(tflite_context);
}

const ::tflite::Subgraph* GetSubgraph(
    const TfLiteOpaqueContext* opaque_context) {
  // The following cast is safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // TfLiteContext::impl_ having type ::tflite::Subgraph*.
  return reinterpret_cast<const ::tflite::Subgraph*>(
      Convert(opaque_context)->impl_);
}

::tflite::Subgraph* GetSubgraph(TfLiteOpaqueContext* opaque_context) {
  // The following cast is safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // TfLiteContext::impl_ having type ::tflite::Subgraph*.
  return reinterpret_cast<::tflite::Subgraph*>(Convert(opaque_context)->impl_);
}
}  // namespace

struct TfLiteOpaqueTensorBuilder {
  TfLiteType type;
  void* data;
  TfLiteAllocationType allocation_type;
  TfLiteQuantizationParams quantization_params;
  TfLiteQuantization quantization;
};

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
  return opaque_tensor != nullptr
             ? TfLiteTensorData(
                   reinterpret_cast<const TfLiteTensor*>(opaque_tensor))
             : nullptr;
}

TfLiteAllocationType TfLiteOpaqueTensorGetAllocationType(
    const TfLiteOpaqueTensor* opaque_tensor) {
  return Convert(opaque_tensor)->allocation_type;
}

TfLiteAllocationStrategy TfLiteOpaqueTensorGetAllocationStrategy(
    const TfLiteOpaqueTensor* t) {
  return TfLiteTensorGetAllocationStrategy(Convert(t));
}

TfLiteRunStability TfLiteOpaqueTensorGetBufferAddressStability(
    const TfLiteOpaqueTensor* t) {
  return TfLiteTensorGetBufferAddressStability(Convert(t));
}

TfLiteRunStability TfLiteOpaqueTensorGetDataStability(
    const TfLiteOpaqueTensor* t) {
  return TfLiteTensorGetDataStability(Convert(t));
}

TfLiteRunStep TfLiteOpaqueTensorGetDataKnownStep(const TfLiteOpaqueTensor* t) {
  return TfLiteTensorGetDataKnownStep(Convert(t));
}

TfLiteRunStep TfLiteOpaqueTensorGetShapeKnownStep(const TfLiteOpaqueTensor* t) {
  return TfLiteTensorGetShapeKnownStep(Convert(t));
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

int TfLiteOpaqueTensorGetStringCount(const TfLiteOpaqueTensor* tensor) {
  return tflite::GetStringCount(Convert(tensor));
}

TfLiteStatus TfLiteOpaqueTensorGetString(const TfLiteOpaqueTensor* tensor,
                                         int index, const char** str,
                                         int* len) {
  tflite::StringRef str_ref = tflite::GetString(Convert(tensor), index);
  *str = str_ref.str;
  *len = str_ref.len;
  return kTfLiteOk;
}

TfLiteStatus TfLiteOpaqueTensorWriteStrings(TfLiteOpaqueTensor* tensor,
                                            const char* const* str_array,
                                            int str_array_len,
                                            const int* str_n_len) {
  tflite::DynamicBuffer buf;
  for (int i = 0; i < str_array_len; ++i) {
    buf.AddString(str_array[i], str_n_len[i]);
  }
  buf.WriteToTensorAsVector(Convert(tensor));
  return kTfLiteOk;
}

TfLiteStatus TfLiteOpaqueTensorWriteString(TfLiteOpaqueTensor* tensor,
                                           const char* str, const int len) {
  TfLiteOpaqueTensorWriteStrings(tensor, &str, 1, &len);
  return kTfLiteOk;
}

TfLiteOpaqueTensorBuilder* TfLiteOpaqueTensorBuilderCreate() {
  return new TfLiteOpaqueTensorBuilder{};
}

void TfLiteOpaqueTensorBuilderDelete(TfLiteOpaqueTensorBuilder* builder) {
  delete builder;
}

TfLiteOpaqueTensorBuilder* TfLiteOpaqueTensorBuilderSetType(
    TfLiteOpaqueTensorBuilder* builder, TfLiteType type) {
  builder->type = type;
  return builder;
}

TfLiteOpaqueTensorBuilder* TfLiteOpaqueTensorBuilderSetData(
    TfLiteOpaqueTensorBuilder* builder, void* data) {
  builder->data = data;
  return builder;
}

TfLiteOpaqueTensorBuilder* TfLiteOpaqueTensorBuilderSetAllocationType(
    TfLiteOpaqueTensorBuilder* builder, TfLiteAllocationType allocation_type) {
  builder->allocation_type = allocation_type;
  return builder;
}

TfLiteOpaqueTensorBuilder* TfLiteOpaqueTensorBuilderSetQuantizationParams(
    TfLiteOpaqueTensorBuilder* builder, TfLiteQuantizationParams params) {
  builder->quantization_params = params;
  return builder;
}

TfLiteOpaqueTensorBuilder* TfLiteOpaqueTensorBuilderSetQuantization(
    TfLiteOpaqueTensorBuilder* builder, TfLiteQuantization quantization) {
  builder->quantization = quantization;
  return builder;
}

void TfLiteOpaqueTensorSetAllocationTypeToDynamic(TfLiteOpaqueTensor* tensor) {
  tflite::SetTensorToDynamic(Convert(tensor));
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

TfLiteStatus TfLiteOpaqueNodeSetTemporaries(TfLiteOpaqueNode* opaque_node,
                                            const int* temporaries,
                                            int num_temporaries) {
  if (num_temporaries < 0) {
    return kTfLiteError;
  }
  TfLiteNode* node = Convert(opaque_node);
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(num_temporaries);
  for (int i = 0; i < num_temporaries; ++i) {
    node->temporaries->data[i] = temporaries[i];
  }
  return kTfLiteOk;
}

int TfLiteOpaqueNodeGetInputTensorIndex(const TfLiteOpaqueNode* opaque_node,
                                        int index_of_input) {
  auto* node = Convert(opaque_node);
  if (index_of_input < 0 || index_of_input >= node->inputs->size) {
    return -1;
  }
  return node->inputs->data[index_of_input];
}

int TfLiteOpaqueNodeGetOutputTensorIndex(const TfLiteOpaqueNode* opaque_node,
                                         int index_of_output) {
  auto* node = Convert(opaque_node);
  if (index_of_output < 0 || index_of_output >= node->outputs->size) {
    return -1;
  }
  return node->outputs->data[index_of_output];
}

TfLiteStatus TfLiteOpaqueContextGetExecutionPlan(
    TfLiteOpaqueContext* opaque_context, TfLiteIntArray** execution_plan) {
  // The following casts are safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // TfLiteOpaqueContext and TfLiteContext being equivalent.
  auto context = reinterpret_cast<TfLiteContext*>(opaque_context);
  return context->GetExecutionPlan(context, execution_plan);
}

TfLiteStatus TfLiteOpaqueContextGetExternalContext(
    TfLiteOpaqueContext* opaque_context, void** external_context,
    TfLiteExternalContextType type) {
  auto context = reinterpret_cast<TfLiteContext*>(opaque_context);
  *external_context = context->GetExternalContext(context, type);
  return kTfLiteOk;
}

TfLiteStatus TfLiteOpaqueContextGetNodeAndRegistration(
    struct TfLiteOpaqueContext* opaque_context, int node_index,
    TfLiteOpaqueNode** node, TfLiteOperator** registration_external) {
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
  // create a TfLiteOperator on the fly, and set its field according
  // to the 'TfLiteRegistration' object.
  auto derived_registration =
      tflite::internal::CommonOpaqueConversionUtil::ObtainOperator(
          context, registration, node_index);

  if (derived_registration == nullptr) return kTfLiteError;

  *registration_external = derived_registration;
  return kTfLiteOk;
}

TfLiteStatus TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
    struct TfLiteOpaqueContext* opaque_context,
    TfLiteOperator* registration_external,
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
  // of 'TfLiteOperator's.  Note that it is important to
  // brace-initialize the 'TfLiteRegistration' so that we pass a registration to
  // 'ReplaceNodeSubsetsWithDelegateKernels' that has all of its fields set to
  // null, except the 'registration_external' one.
  TfLiteRegistration registration{};
  registration.registration_external = registration_external;

  // Takes ownership of registration_external, if delegate is opaque delegate.
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

TfLiteStatus TfLiteOpaqueContextAcquireSubgraphContext(
    struct TfLiteOpaqueContext* opaque_context, int subgraph_index,
    TfLiteOpaqueContext** acquired_opaque_context) {
  auto* subgraph = GetSubgraph(opaque_context);
  TfLiteContext* acquired_context;
  TfLiteStatus status =
      subgraph->AcquireSubgraphContext(subgraph_index, &acquired_context);
  if (status != kTfLiteOk) {
    return status;
  }
  *acquired_opaque_context = Convert(acquired_context);
  return kTfLiteOk;
}

TfLiteStatus TfLiteOpaqueContextReleaseSubgraphContext(
    struct TfLiteOpaqueContext* opaque_context, int subgraph_index) {
  return GetSubgraph(opaque_context)->ReleaseSubgraphContext(subgraph_index);
}

TfLiteStatus TfLiteOpaqueContextMarkSubgraphAsDelegationSkippable(
    TfLiteOpaqueContext* opaque_context, int subgraph_index) {
  auto* subgraph = GetSubgraph(opaque_context);
  return subgraph->MarkSubgraphAsDelegationSkippable(subgraph_index);
}

TfLiteStatus TfLiteOpaqueContextGetNodeInitDataMmapInfo(
    const TfLiteOpaqueContext* context, const TfLiteOpaqueNode* node, int* fd,
    int64_t* custom_initial_data_offset_in_file,
    int64_t* custom_initial_data_size) {
  auto* subgraph = GetSubgraph(context);
  return subgraph->GetNodeInitDataMmapInfo(Convert(node), fd,
                                           custom_initial_data_offset_in_file,
                                           custom_initial_data_size);
}

TfLiteStatus TfLiteOpaqueContextAddTensor(TfLiteOpaqueContext* context,
                                          TfLiteOpaqueTensorBuilder* builder,
                                          int* new_tensor_index) {
  if (builder->allocation_type != kTfLiteDynamic &&
      builder->allocation_type != kTfLiteArenaRw &&
      builder->allocation_type != kTfLiteArenaRwPersistent) {
    TfLiteOpaqueContextReportError(
        context,
        "Invalid allocation type '%d'.  Allocation type for "
        "TfLiteOpaqueContextAddTensor must be one of: "
        "'kTfLiteDynamic', 'kTfLiteArenaRw' or 'kTfLiteArenaRwPersistent'.",
        builder->allocation_type);
    return kTfLiteError;
  }

  if (builder->allocation_type == kTfLiteDynamic && builder->data == nullptr) {
    TfLiteOpaqueContextReportError(context,
                                   "For tensors of allocation type "
                                   "'kTfLiteDynamic' 'data' must be provided.");
    return kTfLiteError;
  }
  if ((builder->allocation_type == kTfLiteArenaRw ||
       builder->allocation_type == kTfLiteArenaRwPersistent) &&
      builder->data != nullptr) {
    TfLiteOpaqueContextReportError(
        context,
        "For tensors of allocation type "
        "'kTfLiteArenaRw' or 'kTfLiteArenaRwPersistent' "
        "'data' must not be provided.");
    return kTfLiteError;
  }

  auto* tflite_context = Convert(context);
  int index = -1;
  auto status = tflite_context->AddTensors(tflite_context, 1, &index);
  if (status != kTfLiteOk) return status;

  tflite_context->tensors[index].type = builder->type;
  tflite_context->tensors[index].data.data = builder->data;
  tflite_context->tensors[index].allocation_type = builder->allocation_type;
  tflite_context->tensors[index].params = builder->quantization_params;
  tflite_context->tensors[index].quantization = builder->quantization;
  if (new_tensor_index != nullptr) {
    *new_tensor_index = index;
  }
  return status;
}

TfLiteStatus TfLiteOpaqueContextGetSizeOfType(TfLiteOpaqueContext* context,
                                              const TfLiteType type,
                                              size_t* bytes) {
  return tflite::GetSizeOfType(Convert(context), type, bytes);
}

TfLiteStatus TfLiteOpaqueContextGetMetadata(TfLiteOpaqueContext* context,
                                            const char* name, const char** ptr,
                                            size_t* bytes) {
  auto* tflite_context = Convert(context);
  return tflite_context->GetModelMetadata(tflite_context, name, ptr, bytes);
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

#ifndef TF_LITE_STATIC_MEMORY
TfLiteOpaqueDelegate* TfLiteOpaqueDelegateCreate(
    const TfLiteOpaqueDelegateBuilder* opaque_delegate_builder) {
  if (!opaque_delegate_builder) return nullptr;

  TfLiteDelegate* result = new TfLiteDelegate{};
  result->opaque_delegate_builder = new TfLiteOpaqueDelegateBuilder{};
  *(result->opaque_delegate_builder) = *opaque_delegate_builder;

  return reinterpret_cast<TfLiteOpaqueDelegate*>(result);
}

void TfLiteOpaqueDelegateDelete(TfLiteOpaqueDelegate* opaque_delegate) {
  if (!opaque_delegate) return;

  const TfLiteDelegate* tflite_delegate =
      reinterpret_cast<const TfLiteDelegate*>(opaque_delegate);
  delete tflite_delegate->opaque_delegate_builder;
  delete tflite_delegate;
}
#endif  // TF_LITE_STATIC_MEMORY

void* TfLiteOpaqueDelegateGetData(const TfLiteOpaqueDelegate* delegate) {
  if (!delegate) return nullptr;

  // The following cast is safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // 'TfLiteOpaqueDelegate' and 'TfLiteDelegate' being equivalent.
  const auto* tflite_delegate =
      reinterpret_cast<const TfLiteDelegate*>(delegate);

  if (!tflite_delegate->opaque_delegate_builder) return tflite_delegate->data_;

  return tflite_delegate->opaque_delegate_builder->data;
}
