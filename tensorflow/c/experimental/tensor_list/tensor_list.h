/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_C_EXPERIMENTAL_TENSOR_LIST_TENSOR_LIST_H_
#define TENSORFLOW_C_EXPERIMENTAL_TENSOR_LIST_TENSOR_LIST_H_

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_tensor.h"

// Macro to control visibility of exported symbols in the shared library (.so,
// .dylib, .dll).
// This duplicates the TF_EXPORT macro definition in
// tensorflow/core/platform/macros.h in order to keep this .h file independent
// of any other includes.
#ifdef SWIG
#define TF_CAPI_EXPORT
#else
#if defined(_WIN32)
#ifdef TF_COMPILE_LIBRARY
#define TF_CAPI_EXPORT __declspec(dllexport)
#else
#define TF_CAPI_EXPORT __declspec(dllimport)
#endif  // TF_COMPILE_LIBRARY
#else
#define TF_CAPI_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32
#endif  // SWIG

#ifdef __cplusplus
extern "C" {
#endif

struct TF_TensorList;

// Create a new TensorList
TF_CAPI_EXPORT TF_TensorList* TF_NewTensorList();

// Delete the TensorList created by TF_NewTensorList
TF_CAPI_EXPORT void TF_DeleteTensorList(TF_TensorList* list);

// Distract the TensorList from the tensor by index.
//
// The TensorList is as a Variant element in Tensor. The Tensor will be viewed
// as a vector, and the tensor of index will be returned. And if the index is
// out of range, the list will be set to nullptr.
TF_CAPI_EXPORT void TF_GetTensorListFromTensor(const TF_Tensor* tensor,
                                               int index, TF_TensorList** list,
                                               TF_Status* status);

// Set the TensorList to Tensor by index.
TF_CAPI_EXPORT void TF_SetTensorListToTensor(TF_TensorList* list, int index,
                                             TF_Tensor* tensor,
                                             TF_Status* status);

// Get the data type of TensorList
TF_CAPI_EXPORT TF_DataType TF_TensorListGetDataType(const TF_TensorList* list);

// Get the dim number of TensorList
TF_CAPI_EXPORT int TF_TensorListNumDims(const TF_TensorList* list);

// Get the dim by dim index
TF_CAPI_EXPORT int64_t TF_TensorListDim(const TF_TensorList* list,
                                        int dim_index);

// Set the data type for TensorList
TF_CAPI_EXPORT void TF_TensorListSetDataType(TF_TensorList* list,
                                             TF_DataType dtype);

// Set the shape for TensorList
TF_CAPI_EXPORT void TF_TensorListSetShape(TF_TensorList* list, int num_dims,
                                          int64_t* dims);

// Get the size of vector of tensor in TensorList
TF_CAPI_EXPORT int TF_TensorListSize(const TF_TensorList* list);

// Deep copy the TensorList and return a new one
TF_CAPI_EXPORT void TF_TensorListCopy(const TF_TensorList* from,
                                      TF_TensorList* to);

// Get the tensor in TensorList by index
//
// If the index is out of range, the tensor will be set to nullptr.
TF_CAPI_EXPORT void TF_TensorListGetTensor(const TF_TensorList* list, int index,
                                           TF_Tensor** tensor,
                                           TF_Status* status);

// Set the tensor in TensorList by index
//
// If out of bound or has nullptr in list and tensor, it will return directly.
TF_CAPI_EXPORT void TF_TensorListSetTensor(TF_TensorList* list, int index,
                                           TF_Tensor* tensor,
                                           TF_Status* status);

// Remove the last tensor of tensor list
//
// If the vector of tensor in list is empty, it will return directly.
TF_CAPI_EXPORT void TF_TensorListPop(TF_TensorList* list);

// Add the tensor to the last of tensor list
TF_CAPI_EXPORT void TF_TensorListPush(TF_TensorList* list, TF_Tensor* tensor);

// Forward the input TensorList or create a new TensorList.
TF_CAPI_EXPORT extern void TF_ForwardInputOrCreateNewList(
    TF_OpKernelContext* context, int input_index, int output_index,
    TF_TensorList* input_list, TF_TensorList** output_list, TF_Status* status);
#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_C_EXPERIMENTAL_TENSOR_LIST_TENSOR_LIST_H_
