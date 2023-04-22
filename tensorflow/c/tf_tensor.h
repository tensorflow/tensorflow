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

#ifndef TENSORFLOW_C_TF_TENSOR_H_
#define TENSORFLOW_C_TF_TENSOR_H_

#include <stdbool.h>
#include <stdint.h>

#include "tensorflow/c/c_api_macros.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

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

// Allocator Attributes used for tensor allocation.
typedef struct TF_AllocatorAttributes {
  size_t struct_size;
  // Set boolean to 1 for CPU allocation, else 0.
  TF_Bool on_host;
} TF_AllocatorAttributes;

#define TF_ALLOCATOR_ATTRIBUTES_STRUCT_SIZE \
  TF_OFFSET_OF_END(TF_AllocatorAttributes, on_host)

// --------------------------------------------------------------------------
// TF_Tensor holds a multi-dimensional array of elements of a single data type.
// For all types other than TF_STRING, the data buffer stores elements
// in row major order.  E.g. if data is treated as a vector of TF_DataType:
//
//   element 0:   index (0, ..., 0)
//   element 1:   index (0, ..., 1)
//   ...
//
// The format for TF_STRING tensors is:
//   start_offset: array[uint64]
//   data:         byte[...]
//
//   The string length (as a varint, start_offset[i + 1] - start_offset[i]),
//   followed by the contents of the string is encoded at data[start_offset[i]].
//   TF_StringEncode and TF_StringDecode facilitate this encoding.

typedef struct TF_Tensor TF_Tensor;

// Return a new tensor that holds the bytes data[0,len-1].
//
// The data will be deallocated by a subsequent call to TF_DeleteTensor via:
//      (*deallocator)(data, len, deallocator_arg)
// Clients must provide a custom deallocator function so they can pass in
// memory managed by something like numpy.
//
// May return NULL (and invoke the deallocator) if the provided data buffer
// (data, len) is inconsistent with a tensor of the given TF_DataType
// and the shape specified by (dima, num_dims).
TF_CAPI_EXPORT extern TF_Tensor* TF_NewTensor(
    TF_DataType, const int64_t* dims, int num_dims, void* data, size_t len,
    void (*deallocator)(void* data, size_t len, void* arg),
    void* deallocator_arg);

// Allocate and return a new Tensor.
//
// This function is an alternative to TF_NewTensor and should be used when
// memory is allocated to pass the Tensor to the C API. The allocated memory
// satisfies TensorFlow's memory alignment preferences and should be preferred
// over calling malloc and free.
//
// The caller must set the Tensor values by writing them to the pointer returned
// by TF_TensorData with length TF_TensorByteSize.
TF_CAPI_EXPORT extern TF_Tensor* TF_AllocateTensor(TF_DataType,
                                                   const int64_t* dims,
                                                   int num_dims, size_t len);

// Deletes `tensor` and returns a new TF_Tensor with the same content if
// possible. Returns nullptr and leaves `tensor` untouched if not.
TF_CAPI_EXPORT extern TF_Tensor* TF_TensorMaybeMove(TF_Tensor* tensor);

// Destroy a tensor.
TF_CAPI_EXPORT extern void TF_DeleteTensor(TF_Tensor*);

// Return the type of a tensor element.
TF_CAPI_EXPORT extern TF_DataType TF_TensorType(const TF_Tensor*);

// Return the number of dimensions that the tensor has.
TF_CAPI_EXPORT extern int TF_NumDims(const TF_Tensor*);

// Return the length of the tensor in the "dim_index" dimension.
// REQUIRES: 0 <= dim_index < TF_NumDims(tensor)
TF_CAPI_EXPORT extern int64_t TF_Dim(const TF_Tensor* tensor, int dim_index);

// Return the size of the underlying data in bytes.
TF_CAPI_EXPORT extern size_t TF_TensorByteSize(const TF_Tensor*);

// Return a pointer to the underlying data buffer.
TF_CAPI_EXPORT extern void* TF_TensorData(const TF_Tensor*);

// Returns the number of elements in the tensor.
TF_CAPI_EXPORT extern int64_t TF_TensorElementCount(const TF_Tensor* tensor);

// Copy the internal data representation of `from` to `to`. `new_dims` and
// `num_new_dims` specify the new shape of the `to` tensor, `type` specifies its
// data type. On success, *status is set to TF_OK and the two tensors share the
// same data buffer.
//
// This call requires that the `from` tensor and the given type and shape (dims
// and num_dims) are "compatible" (i.e. they occupy the same number of bytes).
// Specifically, given from_type_size = TF_DataTypeSize(TF_TensorType(from)):
//
// ShapeElementCount(dims, num_dims) * TF_DataTypeSize(type)
//
// must equal
//
// TF_TensorElementCount(from) * from_type_size
//
// where TF_ShapeElementCount would be the number of elements in a tensor with
// the given shape.
//
// In addition, this function requires:
//   * TF_DataTypeSize(TF_TensorType(from)) != 0
//   * TF_DataTypeSize(type) != 0
//
// If any of the requirements are not met, *status is set to
// TF_INVALID_ARGUMENT.
TF_CAPI_EXPORT extern void TF_TensorBitcastFrom(const TF_Tensor* from,
                                                TF_DataType type, TF_Tensor* to,
                                                const int64_t* new_dims,
                                                int num_new_dims,
                                                TF_Status* status);

// Returns bool iff this tensor is aligned.
TF_CAPI_EXPORT extern bool TF_TensorIsAligned(const TF_Tensor*);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_C_TF_TENSOR_H_
