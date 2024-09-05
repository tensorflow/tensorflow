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

#ifndef TENSORFLOW_C_TF_BUFFER_H_
#define TENSORFLOW_C_TF_BUFFER_H_

#include <stddef.h>

#include "tensorflow/c/c_api_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

// --------------------------------------------------------------------------
// TF_Buffer holds a pointer to a block of data and its associated length.
// Typically, the data consists of a serialized protocol buffer, but other data
// may also be held in a buffer.
//
// By default, TF_Buffer itself does not do any memory management of the
// pointed-to block.  If need be, users of this struct should specify how to
// deallocate the block by setting the `data_deallocator` function pointer.
typedef struct TF_Buffer {
  const void* data;
  size_t length;
  void (*data_deallocator)(void* data, size_t length);
} TF_Buffer;

// Makes a copy of the input and sets an appropriate deallocator.  Useful for
// passing in read-only, input protobufs.
TF_CAPI_EXPORT extern TF_Buffer* TF_NewBufferFromString(const void* proto,
                                                        size_t proto_len);

// Useful for passing *out* a protobuf.
TF_CAPI_EXPORT extern TF_Buffer* TF_NewBuffer(void);

TF_CAPI_EXPORT extern void TF_DeleteBuffer(TF_Buffer*);

TF_CAPI_EXPORT extern TF_Buffer TF_GetBuffer(TF_Buffer* buffer);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_C_TF_BUFFER_H_
