/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_C_C_API_MACROS_H_
#define TENSORFLOW_C_C_API_MACROS_H_

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

// TF_Bool is the C API typedef for unsigned char, while TF_BOOL is
// the datatype for boolean tensors.
#ifndef TF_Bool
#define TF_Bool unsigned char
#endif  // TF_Bool

// Macro used to calculate struct size for maintaining ABI stability across
// different struct implementations.
#ifndef TF_OFFSET_OF_END
#define TF_OFFSET_OF_END(TYPE, MEMBER) \
  (offsetof(TYPE, MEMBER) + sizeof(((TYPE *)0)->MEMBER))
#endif  // TF_OFFSET_OF_END

#endif  // TENSORFLOW_C_C_API_MACROS_H_
