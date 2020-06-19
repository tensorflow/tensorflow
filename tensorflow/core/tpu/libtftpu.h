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

#ifndef TENSORFLOW_CORE_TPU_LIBTFTPU_H_
#define TENSORFLOW_CORE_TPU_LIBTFTPU_H_

// Unfortunately we have to add an Fn suffix because we cannot have the same
// name for both a function and a element within a struct in the global
// namespace in gcc. This restriction doesn't exist in clang.
#define TFTPU_ADD_FN_IN_STRUCT(FnName) decltype(FnName)* FnName##Fn;

#ifdef SWIG
#define TFTPU_CAPI_EXPORT
#else
#if defined(_WIN32)
#ifdef TF_COMPILE_LIBRARY
#define TFTPU_CAPI_EXPORT __declspec(dllexport)
#else
#define TFTPU_CAPI_EXPORT __declspec(dllimport)
#endif  // TF_COMPILE_LIBRARY
#else
#define TFTPU_CAPI_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32
#endif  // SWIG

#ifdef __cplusplus
extern "C" {
#endif

TFTPU_CAPI_EXPORT void TfTpu_Initialize();

#ifdef __cplusplus
}
#endif

struct TfTpu_BaseFn {
  TFTPU_ADD_FN_IN_STRUCT(TfTpu_Initialize);
};

#endif  // TENSORFLOW_CORE_TPU_LIBTFTPU_H_
