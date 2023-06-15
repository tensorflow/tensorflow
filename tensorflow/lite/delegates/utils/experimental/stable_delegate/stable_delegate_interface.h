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
#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_EXPERIMENTAL_STABLE_DELEGATE_STABLE_DELEGATE_INTERFACE_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_EXPERIMENTAL_STABLE_DELEGATE_STABLE_DELEGATE_INTERFACE_H_

#include "tensorflow/lite/acceleration/configuration/c/stable_delegate.h"

// This header file declares the interface that stable delegate shared
// libraries need to implement. The stable delegate loader will dynamically load
// the shared library.

#ifdef __cplusplus
extern "C" {
#endif

// Define TFL_STABLE_DELEGATE_EXPORT macro to export a stable delegate API
// function properly with a shared library.
#ifdef SWIG
#define TFL_STABLE_DELEGATE_EXPORT
#else  // !defined SWIG
#ifdef _WIN32
// On Windows, the TFL_STABLE_DELEGATE_COMPILE_LIBRARY macro should be defined
// when _building_ a stable delegate shared library, but should not be defined
// when _using_ a stable delegate shared library.
#ifdef TFL_STABLE_DELEGATE_COMPILE_LIBRARY
#define TFL_STABLE_DELEGATE_EXPORT __declspec(dllexport)
#else  // !defined TFL_STABLE_DELEGATE_COMPILE_LIBRARY
#define TFL_STABLE_DELEGATE_EXPORT __declspec(dllimport)
#endif  // !defined TFL_STABLE_DELEGATE_COMPILE_LIBRARY
#else   // !defined _WIN32
#define TFL_STABLE_DELEGATE_EXPORT __attribute__((visibility("default")))
#endif  // !defined _WIN32
#endif  // !defined SWIG

// The variable contains stable delegate metadata and implementation.
//
// The variable is dynamically initialized and it will be used as the entrypoint
// for the stable delegate providers to load the symbols. Don't add other
// initializations, which depend on the sequence of this initialization.
extern TFL_STABLE_DELEGATE_EXPORT const TfLiteStableDelegate
    TFL_TheStableDelegate;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_EXPERIMENTAL_STABLE_DELEGATE_STABLE_DELEGATE_INTERFACE_H_
