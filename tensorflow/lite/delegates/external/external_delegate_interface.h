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
#ifndef TENSORFLOW_LITE_DELEGATES_EXTERNAL_EXTERNAL_DELEGATE_INTERFACE_H_
#define TENSORFLOW_LITE_DELEGATES_EXTERNAL_EXTERNAL_DELEGATE_INTERFACE_H_

#include "tensorflow/lite/c/common.h"

// This header file declares the interface that external delegate shared
// libraries need to implement.  The functions declared here are not defined
// in TF Lite itself -- this just declares the interface to functions that
// are defined elsewhere, in a shared library that TfLiteExternalDelegate will
// dynamically load.

#ifdef __cplusplus
extern "C" {
#endif

// Define TFL_EXTERNAL_DELEGATE_EXPORT macro to export an external delegate API
// function properly with a shared library.
#ifdef SWIG
#define TFL_EXTERNAL_DELEGATE_EXPORT
#else  // !defined SWIG
#ifdef _WIN32
// On Windows, the TFL_EXTERNAL_DELEGATE_COMPILE_LIBRARY macro should be
// defined when _building_ an external delegate shared library, but should not
// be defined when _using_ an external delegate shared library.
#ifdef TFL_EXTERNAL_DELEGATE_COMPILE_LIBRARY
#define TFL_EXTERNAL_DELEGATE_EXPORT __declspec(dllexport)
#else  // !defined TFL_EXTERNAL_DELEGATE_COMPILE_LIBRARY
// We may not actually need dllimport,
// since the symbols will looked up dynamically?
#define TFL_EXTERNAL_DELEGATE_EXPORT __declspec(dllimport)
#endif  // !defined TFL_EXTERNAL_DELEGATE_COMPILE_LIBRARY
#else   // !defined _WIN32
#define TFL_EXTERNAL_DELEGATE_EXPORT __attribute__((visibility("default")))
#endif  // !defined _WIN32
#endif  // !defined SWIG

// Creates a delegate object based on provided key-value options.
//
// The delegate is initialized using the option settings specified by the
// names in `options_keys` and the corresponding values in `options_values`,
// which are both arrays of length `num_options` of NUL-terminated C strings.
// This function *should not* modify those arrays, but the caller must not rely
// on that. `options_keys` and `options_values` may be null if `num_options` is
// zero.
//
// On success, returns a non-null value that should be deallocated with
// tflite_plugin_destroy_delegate when no longer needed.
// On failure, returns NULL to indicate an error, with the detailed information
// reported by calling `report_error` if provided.
extern TFL_EXTERNAL_DELEGATE_EXPORT TfLiteDelegate*
tflite_plugin_create_delegate(const char* const* options_keys,
                              const char* const* options_values,
                              size_t num_options,
                              void (*report_error)(const char*));

// Destroys a delegate object that was created by tflite_plugin_create_delegate.
// Calling this with nullptr as the argument value is allowed and has no effect.
extern TFL_EXTERNAL_DELEGATE_EXPORT void tflite_plugin_destroy_delegate(
    TfLiteDelegate* delegate);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_DELEGATES_EXTERNAL_EXTERNAL_DELEGATE_INTERFACE_H_
