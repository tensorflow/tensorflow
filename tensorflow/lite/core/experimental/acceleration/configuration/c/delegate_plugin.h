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
// NOLINTBEGIN(whitespace/line_length)
/// WARNING: Users of TensorFlow Lite should not include this file directly,
/// but should instead include
/// "third_party/tensorflow/lite/experimental/acceleration/configuration/c/delegate_plugin.h".
/// Only the TensorFlow Lite implementation itself should include this
/// file directly.
// NOLINTEND(whitespace/line_length)
#ifndef TENSORFLOW_LITE_CORE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_C_DELEGATE_PLUGIN_H_
#define TENSORFLOW_LITE_CORE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_C_DELEGATE_PLUGIN_H_

// C API types for TF Lite delegate plugins.

#include "tensorflow/lite/core/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Type of function to allocate and construct a delegate.
// The tflite_settings parameter should be a pointer to a FlatBuffer table
// object of type tflite::TFLiteSettings.  (We use 'void *' here since this
// is a C API so we don't want to directly reference C++ types such
// as tflite::TFLiteSettings.)
typedef TfLiteDelegate *TfLiteDelegatePluginCreateFunc(
    const void *tflite_settings);

// Type of function to destroy and deallocate a delegate.
// The delegate argument must have been created with the corresponding
// create function from the same delegate plugin.
typedef void TfLiteDelegatePluginDestroyFunc(TfLiteDelegate *);

// Type of function to return an error code for the last delegate operation.
// The delegate argument must have been created with the corresponding
// create function from the same delegate plugin.
typedef int TfLiteDelegatePluginGetDelegateErrnoFunc(TfLiteDelegate *);

// Struct to hold all the methods for a delegate plugin.
typedef struct TfLiteDelegatePlugin {
  // Function to allocate and construct a delegate.
  TfLiteDelegatePluginCreateFunc *create;

  // Function to deallocate a delegate.
  TfLiteDelegatePluginDestroyFunc *destroy;

  // Function to return an error code for the last delegate operation.
  TfLiteDelegatePluginGetDelegateErrnoFunc *get_delegate_errno;
} TfLiteDelegatePlugin;

// The following block guarded by TFLITE_USE_OPAQUE_DELEGATE has the exact same
// functionality as the concrete types above but only uses truly opaque types.
// Note that it has to be an addition along with the concrete types at this
// point because the in some cases both types are used together in a same build
// target. e.g. TFLite-in-Play Services initialization context.
#if TFLITE_USE_OPAQUE_DELEGATE

// Same as TfLiteDelegatePluginCreateFunc but uses truly opaque types.
typedef TfLiteOpaqueDelegateStruct *TfLiteOpaqueDelegatePluginCreateFunc(
    const void *tflite_settings);

// Same as TfLiteDelegatePluginDestroyFunc but uses truly opaque types.
typedef void TfLiteOpaqueDelegatePluginDestroyFunc(
    TfLiteOpaqueDelegateStruct *delegate);

// Same as TfLiteDelegatePluginGetDelegateErrnoFunc but uses truly opaque types.
typedef int TfLiteOpaqueDelegatePluginGetDelegateErrnoFunc(
    TfLiteOpaqueDelegateStruct *delegate);

// Same as TfLiteDelegatePlugin but uses truly opaque types.
typedef struct TfLiteOpaqueDelegatePlugin {
  TfLiteOpaqueDelegatePluginCreateFunc *create;

  TfLiteOpaqueDelegatePluginDestroyFunc *destroy;

  TfLiteOpaqueDelegatePluginGetDelegateErrnoFunc *get_delegate_errno;
} TfLiteOpaqueDelegatePlugin;

#else

typedef TfLiteDelegatePluginCreateFunc TfLiteOpaqueDelegatePluginCreateFunc;
typedef TfLiteDelegatePluginDestroyFunc TfLiteOpaqueDelegatePluginDestroyFunc;
typedef TfLiteDelegatePluginGetDelegateErrnoFunc
    TfLiteOpaqueDelegatePluginGetDelegateErrnoFunc;
typedef TfLiteDelegatePlugin TfLiteOpaqueDelegatePlugin;

#endif  // TFLITE_USE_OPAQUE_DELEGATE

#ifdef __cplusplus
};  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_CORE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_C_DELEGATE_PLUGIN_H_
