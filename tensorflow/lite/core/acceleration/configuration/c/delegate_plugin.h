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
// WARNING: Users of TensorFlow Lite should not include this file directly,
// but should instead include
// "third_party/tensorflow/lite/acceleration/configuration/c/delegate_plugin.h".
// Only the TensorFlow Lite implementation itself should include this file
// directly.

#ifndef TENSORFLOW_LITE_CORE_ACCELERATION_CONFIGURATION_C_DELEGATE_PLUGIN_H_
#define TENSORFLOW_LITE_CORE_ACCELERATION_CONFIGURATION_C_DELEGATE_PLUGIN_H_

/// C API types for TF Lite delegate plugins.

// clang-format off
// NOLINTBEGIN(whitespace/line_length)
/// \note Users of TensorFlow Lite should use
/// \code
/// #include "tensorflow/lite/acceleration/configuration/c/delegate_plugin.h"
/// \endcode
/// to access the APIs documented on this page.
// NOLINTEND(whitespace/line_length)
// clang-format on

#include "tensorflow/lite/core/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// clang-format off
// NOLINTBEGIN(whitespace/line_length)
/** \defgroup delegate_plugin lite/acceleration/configuration/c/delegate_plugin.h
 *  @{
 */
// NOLINTEND(whitespace/line_length)
// clang-format on

/// Type of delegate creation function used to allocate and construct a
/// delegate.
///
/// The `tflite_settings` parameter passed to the delegate creation function
/// should be a pointer to a FlatBuffer table object of type
/// `tflite::TFLiteSettings`. We use `const void *` here rather than `const
/// tflite::TFLiteSettings*` since this is a C API so we don't want to directly
/// reference C++ types such as `tflite::TFLiteSettings`.  But note that this
/// address should point to the 'parsed' FlatBuffer object, not the raw byte
/// buffer. (Note that 'parsing' FlatBuffers is very cheap, it's just an offset
/// load.)
///
/// If you are using the FlatBuffers C API, then you can alternatively pass
/// in a value of type `tflite_TFLiteSettings_table_t`, which is a typedef for
/// `const struct tflite_TFLiteSettings_table*` -- that is the corresponding
/// type for the 'parsed' FlatBuffer object in the FlatBuffers C API.
///
/// Ownership of the `tflite_settings` flatbuffer remains with the caller.
/// The caller of a delegate creation function may end the lifetime of the
/// `tflite_settings` FlatBuffer immediately after the call to the function.
/// So the delegate creation function should ensure that any settings that the
/// delegate may need to reference later, after the delegate has been
/// constructed, are copied from the FlatBuffer into storage owned by the
/// delegate.
typedef TfLiteDelegate *TfLiteDelegatePluginCreateFunc(
    const void *tflite_settings);

/// Type of function to destroy and deallocate a delegate.
/// The delegate argument must have been created with the corresponding
/// create function from the same delegate plugin.
typedef void TfLiteDelegatePluginDestroyFunc(TfLiteDelegate *);

/// Type of function to return an error code for the last delegate operation.
/// The delegate argument must have been created with the corresponding
/// create function from the same delegate plugin.
typedef int TfLiteDelegatePluginGetDelegateErrnoFunc(TfLiteDelegate *);

/// Struct to hold all the methods for a delegate plugin.
typedef struct TfLiteDelegatePlugin {
  /// Function to allocate and construct a delegate.
  TfLiteDelegatePluginCreateFunc *create;

  /// Function to deallocate a delegate.
  TfLiteDelegatePluginDestroyFunc *destroy;

  /// Function to return an error code for the last delegate operation.
  TfLiteDelegatePluginGetDelegateErrnoFunc *get_delegate_errno;
} TfLiteDelegatePlugin;

// The following block guarded by TFLITE_USE_OPAQUE_DELEGATE has the exact same
// functionality as the concrete types above but only uses truly opaque types.
// Note that it has to be an addition along with the concrete types at this
// point because the in some cases both types are used together in a same build
// target. e.g. TFLite-in-Play Services initialization context.
#if TFLITE_USE_OPAQUE_DELEGATE

/// Same as TfLiteDelegatePluginCreateFunc but uses truly opaque types.
typedef TfLiteOpaqueDelegateStruct *TfLiteOpaqueDelegatePluginCreateFunc(
    const void *tflite_settings);

/// Same as TfLiteDelegatePluginDestroyFunc but uses truly opaque types.
typedef void TfLiteOpaqueDelegatePluginDestroyFunc(
    TfLiteOpaqueDelegateStruct *delegate);

/// Same as TfLiteDelegatePluginGetDelegateErrnoFunc but uses truly opaque
/// types.
typedef int TfLiteOpaqueDelegatePluginGetDelegateErrnoFunc(
    TfLiteOpaqueDelegateStruct *delegate);

/// Same as TfLiteDelegatePlugin but uses truly opaque types.
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

/** @} */

#ifdef __cplusplus
};  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_CORE_ACCELERATION_CONFIGURATION_C_DELEGATE_PLUGIN_H_
