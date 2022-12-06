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
#ifndef TENSORFLOW_LITE_CORE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_C_STABLE_DELEGATE_H_
#define TENSORFLOW_LITE_CORE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_C_STABLE_DELEGATE_H_

// C API types for TFLite delegates that implement stable delegate ABI.

#include "tensorflow/lite/core/shims/c/experimental/acceleration/configuration/delegate_plugin.h"

#ifdef __cplusplus
extern "C" {
#endif

// Constant that identifies the TfLiteStableDelegate ABI version that the
// delegate supports. This will get incremented if there are changes to the
// struct. The version is in semver 2 format (see https://semver.org).
#define TFL_STABLE_DELEGATE_ABI_VERSION "1.0.0"

// Contains stable delegate metadata and implementation.
typedef struct TfLiteStableDelegate {
  // The struct ABI version this delegate supports in semver 2 format. It should
  // be set to TFL_STABLE_DELEGATE_ABI_VERSION.
  const char* delegate_abi_version;

  // Uniquely identifies a delegate.
  const char* delegate_name;

  // Release version of this delegate.
  const char* delegate_version;

  // Provides the implementation of the delegate plugin.
  const TfLiteOpaqueDelegatePlugin* delegate_plugin;
} TfLiteStableDelegate;

#ifdef __cplusplus
};  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_CORE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_C_STABLE_DELEGATE_H_
