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

// This file implements the C API Delegate Plugin for the XNNPACK Delegate.

#include "tensorflow/lite/core/experimental/acceleration/configuration/c/xnnpack_plugin.h"

#include <memory>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"

extern "C" {

static TfLiteDelegate* CreateDelegate(const void* settings) {
  const ::tflite::TFLiteSettings* tflite_settings =
      static_cast<const ::tflite::TFLiteSettings*>(settings);
  auto options(TfLiteXNNPackDelegateOptionsDefault());
  const auto* xnnpack_settings = tflite_settings->xnnpack_settings();
  if (xnnpack_settings) {
    options.num_threads = xnnpack_settings->num_threads();
  }
  return TfLiteXNNPackDelegateCreate(&options);
}

static void DestroyDelegate(TfLiteDelegate* delegate) {
  TfLiteXNNPackDelegateDelete(delegate);
}

static int DelegateErrno(TfLiteDelegate* from_delegate) { return 0; }

static constexpr TfLiteDelegatePlugin kPluginCApi{
    CreateDelegate,
    DestroyDelegate,
    DelegateErrno,
};

const TfLiteDelegatePlugin* TfLiteXnnpackDelegatePluginCApi() {
  return &kPluginCApi;
}

}  // extern "C"
