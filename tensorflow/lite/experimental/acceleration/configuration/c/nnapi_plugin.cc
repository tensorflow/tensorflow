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

// This file implements the Delegate Plugin for the NNAPI Delegate.
// It provides both

#include "tensorflow/lite/experimental/acceleration/configuration/c/nnapi_plugin.h"

#include <memory>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/configuration/nnapi_plugin.h"

extern "C" {

static TfLiteDelegate* CreateDelegate(const void* settings) {
  const ::tflite::TFLiteSettings* tflite_settings =
      static_cast<const ::tflite::TFLiteSettings*>(settings);
  tflite::delegates::NnapiPlugin nnapi_plugin(*tflite_settings);
  return new tflite::StatefulNnApiDelegate(nnapi_plugin.Options());
}

static void DestroyDelegate(TfLiteDelegate* delegate) {
  delete static_cast<tflite::StatefulNnApiDelegate*>(delegate);
}

static int DelegateErrno(TfLiteDelegate* from_delegate) {
  auto nnapi_delegate =
      static_cast<tflite::StatefulNnApiDelegate*>(from_delegate);
  return nnapi_delegate->GetNnApiErrno();
}

static constexpr TfLiteDelegatePlugin kPluginCApi{
    CreateDelegate,
    DestroyDelegate,
    DelegateErrno,
};

const TfLiteDelegatePlugin* TfLiteNnapiDelegatePluginCApi() {
  return &kPluginCApi;
}

}  // extern "C"
