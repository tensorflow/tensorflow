/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/core/acceleration/configuration/c/delegate_plugin.h"
#include "tensorflow/lite/core/c/common.h"

extern "C" {

static TfLiteDelegate* CreateDelegate(const void* settings) {
  static int dummy_delegate_target = 42;
  return reinterpret_cast<TfLiteDelegate*>(&dummy_delegate_target);
}

static void DestroyDelegate(TfLiteDelegate* delegate) {
  // Do nothing.
}

static int DelegateErrno(TfLiteDelegate* from_delegate) { return 0; }

static constexpr TfLiteDelegatePlugin kPluginCApi{
    CreateDelegate,
    DestroyDelegate,
    DelegateErrno,
};

const TfLiteDelegatePlugin* TfLiteGpuDelegatePluginCApi() {
  return &kPluginCApi;
}

}  // extern "C"
