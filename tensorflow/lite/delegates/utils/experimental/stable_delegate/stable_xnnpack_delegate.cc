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
#include "tensorflow/lite/core/shims/c/c_api.h"
#include "tensorflow/lite/core/shims/c/experimental/acceleration/configuration/xnnpack_plugin.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/stable_delegate_interface.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * A stable delegate wrapper of the XNNPack delegate for testing.
 * The built stable delegate binary is used to verify the stable delegate
 * providers by checking if the delegate settings are propagated correctly.
 */
extern TFL_STABLE_DELEGATE_EXPORT const TfLiteStableDelegate
    TFL_TheStableDelegate = {TFL_STABLE_DELEGATE_ABI_VERSION, "XNNPACKDelegate",
                             TfLiteVersion(),
                             TfLiteXnnpackDelegatePluginCApi()};
#ifdef __cplusplus
}  // extern "C"
#endif
