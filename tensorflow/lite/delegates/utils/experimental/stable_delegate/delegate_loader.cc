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
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/delegate_loader.h"

#include <dlfcn.h>

#include <string>

#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace delegates {
namespace utils {

using ::tflite::acceleration::AndroidInfo;
using ::tflite::acceleration::RequestAndroidInfo;

const TfLiteStableDelegate* LoadDelegateFromSharedLibrary(
    const std::string& delegate_path) {
  return LoadDelegateFromSharedLibrary(delegate_path,
                                       kTfLiteStableDelegateSymbol);
}

const TfLiteStableDelegate* LoadDelegateFromSharedLibrary(
    const std::string& delegate_path, const std::string& delegate_symbol) {
  // TODO(b/239825926): Use android_dlopen_ext to support loading from an offset
  // within an apk.
  void* delegate_lib_handle = nullptr;
  // RTLD_NOW: Ensures that any dynamic linking errors occur early rather than
  // crash later.
  // RTLD_LOCAL: Symbols are not available to subsequent objects.
  int dlopen_flags = RTLD_NOW | RTLD_LOCAL;
  int sdk_version;
  AndroidInfo android_info;
  if (RequestAndroidInfo(&android_info).ok() &&
      absl::SimpleAtoi(android_info.android_sdk_version, &sdk_version) &&
      sdk_version >= 23) {
    // RTLD_NODELETE: Not unload the shared object during dlclose to prevent
    // thread specific key leakage. It is supported since Android SDK level 23.
    dlopen_flags |= RTLD_NODELETE;
    TFLITE_LOG(INFO) << "Android SDK level is " << sdk_version
                     << ", using dlopen with RTLD_NODELETE.";
  }
  delegate_lib_handle = dlopen(delegate_path.c_str(), dlopen_flags);
  if (!delegate_lib_handle) {
    TFLITE_LOG(ERROR) << "Failed to open library " << delegate_path << ": "
                      << dlerror();
    return nullptr;
  }

  auto* stable_delegate_pointer = reinterpret_cast<TfLiteStableDelegate*>(
      dlsym(delegate_lib_handle, delegate_symbol.c_str()));
  if (!stable_delegate_pointer) {
    TFLITE_LOG(ERROR) << "Failed to find " << delegate_symbol
                      << " symbol: " << dlerror();
    dlclose(delegate_lib_handle);
    return nullptr;
  }
  return stable_delegate_pointer;
}

}  // namespace utils
}  // namespace delegates
}  // namespace tflite
