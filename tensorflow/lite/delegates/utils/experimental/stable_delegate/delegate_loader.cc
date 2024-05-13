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
#include <stdlib.h>
#include <string.h>

#include <cerrno>
#include <string>

#include "absl/strings/numbers.h"
#include "tensorflow/lite/acceleration/configuration/c/stable_delegate.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace delegates {
namespace utils {
namespace {
void setLibraryPathEnvironmentVariable(const std::string& delegate_path) {
  std::string directory_path = "";
  // Finds the last '/' character in the file path.
  size_t last_slash_index = delegate_path.rfind('/');
  // If there is no '/' character, then the file path is a relative path and
  // the directory path is empty.
  if (last_slash_index != std::string::npos) {
    // Extracts the directory path from the file path.
    directory_path = delegate_path.substr(0, last_slash_index);
  }
  if (setenv(kTfLiteLibraryPathEnvironmentVariable, directory_path.c_str(),
             /* overwrite= */ 1) != 0) {
    // Delegate loading can continue as the environment variable is optional for
    // the stable delegate shared library to use.
    TFLITE_LOG(WARN) << "Error setting environment variable "
                     << kTfLiteLibraryPathEnvironmentVariable
                     << " with error: " << strerror(errno);
  }
}
}  // namespace

using ::tflite::acceleration::AndroidInfo;
using ::tflite::acceleration::RequestAndroidInfo;

const TfLiteStableDelegate* LoadDelegateFromSharedLibrary(
    const std::string& delegate_path) {
  void* symbol_pointer =
      LoadSymbolFromSharedLibrary(delegate_path, kTfLiteStableDelegateSymbol);
  if (!symbol_pointer) {
    return nullptr;
  }
  return reinterpret_cast<const TfLiteStableDelegate*>(symbol_pointer);
}

void* LoadSymbolFromSharedLibrary(const std::string& delegate_path,
                                  const std::string& delegate_symbol) {
  // TODO(b/268483011): Use android_dlopen_ext to support loading from an offset
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
  // Exports the path of the folder containing the stable delegate shared
  // library to the TFLITE_STABLE_DELEGATE_LIBRARY_PATH environment variable.
  setLibraryPathEnvironmentVariable(delegate_path);
  // On the one hand, the handle would not be closed in production. The resource
  // will be released when the process is killed. On the other hand, it may
  // cause issues for leak detection in testing.
  // TODO(b/268483011): Better support for cleanup the shared library handle.
  delegate_lib_handle = dlopen(delegate_path.c_str(), dlopen_flags);
  if (!delegate_lib_handle) {
    TFLITE_LOG(ERROR) << "Failed to open library " << delegate_path << ": "
                      << dlerror();
    return nullptr;
  }

  void* symbol_pointer = dlsym(delegate_lib_handle, delegate_symbol.c_str());
  if (!symbol_pointer) {
    TFLITE_LOG(ERROR) << "Failed to find " << delegate_symbol
                      << " symbol: " << dlerror();
    dlclose(delegate_lib_handle);
    return nullptr;
  }
  return symbol_pointer;
}

}  // namespace utils
}  // namespace delegates
}  // namespace tflite
