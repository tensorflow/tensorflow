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
#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_EXPERIMENTAL_STABLE_DELEGATE_DELEGATE_LOADER_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_EXPERIMENTAL_STABLE_DELEGATE_DELEGATE_LOADER_H_

#include <string>

#include "tensorflow/lite/experimental/acceleration/configuration/c/stable_delegate.h"

namespace tflite {
namespace delegates {
namespace utils {

const char kTfLiteStableDelegateSymbol[] = "TFL_TheStableDelegate";

// Loads the TFLite delegate shared library and returns the pointer to
// TfLiteStableDelegate (defined in
// tensorflow/lite/experimental/acceleration/configuration/c/stable_delegate.h).
// The returned pointer could be null if the delegate shared library cannot be
// opened or the delegate symbol cannot be found.
const TfLiteStableDelegate* LoadDelegateFromSharedLibrary(
    const std::string& delegate_path);

// Loads `delegate_symbol` from the delegate shared library and returns a
// pointer to void. It is caller's responsibility to check and cast the pointer
// to other types. The returned pointer could be null if the delegate shared
// library cannot be opened or the delegate symbol cannot be found.
void* LoadSymbolFromSharedLibrary(const std::string& delegate_path,
                                  const std::string& delegate_symbol);

// TODO(b/239825926): Add ABI version check when loading TfLiteStableDelegate.

}  // namespace utils
}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_EXPERIMENTAL_STABLE_DELEGATE_DELEGATE_LOADER_H_
