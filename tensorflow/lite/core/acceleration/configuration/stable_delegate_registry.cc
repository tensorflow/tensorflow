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
#include "tensorflow/lite/core/acceleration/configuration/stable_delegate_registry.h"

#include <string>

#include "absl/synchronization/mutex.h"
#include "tensorflow/lite/core/acceleration/configuration/c/stable_delegate.h"

namespace tflite {
namespace delegates {

void StableDelegateRegistry::RegisterStableDelegate(
    const TfLiteStableDelegate* delegate) {
  auto* const instance = StableDelegateRegistry::GetSingleton();
  instance->RegisterStableDelegateImpl(delegate);
}

const TfLiteStableDelegate* StableDelegateRegistry::RetrieveStableDelegate(
    const std::string& name) {
  auto* const instance = StableDelegateRegistry::GetSingleton();
  return instance->RetrieveStableDelegateImpl(name);
}

void StableDelegateRegistry::RegisterStableDelegateImpl(
    const TfLiteStableDelegate* delegate) {
  absl::MutexLock lock(&mutex_);
  registry_[delegate->delegate_name] = delegate;
}

const TfLiteStableDelegate* StableDelegateRegistry::RetrieveStableDelegateImpl(
    const std::string& name) {
  absl::MutexLock lock(&mutex_);
  if (registry_.find(name) == registry_.end()) {
    return nullptr;
  } else {
    return registry_[name];
  }
}

StableDelegateRegistry* StableDelegateRegistry::GetSingleton() {
  static auto* instance = new StableDelegateRegistry();
  return instance;
}

}  // namespace delegates
}  // namespace tflite
