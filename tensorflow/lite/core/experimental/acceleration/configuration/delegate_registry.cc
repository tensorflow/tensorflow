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
#include "tensorflow/lite/core/experimental/acceleration/configuration/delegate_registry.h"

#include <functional>
#include <string>

#include "absl/synchronization/mutex.h"

namespace tflite {
namespace delegates {

void DelegatePluginRegistry::RegisterImpl(
    const std::string& name,
    std::function<
        std::unique_ptr<DelegatePluginInterface>(const TFLiteSettings&)>
        creator_function) {
  absl::MutexLock lock(&mutex_);
  factories_[name] = creator_function;
}

std::unique_ptr<DelegatePluginInterface> DelegatePluginRegistry::CreateImpl(
    const std::string& name, const TFLiteSettings& settings) {
  absl::MutexLock lock(&mutex_);
  auto it = factories_.find(name);
  return (it != factories_.end()) ? it->second(settings) : nullptr;
}

DelegatePluginRegistry* DelegatePluginRegistry::GetSingleton() {
  static auto* instance = new DelegatePluginRegistry();
  return instance;
}

std::unique_ptr<DelegatePluginInterface> DelegatePluginRegistry::CreateByName(
    const std::string& name, const TFLiteSettings& settings) {
  auto* const instance = DelegatePluginRegistry::GetSingleton();
  return instance->CreateImpl(name, settings);
}

DelegatePluginRegistry::Register::Register(const std::string& name,
                                           CreatorFunction creator_function) {
  auto* const instance = DelegatePluginRegistry::GetSingleton();
  instance->RegisterImpl(name, creator_function);
}

}  // namespace delegates
}  // namespace tflite
