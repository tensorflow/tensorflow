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
#ifndef TENSORFLOW_LITE_CORE_ACCELERATION_CONFIGURATION_STABLE_DELEGATE_REGISTRY_H_
#define TENSORFLOW_LITE_CORE_ACCELERATION_CONFIGURATION_STABLE_DELEGATE_REGISTRY_H_

#include <string>
#include <unordered_map>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/lite/core/acceleration/configuration/c/stable_delegate.h"

namespace tflite {
namespace delegates {

// A dedicated singleton registry for TfLiteStableDelegate.
// Note that there is also a non-stable delegate registry
// (third_party/tensorflow/lite/core/acceleration/configuration/
// delegate_registry.h)
// but it does not serve very well for TfLiteStableDelegate as it could not
// register all the information of TfLiteStableDelegate and it uses concrete
// types.
class StableDelegateRegistry {
 public:
  // Registers a TfLiteStableDelegate pointer to the registry.
  static void RegisterStableDelegate(const TfLiteStableDelegate* delegate);
  // Retrieves the pointer to the corresponding TfLiteStableDelegate from the
  // registry given a delegate name. Returns nullptr if no registration found.
  static const TfLiteStableDelegate* RetrieveStableDelegate(
      const std::string& name);

 private:
  static StableDelegateRegistry* GetSingleton();
  void RegisterStableDelegateImpl(const TfLiteStableDelegate* delegate);
  const TfLiteStableDelegate* RetrieveStableDelegateImpl(
      const std::string& name);

  absl::Mutex mutex_;
  std::unordered_map<std::string, const TfLiteStableDelegate*> registry_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_ACCELERATION_CONFIGURATION_STABLE_DELEGATE_REGISTRY_H_
