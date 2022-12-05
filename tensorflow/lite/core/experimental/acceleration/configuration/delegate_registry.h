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
// NOLINTBEGIN(whitespace/line_length)
/// WARNING: Users of TensorFlow Lite should not include this file directly,
/// but should instead include
/// "third_party/tensorflow/lite/experimental/acceleration/configuration/delegate_registry.h".
/// Only the TensorFlow Lite implementation itself should include this
/// file directly.
// NOLINTEND(whitespace/line_length)
#ifndef TENSORFLOW_LITE_CORE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_DELEGATE_REGISTRY_H_
#define TENSORFLOW_LITE_CORE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_DELEGATE_REGISTRY_H_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "absl/synchronization/mutex.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"

// Defines an interface for TFLite delegate plugins.
//
// The acceleration library aims to support all TFLite delegates based on
// configuration expressed as data (flatbuffers). However, consumers tend to
// care about size and also use a subset of delegates. Hence we don't want to
// statically build against all delegates.
//
// This interface allows plugins to handle specific delegates.
//
// Goal of this interface is not to abstract away all the differences between
// delegates. The goal is only to avoid static linking.
//
// Note to implementers: this interface may change if new delegates don't fit
// into the same design.
namespace tflite {
namespace delegates {

// Same w/ Interpreter::TfLiteDelegatePtr to avoid pulling
// tensorflow/lite/interpreter.h dependency
using TfLiteDelegatePtr =
    std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>;

class DelegatePluginInterface {
 public:
  virtual TfLiteDelegatePtr Create() = 0;
  virtual int GetDelegateErrno(TfLiteDelegate* from_delegate) = 0;
  virtual ~DelegatePluginInterface() = default;
};

// A stripped-down registry that allows delegate plugins to be created by name.
//
// Limitations:
// - Doesn't allow deregistration.
// - Doesn't check for duplication registration.
//
class DelegatePluginRegistry {
 public:
  typedef std::function<std::unique_ptr<DelegatePluginInterface>(
      const TFLiteSettings&)>
      CreatorFunction;
  // Returns a DelegatePluginInterface registered with `name` or nullptr if no
  // matching plugin found.
  // TFLiteSettings is per-plugin, so that the corresponding delegate options
  // data lifetime is maintained.
  static std::unique_ptr<DelegatePluginInterface> CreateByName(
      const std::string& name, const TFLiteSettings& settings);

  // Struct to be statically allocated for registration.
  struct Register {
    Register(const std::string& name, CreatorFunction creator_function);
  };

 private:
  void RegisterImpl(const std::string& name, CreatorFunction creator_function);
  std::unique_ptr<DelegatePluginInterface> CreateImpl(
      const std::string& name, const TFLiteSettings& settings);
  static DelegatePluginRegistry* GetSingleton();
  absl::Mutex mutex_;
  std::unordered_map<std::string, CreatorFunction> factories_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace delegates
}  // namespace tflite

#define TFLITE_REGISTER_DELEGATE_FACTORY_FUNCTION_VNAME(name, f) \
  static auto* g_delegate_plugin_##name##_ =                     \
      new DelegatePluginRegistry::Register(#name, f);
#define TFLITE_REGISTER_DELEGATE_FACTORY_FUNCTION(name, f) \
  TFLITE_REGISTER_DELEGATE_FACTORY_FUNCTION_VNAME(name, f);

#endif  // TENSORFLOW_LITE_CORE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_DELEGATE_REGISTRY_H_
