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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_DELEGATE_REGISTRY_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_DELEGATE_REGISTRY_H_

#include <memory>
#include <unordered_map>

#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/mutable_op_resolver.h"

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

// A shared pointer to `TfLiteExternalContext`, similar to `TfLiteDelegatePtr`.
using TfLiteExternalContextPtr = std::shared_ptr<TfLiteExternalContext>;

template <typename AcceleratorType, typename AcceleratorPtrType>
class AcceleratorPluginInterface {
 public:
  virtual AcceleratorPtrType Create() = 0;
  // Some accelerators require their own custom ops, such as the Coral plugin.
  // Default to an empty MutableOpResolver.
  virtual std::unique_ptr<MutableOpResolver> CreateOpResolver() {
    return absl::make_unique<MutableOpResolver>();
  }
  virtual int GetDelegateErrno(AcceleratorType* from_delegate) = 0;
  virtual ~AcceleratorPluginInterface() = default;
};

// `AcceleratorPluginInterface` implemented for `TfLiteDelegate`.
using DelegatePluginInterface =
    AcceleratorPluginInterface<TfLiteDelegate, TfLiteDelegatePtr>;
// `AcceleratorPluginInterface` implemented for `TfLiteExternalContext`.
using ContextPluginInterface =
    AcceleratorPluginInterface<TfLiteExternalContext, TfLiteExternalContextPtr>;

// A stripped-down registry that allows delegate plugins to be created by name.
//
// Limitations:
// - Doesn't allow deregistration.
// - Doesn't check for duplication registration.
//
template <typename AcceleratorType, typename AcceleratorPtrType>
class AcceleratorRegistry {
 public:
  typedef std::function<std::unique_ptr<AcceleratorPluginInterface<
      AcceleratorType, AcceleratorPtrType>>(const TFLiteSettings&)>
      CreatorFunction;
  // Returns a AcceleratorPluginInterface registered with `name` or nullptr if
  // no matching plugin found. TFLiteSettings is per-plugin, so that the
  // corresponding delegate options data lifetime is maintained.
  static std::unique_ptr<
      AcceleratorPluginInterface<AcceleratorType, AcceleratorPtrType>>
  CreateByName(const std::string& name, const TFLiteSettings& settings) {
    auto* const instance = AcceleratorRegistry::GetSingleton();
    return instance->CreateImpl(name, settings);
  }

  // Struct to be statically allocated for registration.
  struct Register {
    Register(const std::string& name, CreatorFunction creator_function) {
      auto* const instance = AcceleratorRegistry::GetSingleton();
      instance->RegisterImpl(name, creator_function);
    }
  };

 private:
  void RegisterImpl(const std::string& name, CreatorFunction creator_function) {
    absl::MutexLock lock(&mutex_);
    factories_[name] = creator_function;
  }

  std::unique_ptr<
      AcceleratorPluginInterface<AcceleratorType, AcceleratorPtrType>>
  CreateImpl(const std::string& name, const TFLiteSettings& settings) {
    absl::MutexLock lock(&mutex_);
    auto it = factories_.find(name);
    return (it != factories_.end()) ? it->second(settings) : nullptr;
  }

  static AcceleratorRegistry* GetSingleton() {
    static auto* instance = new AcceleratorRegistry();
    return instance;
  }

  absl::Mutex mutex_;
  std::unordered_map<std::string, CreatorFunction> factories_
      GUARDED_BY(mutex_);
};

using DelegatePluginRegistry =
    AcceleratorRegistry<TfLiteDelegate, TfLiteDelegatePtr>;
using ContextPluginRegistry =
    AcceleratorRegistry<TfLiteExternalContext, TfLiteExternalContextPtr>;

}  // namespace delegates
}  // namespace tflite

#define TFLITE_REGISTER_ACCELERATOR_FACTORY_FUNCTION_VNAME( \
    name, f, accelerator_type, accelerator_ptr_type)        \
  static auto* g_delegate_plugin_##name##_ =                \
      new AcceleratorRegistry<accelerator_type,             \
                              accelerator_ptr_type>::Register(#name, f);
#define TFLITE_REGISTER_DELEGATE_FACTORY_FUNCTION(name, f)                    \
  TFLITE_REGISTER_ACCELERATOR_FACTORY_FUNCTION_VNAME(name, f, TfLiteDelegate, \
                                                     TfLiteDelegatePtr);
#define TFLITE_REGISTER_EXTERNAL_CONTEXT_FACTORY_FUNCTION(name, f) \
  TFLITE_REGISTER_ACCELERATOR_FACTORY_FUNCTION_VNAME(              \
      name, f, TfLiteExternalContext, TfLiteExternalContextPtr);

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_CONFIGURATION_DELEGATE_REGISTRY_H_
