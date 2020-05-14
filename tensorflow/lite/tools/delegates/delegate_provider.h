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

#ifndef TENSORFLOW_LITE_TOOLS_DELEGATES_DELEGATE_PROVIDER_H_
#define TENSORFLOW_LITE_TOOLS_DELEGATES_DELEGATE_PROVIDER_H_

#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/tools/tool_params.h"

namespace tflite {
namespace tools {

// Same w/ Interpreter::TfLiteDelegatePtr to avoid pulling
// tensorflow/lite/interpreter.h dependency
using TfLiteDelegatePtr =
    std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>;

class DelegateProvider {
 public:
  virtual ~DelegateProvider() {}

  // Create a list of command-line parsable flags based on tool params inside
  // 'params' whose value will be set to the corresponding runtime flag value.
  virtual std::vector<Flag> CreateFlags(ToolParams* params) const = 0;

  // Log tool params.
  virtual void LogParams(const ToolParams& params) const = 0;

  // Create a TfLiteDelegate based on tool params.
  virtual TfLiteDelegatePtr CreateTfLiteDelegate(
      const ToolParams& params) const = 0;

  virtual std::string GetName() const = 0;

  const ToolParams& DefaultParams() const { return default_params_; }

 protected:
  template <typename T>
  Flag CreateFlag(const char* name, ToolParams* params,
                  const std::string& usage) const {
    return Flag(
        name, [params, name](const T& val) { params->Set<T>(name, val); },
        default_params_.Get<T>(name), usage, Flag::kOptional);
  }
  ToolParams default_params_;
};

using DelegateProviderPtr = std::unique_ptr<DelegateProvider>;
using DelegateProviderList = std::vector<DelegateProviderPtr>;

class DelegateProviderRegistrar {
 public:
  template <typename T>
  struct Register {
    Register() {
      auto* const instance = DelegateProviderRegistrar::GetSingleton();
      instance->providers_.emplace_back(DelegateProviderPtr(new T()));
    }
  };

  static const DelegateProviderList& GetProviders() {
    return GetSingleton()->providers_;
  }

 private:
  DelegateProviderRegistrar() {}
  DelegateProviderRegistrar(const DelegateProviderRegistrar&) = delete;
  DelegateProviderRegistrar& operator=(const DelegateProviderRegistrar&) =
      delete;

  static DelegateProviderRegistrar* GetSingleton() {
    static auto* instance = new DelegateProviderRegistrar();
    return instance;
  }
  DelegateProviderList providers_;
};

#define REGISTER_DELEGATE_PROVIDER_VNAME(T) gDelegateProvider_##T##_
#define REGISTER_DELEGATE_PROVIDER(T)           \
  static DelegateProviderRegistrar::Register<T> \
      REGISTER_DELEGATE_PROVIDER_VNAME(T);

// A global helper function to get all registered delegate providers.
inline const DelegateProviderList& GetRegisteredDelegateProviders() {
  return DelegateProviderRegistrar::GetProviders();
}
}  // namespace tools
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_DELEGATES_DELEGATE_PROVIDER_H_
