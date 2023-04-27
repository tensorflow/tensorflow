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

#include <memory>
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
    std::unique_ptr<TfLiteOpaqueDelegate, void (*)(TfLiteOpaqueDelegate*)>;

class DelegateProvider {
 public:
  virtual ~DelegateProvider() {}

  // Create a list of command-line parsable flags based on tool params inside
  // 'params' whose value will be set to the corresponding runtime flag value.
  virtual std::vector<Flag> CreateFlags(ToolParams* params) const = 0;

  // Log tool params. If 'verbose' is set to false, the param is going to be
  // only logged if its value has been set, say via being parsed from
  // commandline flags.
  virtual void LogParams(const ToolParams& params, bool verbose) const = 0;

  // Create a TfLiteDelegate based on tool params.
  virtual TfLiteDelegatePtr CreateTfLiteDelegate(
      const ToolParams& params) const = 0;

  // Similar to the above, create a TfLiteDelegate based on tool params. If the
  // same set of tool params could lead to creating multiple TfLite delegates,
  // also return a relative rank of the delegate that indicates the order of the
  // returned delegate that should be applied to the TfLite runtime.
  virtual std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const = 0;

  virtual std::string GetName() const = 0;

  const ToolParams& DefaultParams() const { return default_params_; }

 protected:
  template <typename T>
  Flag CreateFlag(const char* name, ToolParams* params,
                  const std::string& usage) const {
    return Flag(
        name,
        [params, name](const T& val, int argv_position) {
          params->Set<T>(name, val, argv_position);
        },
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
#define REGISTER_DELEGATE_PROVIDER(T)                          \
  static tflite::tools::DelegateProviderRegistrar::Register<T> \
      REGISTER_DELEGATE_PROVIDER_VNAME(T);

// Creates a null delegate, useful for cases where no reasonable delegate can be
// created.
TfLiteDelegatePtr CreateNullDelegate();

// A global helper function to get all registered delegate providers.
inline const DelegateProviderList& GetRegisteredDelegateProviders() {
  return DelegateProviderRegistrar::GetProviders();
}

// A helper class to create a list of TfLite delegates based on the provided
// ToolParams and the global DelegateProviderRegistrar.
class ProvidedDelegateList {
 public:
  struct ProvidedDelegate {
    ProvidedDelegate()
        : provider(nullptr), delegate(CreateNullDelegate()), rank(0) {}
    const DelegateProvider* provider;
    TfLiteDelegatePtr delegate;
    int rank;
  };

  ProvidedDelegateList() : ProvidedDelegateList(/*params*/ nullptr) {}

  // 'params' is the ToolParams instance that this class will operate on,
  // including adding all registered delegate parameters to it etc.
  explicit ProvidedDelegateList(ToolParams* params)
      : providers_(GetRegisteredDelegateProviders()), params_(params) {}

  const DelegateProviderList& providers() const { return providers_; }

  // Add all registered delegate params to the contained 'params_'.
  void AddAllDelegateParams() const;

  // Append command-line parsable flags to 'flags' of all registered delegate
  // providers, and associate the flag values at runtime with the contained
  // 'params_'.
  void AppendCmdlineFlags(std::vector<Flag>& flags) const;

  // Removes command-line parsable flag 'name' from 'flags'
  void RemoveCmdlineFlag(std::vector<Flag>& flags,
                         const std::string& name) const;

  // Return a list of TfLite delegates based on the provided 'params', and the
  // list has been already sorted in ascending order according to the rank of
  // the particular parameter that enables the creation of the delegate.
  std::vector<ProvidedDelegate> CreateAllRankedDelegates(
      const ToolParams& params) const;

  // Similar to the above, the list of TfLite delegates are created based on the
  // contained 'params_'.
  std::vector<ProvidedDelegate> CreateAllRankedDelegates() const {
    return CreateAllRankedDelegates(*params_);
  }

 private:
  const DelegateProviderList& providers_;

  // Represent the set of "ToolParam"s that this helper class will operate on.
  ToolParams* const params_;  // Not own the memory.
};
}  // namespace tools
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_DELEGATES_DELEGATE_PROVIDER_H_
