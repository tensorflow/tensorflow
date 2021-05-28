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

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/external/external_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

// Split a given string to a vector of string using a delimiter character
std::vector<std::string> SplitString(const std::string& str, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream ss(str);
  while (std::getline(ss, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

// External delegate provider used to dynamically load delegate libraries
// Note: Assumes the lifetime of the provider exceeds the usage scope of
// the generated delegates.
class ExternalDelegateProvider : public DelegateProvider {
 public:
  ExternalDelegateProvider() {
    default_params_.AddParam("external_delegate_path",
                             ToolParam::Create<std::string>(""));
    default_params_.AddParam("external_delegate_options",
                             ToolParam::Create<std::string>(""));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "EXTERNAL"; }
};
REGISTER_DELEGATE_PROVIDER(ExternalDelegateProvider);

std::vector<Flag> ExternalDelegateProvider::CreateFlags(
    ToolParams* params) const {
  std::vector<Flag> flags = {
      CreateFlag<std::string>("external_delegate_path", params,
                              "The library path for the underlying external."),
      CreateFlag<std::string>(
          "external_delegate_options", params,
          "A list of comma-separated options to be passed to the external "
          "delegate. Each option is a colon-separated key-value pair, e.g. "
          "option_name:option_value.")};
  return flags;
}

void ExternalDelegateProvider::LogParams(const ToolParams& params,
                                         bool verbose) const {
  LOG_TOOL_PARAM(params, std::string, "external_delegate_path",
                 "External delegate path", verbose);
  LOG_TOOL_PARAM(params, std::string, "external_delegate_options",
                 "External delegate options", verbose);
}

TfLiteDelegatePtr ExternalDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  TfLiteDelegatePtr delegate(nullptr, [](TfLiteDelegate*) {});
  std::string lib_path = params.Get<std::string>("external_delegate_path");
  if (!lib_path.empty()) {
    auto delegate_options =
        TfLiteExternalDelegateOptionsDefault(lib_path.c_str());

    // Parse delegate options
    const std::vector<std::string> options =
        SplitString(params.Get<std::string>("external_delegate_options"), ';');
    std::vector<std::string> keys, values;
    // We reserve the memory here to avoid memory pointer change during
    // insertion to vectors above.
    keys.reserve(options.size());
    values.reserve(options.size());
    for (const auto& option : options) {
      auto key_value = SplitString(option, ':');
      if (key_value.size() == 2) {
        // The inserted (key,value) pair has to outlive the
        // TfLiteExternalDelegateCreate call, therefore, we use two vectors
        // 'keys' and 'values' to achieve this.
        // Also, we will insert the memory pointer of key and value to
        // delegate_options later, we have to ensure the pointer won't change by
        // reserving the memory earlier.
        keys.emplace_back(key_value[0]);
        values.emplace_back(key_value[1]);
        delegate_options.insert(&delegate_options, keys.back().c_str(),
                                values.back().c_str());
      }
    }

    auto external_delegate = TfLiteExternalDelegateCreate(&delegate_options);
    return TfLiteDelegatePtr(external_delegate, [](TfLiteDelegate* delegate) {
      TfLiteExternalDelegateDelete(delegate);
    });
  }
  return delegate;
}

std::pair<TfLiteDelegatePtr, int>
ExternalDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr), params.GetPosition<std::string>(
                                            "external_delegate_path"));
}

}  // namespace tools
}  // namespace tflite
