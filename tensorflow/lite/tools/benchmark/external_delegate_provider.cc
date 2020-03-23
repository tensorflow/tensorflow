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
#include "tensorflow/lite/tools/benchmark/benchmark_model.h"
#include "tensorflow/lite/tools/benchmark/delegate_provider.h"
#include "tensorflow/lite/tools/benchmark/logging.h"

#if defined(_WIN32)
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

#include <string>
#include <type_traits>
#include <vector>

namespace tflite {
namespace benchmark {
namespace {
// Library Support construct to handle dynamic library operations
#if defined(_WIN32)
struct LibSupport {
  static void* Load(const char* lib) { return LoadLibrary(lib); }

  static void* GetSymbol(void* handle, const char* symbol) {
    return (void*)GetProcAddress((HMODULE)handle, symbol);
  }

  static int UnLoad(void* handle) { return FreeLibrary((HMODULE)handle); }
};
#else
struct LibSupport {
  static void* Load(const char* lib) {
    return dlopen(lib, RTLD_LAZY | RTLD_LOCAL);
  }

  static void* GetSymbol(void* handle, const char* symbol) {
    return dlsym(handle, symbol);
  }

  static int UnLoad(void* handle) { return dlclose(handle); }
};
#endif

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

// External delegate library construct
struct ExternalLib {
  using CreateDelegatePtr = std::add_pointer<TfLiteDelegate*(
      const char**, const char**, size_t,
      void (*report_error)(const char*))>::type;
  using DestroyDelegatePtr = std::add_pointer<void(TfLiteDelegate*)>::type;

  // Open a given delegate library and load the create/destroy symbols
  bool load(const std::string library) {
    void* handle = LibSupport::Load(library.c_str());
    if (handle == nullptr) {
      TFLITE_LOG(INFO) << "Unable to load external delegate from : " << library;
    } else {
      create = reinterpret_cast<decltype(create)>(
          LibSupport::GetSymbol(handle, "tflite_plugin_create_delegate"));
      destroy = reinterpret_cast<decltype(destroy)>(
          LibSupport::GetSymbol(handle, "tflite_plugin_destroy_delegate"));
      return create && destroy;
    }
    return false;
  }

  CreateDelegatePtr create{nullptr};
  DestroyDelegatePtr destroy{nullptr};
};
}  // namespace

// External delegate provider used to dynamically load delegate libraries
// Note: Assumes the lifetime of the provider exceeds the usage scope of
// the generated delegates.
class ExternalDelegateProvider : public DelegateProvider {
 public:
  std::vector<Flag> CreateFlags(BenchmarkParams* params) const final;

  void AddParams(BenchmarkParams* params) const final;

  void LogParams(const BenchmarkParams& params) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(
      const BenchmarkParams& params) const final;

  std::string GetName() const final { return "EXTERNAL"; }
};
REGISTER_DELEGATE_PROVIDER(ExternalDelegateProvider);

std::vector<Flag> ExternalDelegateProvider::CreateFlags(
    BenchmarkParams* params) const {
  std::vector<Flag> flags = {
      CreateFlag<std::string>("external_delegate_path", params,
                              "The library path for the underlying external."),
      CreateFlag<std::string>(
          "external_delegate_options", params,
          "Comma-seperated options to be passed to the external delegate")};
  return flags;
}

void ExternalDelegateProvider::AddParams(BenchmarkParams* params) const {
  params->AddParam("external_delegate_path",
                   BenchmarkParam::Create<std::string>(""));
  params->AddParam("external_delegate_options",
                   BenchmarkParam::Create<std::string>(""));
}

void ExternalDelegateProvider::LogParams(const BenchmarkParams& params) const {
  TFLITE_LOG(INFO) << "External delegate path : ["
                   << params.Get<std::string>("external_delegate_path") << "]";
  TFLITE_LOG(INFO) << "External delegate options : ["
                   << params.Get<std::string>("external_delegate_options")
                   << "]";
}

TfLiteDelegatePtr ExternalDelegateProvider::CreateTfLiteDelegate(
    const BenchmarkParams& params) const {
  TfLiteDelegatePtr delegate(nullptr, [](TfLiteDelegate*) {});
  std::string lib_path = params.Get<std::string>("external_delegate_path");
  if (!lib_path.empty()) {
    ExternalLib delegate_lib;
    if (delegate_lib.load(lib_path)) {
      // Parse delegate options
      const std::vector<std::string> options = SplitString(
          params.Get<std::string>("external_delegate_options"), ';');
      std::vector<std::string> keys, values;
      for (const auto& option : options) {
        auto key_value = SplitString(option, ':');
        if (key_value.size() == 2) {
          values.push_back(std::move(key_value[1]));
          keys.push_back(std::move(key_value[0]));
        }
      }

      const size_t num_options = keys.size();
      std::vector<const char*> ckeys, cvalues;
      for (int i = 0; i < num_options; ++i) {
        ckeys.push_back(keys[i].c_str());
        cvalues.push_back(values[i].c_str());
      }

      // Create delegate
      delegate =
          TfLiteDelegatePtr(delegate_lib.create(ckeys.data(), cvalues.data(),
                                                num_options, nullptr),
                            delegate_lib.destroy);
    }
  }
  return delegate;
}
}  // namespace benchmark
}  // namespace tflite
