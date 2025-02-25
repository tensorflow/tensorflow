// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/vendors/google_tensor/adapter.h"

#include <dlfcn.h>

#include <optional>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert {
namespace google_tensor {

Adapter::Adapter() : api_(new Api) {}

Adapter::~Adapter() {
  if (dlib_handle_) {
    dlclose(dlib_handle_);  // Use dlclose directly
  }
}

litert::Expected<Adapter::Ptr> Adapter::Create(
    std::optional<std::string> shared_library_dir) {
  Ptr adapter(new Adapter);
  auto status = adapter->LoadSymbols(shared_library_dir);
  if (!status.HasValue()) {
    LITERT_LOG(LITERT_ERROR, "Failed to create Adapter: %s",
               status.Error().Message().c_str());
    return status.Error();
  }
  return adapter;
}

litert::Expected<void> Adapter::LoadSymbols(
    std::optional<std::string> shared_library_dir) {
  constexpr auto kLibTensorTPUCompiler = "libcompiler_api_wrapper.so";

  const std::vector<std::string> so_paths = {
      shared_library_dir.has_value()
          ? absl::StrCat(*shared_library_dir, "/", kLibTensorTPUCompiler)
          : kLibTensorTPUCompiler};

  // Use dlopen directly
  for (const auto& path : so_paths) {
    dlib_handle_ = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (dlib_handle_) {
      void* init_func = dlsym(dlib_handle_, "Initialize");
      if (init_func) {
        (*reinterpret_cast<void (*)()>(init_func))();
      }
      break;  // Found the library
    }
  }

  if (!dlib_handle_) {
    const std::string error_message =
        "Failed to load Tensor TPU compiler library: " + std::string(dlerror());
    LITERT_LOG(LITERT_ERROR, "Failed to load Tensor TPU compiler library: %s",
               error_message.c_str());  // Include dlerror() for more info
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, error_message);
  }

  api_->compile =
      reinterpret_cast<Compile>(dlsym(dlib_handle_, "CompileFlatbuffer"));
  if (!api_->compile) {
    const std::string error_message =
        "Failed to load Tensor TPU compiler API: " + std::string(dlerror());
    LITERT_LOG(LITERT_ERROR, "Failed to load Tensor TPU compiler API: %s",
               error_message.c_str());  // Include dlerror()
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, error_message);
  }

  LITERT_LOG(LITERT_INFO, "Tensor TPU compiler API symbols loaded");
  return {};
}

}  // namespace google_tensor
}  // namespace litert
