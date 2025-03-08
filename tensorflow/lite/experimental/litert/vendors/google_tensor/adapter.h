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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_GOOGLE_TENSOR_ADAPTER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_GOOGLE_TENSOR_ADAPTER_H_
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert {
namespace google_tensor {
// Flags is a vector of key-value pairs. where key is the flag name and value is
// the flag value. eg. {{"enable_reference", "true"}}
using Flags = std::vector<std::pair<std::string, std::string>>;
typedef absl::Status (*Compile)(absl::string_view serialized_tfl_buffer,
                                absl::string_view soc_model, const Flags& flags,
                                std::string* compiled_code);

// This class adapts the google tensor compiler API for dynamic loading.
class Adapter {
 public:
  // A smart pointer for managing TensorAdapter objects.
  using Ptr = std::unique_ptr<Adapter>;
  struct Api;

  Adapter();
  ~Adapter();

  // Creates a new TensorAdapter and loads the compiler API symbols.
  static litert::Expected<Ptr> Create(
      std::optional<std::string> shared_library_dir);

  // Returns a reference to the loaded API.
  const Api& api() const { return *api_; }

 private:
  // Loads the symbols from the compiler library.
  litert::Expected<void> LoadSymbols(
      std::optional<std::string> shared_library_dir);

  void* dlib_handle_ = nullptr;
  std::unique_ptr<Api> api_;
};

struct Adapter::Api {
  // The function pointer to the compiler wrapper API.
  Compile compile = nullptr;
};

}  // namespace google_tensor
}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_GOOGLE_TENSOR_ADAPTER_H_
