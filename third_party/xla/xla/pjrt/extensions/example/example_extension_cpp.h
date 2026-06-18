/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PJRT_EXTENSIONS_EXAMPLE_EXAMPLE_EXTENSION_CPP_H_
#define XLA_PJRT_EXTENSIONS_EXAMPLE_EXAMPLE_EXTENSION_CPP_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace xla {

// This is an example extension that can be implemented by subclassing the
// extension's cpp implementation. The C API is provided via wrapper functions,
// into which the plugin's custom implementation of the extension can be
// injected. In order to use this extension, the plugin author must only
// subclass this class and implement the ExampleMethod method.
class ExampleExtensionCpp {
 public:
  virtual ~ExampleExtensionCpp() = default;

  // Creates an extension with the given print prefix.
  explicit ExampleExtensionCpp(absl::string_view print_prefix) {
    print_prefix_ = std::string(print_prefix);
  }

  virtual absl::Status ExampleMethod(int64_t value) = 0;

 protected:
  std::string print_prefix_;
};

}  // namespace xla

#endif  // XLA_PJRT_EXTENSIONS_EXAMPLE_EXAMPLE_EXTENSION_CPP_H_
