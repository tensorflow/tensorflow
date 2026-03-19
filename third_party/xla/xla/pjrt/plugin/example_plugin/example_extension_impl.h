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

#ifndef XLA_PJRT_PLUGIN_EXAMPLE_PLUGIN_EXAMPLE_EXTENSION_IMPL_H_
#define XLA_PJRT_PLUGIN_EXAMPLE_PLUGIN_EXAMPLE_EXTENSION_IMPL_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/extensions/example/example_extension_cpp.h"

namespace xla {

// ExampleExtensionImpl is an implementation of ExampleExtensionCpp that
// demonstrates how to implement a plugin extension that is customized for
// MyPlugin.
class ExampleExtensionImpl : public ExampleExtensionCpp {
 public:
  explicit ExampleExtensionImpl(absl::string_view print_prefix,
                                absl::string_view my_print_prefix);

  absl::Status ExampleMethod(int64_t value) override;

 private:
  std::string my_print_prefix_;
};

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_EXAMPLE_PLUGIN_EXAMPLE_EXTENSION_IMPL_H_
