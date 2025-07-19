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

#include "xla/pjrt/plugin/example_plugin/example_extension_impl.h"

#include <cstdint>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/extensions/example/example_extension_cpp.h"

namespace xla {

ExampleExtensionImpl::ExampleExtensionImpl(absl::string_view print_prefix,
                                           absl::string_view my_print_prefix)
    : ExampleExtensionCpp(print_prefix),
      my_print_prefix_(std::string(my_print_prefix)) {}

absl::Status ExampleExtensionImpl::ExampleMethod(int64_t value) {
  LOG(INFO) << print_prefix_ << " value=" << value;
  LOG(INFO) << my_print_prefix_ << " value=" << value;
  return absl::OkStatus();
}

}  // namespace xla
