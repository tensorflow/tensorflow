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

#ifndef XLA_CODEGEN_TOOLS_TEST_LIB_H_
#define XLA_CODEGEN_TOOLS_TEST_LIB_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

// Loads a test module from the given filename, ensuring it has a single fusion.
// If the file contains more than one fusion, the function fails. If the file
// contains no fusions, the function generates a fusion from the entry
// computation.
absl::StatusOr<std::unique_ptr<HloModule>> LoadTestModule(
    absl::string_view filename);

}  // namespace xla

#endif  // XLA_CODEGEN_TOOLS_TEST_LIB_H_
