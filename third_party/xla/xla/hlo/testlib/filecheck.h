/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_HLO_TESTLIB_FILECHECK_H_
#define XLA_HLO_TESTLIB_FILECHECK_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/types.h"

namespace xla {

// Runs FileCheck with the given pattern over given input string. Provided that
// FileCheck can execute, returns true if and only if FileCheck succeeded in
// matching the input.
absl::StatusOr<bool> RunFileCheck(const std::string& input,
                                  absl::string_view pattern);

// Runs FileCheck with the given pattern file over given input string. Provided
// that FileCheck can execute, returns true if and only if FileCheck succeeded
// in matching the input.
absl::StatusOr<bool> RunFileCheckWithPatternFile(
    const std::string& input, const std::string& pattern_file);

}  // namespace xla

#endif  // XLA_HLO_TESTLIB_FILECHECK_H_
