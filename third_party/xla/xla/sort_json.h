/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SORT_JSON_H_
#define XLA_SORT_JSON_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace xla {

// Sorts the given JSON string or returns an error if the JSON could not be
// parsed. Note that this function expects the input JSON to be valid and not
// all forms of invalid JSON are correctly recognized. This function completely
// ignores whitespace and the resulting JSON does not have any whitespace.
// Comments are not supported in the input JSON.
absl::StatusOr<std::string> SortJson(absl::string_view json);

}  // namespace xla

#endif  // XLA_SORT_JSON_H_
