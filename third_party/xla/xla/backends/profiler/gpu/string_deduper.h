/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_STRING_DEDUPER_H_
#define XLA_BACKENDS_PROFILER_GPU_STRING_DEDUPER_H_

#include <cstddef>
#include <string>

#include "absl/container/node_hash_set.h"
#include "absl/strings/string_view.h"

namespace xla {
namespace profiler {

// Retains unique copies of string_views so that returned string_views remain
// valid for the lifetime of this object.
class StringDeduper {
 public:
  void Clear() { strings_.clear(); }

  // max_unique_count is not put into data member to make it consistent with
  // existing logic.
  absl::string_view Dedup(absl::string_view str, size_t max_unique_count = 0);

  size_t Size() const { return strings_.size(); }

 private:
  absl::node_hash_set<std::string> strings_;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_STRING_DEDUPER_H_
