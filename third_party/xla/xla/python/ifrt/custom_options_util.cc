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

#include "xla/python/ifrt/custom_options_util.h"

#include <optional>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/python/ifrt/attribute_map.h"

namespace xla {
namespace ifrt {

namespace {

struct Registry {
  absl::Mutex mu;
  absl::flat_hash_set<std::string> keys ABSL_GUARDED_BY(mu);
};

Registry* registry() {
  static auto* const r = new Registry();
  return r;
}

}  // namespace

void RegisterIfrtInternalCustomOption(absl::string_view key) {
  Registry* const r = registry();
  absl::MutexLock lock(r->mu);
  CHECK(r->keys.insert(std::string(key)).second)
      << "xla::ifrt internal custom option key cannot be registered more than "
         "once: "
      << key;
}

bool IsIfrtInternalCustomOption(absl::string_view key) {
  Registry* const r = registry();
  absl::ReaderMutexLock lock(r->mu);
  return r->keys.contains(key);
}

std::optional<AttributeMap> FilterOutIfrtInternalCustomOptions(
    const AttributeMap& custom_options) {
  if (custom_options.IsEmpty()) {
    return std::nullopt;
  }
  AttributeMap::Map filtered;
  custom_options.ForEach(
      [&](const std::string& key, const AttributeMap::Value& value) {
        // Keep `Registry::mu` strictly a leaf lock (`AttributeMap::mu_` ->
        // `Registry::mu`) so callers invoking `IsIfrtInternalCustomOption`
        // inside `AttributeMap::ForEach` never invert lock order.
        if (!IsIfrtInternalCustomOption(key)) {
          filtered.insert({key, value});
        }
      });
  if (filtered.empty()) {
    return std::nullopt;
  }
  return AttributeMap(std::move(filtered));
}

}  // namespace ifrt
}  // namespace xla
