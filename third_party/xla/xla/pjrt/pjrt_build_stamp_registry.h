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

#ifndef XLA_PJRT_PJRT_BUILD_STAMP_REGISTRY_H_
#define XLA_PJRT_PJRT_BUILD_STAMP_REGISTRY_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/call_once.h"
#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/pjrt_build_stamp.h"
#include "xla/pjrt/pjrt_compiler_variant.h"

namespace xla {

struct PjRtBuildStampKey {
  // Platform name, e.g. "cuda", "tpu".
  std::string platform_name;
  // PjRt compiler variant name, e.g. "linked".
  std::string variant_name;

  PjRtBuildStampKey(absl::string_view platform, absl::string_view variant)
      : platform_name(platform), variant_name(variant) {}

  template <typename H>
  friend H AbslHashValue(H h, const PjRtBuildStampKey& k) {
    return H::combine(std::move(h), k.platform_name, k.variant_name);
  }

  bool operator==(const PjRtBuildStampKey& other) const {
    return platform_name == other.platform_name &&
           variant_name == other.variant_name;
  }
};

// A registry to manage and look up PjRtBuildStamps. The factory is run on the
// first lookup for a given key, and the result, even if an error, is memoized.
class PjRtBuildStampRegistry {
 public:
  static PjRtBuildStampRegistry& Global();

  PjRtBuildStampRegistry() = default;
  ~PjRtBuildStampRegistry() = default;

  // Not copyable or movable.
  PjRtBuildStampRegistry(const PjRtBuildStampRegistry&) = delete;
  PjRtBuildStampRegistry& operator=(const PjRtBuildStampRegistry&) = delete;

  using Factory = std::function<absl::StatusOr<PjRtBuildStamp>()>;

  absl::Status RegisterBuildStampFactory(absl::string_view platform_name,
                                         Factory factory);

  absl::Status RegisterBuildStampFactory(absl::string_view platform_name,
                                         absl::string_view variant_name,
                                         Factory factory);

  absl::StatusOr<const PjRtBuildStamp* absl_nonnull> GetBuildStamp(
      absl::string_view platform_name,
      absl::string_view variant_name = kLinkedVariant)
      ABSL_ATTRIBUTE_LIFETIME_BOUND;

 private:
  struct Entry {
    explicit Entry(Factory f) : factory(std::move(f)) {}
    Factory factory;
    absl::once_flag once;
    absl::StatusOr<PjRtBuildStamp> memoized =
        absl::UnknownError("Uninitialized build stamp");
  };

  absl::Mutex mu_;
  absl::node_hash_map<PjRtBuildStampKey, std::unique_ptr<Entry>> entries_
      ABSL_GUARDED_BY(mu_);
};

// Convenience global registration and lookup functions.

void PjRtRegisterBuildStampFactory(absl::string_view platform_name,
                                   PjRtBuildStampRegistry::Factory factory);

void PjRtRegisterBuildStampFactory(absl::string_view platform_name,
                                   absl::string_view variant_name,
                                   PjRtBuildStampRegistry::Factory factory);

absl::StatusOr<const PjRtBuildStamp* absl_nonnull> PjRtGetBuildStamp(
    absl::string_view platform_name,
    absl::string_view variant_name = kLinkedVariant);

absl::StatusOr<const PjRtBuildStamp* absl_nonnull> PjRtGetBuildStamp(
    absl::string_view platform_name, PjRtCompilerVariant variant);

}  // namespace xla

#endif  // XLA_PJRT_PJRT_BUILD_STAMP_REGISTRY_H_
