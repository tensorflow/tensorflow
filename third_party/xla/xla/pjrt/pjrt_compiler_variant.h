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

#ifndef XLA_PJRT_PJRT_COMPILER_VARIANT_H_
#define XLA_PJRT_PJRT_COMPILER_VARIANT_H_

#include <cstdint>
#include <functional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/pjrt_common.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

// Represents the PjRt compiler variant to use.
enum class PjRtCompilerVariant : uint64_t {};

// XLA linked in as a library.
inline constexpr absl::string_view kLinkedVariant = "linked";
inline constexpr absl::string_view kUnknownVariant = "unknown";

inline PjRtCompilerVariant LinkedCompilerVariantId() {
  static const PjRtCompilerVariant kLinkedVariantId =
      static_cast<PjRtCompilerVariant>(tsl::Fingerprint64(kLinkedVariant));
  return kLinkedVariantId;
}

// Converts a compiler-registry `variant_name` (where both `""` and `"linked"`
// denote the linked-in compiler) to its `PjRtCompilerVariant` ID.
inline PjRtCompilerVariant CompilerVariantFromName(
    absl::string_view variant_name) {
  if (variant_name.empty() || variant_name == kLinkedVariant) {
    return LinkedCompilerVariantId();
  }
  return static_cast<PjRtCompilerVariant>(tsl::Fingerprint64(variant_name));
}

// This function has a weak default implementation in the
// pjrt_compiler_variant.cc file.
// Users can override it by providing a strong definition elsewhere.
// It is deliberately not marked as weak in this header so that any
// overriding definitions are correctly treated as strong symbols by the linker.
std::string CompilerVariantToString(xla::PjRtCompilerVariant variant);

// Returns the active compiler variant name for a platform (where `""` and
// `"linked"` both denote the linked compiler).
using PjRtCompilerVariantPicker = std::function<absl::StatusOr<std::string>()>;

// A thread-safe registry mapping each platform to a picker that chooses which
// compiler variant to use at runtime. A picker returns the string variant name
// used to key `PjRtCompilerRegistry` (such as `"linked"` for the linked
// compiler, or `"remote"`), and `PickVariant()` maps that name to a
// `PjRtCompilerVariant` ID.
//
// A platform's `PjRtPlatformId` is the `tsl::Fingerprint64` hash of its
// `platform_name` (for example, `TpuId()` is the hash of `"tpu"`). This
// registry stores entries by `PjRtPlatformId`, and the `platform_name`
// overloads hash the name first, so callers can register or look up a picker by
// either name or ID and hit the same entry.
//
// Strong registrations (`is_weak = false`) always win over weak defaults
// (`is_weak = true`), regardless of static-initializer order.
class PjRtCompilerVariantPickerRegistry {
 public:
  PjRtCompilerVariantPickerRegistry() = default;

  // Not copyable or movable.
  PjRtCompilerVariantPickerRegistry(const PjRtCompilerVariantPickerRegistry&) =
      delete;
  PjRtCompilerVariantPickerRegistry& operator=(
      const PjRtCompilerVariantPickerRegistry&) = delete;

  // Returns the process-wide registry.
  static PjRtCompilerVariantPickerRegistry& Global();

  void RegisterVariantPicker(PjRtPlatformId platform_id,
                             PjRtCompilerVariantPicker picker,
                             bool is_weak = false);
  void RegisterVariantPicker(absl::string_view platform_name,
                             PjRtCompilerVariantPicker picker,
                             bool is_weak = false);

  // Picks the active compiler variant name using its registered picker,
  // returning `kLinkedVariant` if no picker is registered or the picker returns
  // `""`.
  absl::StatusOr<std::string> PickVariantName(PjRtPlatformId platform_id) const;
  absl::StatusOr<std::string> PickVariantName(
      absl::string_view platform_name) const;

  // Picks the active `PjRtCompilerVariant` using `PickVariantName()`.
  absl::StatusOr<PjRtCompilerVariant> PickVariant(
      PjRtPlatformId platform_id) const;
  absl::StatusOr<PjRtCompilerVariant> PickVariant(
      absl::string_view platform_name) const;

 private:
  struct Entry {
    PjRtCompilerVariantPicker picker;
    bool is_weak = false;
  };

  mutable absl::Mutex mu_;
  absl::flat_hash_map<PjRtPlatformId, Entry> variant_pickers_
      ABSL_GUARDED_BY(mu_);
};

// Registers a compiler variant picker for `platform_id` / `platform_name` in
// `registry`.
void PjRtRegisterCompilerVariantPicker(
    PjRtPlatformId platform_id, PjRtCompilerVariantPicker picker,
    bool is_weak = false,
    PjRtCompilerVariantPickerRegistry& registry =
        PjRtCompilerVariantPickerRegistry::Global());
void PjRtRegisterCompilerVariantPicker(
    absl::string_view platform_name, PjRtCompilerVariantPicker picker,
    bool is_weak = false,
    PjRtCompilerVariantPickerRegistry& registry =
        PjRtCompilerVariantPickerRegistry::Global());

}  // namespace xla

#endif  // XLA_PJRT_PJRT_COMPILER_VARIANT_H_
