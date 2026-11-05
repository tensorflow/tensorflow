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

#ifndef XLA_PJRT_PJRT_BUILD_STAMP_H_
#define XLA_PJRT_PJRT_BUILD_STAMP_H_

#include <string>
#include <utility>
#include <variant>

#include "google/protobuf/timestamp.pb.h"
#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/proto/pjrt_build_stamp.pb.h"
#include "xla/pjrt/proto/pjrt_value_type.pb.h"

namespace xla {

// Validates that `vcs_revision` is non-empty and not the `"<unknown>"` sentinel
// returned by `tsl::builddata::SourceRevision()` on unstamped builds.
absl::Status ValidateVcsRevision(absl::string_view vcs_revision);

// Validates that `timestamp` has `nanos` in `[0, 999'999'999]` and non-negative
// `seconds` (on or after the Unix epoch; unstamped builds have `seconds == 0`).
absl::Status ValidateBuildTimestamp(
    const google::protobuf::Timestamp& timestamp);

// Validates that `version` has a non-empty name and non-negative major and
// minor version numbers.
absl::Status ValidateCompatibilityVersion(
    const PjRtCompatibilityVersionProto& version);

// Validates that `key` is non-empty and `value` has a value set.
absl::Status ValidateAttribute(absl::string_view key,
                               const PjRtValueTypeProto& value);

// Immutable value type representing the build provenance, binary hash,
// compile-time components (`PjRtComponentBuildStampProto`), compatibility
// versions (`PjRtCompatibilityVersionProto`), and extensible attributes for a
// given PJRT plugin or compiler.
//
// `components` hold opaque source hashes of non-direct compile-time
// dependencies (e.g. FFI/custom-call handlers) folded into `full_source_hash`
// for cache invalidation, whereas `compatibility_versions` hold ordered
// `(major, minor)` numbers queried via `GetVersion()` for IR/ABI compatibility
// and target negotiation.
//
// Two PJRT plugins or compilers are equivalent if and only if they have the
// same `full_source_hash`. These two plugins/compilers have the same
// source_hash, same components, and same compatibility_versions, after
// deduping. They can have different build_label, build_timestamp, vcs_revision,
// and/or attributes. Attributes are informational-only metadata and do not
// affect compilation or full_source_hash.
//
// The authoritative build stamp of a compiler is the one returned by
// `PjRtGetBuildStamp()`, which is produced by the factory the compiler
// registered. Consumers should read stamps from there rather than construct
// their own; a stamp built from a hand-constructed or modified proto does not
// describe any compiler. Compiler build stamp factories populate the proto
// using the helpers in `pjrt_build_stamp_writer.h`.
class PjRtBuildStamp final {
 public:
  PjRtBuildStamp(const PjRtBuildStamp&) = default;
  PjRtBuildStamp& operator=(const PjRtBuildStamp&) = default;
  PjRtBuildStamp(PjRtBuildStamp&&) = default;
  PjRtBuildStamp& operator=(PjRtBuildStamp&&) = default;

  // Validates invariants, canonicalizes components and compatibility_versions,
  // calculates a comprehensive source hash, and creates an immutable
  // PjRtBuildStamp from a proto.
  //
  // Intended for compiler build stamp factories and for deserializing a stamp
  // produced by one (e.g. across the PJRT C API), not for constructing or
  // modifying stamps elsewhere.
  static absl::StatusOr<PjRtBuildStamp> FromProto(PjRtBuildStampProto proto);

  // Returns a validated proto.
  const PjRtBuildStampProto& ToProto() const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  absl::string_view build_label() const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  absl::Time build_timestamp() const;

  absl::string_view vcs_revision() const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  absl::string_view source_hash() const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  // Returns a deterministic string combining `source_hash` (or a fallback
  // built from `vcs_revision` or `build_timestamp`) with sorted, deduplicated
  // `components` and `compatibility_versions`. Does not include `attributes`.
  absl::string_view full_source_hash() const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  // Returns the compatibility version for the given version name if present.
  // For well-known versions, prefer the typed getters in
  // `pjrt_compatibility_versions.h` (e.g. `GetMosaicVersion()`).
  absl::StatusOr<PjRtCompatibilityVersionProto> GetVersion(
      absl::string_view name) const;

  // Returns the value of the attribute with the given key if it exists and is
  // of type `T`.
  template <typename T>
  absl::StatusOr<T> GetAttribute(absl::string_view key) const {
    auto it = proto_.attributes().find(key);
    if (it == proto_.attributes().end()) {
      return absl::NotFoundError(
          absl::StrCat("Attribute '", key, "' not found."));
    }
    PjRtValueType val = PjRtValueTypeFromProto(it->second);
    if (T* typed_val = std::get_if<T>(&val)) {
      return std::move(*typed_val);
    }
    return absl::InvalidArgumentError(
        absl::StrCat("Attribute '", key, "' has type ",
                     PjRtValueTypeCaseName(it->second.value_case()),
                     ", not the requested type \"", PjRtTypeName<T>(), "\"."));
  }

  // Zero-copy lookup for a string attribute.
  absl::StatusOr<absl::string_view> GetStringAttribute(
      absl::string_view key) const ABSL_ATTRIBUTE_LIFETIME_BOUND;

 private:
  explicit PjRtBuildStamp(PjRtBuildStampProto proto,
                          std::string full_source_hash);

  // Not `const`: the class is already immutable (no mutators). `const` members
  // would make PjRtBuildStamp non-assignable (so it couldn't be assigned into
  // an existing `absl::StatusOr<PjRtBuildStamp>`) and would turn moves into
  // copies.
  PjRtBuildStampProto proto_;
  std::string full_source_hash_;
};

}  // namespace xla

#endif  // XLA_PJRT_PJRT_BUILD_STAMP_H_
