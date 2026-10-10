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
#include <variant>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "google/protobuf/map.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/proto/pjrt_build_stamp.pb.h"
#include "xla/pjrt/proto/pjrt_value_type.pb.h"

namespace xla {

// Well-known version name for the Mosaic compiler dialect version.
inline constexpr absl::string_view kBuildStampVersionMosaic = "mosaic";

inline bool PjRtVersionProtoEquals(const PjRtVersionProto& a,
                                   const PjRtVersionProto& b) {
  return a.name() == b.name() && a.major_version() == b.major_version() &&
         a.minor_version() == b.minor_version();
}

// Immutable value type representing the build provenance, binary hash,
// compile-time components, versioned dialects/interfaces, and extensible
// attributes for a given PJRT plugin or compiler.
//
// Two PJRT plugins or compilers are equivalent if and only if they have the
// same `full_source_hash`. These two plugins/compilers have the same
// source_hash, same components, and same versions, after deduping. They can
// have different build_label, build_timestamp, vcs_revision, and/or attributes.
// Attributes are informational-only metadata and do not affect compilation
// or full_source_hash.
class PjRtBuildStamp final {
 public:
  // Populates build provenance, e.g. timestamp, VCS revision, etc.
  static PjRtBuildStampProto BuildStampProtoFromBuildData();

  // Validates invariants, canonicalizes components and versions, calculates a
  // comprehensive source hash, and creates an immutable PjRtBuildStamp.
  static absl::StatusOr<PjRtBuildStamp> Create(PjRtBuildStampProto proto);

  // Returns a validated proto.
  const PjRtBuildStampProto& ToProto() const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  absl::string_view build_label() const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  absl::Time build_timestamp() const;

  absl::string_view vcs_revision() const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  absl::string_view source_hash() const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  // Returns a deterministic string combining `source_hash` (or a fallback
  // built from `vcs_revision` or `build_timestamp`) with sorted, deduplicated
  // `components` and `versions`. Does not include `attributes`.
  absl::string_view full_source_hash() const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  // Returns sorted and deduplicated compile-time components (e.g. custom call
  // handlers).
  const google::protobuf::RepeatedPtrField<PjRtComponentBuildStampProto>& components()
      const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  // Returns sorted and deduplicated version specifications (e.g. Mosaic
  // dialect version).
  const google::protobuf::RepeatedPtrField<PjRtVersionProto>& versions() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND;

  // Returns environment-specific attributes. Attributes are informational-only
  // metadata and do not affect compilation or `full_source_hash`.
  const google::protobuf::Map<std::string, PjRtValueTypeProto>& attributes() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND;

  // Returns the version for the given version name if present.
  absl::StatusOr<PjRtVersionProto> GetVersion(absl::string_view name) const;

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
    if (const T* typed_val = std::get_if<T>(&val)) {
      return *typed_val;
    }
    return absl::InvalidArgumentError(
        absl::StrCat("Attribute '", key, "' has type ",
                     PjRtValueTypeCaseName(it->second.value_case()),
                     ", not the requested type \"", PjRtTypeName<T>(), "\"."));
  }

  // Zero-copy lookup for a string attribute in `attributes()`.
  absl::StatusOr<absl::string_view> GetStringAttribute(
      absl::string_view key) const ABSL_ATTRIBUTE_LIFETIME_BOUND;

 private:
  explicit PjRtBuildStamp(PjRtBuildStampProto proto);

  PjRtBuildStampProto proto_;
};

// Gets the Mosaic IR compatibility version.
absl::StatusOr<PjRtVersionProto> GetMosaicVersion(const PjRtBuildStamp& stamp);

}  // namespace xla

#endif  // XLA_PJRT_PJRT_BUILD_STAMP_H_
