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

#include <cstdint>
#include <optional>
#include <string>
#include <variant>

#include "absl/base/attributes.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "google/protobuf/map.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/proto/pjrt_build_stamp.pb.h"
#include "xla/pjrt/proto/pjrt_value_type.pb.h"

namespace xla {

// Well-known attribute key for the Mosaic compiler dialect version.
inline constexpr absl::string_view kBuildStampAttrMosaicVersion =
    "mosaic_version";

// Immutable value type representing the build provenance, binary hash,
// compile-time components, and extensible attributes for a given PJRT plugin or
// compiler.
class PjRtBuildStamp final {
 public:
  // Populates build provenance, e.g. timestamp, VCS revision, etc.
  static PjRtBuildStampProto BuildStampProtoFromBuildData();

  // Validates invariants, canonicalizes components, calculates a comprehensive
  // source hash, and creates an immutable PjRtBuildStamp.
  static absl::StatusOr<PjRtBuildStamp> Create(PjRtBuildStampProto proto);

  // Returns a validated proto.
  const PjRtBuildStampProto& ToProto() const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  absl::string_view build_label() const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  absl::Time build_timestamp() const;

  absl::string_view vcs_revision() const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  absl::string_view source_hash() const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  // Returns a deterministic string combining `source_hash` (or a fallback
  // built from `vcs_revision` or `build_timestamp`) with sorted, deduplicated
  // `components`. Does not include `attributes`.
  absl::string_view full_source_hash() const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  const google::protobuf::RepeatedPtrField<PjRtComponentBuildStampProto>& components()
      const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  const google::protobuf::Map<std::string, PjRtValueTypeProto>& attributes() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND;

  template <typename T>
  std::optional<T> GetAttribute(absl::string_view key) const {
    auto it = proto_.attributes().find(std::string(key));
    if (it == proto_.attributes().end()) {
      return std::nullopt;
    }
    PjRtValueType val = PjRtValueTypeFromProto(it->second);
    if (const T* typed_val = std::get_if<T>(&val)) {
      return *typed_val;
    }
    return std::nullopt;
  }

  // Zero-copy lookup for a string attribute in `attributes()`.
  // Returns `std::nullopt` if `key` is not present or does not hold a string.
  std::optional<absl::string_view> GetStringAttribute(
      absl::string_view key) const ABSL_ATTRIBUTE_LIFETIME_BOUND;

 private:
  explicit PjRtBuildStamp(PjRtBuildStampProto proto);

  PjRtBuildStampProto proto_;
};

// Sets the Mosaic IR compatibility version in `proto.mutable_attributes()`.
void SetMosaicVersion(PjRtBuildStampProto& proto, int64_t mosaic_version);

// Gets the Mosaic IR compatibility version from `stamp.attributes()`.
// Returns std::nullopt if unset or if the attribute is not an int64_t.
std::optional<int64_t> GetMosaicVersion(const PjRtBuildStamp& stamp);

}  // namespace xla

#endif  // XLA_PJRT_PJRT_BUILD_STAMP_H_
