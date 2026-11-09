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

#include "xla/pjrt/pjrt_build_stamp.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>

#include "google/protobuf/timestamp.pb.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "google/protobuf/map.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/proto/pjrt_build_stamp.pb.h"
#include "xla/pjrt/proto/pjrt_value_type.pb.h"
#include "tsl/platform/fingerprint.h"

namespace xla {
namespace {

// Sentinels returned by `tsl::builddata::SourceRevision()` or Bazel workspace
// status scripts when VCS metadata is unavailable.
constexpr absl::string_view kBracketedUnknownVcsRevision = "<unknown>";
constexpr absl::string_view kBareUnknownVcsRevision = "unknown";

// Upper bound of `google.protobuf.Timestamp.seconds` (9999-12-31T23:59:59Z), as
// documented in `google/protobuf/timestamp.proto`.
constexpr int64_t kMaxTimestampSeconds = 253402300799;

// Delimiter used to separate fields and sections in the serialized hash input.
constexpr absl::string_view kHashDelimiter = "|";

}  // namespace

absl::Status ValidateVcsRevision(absl::string_view vcs_revision) {
  if (vcs_revision.empty()) {
    return absl::InvalidArgumentError("vcs_revision must not be empty.");
  }
  if (vcs_revision == kBracketedUnknownVcsRevision ||
      vcs_revision == kBareUnknownVcsRevision) {
    return absl::InvalidArgumentError(absl::StrCat(
        "vcs_revision cannot be the '", vcs_revision, "' sentinel."));
  }
  return absl::OkStatus();
}

absl::Status ValidateBuildTimestamp(
    const google::protobuf::Timestamp& timestamp) {
  const int64_t seconds = timestamp.seconds();
  const int32_t nanos = timestamp.nanos();
  if (nanos < 0 || nanos >= 1'000'000'000) {
    return absl::InvalidArgumentError(absl::StrCat(
        "build_timestamp.nanos must be in [0, 999999999], got: ", nanos));
  }
  // Stricter than the `google.protobuf.Timestamp` lower bound
  // (0001-01-01T00:00:00Z), since unstamped builds report `seconds == 0`.
  if (seconds <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "build_timestamp.seconds must be positive, got: ", seconds));
  }
  if (seconds > kMaxTimestampSeconds) {
    return absl::InvalidArgumentError(absl::StrCat(
        "build_timestamp.seconds must be at most ", kMaxTimestampSeconds,
        " (9999-12-31T23:59:59Z), got: ", seconds));
  }
  return absl::OkStatus();
}

absl::Status ValidateCompatibilityVersion(
    const PjRtCompatibilityVersionProto& version) {
  if (version.name().empty()) {
    return absl::InvalidArgumentError("Version name must not be empty.");
  }
  if (version.major_version() < 0 || version.minor_version() < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Version numbers must be non-negative. name='",
                     version.name(), "', major=", version.major_version(),
                     ", minor=", version.minor_version()));
  }
  return absl::OkStatus();
}

absl::Status ValidateAttribute(absl::string_view key,
                               const PjRtValueTypeProto& value) {
  if (key.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Attribute key must not be empty. value='", value.DebugString(), "'"));
  }
  if (value.value_case() == PjRtValueTypeProto::VALUE_NOT_SET) {
    return absl::InvalidArgumentError(
        absl::StrCat("Attribute value for key='", key, "' must be set."));
  }
  return absl::OkStatus();
}

namespace {

// Validates that each component has a non-empty name and hash, and sorts and
// deduplicates `components` in-place. Also verifies no duplicate
// (kind, name, owner) with different hashes.
absl::Status ValidateComponents(
    google::protobuf::RepeatedPtrField<PjRtComponentBuildStampProto>& components) {
  // Verify that each component has a non-empty name and hash.
  for (const PjRtComponentBuildStampProto& comp : components) {
    if (comp.name().empty() || comp.hash().empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Component build stamp name and hash must not be empty. kind='",
          comp.kind(), "', name='", comp.name(), "', owner='", comp.owner(),
          "', hash='", comp.hash(), "'"));
    }
  }

  // Sort and deduplicate identical components in-place.
  auto comp_key = [](const PjRtComponentBuildStampProto& c) {
    return std::make_tuple(
        absl::string_view(c.kind()), absl::string_view(c.name()),
        absl::string_view(c.owner()), absl::string_view(c.hash()));
  };
  std::sort(components.pointer_begin(), components.pointer_end(),
            [&comp_key](const PjRtComponentBuildStampProto* a,
                        const PjRtComponentBuildStampProto* b) {
              return comp_key(*a) < comp_key(*b);
            });
  components.erase(
      std::unique(components.begin(), components.end(),
                  [&comp_key](const PjRtComponentBuildStampProto& a,
                              const PjRtComponentBuildStampProto& b) {
                    return comp_key(a) == comp_key(b);
                  }),
      components.end());

  // Verify no duplicate (kind, name, owner) with different hashes.
  for (int i = 1; i < components.size(); ++i) {
    const PjRtComponentBuildStampProto& prev = components[i - 1];
    const PjRtComponentBuildStampProto& curr = components[i];
    if (prev.kind() == curr.kind() && prev.name() == curr.name() &&
        prev.owner() == curr.owner() && prev.hash() != curr.hash()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Duplicate component found with conflicting hashes for kind='",
          curr.kind(), "', name='", curr.name(), "', owner='", curr.owner(),
          "': '", prev.hash(), "' vs '", curr.hash(), "'"));
    }
  }

  return absl::OkStatus();
}

// Validates that each attribute has a non-empty key and a set value.
absl::Status ValidateAttributes(
    const google::protobuf::Map<std::string, PjRtValueTypeProto>& attributes) {
  // Verify that each attribute has a non-empty key and a set value.
  // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
  for (const auto& [key, value] : attributes) {
    ABSL_RETURN_IF_ERROR(ValidateAttribute(key, value));
  }
  return absl::OkStatus();
}

// Validates that each version has a non-empty name and non-negative version
// numbers, and sorts and deduplicates `versions` in-place. Also verifies no
// duplicate name with conflicting version numbers.
absl::Status ValidateVersions(
    google::protobuf::RepeatedPtrField<PjRtCompatibilityVersionProto>& versions) {
  for (const PjRtCompatibilityVersionProto& version : versions) {
    ABSL_RETURN_IF_ERROR(ValidateCompatibilityVersion(version));
  }

  // Sort and deduplicate identical versions in-place.
  auto ver_key = [](const PjRtCompatibilityVersionProto& version) {
    return std::make_tuple(absl::string_view(version.name()),
                           version.major_version(), version.minor_version());
  };
  std::sort(versions.pointer_begin(), versions.pointer_end(),
            [&ver_key](const PjRtCompatibilityVersionProto* a,
                       const PjRtCompatibilityVersionProto* b) {
              return ver_key(*a) < ver_key(*b);
            });
  versions.erase(
      std::unique(versions.begin(), versions.end(),
                  [&ver_key](const PjRtCompatibilityVersionProto& a,
                             const PjRtCompatibilityVersionProto& b) {
                    return ver_key(a) == ver_key(b);
                  }),
      versions.end());

  // Verify no duplicate name with conflicting version numbers.
  for (int i = 1; i < versions.size(); ++i) {
    const PjRtCompatibilityVersionProto& prev = versions[i - 1];
    const PjRtCompatibilityVersionProto& curr = versions[i];
    if (prev.name() == curr.name()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Duplicate version found with conflicting values for '", curr.name(),
          "': ", prev.major_version(), ".", prev.minor_version(), " vs ",
          curr.major_version(), ".", curr.minor_version()));
    }
  }

  return absl::OkStatus();
}

// Builds an injective byte encoding of the `PjRtBuildStampProto` fields that
// contribute to `full_source_hash` and fingerprints it. Distinct sequences of
// `Append*` calls always produce distinct encodings, so two stamps can only
// share a `full_source_hash` through a collision in `tsl::Fingerprint128`
// itself.
//
// Every element is terminated by `kHashDelimiter`:
//   - Tag:     a single uppercase section tag character (`Tag`), e.g. `S|`.
//   - String:  `<byte length>:<bytes>|`. The length prefix fixes the field
//              boundary, so embedded delimiters in `<bytes>` cannot shift it.
//   - Integer: signed decimal digits, e.g. `-42|`. Digits and `-` never
//              contain the delimiter.
// Callers must precede each repeated field with its element count
// (`AppendInt(size)`) so list boundaries are unambiguous.
class FingerprintBuilder {
 public:
  enum class Tag : char {
    kSourceHash = 'S',
    kVcsRevision = 'R',
    kBuildTimestamp = 'T',
    kComponents = 'C',
    kVersions = 'V',
  };

  explicit FingerprintBuilder(size_t initial_capacity) {
    buffer_.reserve(initial_capacity);
  }

  void AppendTag(Tag tag) {
    buffer_.push_back(static_cast<char>(tag));
    buffer_.append(kHashDelimiter);
  }

  void AppendString(absl::string_view value) {
    absl::StrAppend(&buffer_, value.size(), ":", value, kHashDelimiter);
  }

  void AppendInt(int64_t value) {
    absl::StrAppend(&buffer_, value, kHashDelimiter);
  }

  // Returns the 128-bit fingerprint of the encoded input as 32 lowercase hex
  // characters.
  std::string FingerprintHex() const {
    const tsl::Fprint128 fprint = tsl::Fingerprint128(buffer_);
    return absl::StrCat(absl::Hex(fprint.high64, absl::kZeroPad16),
                        absl::Hex(fprint.low64, absl::kZeroPad16));
  }

 private:
  std::string buffer_;
};

// Computes `full_source_hash` of a validated and canonicalized `proto`.
// Preconditions (enforced by `FromProto()`):
//   - At least one of `source_hash`, `vcs_revision`, or `build_timestamp` is
//     present and validated.
//   - `components` and `compatibility_versions` are sorted and deduplicated.
std::string ComputeFullSourceHash(const PjRtBuildStampProto& proto) {
  // Note: `proto.source_hash` remains empty if a fallback identity is used.
  //
  // Estimate encoded buffer size from repeated-field counts (~128B base + ~128B
  // per component + ~32B per compatibility version) to avoid repeated heap
  // reallocations without an extra pass to sum string lengths. If the actual
  // encoding exceeds this estimate, `std::string` grows automatically.
  const size_t estimated_bytes = 128 + proto.components_size() * 128 +
                                 proto.compatibility_versions_size() * 32;
  FingerprintBuilder fp_builder(estimated_bytes);
  if (!proto.source_hash().empty()) {
    fp_builder.AppendTag(FingerprintBuilder::Tag::kSourceHash);
    fp_builder.AppendString(proto.source_hash());
  } else if (!proto.vcs_revision().empty()) {
    VLOG(1) << "PjRtBuildStamp source_hash fallback to vcs_revision: "
            << proto.vcs_revision();
    fp_builder.AppendTag(FingerprintBuilder::Tag::kVcsRevision);
    fp_builder.AppendString(proto.vcs_revision());
  } else {
    VLOG(1) << "PjRtBuildStamp source_hash fallback to build_timestamp: "
            << proto.build_timestamp().seconds() << "."
            << absl::Dec(proto.build_timestamp().nanos(), absl::kZeroPad9);
    fp_builder.AppendTag(FingerprintBuilder::Tag::kBuildTimestamp);
    fp_builder.AppendInt(proto.build_timestamp().seconds());
    fp_builder.AppendInt(proto.build_timestamp().nanos());
  }

  fp_builder.AppendTag(FingerprintBuilder::Tag::kComponents);
  fp_builder.AppendInt(proto.components_size());
  for (const PjRtComponentBuildStampProto& comp : proto.components()) {
    VLOG(2) << "PjRtBuildStamp component [" << comp.kind() << ":" << comp.name()
            << " (" << comp.owner() << ")]: hash=" << comp.hash();
    fp_builder.AppendString(comp.kind());
    fp_builder.AppendString(comp.name());
    fp_builder.AppendString(comp.owner());
    fp_builder.AppendString(comp.hash());
  }

  fp_builder.AppendTag(FingerprintBuilder::Tag::kVersions);
  fp_builder.AppendInt(proto.compatibility_versions_size());
  for (const PjRtCompatibilityVersionProto& v :
       proto.compatibility_versions()) {
    VLOG(2) << "PjRtBuildStamp version [" << v.name()
            << "]: " << v.major_version() << "." << v.minor_version();
    fp_builder.AppendString(v.name());
    fp_builder.AppendInt(v.major_version());
    fp_builder.AppendInt(v.minor_version());
  }

  std::string full_source_hash = fp_builder.FingerprintHex();
  VLOG(1) << "PjRtBuildStamp final full_source_hash: " << full_source_hash;
  return full_source_hash;
}

}  // namespace

PjRtBuildStamp::PjRtBuildStamp(PjRtBuildStampProto proto,
                               std::string full_source_hash)
    : proto_(std::move(proto)),
      full_source_hash_(std::move(full_source_hash)) {}

absl::StatusOr<PjRtBuildStamp> PjRtBuildStamp::FromProto(
    PjRtBuildStampProto proto) {
  // TODO(b/567130044): Validate that `source_hash` is not set to an "unknown"
  // sentinel by an OSS or custom build.
  if (!proto.vcs_revision().empty()) {
    ABSL_RETURN_IF_ERROR(ValidateVcsRevision(proto.vcs_revision()));
  }
  if (proto.has_build_timestamp()) {
    ABSL_RETURN_IF_ERROR(ValidateBuildTimestamp(proto.build_timestamp()));
  }
  if (proto.source_hash().empty() && proto.vcs_revision().empty() &&
      !proto.has_build_timestamp()) {
    return absl::InvalidArgumentError(
        "PjRtBuildStamp requires at least one build identity field "
        "(source_hash, vcs_revision, or build_timestamp)");
  }

  ABSL_RETURN_IF_ERROR(ValidateAttributes(proto.attributes()));
  ABSL_RETURN_IF_ERROR(ValidateComponents(*proto.mutable_components()));
  ABSL_RETURN_IF_ERROR(ValidateVersions(*proto.mutable_compatibility_versions()));

  // The full source hash is computed after validation since we rely on the
  // repeated fields being sorted and deduped.
  std::string full_source_hash = ComputeFullSourceHash(proto);
  return PjRtBuildStamp(std::move(proto), std::move(full_source_hash));
}

const PjRtBuildStampProto& PjRtBuildStamp::ToProto() const { return proto_; }

absl::string_view PjRtBuildStamp::build_label() const {
  return proto_.build_label();
}

absl::Time PjRtBuildStamp::build_timestamp() const {
  if (!proto_.has_build_timestamp()) {
    return absl::InfinitePast();
  }
  return absl::FromUnixSeconds(proto_.build_timestamp().seconds()) +
         absl::Nanoseconds(proto_.build_timestamp().nanos());
}

absl::string_view PjRtBuildStamp::vcs_revision() const {
  return proto_.vcs_revision();
}

absl::string_view PjRtBuildStamp::source_hash() const {
  return proto_.source_hash();
}

absl::string_view PjRtBuildStamp::full_source_hash() const {
  return full_source_hash_;
}

absl::StatusOr<PjRtCompatibilityVersionProto> PjRtBuildStamp::GetVersion(
    absl::string_view name) const {
  for (const PjRtCompatibilityVersionProto& ver :
       proto_.compatibility_versions()) {
    if (ver.name() == name) {
      return ver;
    }
  }
  return absl::NotFoundError(absl::StrCat("Version '", name, "' not found."));
}

absl::StatusOr<absl::string_view> PjRtBuildStamp::GetStringAttribute(
    absl::string_view key) const {
  auto it = proto_.attributes().find(key);
  if (it == proto_.attributes().end()) {
    return absl::NotFoundError(
        absl::StrCat("Attribute '", key, "' not found."));
  }
  if (it->second.has_string_value()) {
    return it->second.string_value();
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Attribute '", key, "' has type ",
                   PjRtValueTypeCaseName(it->second.value_case()),
                   ", not the requested type \"string\"."));
}

}  // namespace xla
