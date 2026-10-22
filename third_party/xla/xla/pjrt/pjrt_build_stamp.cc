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
#include <string>
#include <tuple>
#include <utility>

#include "google/protobuf/timestamp.pb.h"
#include "absl/base/attributes.h"
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
#include "xla/tsl/builddata/builddata.h"
#include "tsl/platform/fingerprint.h"

namespace xla {
namespace {

// Validates that each component has a non-empty name and hash, and sorts and
// deduplicates `components` in-place. Also verifies no duplicate
// (kind, name, owner) with different hashes.
absl::Status ValidateComponents(
    google::protobuf::RepeatedPtrField<PjRtComponentBuildStampProto>& components) {
  // Verify that each component has a non-empty name and hash.
  for (const auto& comp : components) {
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
  std::stable_sort(components.pointer_begin(), components.pointer_end(),
                   [&comp_key](const PjRtComponentBuildStampProto* a,
                               const PjRtComponentBuildStampProto* b) {
                     return comp_key(*a) < comp_key(*b);
                   });
  components.erase(std::unique(components.begin(), components.end(),
                               [&comp_key](const auto& a, const auto& b) {
                                 return comp_key(a) == comp_key(b);
                               }),
                   components.end());

  // Verify no duplicate (kind, name, owner) with different hashes.
  for (int i = 1; i < components.size(); ++i) {
    const auto& prev = components[i - 1];
    const auto& curr = components[i];
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
    if (key.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Attribute key must not be empty. value='",
                       value.DebugString(), "'"));
    }
    if (value.value_case() == PjRtValueTypeProto::VALUE_NOT_SET) {
      return absl::InvalidArgumentError(
          absl::StrCat("Attribute value for key='", key, "' must be set."));
    }
  }
  return absl::OkStatus();
}

// Validates that each version has a non-empty name and non-negative version
// numbers, and sorts and deduplicates `versions` in-place. Also verifies no
// duplicate name with conflicting version numbers.
absl::Status ValidateVersions(
    google::protobuf::RepeatedPtrField<PjRtVersionProto>& versions) {
  for (const auto& v : versions) {
    if (v.name().empty()) {
      return absl::InvalidArgumentError("Version name must not be empty.");
    }
    if (v.major_version() < 0 || v.minor_version() < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Version numbers must be non-negative. name='", v.name(),
          "', major=", v.major_version(), ", minor=", v.minor_version()));
    }
  }

  // Sort and deduplicate identical versions in-place.
  auto ver_key = [](const PjRtVersionProto& v) {
    return std::make_tuple(absl::string_view(v.name()), v.major_version(),
                           v.minor_version());
  };
  std::stable_sort(
      versions.pointer_begin(), versions.pointer_end(),
      [&ver_key](const PjRtVersionProto* a, const PjRtVersionProto* b) {
        return ver_key(*a) < ver_key(*b);
      });
  versions.erase(std::unique(versions.begin(), versions.end(),
                             [&ver_key](const auto& a, const auto& b) {
                               return ver_key(a) == ver_key(b);
                             }),
                 versions.end());

  // Verify no duplicate name with conflicting version numbers.
  for (size_t i = 1; i < versions.size(); ++i) {
    const auto& prev = versions[i - 1];
    const auto& curr = versions[i];
    if (prev.name() == curr.name()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Duplicate version found with conflicting values for '", curr.name(),
          "': ", prev.major_version(), ".", prev.minor_version(), " vs ",
          curr.major_version(), ".", curr.minor_version()));
    }
  }

  return absl::OkStatus();
}

}  // namespace

PjRtBuildStampProto PjRtBuildStamp::BuildStampProtoFromBuildData() {
  PjRtBuildStampProto proto;
  proto.mutable_build_timestamp()->set_seconds(
      tsl::builddata::TimestampAsInt());
  proto.set_vcs_revision(tsl::builddata::SourceRevision());
  proto.set_build_label(tsl::builddata::BuildLabel());
  return proto;
}

PjRtBuildStamp::PjRtBuildStamp(PjRtBuildStampProto proto)
    : proto_(std::move(proto)) {}

absl::StatusOr<PjRtBuildStamp> PjRtBuildStamp::Create(
    PjRtBuildStampProto proto) {
  if (proto.source_hash().empty() && proto.vcs_revision().empty() &&
      !proto.has_build_timestamp()) {
    return absl::InvalidArgumentError(
        "PjRtBuildStamp requires at least one build identity field "
        "(source_hash, vcs_revision, or build_timestamp)");
  }

  ABSL_RETURN_IF_ERROR(ValidateAttributes(proto.attributes()));
  ABSL_RETURN_IF_ERROR(ValidateComponents(*proto.mutable_components()));
  ABSL_RETURN_IF_ERROR(ValidateVersions(*proto.mutable_versions()));

  // Note: source_hash remains empty if fallback is used.
  std::string fallback_source_hash;
  absl::string_view source_hash = proto.source_hash();
  if (source_hash.empty()) {
    if (!proto.vcs_revision().empty()) {
      fallback_source_hash = absl::StrCat("vcs:", proto.vcs_revision());
      source_hash = fallback_source_hash;
      VLOG(1) << "PjRtBuildStamp source_hash fallback to vcs_revision: "
              << source_hash;
    } else if (proto.has_build_timestamp()) {
      fallback_source_hash = absl::StrCat(
          "ts:", proto.build_timestamp().seconds(), ".",
          absl::Dec(proto.build_timestamp().nanos(), absl::kZeroPad9));
      source_hash = fallback_source_hash;
      VLOG(1) << "PjRtBuildStamp source_hash fallback to build_timestamp: "
              << source_hash;
    }
  }

  if (proto.components().empty() && proto.versions().empty()) {
    proto.set_full_source_hash(source_hash);
  } else {
    std::string hash_input(source_hash);
    for (const auto& comp : proto.components()) {
      VLOG(2) << "PjRtBuildStamp component [" << comp.kind() << ":"
              << comp.name() << " (" << comp.owner()
              << ")]: hash=" << comp.hash();
      absl::StrAppend(&hash_input, comp.hash());
    }
    for (const auto& v : proto.versions()) {
      VLOG(2) << "PjRtBuildStamp version [" << v.name()
              << "]: " << v.major_version() << "." << v.minor_version();
      absl::StrAppend(&hash_input, "|ver:", v.name(), ":", v.major_version(),
                      ".", v.minor_version());
    }
    const tsl::Fprint128 fprint = tsl::Fingerprint128(hash_input);
    proto.set_full_source_hash(
        absl::StrCat(absl::Hex(fprint.high64, absl::kZeroPad16),
                     absl::Hex(fprint.low64, absl::kZeroPad16)));
  }
  VLOG(1) << "PjRtBuildStamp final full_source_hash: "
          << proto.full_source_hash();

  return PjRtBuildStamp(std::move(proto));
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
  return proto_.full_source_hash();
}

const google::protobuf::RepeatedPtrField<PjRtComponentBuildStampProto>&
PjRtBuildStamp::components() const {
  return proto_.components();
}

const google::protobuf::RepeatedPtrField<PjRtVersionProto>& PjRtBuildStamp::versions()
    const {
  return proto_.versions();
}

const google::protobuf::Map<std::string, PjRtValueTypeProto>& PjRtBuildStamp::attributes()
    const {
  return proto_.attributes();
}

absl::StatusOr<PjRtVersionProto> PjRtBuildStamp::GetVersion(
    absl::string_view name) const {
  for (const auto& ver : proto_.versions()) {
    if (ver.name() == name) {
      return ver;
    }
  }
  return absl::NotFoundError(absl::StrCat("Version '", name, "' not found."));
}

absl::StatusOr<absl::string_view> PjRtBuildStamp::GetStringAttribute(
    absl::string_view key) const ABSL_ATTRIBUTE_LIFETIME_BOUND {
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

absl::StatusOr<PjRtVersionProto> GetMosaicVersion(const PjRtBuildStamp& stamp) {
  return stamp.GetVersion(kBuildStampVersionMosaic);
}

}  // namespace xla
