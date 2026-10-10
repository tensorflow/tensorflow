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
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "google/protobuf/timestamp.pb.h"
#include "absl/algorithm/container.h"
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

// Validates that each component has a non-empty name and hash, and returns a
// sorted, deduplicated list of components. Also verifies no duplicate
// (kind, name, owner) with different hashes.
absl::StatusOr<std::vector<PjRtComponentBuildStampProto>> ValidateComponents(
    const google::protobuf::RepeatedPtrField<PjRtComponentBuildStampProto>& proto) {
  std::vector<PjRtComponentBuildStampProto> components;
  components.reserve(proto.size());

  // Verify that each component has a non-empty name and hash.
  for (const auto& comp : proto) {
    if (comp.name().empty() || comp.hash().empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Component build stamp name and hash must not be empty. kind='",
          comp.kind(), "', name='", comp.name(), "', owner='", comp.owner(),
          "', hash='", comp.hash(), "'"));
    }
    components.push_back(comp);
  }

  // Sort and deduplicate identical components.
  auto comp_key = [](const PjRtComponentBuildStampProto& c) {
    return std::make_tuple(c.kind(), c.name(), c.owner(), c.hash());
  };
  absl::c_stable_sort(components, [&comp_key](const auto& a, const auto& b) {
    return comp_key(a) < comp_key(b);
  });
  components.erase(std::unique(components.begin(), components.end(),
                               [&comp_key](const auto& a, const auto& b) {
                                 return comp_key(a) == comp_key(b);
                               }),
                   components.end());

  // Verify no duplicate (kind, name, owner) with different hashes.
  for (size_t i = 1; i < components.size(); ++i) {
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

  return components;
}

// Validates that each attribute has a non-empty key and a set value.
absl::Status ValidateAttributes(
    const google::protobuf::Map<std::string, PjRtValueTypeProto>& attributes) {
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

  ABSL_ASSIGN_OR_RETURN(std::vector<PjRtComponentBuildStampProto> components,
                   ValidateComponents(proto.components()), _);

  proto.clear_components();
  proto.mutable_components()->Reserve(components.size());

  std::string source_hash = std::string(proto.source_hash());
  if (source_hash.empty()) {
    if (!proto.vcs_revision().empty()) {
      source_hash = absl::StrCat("vcs:", proto.vcs_revision());
      VLOG(1) << "PjRtBuildStamp source_hash fallback to vcs_revision: "
              << source_hash;
    } else if (proto.has_build_timestamp()) {
      source_hash = absl::StrCat(
          "ts:", proto.build_timestamp().seconds(), ".",
          absl::Dec(proto.build_timestamp().nanos(), absl::kZeroPad9));
      VLOG(1) << "PjRtBuildStamp source_hash fallback to build_timestamp: "
              << source_hash;
    }
  }

  std::string full_source_hash = source_hash;
  std::string components_hash;
  for (auto& comp : components) {
    VLOG(2) << "PjRtBuildStamp component [" << comp.kind() << ":" << comp.name()
            << " (" << comp.owner() << ")]: hash=" << comp.hash();
    absl::StrAppend(&components_hash, comp.hash());
    *proto.add_components() = std::move(comp);
  }
  if (!components_hash.empty()) {
    const tsl::Fprint128 fprint =
        tsl::Fingerprint128(absl::StrCat(source_hash, components_hash));
    full_source_hash = absl::StrCat(absl::Hex(fprint.high64, absl::kZeroPad16),
                                    absl::Hex(fprint.low64, absl::kZeroPad16));
  }

  proto.set_full_source_hash(full_source_hash);
  VLOG(1) << "PjRtBuildStamp final full_source_hash: " << full_source_hash;

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

const google::protobuf::Map<std::string, PjRtValueTypeProto>& PjRtBuildStamp::attributes()
    const {
  return proto_.attributes();
}

std::optional<absl::string_view> PjRtBuildStamp::GetStringAttribute(
    absl::string_view key) const ABSL_ATTRIBUTE_LIFETIME_BOUND {
  auto it = proto_.attributes().find(std::string(key));
  if (it == proto_.attributes().end()) {
    return std::nullopt;
  }
  if (it->second.has_string_value()) {
    return it->second.string_value();
  }
  return std::nullopt;
}

void SetMosaicVersion(PjRtBuildStampProto& proto, int64_t mosaic_version) {
  (*proto.mutable_attributes())[std::string(kBuildStampAttrMosaicVersion)] =
      PjRtValueTypeToProto(mosaic_version);
}

std::optional<int64_t> GetMosaicVersion(const PjRtBuildStamp& stamp) {
  return stamp.GetAttribute<int64_t>(kBuildStampAttrMosaicVersion);
}

}  // namespace xla
