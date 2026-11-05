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

#include "xla/pjrt/pjrt_build_stamp_writer.h"

#include <cstdint>
#include <utility>

#include "google/protobuf/timestamp.pb.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_build_stamp.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compatibility_versions.h"
#include "xla/pjrt/proto/pjrt_build_stamp.pb.h"
#include "xla/pjrt/proto/pjrt_value_type.pb.h"
#include "xla/tsl/builddata/builddata.h"

namespace xla {
namespace {

// Sets the compatibility version `name` to `(major_version, minor_version)`.
// Each version may be set at most once.
absl::Status SetCompatibilityVersion(PjRtBuildStampProto& proto,
                                     absl::string_view name,
                                     int64_t major_version,
                                     int64_t minor_version) {
  PjRtCompatibilityVersionProto version;
  version.set_name(name);
  version.set_major_version(major_version);
  version.set_minor_version(minor_version);
  ABSL_RETURN_IF_ERROR(ValidateCompatibilityVersion(version));
  for (const PjRtCompatibilityVersionProto& existing :
       proto.compatibility_versions()) {
    if (existing.name() == name) {
      return absl::AlreadyExistsError(absl::StrCat(
          "Compatibility version '", name, "' is already set to ",
          existing.major_version(), ".", existing.minor_version(), "."));
    }
  }
  *proto.add_compatibility_versions() = std::move(version);
  return absl::OkStatus();
}

}  // namespace

PjRtBuildStampProto PjRtBuildStampProtoFromBuildData() {
  PjRtBuildStampProto proto;
  google::protobuf::Timestamp timestamp;
  timestamp.set_seconds(tsl::builddata::TimestampAsInt());
  if (ValidateBuildTimestamp(timestamp).ok()) {
    *proto.mutable_build_timestamp() = std::move(timestamp);
  }
  if (absl::string_view vcs_revision = tsl::builddata::SourceRevision();
      ValidateVcsRevision(vcs_revision).ok()) {
    proto.set_vcs_revision(vcs_revision);
  }
  proto.set_build_label(tsl::builddata::BuildLabel());
  return proto;
}

absl::Status SetAttribute(PjRtBuildStampProto& proto, absl::string_view key,
                          const PjRtValueType& value) {
  PjRtValueTypeProto value_proto = PjRtValueTypeToProto(value);
  ABSL_RETURN_IF_ERROR(ValidateAttribute(key, value_proto));
  auto [it, inserted] =
      proto.mutable_attributes()->try_emplace(key, std::move(value_proto));
  if (!inserted) {
    return absl::AlreadyExistsError(
        absl::StrCat("Attribute '", key, "' is already set."));
  }
  return absl::OkStatus();
}

absl::Status SetMosaicVersion(PjRtBuildStampProto& proto, int64_t major_version,
                              int64_t minor_version) {
  return SetCompatibilityVersion(proto, kMosaicCompatibilityVersionName,
                                 major_version, minor_version);
}

}  // namespace xla
