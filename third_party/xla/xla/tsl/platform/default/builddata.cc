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

// Provides various information about time and in environment in which an
// executable was built.

// NOTE: Implementation of the magic aspects of BuildData are provided
// in builddata_globals.cc; which is handled specially by the build
// system to provide details such as the build time and date.

#include "xla/tsl/builddata/builddata.h"

#include <cstdint>
#include <ctime>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/tsl/builddata/builddata_globals.h"

namespace tsl::builddata {

absl::string_view BuildInfo() { return kTslBuildStampInfoStr; }

absl::string_view BuildId() { return kTslBuildStampIdStr; }

absl::string_view BuildDir() { return kTslBuildStampDirectoryStr; }

absl::string_view SourceUri() { return kTslBuildStampSourceUriStr; }

absl::string_view BuildHost() { return kTslBuildStampHostnameStr; }

absl::string_view BuildTarget() { return kTslBuildStampG3BuildTargetStr; }

absl::string_view TargetName() { return kTslBuildStampTargetStr; }

absl::string_view BuildLabel() { return kTslBuildStampLabelStr; }

absl::string_view BuildClient() { return kTslBuildStampClientStr; }

std::string Timestamp() {
  int64_t timestamp = kTslBuildStampTimestampInt;
  absl::Time build_time = absl::FromUnixSeconds(timestamp);

  struct ZoneInfo {
    absl::TimeZone tz;
    bool loaded;
  };
  // Cache the timezone info to avoid repeated loading.
  static const ZoneInfo zone_info = []() {
    ZoneInfo zi;
    zi.loaded = absl::LoadTimeZone("America/Los_Angeles", &zi.tz);
    return zi;
  }();

  if (!zone_info.loaded) {
    return absl::StrCat("(", timestamp, ")");
  }
  return absl::FormatTime(
      "Built on %b %d %Y %H:%M:%S [TZ=America/Los_Angeles] (%s)", build_time,
      zone_info.tz);
}

time_t TimestampAsInt() { return kTslBuildStampTimestampInt; }

absl::string_view SourceRevision() {
  if (kTslBuildStampChangelistInt == -1) {
    return "";
  }
  if (kTslBuildStampChangelistInt == 0) {
    return "<unknown>";
  }
  return kTslBuildStampChangelistStr;
}

int64_t SourceRevisionAsInt() { return kTslBuildStampChangelistInt; }

absl::string_view BaselineSourceRevision() {
  if (kTslBuildStampBaselineChangelistInt == -1) {
    return "";
  }
  if (kTslBuildStampBaselineChangelistInt == 0) {
    return "<unknown>";
  }
  return kTslBuildStampBaselineChangelistStr;
}

int64_t BaselineSourceRevisionAsInt() {
  return kTslBuildStampBaselineChangelistInt;
}

ClientStatusType ClientStatus() {
  if (kTslBuildStampClientMintStatusInt == 1) {
    return MINT;
  }
  if (kTslBuildStampClientMintStatusInt == 0) {
    return MODIFIED;
  }
  return UNKNOWN;
}

absl::string_view ClientStatusAsString() {
  if (kTslBuildStampClientMintStatusInt == 1) {
    return "mint";
  }
  if (kTslBuildStampClientMintStatusInt == 0) {
    return "modified";
  }
  return "unknown";
}

absl::string_view CompilerTarget() { return kTslBuildStampCompilerStr; }

}  // namespace tsl::builddata
