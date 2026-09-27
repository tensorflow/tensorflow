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

// This file is somewhat magic: it gets compiled into an object file
// during the bazel *link* step, immediately before the linker is
// run. It is never built/cached separately, so that the data will
// always be accurate as of when the binary was linked.
//
// Therefore, it should be kept extremely simple, and should not
// depend on any system or other header files, so that the link step
// does not need to have a full compilation environment available.
//
// These constants allow access to the date and time the executable
// was built, and information about the source control client where the build
// occurred.
//
// These strings are used directly as exported variables.

#include "xla/tsl/builddata/builddata_globals.h"

#include "xla/tsl/builddata/utils.h"

using tsl::builddata::ParseChangelist;
using tsl::builddata::ParseMintStatus;

#define AS_STRING2(x) #x
#define AS_STRING(x) AS_STRING2(x)

#ifndef BUILD_TIMESTAMP
// NOLINTNEXTLINE
#error Must specify -DBUILD_TIMESTAMP=`date +%s` when linking in builddata_globals.cc
#endif

#ifndef BUILD_USERNAME
#ifndef BUILD_USER
#define BUILD_USERNAME "unknown"
#else
#define BUILD_USERNAME BUILD_USER
#endif  // BUILD_USER
#endif  // BUILD_USERNAME

#ifndef BUILD_HOSTNAME
#ifndef BUILD_HOST
#define BUILD_HOSTNAME "unknown"
#else
#define BUILD_HOSTNAME BUILD_HOST
#endif  // BUILD_HOST
#endif  // BUILD_HOSTNAME

#ifndef BUILD_EMBED_LABEL
#define BUILD_EMBED_LABEL ""
#endif

#ifndef BUILD_SCM_STATUS
#define BUILD_SCM_STATUS "unknown"
#endif

#ifndef G3_TARGET_NAME
// NOLINTNEXTLINE
#error Must specify -DG3_TARGET_NAME=... when linking in builddata_globals.cc
#endif

#ifndef G3_BUILD_TARGET
// NOLINTNEXTLINE
#error Must specify -DG3_BUILD_TARGET=... when linking in builddata_globals.cc
#endif

#ifndef GPLATFORM
#define GPLATFORM "Unknown"
#endif

#ifdef BUILD_CHANGELIST
#define BUILD_CHANGELIST_INT BUILD_CHANGELIST
#define BUILD_CHANGELIST_STR AS_STRING(BUILD_CHANGELIST)
#else
#ifdef BUILD_SCM_REVISION
#define BUILD_CHANGELIST_INT (ParseChangelist(BUILD_SCM_REVISION))
#define BUILD_CHANGELIST_STR BUILD_SCM_REVISION
#else
#define BUILD_CHANGELIST_INT -1
#define BUILD_CHANGELIST_STR ""
#endif  // BUILD_SCM_REVISION
#endif  // BUILD_CHANGELIST

#ifndef BUILD_BASELINE_CHANGELIST  // as good as undefined
#define BUILD_BASELINE_CHANGELIST_INT BUILD_CHANGELIST_INT
#define BUILD_BASELINE_CHANGELIST_STR BUILD_CHANGELIST_STR
#else
#define BUILD_BASELINE_CHANGELIST_INT BUILD_BASELINE_CHANGELIST
#define BUILD_BASELINE_CHANGELIST_STR AS_STRING(BUILD_BASELINE_CHANGELIST)
#endif

#ifndef BUILD_CLIENT_MINT_STATUS
#ifndef BUILD_SCM_STATUS
#define BUILD_CLIENT_MINT_STATUS -1
#else
#define BUILD_CLIENT_MINT_STATUS BUILD_SCM_STATUS
#endif  // BUILD_SCM_STATUS
#endif  // BUILD_CLIENT_MINT_STATUS

#ifndef SOURCE_URI
#ifndef BUILD_SCM_REVISION
#define SOURCE_URI "unknown"
#else
#define SOURCE_URI BUILD_SCM_REVISION
#endif
#endif

#ifndef BUILD_DIRECTORY
#define BUILD_DIRECTORY "."
#endif

#ifndef BUILD_CLIENT
#define BUILD_CLIENT ""
#endif

#ifndef BUILD_ID
#define BUILD_ID ""
#endif

#define TSL_ATTRIBUTE_USED_RETAIN __attribute__((used, retain))

// Define extern variables for use by tools that wish to obtain these
// values via symbol lookup without running the executable.

extern "C" {
TSL_ATTRIBUTE_USED_RETAIN const char kTslBuildStampTimestampStr[32] =
    AS_STRING(BUILD_TIMESTAMP);
TSL_ATTRIBUTE_USED_RETAIN const char kTslBuildStampUsernameStr[512] =
    BUILD_USERNAME;
TSL_ATTRIBUTE_USED_RETAIN const char kTslBuildStampHostnameStr[512] =
    BUILD_HOSTNAME;
TSL_ATTRIBUTE_USED_RETAIN const char kTslBuildStampLabelStr[512] =
    BUILD_EMBED_LABEL;
TSL_ATTRIBUTE_USED_RETAIN const char kTslBuildStampTargetStr[512] =
    G3_TARGET_NAME;
TSL_ATTRIBUTE_USED_RETAIN const char kTslBuildStampG3BuildTargetStr[512] =
    G3_BUILD_TARGET;
TSL_ATTRIBUTE_USED_RETAIN const char kTslBuildStampCompilerStr[512] = GPLATFORM;
TSL_ATTRIBUTE_USED_RETAIN const char kTslBuildStampChangelistStr[32] =
    BUILD_CHANGELIST_STR;
TSL_ATTRIBUTE_USED_RETAIN const char kTslBuildStampBaselineChangelistStr[32] =
    BUILD_BASELINE_CHANGELIST_STR;
TSL_ATTRIBUTE_USED_RETAIN const char kTslBuildStampSourceUriStr[512] =
    SOURCE_URI;
TSL_ATTRIBUTE_USED_RETAIN const char kTslBuildStampDirectoryStr[512] =
    BUILD_DIRECTORY;
TSL_ATTRIBUTE_USED_RETAIN const char kTslBuildStampInfoStr[512] =
    BUILD_USERNAME "@" BUILD_HOSTNAME ":" BUILD_DIRECTORY;
TSL_ATTRIBUTE_USED_RETAIN const char kTslBuildStampClientStr[512] =
    BUILD_CLIENT;
TSL_ATTRIBUTE_USED_RETAIN const char kTslBuildStampIdStr[512] = BUILD_ID;

// Now, the same sorts of things, as integers.
TSL_ATTRIBUTE_USED_RETAIN const long long  // NOLINT(runtime/int)
    kTslBuildStampTimestampInt = BUILD_TIMESTAMP;
TSL_ATTRIBUTE_USED_RETAIN const long long  // NOLINT(runtime/int)
    kTslBuildStampChangelistInt = BUILD_CHANGELIST_INT;
TSL_ATTRIBUTE_USED_RETAIN const long long  // NOLINT(runtime/int)
    kTslBuildStampBaselineChangelistInt = BUILD_BASELINE_CHANGELIST_INT;

TSL_ATTRIBUTE_USED_RETAIN const int kTslBuildStampClientMintStatusInt =
    ParseMintStatus(BUILD_CLIENT_MINT_STATUS);
}  // extern "C"

#undef TSL_ATTRIBUTE_USED_RETAIN

#ifdef __APPLE__
// On Apple, due to quirks of its build rules, linkstamp appears first in link
// ordering.  Having an empty file at the beginning of the linker commandine is
// triggering a linker (lld.ld64) bug that causes some tests to crash.  For
// Apple this file is empty.  Work around the bug by adding a dummy function to
// this file.
namespace {
__attribute__((used)) void dummy() {}
}  // namespace
#endif  // __APPLE__
