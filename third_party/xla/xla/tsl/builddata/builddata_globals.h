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

#ifndef XLA_TSL_BUILDDATA_BUILDDATA_GLOBALS_H_
#define XLA_TSL_BUILDDATA_BUILDDATA_GLOBALS_H_

extern "C" const char kTslBuildStampChangelistStr[];
extern "C" const char kTslBuildStampSourceUriStr[];
extern "C" const char kTslBuildStampClientStr[];
extern "C" const char kTslBuildStampCompilerStr[];
extern "C" const char kTslBuildStampIdStr[];
extern "C" const char kTslBuildStampInfoStr[];
extern "C" const char kTslBuildStampLabelStr[];
extern "C" const char kTslBuildStampTargetStr[];
extern "C" const char kTslBuildStampTimestampStr[];
extern "C" const char kTslBuildStampG3BuildTargetStr[];
extern "C" const char kTslBuildStampBaselineChangelistStr[];
extern "C" const char kTslBuildStampUsernameStr[];
extern "C" const char kTslBuildStampHostnameStr[];
extern "C" const char kTslBuildStampDirectoryStr[];

// We use `long long` instead of `int64_t` to avoid pulling in `<cstdint>`,
// which minimizes the overhead of link-time compilation for the linkstamp
// file.
extern "C" const long long  // NOLINT(runtime/int) NOLINT(google-runtime-int)
    kTslBuildStampChangelistInt;
extern "C" const int kTslBuildStampClientMintStatusInt;
extern "C" const long long  // NOLINT(runtime/int) NOLINT(google-runtime-int)
    kTslBuildStampTimestampInt;
extern "C" const long long  // NOLINT(runtime/int) NOLINT(google-runtime-int)
    kTslBuildStampBaselineChangelistInt;

#endif  // XLA_TSL_BUILDDATA_BUILDDATA_GLOBALS_H_
