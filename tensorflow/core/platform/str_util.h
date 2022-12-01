/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_STR_UTIL_H_
#define TENSORFLOW_CORE_PLATFORM_STR_UTIL_H_

#include <string>
#include <vector>

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/tsl/platform/str_util.h"

// Basic string utility routines
namespace tensorflow {
namespace str_util {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::str_util::AllowEmpty;
using tsl::str_util::ArgDefCase;
using tsl::str_util::CEscape;
using tsl::str_util::ConsumeLeadingDigits;
using tsl::str_util::ConsumeNonWhitespace;
using tsl::str_util::ConsumePrefix;
using tsl::str_util::ConsumeSuffix;
using tsl::str_util::CUnescape;
using tsl::str_util::EndsWith;
using tsl::str_util::Join;
using tsl::str_util::Lowercase;
using tsl::str_util::RemoveLeadingWhitespace;
using tsl::str_util::RemoveTrailingWhitespace;
using tsl::str_util::RemoveWhitespaceContext;
using tsl::str_util::SkipEmpty;
using tsl::str_util::SkipWhitespace;
using tsl::str_util::Split;
using tsl::str_util::StartsWith;
using tsl::str_util::StrContains;
using tsl::str_util::StringReplace;
using tsl::str_util::StripPrefix;
using tsl::str_util::StripSuffix;
using tsl::str_util::StripTrailingWhitespace;
using tsl::str_util::Strnlen;
using tsl::str_util::TitlecaseString;
using tsl::str_util::Uppercase;
// NOLINTEND(misc-unused-using-decls)
}  // namespace str_util
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_STR_UTIL_H_
