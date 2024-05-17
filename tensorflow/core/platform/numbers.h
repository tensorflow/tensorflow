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

#ifndef TENSORFLOW_CORE_PLATFORM_NUMBERS_H_
#define TENSORFLOW_CORE_PLATFORM_NUMBERS_H_

#include <string>

#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/numbers.h"

namespace tensorflow {
namespace strings {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::strings::DoubleToBuffer;
using tsl::strings::FastInt32ToBufferLeft;
using tsl::strings::FastInt64ToBufferLeft;
using tsl::strings::FastUInt32ToBufferLeft;
using tsl::strings::FastUInt64ToBufferLeft;
using tsl::strings::FloatToBuffer;
using tsl::strings::FpToString;
using tsl::strings::HexStringToUint64;
using tsl::strings::HumanReadableElapsedTime;
using tsl::strings::HumanReadableNum;
using tsl::strings::HumanReadableNumBytes;
using tsl::strings::kFastToBufferSize;
using tsl::strings::ProtoParseNumeric;
using tsl::strings::safe_strto32;
using tsl::strings::safe_strto64;
using tsl::strings::safe_strtod;
using tsl::strings::safe_strtof;
using tsl::strings::safe_strtou32;
using tsl::strings::safe_strtou64;
using tsl::strings::SafeStringToNumeric;
using tsl::strings::StringToFp;
using tsl::strings::Uint64ToHexString;
// NOLINTEND(misc-unused-using-decls)
}  // namespace strings
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_NUMBERS_H_
