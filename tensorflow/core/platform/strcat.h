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

#ifndef TENSORFLOW_CORE_PLATFORM_STRCAT_H_
#define TENSORFLOW_CORE_PLATFORM_STRCAT_H_

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/numbers.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/tsl/platform/strcat.h"

namespace tensorflow {
namespace strings {

// NOLINTBEGIN(misc-unused-using-decls)
using tsl::strings::AlphaNum;
using tsl::strings::Hex;
using tsl::strings::kZeroPad10;
using tsl::strings::kZeroPad11;
using tsl::strings::kZeroPad12;
using tsl::strings::kZeroPad13;
using tsl::strings::kZeroPad14;
using tsl::strings::kZeroPad15;
using tsl::strings::kZeroPad16;
using tsl::strings::kZeroPad2;
using tsl::strings::kZeroPad3;
using tsl::strings::kZeroPad4;
using tsl::strings::kZeroPad5;
using tsl::strings::kZeroPad6;
using tsl::strings::kZeroPad7;
using tsl::strings::kZeroPad8;
using tsl::strings::kZeroPad9;
using tsl::strings::PadSpec;
using tsl::strings::StrAppend;
using tsl::strings::StrCat;
// NOLINTEND(misc-unused-using-decls)

}  // namespace strings
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_STRCAT_H_
