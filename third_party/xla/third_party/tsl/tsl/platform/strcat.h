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

// #status: RECOMMENDED
// #category: operations on strings
// #summary: Merges strings or numbers with no delimiter.
//
#ifndef TENSORFLOW_TSL_PLATFORM_STRCAT_H_
#define TENSORFLOW_TSL_PLATFORM_STRCAT_H_

#include "absl/strings/str_cat.h"

namespace tsl {
namespace strings {

// NOLINTBEGIN(misc-unused-using-decls)
using absl::AlphaNum;
using absl::Hex;
using absl::PadSpec;
using absl::StrAppend;
using absl::StrCat;
// NOLINTEND(misc-unused-using-decls)

}  // namespace strings
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_STRCAT_H_
