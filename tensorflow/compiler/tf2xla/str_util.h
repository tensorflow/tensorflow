/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// String utilities that are esoteric enough that they don't belong in
// third_party/tensorflow/core/lib/strings/str_util.h, but are still generally
// useful under xla.

#ifndef TENSORFLOW_COMPILER_TF2XLA_STR_UTIL_H_
#define TENSORFLOW_COMPILER_TF2XLA_STR_UTIL_H_

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {
namespace str_util {

// Replace all non-overlapping occurrences of the given (from,to) pairs in-place
// in text.  If from is empty, it matches at the beginning of the text and after
// every byte.  Each (from,to) replacement pair is processed in the order it is
// given.
void ReplaceAllPairs(string* text,
                     const std::vector<std::pair<string, string>>& replace);

}  // namespace str_util
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_STR_UTIL_H_
