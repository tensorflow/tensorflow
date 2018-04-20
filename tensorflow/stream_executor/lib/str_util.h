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

#ifndef TENSORFLOW_STREAM_EXECUTOR_LIB_STR_UTIL_H_
#define TENSORFLOW_STREAM_EXECUTOR_LIB_STR_UTIL_H_

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/stream_executor/lib/stringpiece.h"

namespace perftools {
namespace gputools {
namespace port {

using tensorflow::str_util::Join;
using tensorflow::str_util::Split;

// Returns a copy of the input string 'str' with the given 'suffix'
// removed. If the suffix doesn't match, returns a copy of the original string.
inline string StripSuffixString(port::StringPiece str, port::StringPiece suffix) {
  if (tensorflow::str_util::EndsWith(str, suffix)) {
    str.remove_suffix(suffix.size());
  }
  return str.ToString();
}

using tensorflow::str_util::Lowercase;

}  // namespace port
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_LIB_STR_UTIL_H_
