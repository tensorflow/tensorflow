/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_HTML_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_HTML_UTILS_H_

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace tensorflow {
namespace profiler {

// Creates a html that links to the given url with the given text.
inline std::string AnchorElement(absl::string_view url,
                                 absl::string_view text) {
  return absl::StrCat("<a href=\"", url, "\" target=\"_blank\">", text, "</a>");
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_HTML_UTILS_H_
