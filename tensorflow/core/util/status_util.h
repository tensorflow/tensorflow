/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_STATUS_UTIL_H_
#define TENSORFLOW_CORE_UTIL_STATUS_UTIL_H_

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

// Creates a tag to be used in an exception error message. This can be parsed by
// the Python layer and replaced with information about the node.
//
// For example, error_format_tag(node, "${file}") returns
// "^^node:NODE_NAME:${line}^^" which would be rewritten by the Python layer as
// e.g. "file/where/node/was/created.py".
inline string error_format_tag(const Node& node, const string& format) {
  return strings::StrCat("^^node:", node.name(), ":", format, "^^");
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_STATUS_UTIL_H_
