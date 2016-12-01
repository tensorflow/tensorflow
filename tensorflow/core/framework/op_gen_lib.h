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

#ifndef TENSORFLOW_FRAMEWORK_OP_GEN_LIB_H_
#define TENSORFLOW_FRAMEWORK_OP_GEN_LIB_H_

#include <string>
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {

inline string Spaces(int n) { return string(n, ' '); }

// Wrap prefix + str to be at most width characters, indenting every line
// after the first by prefix.size() spaces.  Intended use case is something
// like prefix = "  Foo(" and str is a list of arguments (terminated by a ")").
// TODO(josh11b): Option to wrap on ", " instead of " " when possible.
string WordWrap(StringPiece prefix, StringPiece str, int width);

// Looks for an "=" at the beginning of *description.  If found, strips it off
// (and any following spaces) from *description and return true.  Otherwise
// returns false.
bool ConsumeEquals(StringPiece* description);

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_OP_GEN_LIB_H_
