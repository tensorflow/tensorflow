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

#include "tensorflow/compiler/xla/service/name_uniquer.h"

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

string NameUniquer::GetUniqueName(tensorflow::StringPiece prefix) {
  string root = prefix.empty() ? "name" : prefix.ToString();
  int* count = &(generated_names_[root]);
  if (*count == 0) {
    *count = 1;
    return root;
  } else {
    tensorflow::strings::StrAppend(&root, separator_, *count);
    // Increment lookup under old 'root' name.
    (*count)++;
    // Initialize count under new 'root' name.
    count = &(generated_names_[root]);
    *count = 1;
    return root;
  }
}

}  // namespace xla
