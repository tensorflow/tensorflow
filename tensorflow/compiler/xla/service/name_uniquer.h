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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_NAME_UNIQUER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_NAME_UNIQUER_H_

#include <string>
#include <unordered_map>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

// Simple stateful class that helps generate "unique" names. To use it, simply
// call GetUniqueName as many times as needed. The names returned by
// GetUniqueName are guaranteed to be distinct for this instance of the class.
class NameUniquer {
 public:
  explicit NameUniquer(const string& separator = "__")
      : separator_(separator) {}

  // Get a unique name in a string, with an optional prefix for convenience.
  string GetUniqueName(tensorflow::StringPiece prefix = "");

 private:
  // The string to use to separate the prefix of the name from the uniquing
  // integer value.
  string separator_;

  // Map from name prefix to the number of names generated using that prefix
  // so far.
  std::unordered_map<string, int> generated_names_;

  TF_DISALLOW_COPY_AND_ASSIGN(NameUniquer);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_NAME_UNIQUER_H_
