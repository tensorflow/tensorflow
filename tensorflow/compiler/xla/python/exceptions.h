/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_EXCEPTIONS_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_EXCEPTIONS_H_

#include <stdexcept>
#include <string>

#include "tensorflow/compiler/xla/status.h"

namespace xla {

// Custom exception type used such that we can raise XlaRuntimeError in
// Python code instead of RuntimeError.
class XlaRuntimeError : public std::runtime_error {
 public:
  explicit XlaRuntimeError(const Status status)
      : std::runtime_error(status.ToString()) {
    CHECK(!status.ok());
  }

  explicit XlaRuntimeError(const std::string what) : std::runtime_error(what) {}
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_EXCEPTIONS_H_
