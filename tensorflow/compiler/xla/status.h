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

#ifndef TENSORFLOW_COMPILER_XLA_STATUS_H_
#define TENSORFLOW_COMPILER_XLA_STATUS_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

#if defined(__clang__)
// Only clang supports warn_unused_result as a type annotation.
class TF_MUST_USE_RESULT Status;
#endif

// Simple wrapper around tensorflow::Status that has the MUST_USE_RESULT
// annotation above. When tensorflow::Status adopts this annotation, this can
// simply become a "using tensorflow::Status".
class Status : public tensorflow::Status {
 public:
  static Status OK() { return tensorflow::Status::OK(); }

  // Note: implicit constructor.
  Status(tensorflow::Status other) : tensorflow::Status(other) {}

  Status() : tensorflow::Status() {}
  Status(tensorflow::error::Code code, tensorflow::StringPiece msg)
      : tensorflow::Status(code, msg) {}
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_STATUS_H_
