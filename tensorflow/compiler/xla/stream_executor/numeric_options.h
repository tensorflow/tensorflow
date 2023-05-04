/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
------------------------------------------------------------------------------*/

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_NUMERIC_OPTIONS_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_NUMERIC_OPTIONS_H_

namespace stream_executor {

// Options that specify the numeric behavior of operations like matrix
// multiplications and convolutions
struct NumericOptions {
  explicit NumericOptions(bool require_determinism)
      : require_determinism(require_determinism) {}

  bool require_determinism;
};

}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_NUMERIC_OPTIONS_H_
