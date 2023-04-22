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

#ifndef TENSORFLOW_COMPILER_XLA_TEXT_LITERAL_WRITER_H_
#define TENSORFLOW_COMPILER_XLA_TEXT_LITERAL_WRITER_H_

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

// Writes a literal to textual form at a file path.
//
// The format is roughly:
//
//    f32[1,2,3,4]
//    (0, 0, 0, 0): 1.234
//    (0, 0, 0, 1): 0xf00p-2
//    ...
//
// This should be readable by xla::TextLiteralReader.
class TextLiteralWriter {
 public:
  static Status WriteToPath(const Literal& literal, absl::string_view path);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TextLiteralWriter);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TEXT_LITERAL_WRITER_H_
