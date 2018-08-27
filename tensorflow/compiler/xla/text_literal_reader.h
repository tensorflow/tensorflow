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

#ifndef TENSORFLOW_COMPILER_XLA_TEXT_LITERAL_READER_H_
#define TENSORFLOW_COMPILER_XLA_TEXT_LITERAL_READER_H_

#include <memory>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

// Reads a textual literal from a file path.  The format of the file must be:
//
//    f32[1,2,3,4]
//    (0, 0, 0, 0): 1.234
//    (0, 0, 0, 1): 0xf00p-2
//    ...
//
// Note that for floating values the hex output (as in the second value above)
// will more precisely convey the exact values.
class TextLiteralReader {
 public:
  // See class comment -- reads a file in its entirety (there must be only one
  // literal in the text file path provided).
  static StatusOr<std::unique_ptr<Literal>> ReadPath(absl::string_view path);

 private:
  // Ownership of file is transferred.
  explicit TextLiteralReader(tensorflow::RandomAccessFile* file);

  // Parses a shape string on the first line, followed by lines of values to the
  // end of the file.
  StatusOr<std::unique_ptr<Literal>> ReadAllLines();

  // Owns the file being read
  std::unique_ptr<tensorflow::RandomAccessFile> file_;

  TF_DISALLOW_COPY_AND_ASSIGN(TextLiteralReader);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TEXT_LITERAL_READER_H_
