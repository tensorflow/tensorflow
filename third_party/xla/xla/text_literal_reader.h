/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_TEXT_LITERAL_READER_H_
#define XLA_TEXT_LITERAL_READER_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/literal.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/file_system.h"

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
  static absl::StatusOr<Literal> ReadPath(absl::string_view path);

 private:
  // Ownership of file is transferred.
  explicit TextLiteralReader(tsl::RandomAccessFile* file);

  // Parses a shape string on the first line, followed by lines of values to the
  // end of the file.
  absl::StatusOr<Literal> ReadAllLines();

  // Owns the file being read
  std::unique_ptr<tsl::RandomAccessFile> file_;

  TextLiteralReader(const TextLiteralReader&) = delete;
  TextLiteralReader& operator=(const TextLiteralReader&) = delete;
};

}  // namespace xla

#endif  // XLA_TEXT_LITERAL_READER_H_
