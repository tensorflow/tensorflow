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

#include <string>

#include "tensorflow/java/src/gen/cc/source_writer.h"

namespace tensorflow {

SourceWriter& SourceWriter::Append(const StringPiece& str) {
  if (!str.empty()) {
    if (newline_) {
      DoAppend(left_margin_ + line_prefix_);
      newline_ = false;
    }
    DoAppend(str);
  }
  return *this;
}

SourceWriter& SourceWriter::Write(const string& str) {
  size_t line_pos = 0;
  do {
    size_t start_pos = line_pos;
    line_pos = str.find('\n', start_pos);
    if (line_pos != string::npos) {
      ++line_pos;
      Append(StringPiece(str.data() + start_pos, line_pos - start_pos));
      newline_ = true;
    } else {
      Append(StringPiece(str.data() + start_pos, str.size() - start_pos));
    }
  } while (line_pos != string::npos && line_pos < str.size());

  return *this;
}

SourceWriter& SourceWriter::EndLine() {
  Append("\n");
  newline_ = true;
  return *this;
}

SourceWriter& SourceWriter::Indent(int tab) {
  left_margin_.resize(std::max(static_cast<int>(left_margin_.size() + tab), 0),
                      ' ');
  return *this;
}

}  // namespace tensorflow
