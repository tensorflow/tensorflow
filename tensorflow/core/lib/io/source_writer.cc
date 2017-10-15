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

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/source_writer.h"

namespace tensorflow {
namespace io {

SourceWriter* SourceWriter::Write(const StringPiece& str) {
  if (!str.empty()) {
    if (new_line) {
      Append(left_margin + line_prefix);
      new_line = false;
    }
    Append(str);
  }
  return this;
}

SourceWriter* SourceWriter::Inline(const string& str) {
  size_t line_pos = 0;
  do {
    size_t start_pos = line_pos;
    line_pos = str.find('\n', start_pos);
    if (line_pos != string::npos) {
      ++line_pos;
      Write(StringPiece(str.data() + start_pos, line_pos - start_pos));
      new_line = true;
    } else {
      Write(StringPiece(str.data() + start_pos, str.size() - start_pos));
    }
  } while (line_pos != string::npos && line_pos < str.size());

  return this;
}

SourceFileWriter::SourceFileWriter(const string& fname, Env* env) {
  TF_CHECK_OK(env->NewWritableFile(fname, &file_));
}

SourceFileWriter::~SourceFileWriter() {
  TF_CHECK_OK(file_->Close());
}

}  // namespace io
}  // namespace tensorflow
