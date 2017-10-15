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

#ifndef TENSORFLOW_CORE_LIB_IO_SOURCE_WRITER_H_
#define TENSORFLOW_CORE_LIB_IO_SOURCE_WRITER_H_

#include <memory>
#include <string>
#include <algorithm>

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {
namespace io {

// A utility class for writing source code, often generated at compile-time.
//
// Source writers are language-agnostic and therefore only expose generic
// methods common to most languages. Use a decorator class to implement
// language-specific features.
class SourceWriter {
 public:
  SourceWriter() : new_line(true) {}
  virtual ~SourceWriter() {}

  // Appends a piece of code or text.
  //
  // It is expected that no newline character is present in the data provided,
  // otherwise Inline() must be used.
  SourceWriter* Write(const StringPiece& str);

  // Appends a block of code or text.
  //
  // The data might potentially contain newline characters, therefore it will
  // be scanned to ensure that each line is indented and prefixed properly,
  // making it a bit slower that Write().
  SourceWriter* Inline(const string& text);

  // Appends a newline character.
  SourceWriter* EndOfLine() {
    static const StringPiece eol("\n");
    Write(eol);
    new_line = true;
    return this;
  }

  // Indents following lines with white spaces.
  //
  // Indentation is cumulative, i.e. the provided tabulation is added to the
  // current indentation value. If the tabulation is negative, the operation
  // will outdent the source code, until the indentation reaches 0 again.
  //
  // For example, calling Indent(2) twice will indent code with 4 white
  // spaces. Then calling Indent(-2) will outdent the code back to 2 white
  // spaces.
  SourceWriter* Indent(int tab) {
    left_margin.resize(
        std::max(static_cast<int>(left_margin.size() + tab), 0), ' ');
    return this;
  }

  // Prefixes following lines with character(s).
  //
  // A common use case of a prefix is for writing comments into the source code.
  //
  // The prefix is written after the indentation, For example, invoking
  // Indent(2)->Prefix("//") will result in prefixing lines with "  //".
  SourceWriter* LinePrefix(const char* line_prefix) {
    this->line_prefix = line_prefix;
    return this;
  }

  // Removes the actual line prefix, if any.
  SourceWriter* RemoveLinePrefix() {
    this->line_prefix.clear();
    return this;
  }

 protected:
  // Appends a piece of text to the source destination.
  virtual void Append(const StringPiece& str) = 0;

 private:
  string left_margin;
  string line_prefix;
  bool new_line;
};

// A writer outputing source code into a writable file.
class SourceFileWriter : public SourceWriter {
 public:
  explicit SourceFileWriter(const string& fname, Env* env = Env::Default());
  virtual ~SourceFileWriter();

 protected:
  void Append(const StringPiece& str) override {
    TF_CHECK_OK(file_->Append(str));
  }
 private:
  std::unique_ptr<WritableFile> file_;
};

// A writer outputing source code into a string buffer.
class SourceBufferWriter : public SourceWriter {
 public:
  SourceBufferWriter() = default;
  virtual ~SourceBufferWriter() = default;

  // Returns the string buffer of this writer, with all the code written so far.
  const string& ToString() const {
    return buffer_;
  }
 protected:
  void Append(const StringPiece& str) override {
    buffer_.append(str.begin(), str.end());
  }
 private:
  string buffer_;
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_IO_SOURCE_WRITER_H_
