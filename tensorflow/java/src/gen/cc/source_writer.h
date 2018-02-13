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

#ifndef TENSORFLOW_JAVA_SRC_GEN_CC_SOURCE_WRITER_H_
#define TENSORFLOW_JAVA_SRC_GEN_CC_SOURCE_WRITER_H_

#include <string>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

// A utility class for writing source code, normally generated at
// compile-time.
//
// Source writers are language-agnostic and therefore only expose generic
// methods common to most languages. Extend or wrap this class to implement
// language-specific features.
//
// Note: if you are looking to reuse this class for generating code in another
// language than Java, please do by moving it at the '//tensorflow/core/lib/io'
// level.
class SourceWriter {
 public:
  virtual ~SourceWriter() = default;

  // Returns true if the writer is at the beginnig of a new line
  bool newline() const { return newline_; }

  // Appends a piece of code or text.
  //
  // It is expected that no newline character is present in the data provided,
  // otherwise Write() must be used.
  SourceWriter& Append(const StringPiece& str);

  // Writes a block of code or text.
  //
  // The data might potentially contain newline characters, therefore it will
  // be scanned to ensure that each line is indented and prefixed properly,
  // making it a bit slower than Append().
  SourceWriter& Write(const string& text);

  // Appends a newline character and start writing on a new line.
  SourceWriter& EndLine();

  // Indents following lines with white spaces.
  //
  // Indentation is cumulative, i.e. the provided tabulation is added to the
  // current indentation value. If the tabulation is negative, the operation
  // will outdent the source code, until the indentation reaches 0 again.
  //
  // For example, calling Indent(2) twice will indent code with 4 white
  // spaces. Then calling Indent(-2) will outdent the code back to 2 white
  // spaces.
  SourceWriter& Indent(int tab);

  // Prefixes following lines with provided character(s).
  //
  // A common use case of a prefix is for commenting or documenting the code.
  //
  // The prefix is written after the indentation, For example, invoking
  // Indent(2)->Prefix("//") will result in prefixing lines with "  //".
  //
  // An empty value ("") will remove any line prefix that was previously set.
  SourceWriter& Prefix(const char* line_prefix) {
    line_prefix_ = line_prefix;
    return *this;
  }

 protected:
  virtual void DoAppend(const StringPiece& str) = 0;

 private:
  string left_margin_;
  string line_prefix_;
  bool newline_ = true;
};

// A writer that outputs source code into a file.
//
// Note: the writer does not acquire the ownership of the file being passed in
// parameter.
class SourceFileWriter : public SourceWriter {
 public:
  explicit SourceFileWriter(WritableFile* file) : file_(file) {}
  virtual ~SourceFileWriter() = default;

 protected:
  void DoAppend(const StringPiece& str) override {
    TF_CHECK_OK(file_->Append(str));
  }

 private:
  WritableFile* file_;
};

// A writer that outputs source code into a string buffer.
class SourceBufferWriter : public SourceWriter {
 public:
  SourceBufferWriter() : owns_buffer_(true), buffer_(new string()) {}
  explicit SourceBufferWriter(string* buffer)
      : owns_buffer_(false), buffer_(buffer) {}
  virtual ~SourceBufferWriter() {
    if (owns_buffer_) delete buffer_;
  }
  const string& str() { return *buffer_; }

 protected:
  void DoAppend(const StringPiece& str) override {
    buffer_->append(str.begin(), str.end());
  }

 private:
  bool owns_buffer_;
  string* buffer_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_SOURCE_WRITER_H_
