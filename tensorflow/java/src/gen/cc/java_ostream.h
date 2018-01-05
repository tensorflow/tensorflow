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

#ifndef TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_OSTREAM_H_
#define TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_OSTREAM_H_

#include <string>
#include <set>
#include <vector>

#include "tensorflow/java/src/gen/cc/source_writer.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"

namespace tensorflow {
namespace java {

// A base class for writing Java source code using a stream-like API.
//
// It wraps a SourceWriter and exposes a set of basic operations for writing
// Java-specific features.
class JavaOutputStream {
 public:
  explicit JavaOutputStream(SourceWriter* writer) : writer_(writer) {}
  virtual ~JavaOutputStream() = default;

  // Returns the underlying source writer for direct access.
  SourceWriter& writer() { return *writer_; }

  // Appends a piece of text to the current line.
  //
  // See SourceWriter.Append(const StringPiece&)
  JavaOutputStream& operator<<(const string& str) {
    writer_->Append(str);
    return *this;
  }

  // Appends a piece of text to the current line.
  //
  // See SourceWriter.Append(const StringPiece&)
  JavaOutputStream& operator<<(const char* str) {
    writer_->Append(str);
    return *this;
  }

  // Appends the signature of a Java type to the current line.
  //
  // The type is written in its simple form (i.e. not prefixed by its package)
  // and followed by any parameter types it has enclosed in brackets (<>).
  JavaOutputStream& operator<<(const Type& type);

  // Appends the content of a Java snippet to the source output.
  //
  // If content contains more than one line, it will be automatically indented
  // and prefixed to match the current writing context.
  //
  // See SourceWriter.Write(const string&)
  JavaOutputStream& operator<<(const Snippet& snippet) {
    writer_->Write(snippet.data());
    return *this;
  }

  // Invokes the given manipulator with this stream.
  //
  // A list of common manipulators is provided at the end of this file.
  JavaOutputStream& operator<<(void (*f)(JavaOutputStream*)) {
    f(this);
    return *this;
  }

 protected:
  SourceWriter* writer_;
};

// A Java output stream specialized for writing class methods.
//
// This class should be instantiated by invoking ClassOutputStream.BeginMethod()
// and must be deleted implicitely by calling its EndMethod() function.
class MethodOutputStream : public JavaOutputStream {
 public:
  // Ends the current method.
  //
  // The stream is automatically deleted and therefore should not be reused.
  void EndMethod();

 private:
  explicit MethodOutputStream(SourceWriter* writer)
    : JavaOutputStream(writer) {}
  MethodOutputStream(SourceWriter* writer, std::set<string> generics)
    : JavaOutputStream(writer), declared_generics_names_(generics) {}
  virtual ~MethodOutputStream() = default;

  MethodOutputStream* Begin(const Method& method, int modifiers);

  std::set<string> declared_generics_names_;

  friend class ClassOutputStream;
};

// A Java output stream specialized for writing Java classes.
//
// This class should be instantiated by invoking SourceOutputStream.BeginClass()
// or ClassOutputStream.BeginInnerClass() and must be deleted implicitely by
// calling its EndClass() function.
class ClassOutputStream : public JavaOutputStream {
 public:
  // Writes a list of variables as fields of this class.
  ClassOutputStream* WriteFields(const std::vector<Variable>& fields,
      int modifiers = 0);

  // Begins to write a method of this class.
  MethodOutputStream* BeginMethod(const Method& method, int modifiers = 0);

  // Begins to write an inner class of this class.
  ClassOutputStream* BeginInnerClass(const Type& clazz, int modifiers = 0);

  // Ends the current class.
  //
  // The stream is automatically deleted and therefore should not be reused.
  void EndClass();

 private:
  explicit ClassOutputStream(SourceWriter* writer)
    : JavaOutputStream(writer) {}
  ClassOutputStream(SourceWriter* writer, std::set<string> generics)
    : JavaOutputStream(writer), declared_generics_names_(generics) {}
  virtual ~ClassOutputStream() = default;

  ClassOutputStream* Begin(const Type& clazz, int modifiers);

  std::set<string> declared_generics_names_;

  friend class SourceOutputStream;
};

// A Java output stream specialized for writing source files.
//
// This class is at the root of the JavaOutputStream hierarchy. The destination
// of the source being written is provided by the SourceWriter being passed
// to its constructor.
//
// As oppose to other Java streams, instances of this class must be deleted
// explicitly.
class SourceOutputStream : public JavaOutputStream {
 public:
  explicit SourceOutputStream(SourceWriter* writer)
    : JavaOutputStream(writer) {}
  virtual ~SourceOutputStream() = default;

  // Begins to write the main class of this file.
  ClassOutputStream* BeginClass(const Type& clazz,
      const std::vector<Type>* dependencies, int modifiers = 0);
};

//
// Common Java stream manipulators
//

// Manipulator inserting a newline character.
inline void endl(JavaOutputStream* stream) {
  stream->writer().EndLine();
}

// Manipulator beginning a new indented block of code.
inline void beginb(JavaOutputStream* stream) {
  stream->writer().Append(stream->writer().newline() ? "{" : " {")
      .EndLine()
      .Indent(2);
}

// Manipulator ending the current block of code.
inline void endb(JavaOutputStream* stream) {
  stream->writer().Indent(-2).Append("}").EndLine();
}

}  // namespace java
}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_OSTREAM_H_
