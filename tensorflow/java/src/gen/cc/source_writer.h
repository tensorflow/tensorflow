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
#include <stack>
#include <list>
#include <set>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"

namespace tensorflow {
namespace java {

// A class for writing Java source code.
class SourceWriter {
 public:
  SourceWriter();

  virtual ~SourceWriter();

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
  SourceWriter& Prefix(const char* line_prefix);

  // Writes a source code snippet.
  //
  // The data might potentially contain newline characters, therefore it will
  // be scanned to ensure that each line is indented and prefixed properly,
  // making it a bit slower than Append().
  SourceWriter& Write(const StringPiece& str);

  // Writes a source code snippet read from a file.
  //
  // All lines of the file at the provided path will be read and written back
  // to the output of this writer in regard of its current attributes (e.g.
  // the indentation, prefix, etc.)
  SourceWriter& WriteFromFile(const string& fname, Env* env = Env::Default());

  // Appends a piece of source code.
  //
  // It is expected that no newline character is present in the data provided,
  // otherwise Write() must be used.
  SourceWriter& Append(const StringPiece& str);

  // Appends a type to the current line.
  //
  // The type is written in its simple form (i.e. not prefixed by its package)
  // and followed by any parameter types it has enclosed in brackets (<>).
  SourceWriter& AppendType(const Type& type);

  // Appends a newline character.
  //
  // Data written after calling this method will start on a new line, in respect
  // of the current indentation.
  SourceWriter& EndLine();

  // Begins a block of source code.
  //
  // This method appends a new opening brace to the current data and indent the
  // next lines according to Google Java Style Guide. The block can optionally
  // be preceded by an expression (e.g. Append("if(true)").BeginBlock();)
  SourceWriter& BeginBlock() {
    return Append(newline_ ? "{" : " {").EndLine().Indent(2);
  }

  // Ends the current block of source code.
  //
  // This method appends a new closing brace to the current data and outdent the
  // next lines back to the margin used before BeginBlock() was invoked.
  SourceWriter& EndBlock() {
    return Indent(-2).Append("}").EndLine();
  }

  // Begins to write a method.
  //
  // This method outputs the signature of the Java method from the data passed
  // in the 'method' parameter and starts a new block. Additionnal modifiers can
  // also be passed in parameter to define the accesses and the scope of this
  // method.
  SourceWriter& BeginMethod(const Method& method, int modifiers = 0);

  // Ends the current method.
  //
  // This method ends the block of code that has begun when invoking
  // BeginMethod() prior to this.
  SourceWriter& EndMethod();

  // Begins to write the main type of a source file.
  //
  // This method outputs the declaration of the Java type from the data passed
  // in the 'type' parameter and starts a new block. Additionnal modifiers can
  // also be passed in parameter to define the accesses and the scope of this
  // type.
  //
  // If not null, all types found in the 'dependencies' list will be imported
  // before declaring the new type.
  SourceWriter& BeginType(const Type& clazz,
      const std::list<Type>* dependencies, int modifiers = 0);

  // Begins to write a new inner type.
  //
  // This method outputs the declaration of the Java type from the data passed
  // in the 'type' parameter and starts a new block. Additionnal modifiers can
  // also be passed in parameter to define the accesses and the scope of this
  // type.
  SourceWriter& BeginInnerType(const Type& type, int modifiers = 0);

  // Ends the current type.
  //
  // This method ends the block of code that has begun when invoking
  // BeginType() or BeginInnerType() prior to this.
  SourceWriter& EndType();

  // Writes a list of variables as fields of a type.
  //
  // This method must be called within the definition of a type (see BeginType()
  // or BeginInnerType()). Additional modifiers can also be passed in parameter
  // to define the accesses and the scope of those fields.
  SourceWriter& WriteFields(const std::list<Variable>& fields,
      int modifiers = 0);

 protected:
  virtual void DoAppend(const StringPiece& str) = 0;

 private:
  // A utility base class for visiting elements of a type.
  class TypeVisitor {
   public:
    virtual ~TypeVisitor() = default;
    void Visit(const Type& type);

   protected:
    virtual void DoVisit(const Type& type) = 0;
  };

  // A utility class for keeping track of declared generics in a given scope.
  class GenericNamespace : public TypeVisitor {
   public:
    GenericNamespace() = default;
    explicit GenericNamespace(const GenericNamespace* parent)
      : generic_names_(parent->generic_names_) {}
    std::list<const Type*> declared_types() {
      return declared_types_;
    }
   protected:
    virtual void DoVisit(const Type& type);

   private:
    std::list<const Type*> declared_types_;
    std::set<string> generic_names_;
  };

  // A utility class for collecting a list of import statements to declare.
  class TypeImporter : public TypeVisitor {
   public:
    explicit TypeImporter(const string& current_package)
      : current_package_(current_package) {}
    virtual ~TypeImporter() = default;
    const std::set<string> imports() {
      return imports_;
    }
   protected:
    virtual void DoVisit(const Type& type);

   private:
    string current_package_;
    std::set<string> imports_;
  };

  string left_margin_;
  string line_prefix_;
  bool newline_ = true;
  std::stack<GenericNamespace*> generic_namespaces_;

  SourceWriter& WriteModifiers(int modifiers);
  SourceWriter& WriteDoc(const string& description,
    const string& return_description = "",
    const std::list<Variable>* parameters = nullptr);
  SourceWriter& WriteAnnotations(const std::list<Annotation>& annotations);
  SourceWriter& WriteGenerics(const std::list<const Type*>& generics);
  GenericNamespace* PushGenericNamespace(int modifiers);
  void PopGenericNamespace();
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

}  // namespace java
}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_SOURCE_WRITER_H_
