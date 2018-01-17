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
#include <set>
#include <vector>
#include <deque>

#include "tensorflow/java/src/gen/cc/source_writer.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"
#include "tensorflow/java/src/gen/cc/java_ostream.h"

namespace tensorflow {
namespace java {
namespace {

template <typename TypeScanner>
void ScanType(const Type& type, TypeScanner* scanner) {
  (*scanner)(type);
  for (auto it = type.parameters().cbegin(); it != type.parameters().cend();
      ++it) {
    ScanType(*it, scanner);
  }
  for (auto it = type.annotations().cbegin(); it != type.annotations().cend();
      ++it) {
    ScanType(*it, scanner);
  }
  for (auto it = type.supertypes().cbegin(); it != type.supertypes().cend();
      ++it) {
    ScanType(*it, scanner);
  }
}

// A type scanner used to discover all of its undeclared generic parameters (see
// the ScanType() utilities in java_defs.h)
class GenericTypeScanner {
 public:
  explicit GenericTypeScanner(std::set<string>* generic_namespace)
    : generic_namespace_(generic_namespace) {}
  const std::vector<const Type*>& discoveredTypes() const {
    return discovered_generics_;
  }
  void operator()(const Type& type) {
    // ignore non-generic parameters, wildcards and generics already declared
    if (type.kind() == Type::GENERIC
        && !type.IsWildcard()
        && generic_namespace_->find(type.name()) == generic_namespace_->end()) {
      discovered_generics_.push_back(&type);
      generic_namespace_->insert(type.name());
    }
  }
 private:
  std::vector<const Type*> discovered_generics_;
  std::set<string>* generic_namespace_;
};

// Writes a list of Java modifiers.
void WriteModifiers(int modifiers, SourceWriter* writer) {
  if (modifiers & PUBLIC) {
    writer->Append("public ");
  } else if (modifiers & PROTECTED) {
    writer->Append("protected ");
  } else if (modifiers & PRIVATE) {
    writer->Append("private ");
  }
  if (modifiers & STATIC) {
    writer->Append("static ");
  }
  if (modifiers & FINAL) {
    writer->Append("final ");
  }
}

// Writes the signature of a Java type.
void WriteType(const Type& type, SourceWriter* writer) {
  if (type.kind() == Type::Kind::GENERIC && type.name().empty()) {
    writer->Append("?");
  } else {
    writer->Append(type.name());
  }
  if (!type.parameters().empty()) {
    writer->Append("<");
    for (std::vector<Type>::const_iterator it = type.parameters().cbegin();
        it != type.parameters().cend(); ++it) {
      if (it != type.parameters().cbegin()) {
        writer->Append(", ");
      }
      WriteType(*it, writer);
    }
    writer->Append(">");
  }
}

// Writes the usage of a Java annotation.
void WriteAnnotations(const std::vector<Annotation>& annotations,
    SourceWriter* writer) {
  for (std::vector<Annotation>::const_iterator it = annotations.cbegin();
      it != annotations.cend(); ++it) {
    writer->Append("@" + it->name());
    if (!it->attributes().empty()) {
      writer->Append("(").Append(it->attributes()).Append(")");
    }
    writer->EndLine();
  }
}

// Writes documentation in the Javadoc format.
void WriteDoc(const string& description,
    SourceWriter* writer,
    const string& return_description = "",
    const std::vector<Variable>* parameters = nullptr) {
  if (description.empty() && return_description.empty()
      && (parameters == nullptr || parameters->empty())) {
    return;  // no doc to write
  }
  bool do_line_break = false;
  writer->Append("/**").EndLine().Prefix(" * ");
  if (!description.empty()) {
    writer->Write(description).EndLine();
    do_line_break = true;
  }
  if (parameters != nullptr && !parameters->empty()) {
    if (do_line_break) {
      writer->EndLine();
      do_line_break = false;
    }
    for (std::vector<Variable>::const_iterator it = parameters->begin();
        it != parameters->end(); ++it) {
      writer->Append("@param ").Append(it->name());
      if (!it->description().empty()) {
        writer->Append(" ").Write(it->description());
      }
      writer->EndLine();
    }
  }
  if (!return_description.empty()) {
    if (do_line_break) {
      writer->EndLine();
      do_line_break = false;
    }
    writer->Append("@return ").Write(return_description).EndLine();
  }
  writer->Prefix("").Append(" **/").EndLine();
}

// Declares a list of Java generic parameters.
void WriteGenerics(const std::vector<const Type*>& generics,
    SourceWriter* writer) {
  writer->Append("<");
  for (std::vector<const Type*>::const_iterator it = generics.cbegin();
      it != generics.cend(); ++it) {
    if (it != generics.cbegin()) {
      writer->Append(", ");
    }
    writer->Append((*it)->name());
    if (!(*it)->supertypes().empty()) {
      writer->Append(" extends ");
      WriteType((*it)->supertypes().front(), writer);
    }
  }
  writer->Append(">");
}

}  // namespace

JavaOutputStream& JavaOutputStream::operator<<(const Type& type) {
  WriteType(type, writer_);
  return *this;
}

void MethodOutputStream::EndMethod() {
  *this << endb;
  delete this;
}

MethodOutputStream* MethodOutputStream::Begin(const Method& method,
    int modifiers) {
  WriteModifiers(modifiers, writer_);
  // Look for generics we need to declare at the beginning of the signature
  GenericTypeScanner generic_scanner(&generic_namespace_);
  if (!method.constructor()) {
    ScanType(method.return_type(), &generic_scanner);
  }
  for (std::vector<Variable>::const_iterator it = method.arguments().cbegin();
      it != method.arguments().cend(); ++it) {
    ScanType(it->type(), &generic_scanner);
  }
  if (!generic_scanner.discoveredTypes().empty()) {
    WriteGenerics(generic_scanner.discoveredTypes(), writer_);
    *this << " ";
  }
  // Complete signature with the return type (if not a constructor) and the list
  // of arguments
  if (!method.constructor()) {
    *this << method.return_type() << " ";
  }
  *this << method.name() << "(";
  for (std::vector<Variable>::const_iterator it = method.arguments().cbegin();
      it != method.arguments().cend(); ++it) {
    if (it != method.arguments().cbegin()) {
      *this << ", ";
    }
    *this << it->type() << (it->variadic() ? "... " : " ") << it->name();
  }
  *this << ")" << beginb;
  return this;
}

ClassOutputStream* ClassOutputStream::WriteFields(
    const std::vector<Variable>& fields, int modifiers) {
  *this << endl;
  for (std::vector<Variable>::const_iterator it = fields.cbegin();
      it != fields.cend(); ++it) {
    WriteModifiers(modifiers, writer_);
    *this << it->type() << " " << it->name() << ";" << endl;
  }
  return this;
}

MethodOutputStream* ClassOutputStream::BeginMethod(const Method& method,
    int modifiers) {
  *this << endl;
  WriteDoc(method.description(), writer_, method.return_description(),
      &method.arguments());
  if (!method.annotations().empty()) {
    WriteAnnotations(method.annotations(), writer_);
  }
  MethodOutputStream* method_ostream;
  if (modifiers & STATIC) {
    method_ostream = new MethodOutputStream(writer_);
  } else {
    method_ostream = new MethodOutputStream(writer_, generic_namespace_);
  }
  return method_ostream->Begin(method, modifiers);
}

ClassOutputStream* ClassOutputStream::BeginInnerClass(const Type& clazz,
    int modifiers) {
  *this << endl;
  ClassOutputStream* class_ostream;
  if (modifiers & STATIC) {
    class_ostream = new ClassOutputStream(writer_);
  } else {
    class_ostream = new ClassOutputStream(writer_, generic_namespace_);
  }
  return class_ostream->Begin(clazz, modifiers);
}

void ClassOutputStream::EndClass() {
  *this << endb;
  delete this;
}

ClassOutputStream* ClassOutputStream::Begin(const Type& clazz, int modifiers) {
  WriteDoc(clazz.description(), writer_);
  if (!clazz.annotations().empty()) {
    WriteAnnotations(clazz.annotations(), writer_);
  }
  WriteModifiers(modifiers, writer_);
  *this << "class " << clazz.name();

  // Look for generics to declare with this class
  GenericTypeScanner generic_scanner(&generic_namespace_);
  ScanType(clazz, &generic_scanner);
  if (!generic_scanner.discoveredTypes().empty()) {
    WriteGenerics(generic_scanner.discoveredTypes(), writer_);
  }

  if (!clazz.supertypes().empty()) {
    std::deque<Type>::const_iterator first = clazz.supertypes().cbegin();
    if (first->kind() == Type::CLASS) {  // superclass is always first in list
      *this << " extends " << *first++;
    }
    for (std::deque<Type>::const_iterator it = first;
        it != clazz.supertypes().cend(); ++it) {
      *this << ((it == first) ? " implements " : ", ") << *it;
    }
  }
  *this << beginb;
  return this;
}

ClassOutputStream* SourceOutputStream::BeginClass(const Type& clazz,
    const std::vector<Type>* dependencies, int modifiers) {
  if (!clazz.package().empty()) {
    *this << "package " << clazz.package() << ";" << endl << endl;
  }
  if (dependencies != nullptr && !dependencies->empty()) {
    std::set<Type, Type::Comparator> imports;
    auto import_scanner = [&](const Type& type) {
      if (!type.package().empty() && type.package() != clazz.package()) {
        imports.insert(type);
      }
    };
    for (std::vector<Type>::const_iterator it = dependencies->cbegin();
        it != dependencies->cend(); ++it) {
      ScanType(*it, &import_scanner);
    }
    for (std::set<Type, Type::Comparator>::const_iterator it = imports.cbegin();
        it != imports.cend(); ++it) {
      *this << "import " << it->package() << "." << it->name() << ";" << endl;
    }
    *this << endl;
  }
  ClassOutputStream* class_ostream = new ClassOutputStream(writer_);
  return class_ostream->Begin(clazz, modifiers);
}

}  // namespace java
}  // namespace tensorflow
