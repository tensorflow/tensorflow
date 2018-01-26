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
#include <algorithm>
#include <deque>

#include "tensorflow/java/src/gen/cc/source_writer.h"

namespace tensorflow {
namespace java {
namespace {

template <typename TypeVisitor>
void VisitType(const Type& type, TypeVisitor* visitor) {
  (*visitor)(type);
  for (auto it = type.parameters().cbegin(); it != type.parameters().cend();
      ++it) {
    VisitType(*it, visitor);
  }
  for (auto it = type.annotations().cbegin(); it != type.annotations().cend();
      ++it) {
    VisitType(*it, visitor);
  }
  for (auto it = type.supertypes().cbegin(); it != type.supertypes().cend();
      ++it) {
    VisitType(*it, visitor);
  }
}

}  // namespace

SourceWriter::SourceWriter() {
  // push an empty generic namespace at start, for simplification
  generic_namespaces_.push(new GenericNamespace());
}

SourceWriter& SourceWriter::Indent(int tab) {
  left_margin_.resize(
      std::max(static_cast<int>(left_margin_.size() + tab), 0), ' ');
  return *this;
}

SourceWriter& SourceWriter::Prefix(const char* line_prefix) {
  line_prefix_ = line_prefix;
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

// Writes the signature of a Java type.
SourceWriter& SourceWriter::Append(const Type& type) {
  if (type.kind() == Type::Kind::GENERIC && type.name().empty()) {
    Append("?");
  } else {
    Append(type.name());
  }
  if (!type.parameters().empty()) {
    Append("<");
    for (std::vector<Type>::const_iterator it = type.parameters().cbegin();
        it != type.parameters().cend(); ++it) {
      if (it != type.parameters().cbegin()) {
        Append(", ");
      }
      Append(*it);
    }
    Append(">");
  }
  return *this;
}

SourceWriter& SourceWriter::EndLine() {
  Append("\n");
  newline_ = true;
  return *this;
}

SourceWriter& SourceWriter::BeginMethod(const Method& method, int modifiers) {
  GenericNamespace* generic_namespace = PushGenericNamespace(modifiers);
  if (!method.constructor()) {
    VisitType(method.return_type(), generic_namespace);
  }
  for (std::vector<Variable>::const_iterator it = method.arguments().cbegin();
      it != method.arguments().cend(); ++it) {
    VisitType(it->type(), generic_namespace);
  }
  EndLine();
  WriteDoc(method.description(), method.return_description(),
      &method.arguments());
  if (!method.annotations().empty()) {
    WriteAnnotations(method.annotations());
  }
  WriteModifiers(modifiers);
  if (!generic_namespace->declared_types().empty()) {
    WriteGenerics(generic_namespace->declared_types());
    Append(" ");
  }
  if (!method.constructor()) {
    Append(method.return_type()).Append(" ");
  }
  Append(method.name()).Append("(");
  for (std::vector<Variable>::const_iterator it = method.arguments().cbegin();
      it != method.arguments().cend(); ++it) {
    if (it != method.arguments().cbegin()) {
      Append(", ");
    }
    Append(it->type()).Append(it->variadic() ? "... " : " ").Append(it->name());
  }
  return Append(")").BeginBlock();
}

SourceWriter& SourceWriter::EndMethod() {
  EndBlock();
  PopGenericNamespace();
  return *this;
}

SourceWriter& SourceWriter::BeginType(const Type& type,
    const std::vector<Type>* dependencies, int modifiers) {
  if (!type.package().empty()) {
    Append("package ").Append(type.package()).Append(";").EndLine();
  }
  if (dependencies != nullptr && !dependencies->empty()) {
    std::set<string> imports;
    auto import_scanner = [&](const Type& t) {
      if (!t.package().empty() && t.package() != type.package()) {
        imports.insert(t.package() + '.' + t.name());
      }
    };
    for (std::vector<Type>::const_iterator it = dependencies->cbegin();
        it != dependencies->cend(); ++it) {
      VisitType(*it, &import_scanner);
    }
    EndLine();
    for (std::set<string>::const_iterator it = imports.cbegin();
        it != imports.cend(); ++it) {
      Append("import ").Append(*it).Append(";").EndLine();
    }
  }
  return BeginInnerType(type, modifiers);
}

SourceWriter& SourceWriter::BeginInnerType(const Type& type, int modifiers) {
  GenericNamespace* generic_namespace = PushGenericNamespace(modifiers);
  VisitType(type, generic_namespace);
  EndLine();
  WriteDoc(type.description());
  if (!type.annotations().empty()) {
    WriteAnnotations(type.annotations());
  }
  WriteModifiers(modifiers);
  if (type.kind() != Type::Kind::CLASS) {
    // Add support for other kind of types only when required
    LOG(FATAL) << type.kind() << " types are not supported yet";
  }
  Append("class ").Append(type.name());
  if (!generic_namespace->declared_types().empty()) {
    WriteGenerics(generic_namespace->declared_types());
  }
  if (!type.supertypes().empty()) {
    std::deque<Type>::const_iterator first = type.supertypes().cbegin();
    if (first->kind() == Type::CLASS) {  // superclass is always first in list
      Append(" extends ").Append(*first++);
    }
    for (std::deque<Type>::const_iterator it = first;
        it != type.supertypes().cend(); ++it) {
      Append((it == first) ? " implements " : ", ").Append(*it);
    }
  }
  return BeginBlock();
}

SourceWriter& SourceWriter::EndType() {
  EndBlock();
  PopGenericNamespace();
  return *this;
}

SourceWriter& SourceWriter::WriteFields(
    const std::vector<Variable>& fields, int modifiers) {
  EndLine();
  for (std::vector<Variable>::const_iterator it = fields.cbegin();
      it != fields.cend(); ++it) {
    WriteModifiers(modifiers);
    Append(it->type()).Append(" ").Append(it->name()).Append(";").EndLine();
  }
  return *this;
}

SourceWriter& SourceWriter::WriteModifiers(int modifiers) {
  if (modifiers & PUBLIC) {
    Append("public ");
  } else if (modifiers & PROTECTED) {
    Append("protected ");
  } else if (modifiers & PRIVATE) {
    Append("private ");
  }
  if (modifiers & STATIC) {
    Append("static ");
  }
  if (modifiers & FINAL) {
    Append("final ");
  }
  return *this;
}

SourceWriter& SourceWriter::WriteDoc(const string& description,
    const string& return_description, const std::vector<Variable>* parameters) {
  if (description.empty() && return_description.empty()
      && (parameters == nullptr || parameters->empty())) {
    return *this;  // no doc to write
  }
  bool do_line_break = false;
  Append("/**").EndLine().Prefix(" * ");
  if (!description.empty()) {
    Write(description).EndLine();
    do_line_break = true;
  }
  if (parameters != nullptr && !parameters->empty()) {
    if (do_line_break) {
      EndLine();
      do_line_break = false;
    }
    for (std::vector<Variable>::const_iterator it = parameters->begin();
        it != parameters->end(); ++it) {
      Append("@param ").Append(it->name());
      if (!it->description().empty()) {
        Append(" ").Write(it->description());
      }
      EndLine();
    }
  }
  if (!return_description.empty()) {
    if (do_line_break) {
      EndLine();
      do_line_break = false;
    }
    Append("@return ").Write(return_description).EndLine();
  }
  return Prefix("").Append(" **/").EndLine();
}

SourceWriter& SourceWriter::WriteAnnotations(
    const std::vector<Annotation>& annotations) {
  for (std::vector<Annotation>::const_iterator it = annotations.cbegin();
      it != annotations.cend(); ++it) {
    Append("@" + it->name());
    if (!it->attributes().empty()) {
      Append("(").Append(it->attributes()).Append(")");
    }
    EndLine();
  }
  return *this;
}

SourceWriter& SourceWriter::WriteGenerics(
    const std::vector<const Type*>& generics) {
  Append("<");
  for (std::vector<const Type*>::const_iterator it = generics.cbegin();
      it != generics.cend(); ++it) {
    if (it != generics.cbegin()) {
      Append(", ");
    }
    Append((*it)->name());
    if (!(*it)->supertypes().empty()) {
      Append(" extends ").Append((*it)->supertypes().front());
    }
  }
  return Append(">");
}

SourceWriter::GenericNamespace* SourceWriter::PushGenericNamespace(
    int modifiers) {
  GenericNamespace* generic_namespace;
  if (modifiers & STATIC) {
    generic_namespace = new GenericNamespace();
  } else {
    generic_namespace = new GenericNamespace(generic_namespaces_.top());
  }
  generic_namespaces_.push(generic_namespace);
  return generic_namespace;
}

void SourceWriter::PopGenericNamespace() {
  GenericNamespace* generic_namespace = generic_namespaces_.top();
  generic_namespaces_.pop();
  delete generic_namespace;
}

void SourceWriter::GenericNamespace::operator()(const Type& type) {
  // ignore non-generic parameters, wildcards and generics already declared
  if (type.kind() == Type::GENERIC
      && !type.IsWildcard()
      && generic_names_.find(type.name()) == generic_names_.end()) {
    declared_types_.push_back(&type);
    generic_names_.insert(type.name());
  }
}

}  // namespace java
}  // namespace tensorflow
