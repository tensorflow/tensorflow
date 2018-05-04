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
#include <list>

#include "tensorflow/java/src/gen/cc/source_writer.h"

namespace tensorflow {
namespace java {

SourceWriter::SourceWriter() {
  // Push an empty generic namespace at start, for simplification.
  generic_namespaces_.push(new GenericNamespace());
}

SourceWriter::~SourceWriter() {
  // Remove empty generic namespace added at start as well as any other
  // namespace objects that haven't been removed.
  while (!generic_namespaces_.empty()) {
    GenericNamespace* generic_namespace = generic_namespaces_.top();
    generic_namespaces_.pop();
    delete generic_namespace;
  }
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

SourceWriter& SourceWriter::Write(const StringPiece& str) {
  size_t line_pos = 0;
  do {
    size_t start_pos = line_pos;
    line_pos = str.find('\n', start_pos);
    if (line_pos != string::npos) {
      ++line_pos;
      Append(str.substr(start_pos, line_pos - start_pos));
      newline_ = true;
    } else {
      Append(str.substr(start_pos, str.size() - start_pos));
    }
  } while (line_pos != string::npos && line_pos < str.size());

  return *this;
}

SourceWriter& SourceWriter::WriteFromFile(const string& fname, Env* env) {
  string data_;
  TF_CHECK_OK(ReadFileToString(env, fname, &data_));
  return Write(data_);
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

SourceWriter& SourceWriter::AppendType(const Type& type) {
  if (type.wildcard()) {
    Append("?");
  } else {
    Append(type.name());
    if (!type.parameters().empty()) {
      Append("<");
      bool first = true;
      for (const Type& t : type.parameters()) {
        if (!first) {
          Append(", ");
        }
        AppendType(t);
        first = false;
      }
      Append(">");
    }
  }
  return *this;
}

SourceWriter& SourceWriter::EndLine() {
  Append("\n");
  newline_ = true;
  return *this;
}

SourceWriter& SourceWriter::BeginBlock(const string& expression) {
  if (!expression.empty()) {
    Append(expression + " {");
  } else {
    Append(newline_ ? "{" : " {");
  }
  return EndLine().Indent(2);
}

SourceWriter& SourceWriter::EndBlock() {
  return Indent(-2).Append("}").EndLine();
}

SourceWriter& SourceWriter::BeginMethod(const Method& method, int modifiers,
    const Javadoc* javadoc) {
  GenericNamespace* generic_namespace = PushGenericNamespace(modifiers);
  if (!method.constructor()) {
    generic_namespace->Visit(method.return_type());
  }
  for (const Variable& v : method.arguments()) {
    generic_namespace->Visit(v.type());
  }
  EndLine();
  if (javadoc != nullptr) {
    WriteJavadoc(*javadoc);
  }
  if (!method.annotations().empty()) {
    WriteAnnotations(method.annotations());
  }
  WriteModifiers(modifiers);
  if (!generic_namespace->declared_types().empty()) {
    WriteGenerics(generic_namespace->declared_types());
    Append(" ");
  }
  if (!method.constructor()) {
    AppendType(method.return_type()).Append(" ");
  }
  Append(method.name()).Append("(");
  bool first = true;
  for (const Variable& v : method.arguments()) {
    if (!first) {
      Append(", ");
    }
    AppendType(v.type()).Append(v.variadic() ? "... " : " ").Append(v.name());
    first = false;
  }
  return Append(")").BeginBlock();
}

SourceWriter& SourceWriter::EndMethod() {
  EndBlock();
  PopGenericNamespace();
  return *this;
}

SourceWriter& SourceWriter::BeginType(const Type& type, int modifiers,
    const std::list<Type>* extra_dependencies, const Javadoc* javadoc) {
  if (!type.package().empty()) {
    Append("package ").Append(type.package()).Append(";").EndLine();
  }
  TypeImporter type_importer(type.package());
  type_importer.Visit(type);
  if (extra_dependencies != nullptr) {
    for (const Type& t : *extra_dependencies) {
      type_importer.Visit(t);
    }
  }
  if (!type_importer.imports().empty()) {
    EndLine();
    for (const string& s : type_importer.imports()) {
      Append("import ").Append(s).Append(";").EndLine();
    }
  }
  return BeginInnerType(type, modifiers, javadoc);
}

SourceWriter& SourceWriter::BeginInnerType(const Type& type, int modifiers,
    const Javadoc* javadoc) {
  GenericNamespace* generic_namespace = PushGenericNamespace(modifiers);
  generic_namespace->Visit(type);
  EndLine();
  if (javadoc != nullptr) {
    WriteJavadoc(*javadoc);
  }
  if (!type.annotations().empty()) {
    WriteAnnotations(type.annotations());
  }
  WriteModifiers(modifiers);
  CHECK_EQ(Type::Kind::CLASS, type.kind()) << ": Not supported yet";
  Append("class ").Append(type.name());
  if (!generic_namespace->declared_types().empty()) {
    WriteGenerics(generic_namespace->declared_types());
  }
  if (!type.supertypes().empty()) {
    bool first_interface = true;
    for (const Type& t : type.supertypes()) {
      if (t.kind() == Type::CLASS) {  // superclass is always first in list
        Append(" extends ");
      } else if (first_interface) {
        Append(" implements ");
        first_interface = false;
      } else {
        Append(", ");
      }
      AppendType(t);
    }
  }
  return BeginBlock();
}

SourceWriter& SourceWriter::EndType() {
  EndBlock();
  PopGenericNamespace();
  return *this;
}

SourceWriter& SourceWriter::WriteField(const Variable& field, int modifiers,
    const Javadoc* javadoc) {
  // If present, write field javadoc only as one brief line
  if (javadoc != nullptr && !javadoc->brief().empty()) {
    Append("/** ").Append(javadoc->brief()).Append(" */").EndLine();
  }
  WriteModifiers(modifiers);
  AppendType(field.type()).Append(" ").Append(field.name()).Append(";");
  EndLine();
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

SourceWriter& SourceWriter::WriteJavadoc(const Javadoc& javadoc) {
  Append("/**").Prefix(" * ").EndLine();
  bool do_line_break = false;
  if (!javadoc.brief().empty()) {
    Write(javadoc.brief()).EndLine();
    do_line_break = true;
  }
  if (!javadoc.details().empty()) {
    if (do_line_break) {
      Append("<p>").EndLine();
    }
    Write(javadoc.details()).EndLine();
    do_line_break = true;
  }
  if (!javadoc.tags().empty()) {
    if (do_line_break) {
      EndLine();
    }
    for (const auto& p : javadoc.tags()) {
      Append("@" + p.first);
      if (!p.second.empty()) {
        Append(" ").Write(p.second);
      }
      EndLine();
    }
  }
  return Prefix("").Append(" */").EndLine();
}

SourceWriter& SourceWriter::WriteAnnotations(
    const std::list<Annotation>& annotations) {
  for (const Annotation& a : annotations) {
    Append("@" + a.name());
    if (!a.attributes().empty()) {
      Append("(").Append(a.attributes()).Append(")");
    }
    EndLine();
  }
  return *this;
}

SourceWriter& SourceWriter::WriteGenerics(
    const std::list<const Type*>& generics) {
  Append("<");
  bool first = true;
  for (const Type* pt : generics) {
    if (!first) {
      Append(", ");
    }
    Append(pt->name());
    if (!pt->supertypes().empty()) {
      Append(" extends ").AppendType(pt->supertypes().front());
    }
    first = false;
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

void SourceWriter::TypeVisitor::Visit(const Type& type) {
  DoVisit(type);
  for (const Type& t : type.parameters()) {
    Visit(t);
  }
  for (const Annotation& t : type.annotations()) {
    DoVisit(t);
  }
  for (const Type& t : type.supertypes()) {
    Visit(t);
  }
}

void SourceWriter::GenericNamespace::DoVisit(const Type& type) {
  // ignore non-generic parameters, wildcards and generics already declared
  if (type.kind() == Type::GENERIC && !type.wildcard()
      && generic_names_.find(type.name()) == generic_names_.end()) {
    declared_types_.push_back(&type);
    generic_names_.insert(type.name());
  }
}

void SourceWriter::TypeImporter::DoVisit(const Type& type) {
  if (!type.package().empty() && type.package() != current_package_) {
    imports_.insert(type.canonical_name());
  }
}

}  // namespace java
}  // namespace tensorflow
