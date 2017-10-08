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

#ifndef TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_DEFS_H_
#define TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_DEFS_H_

#include <string>
#include <vector>
#include <set>
#include <deque>

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {
namespace java {

/// \brief An enumeration of different modifiers commonly used in Java
enum JavaModifier {
  PUBLIC    = (1 << 0),
  PROTECTED = (1 << 1),
  PRIVATE   = (1 << 2),
  STATIC    = (1 << 3),
  FINAL     = (1 << 4),
};

/// \brief A definition of a Java documentation block
///
/// Any vector of parameters (@param) that should be included in this block
/// can be provided separately (e.g. a vector of documented variables, see
/// JavaVariable).
class JavaDoc {
 public:
  const string& brief() const { return brief_; }
  JavaDoc& brief(const string& brief) { brief_ = brief; return *this; }
  const string& description() const { return description_; }
  JavaDoc& description(const string& txt) { description_ = txt; return *this; }
  const string& value() const { return value_; }
  JavaDoc& value(const string& value) { value_ = value; return *this; }
  bool empty() const {
    return brief().empty() && description().empty() && value().empty();
  }

 private:
  string brief_;
  string description_;
  string value_;
};

class JavaAnnot;

/// \brief A definition of any kind of Java type (classes, interfaces...)
///
/// Note that most of the data fields of this class are only useful in specific
/// contexts and are not required in many cases. For example, annotations and
/// supertypes are only useful when declaring a type.
class JavaType {
 public:
  enum Kind {
    PRIMITIVE, CLASS, INTERFACE, GENERIC, ANNOTATION, NONE
  };
  JavaType() = default;
  const Kind& kind() const { return kind_; }
  const string& name() const { return name_; }
  const string& package() const { return package_; }
  const JavaDoc& doc() const { return doc_; }
  JavaDoc* doc_ptr() { return &doc_; }
  JavaType& doc(const JavaDoc& doc) { doc_ = doc; return *this; }
  const std::vector<JavaType>& params() const { return params_; }
  JavaType& param(const JavaType& param) {
    params_.push_back(param);
    return *this;
  }
  const std::vector<JavaAnnot>& annotations() const { return annotations_; }
  JavaType& annotation(const JavaAnnot& annt) {
    annotations_.push_back(annt);
    return *this;
  }
  const std::deque<JavaType>& supertypes() const { return supertypes_; }
  JavaType& supertype(const JavaType& type) {
    if (type.kind_ == CLASS) {
      supertypes_.push_front(type);  // keep superclass at the front of the list
    } else if (type.kind_ == INTERFACE) {
      supertypes_.push_back(type);
    }
    return *this;
  }
  bool empty() const { return kind_ == NONE; }

  /// Scans this type and any of its parameter types.
  template <class TypeScanner>
  void Scan(TypeScanner* scanner) const;

  /// For sets
  bool operator<(const JavaType& type) const { return name() < type.name(); }

 private:
  Kind kind_ = NONE;
  string name_;
  string package_;
  std::vector<JavaType> params_;
  std::vector<JavaAnnot> annotations_;
  std::deque<JavaType> supertypes_;
  JavaDoc doc_;

  explicit JavaType(Kind kind, const string& name = "", const string& pkg = "")
    : kind_(kind), name_(name), package_(pkg) {}

  friend class Java;
};

/// \brief Definition of a Java annotation
///
/// This class only defines the usage of an annotation in a specific context,
/// giving optionally a set of attributes to initialize.
class JavaAnnot {
 public:
  JavaAnnot() = default;
  const JavaType& type() const { return type_; }
  const string& attrs() const { return attrs_; }
  JavaAnnot& attrs(const string& attrs) { attrs_ = attrs; return *this; }

 private:
  JavaType type_;
  string attrs_;

  explicit JavaAnnot(const JavaType& type) : type_(type) {}

  friend class Java;
};

/// \brief A definition of a Java variable
///
/// This class defines an instance of a type, which could be documented.
class JavaVar {
 public:
  JavaVar() = default;
  const string& name() const { return name_; }
  const JavaType& type() const { return type_; }
  const JavaDoc& doc() const { return doc_; }
  JavaDoc* doc_ptr() { return &doc_; }
  JavaVar& doc(const JavaDoc& doc) { doc_ = doc; return *this; }

 private:
  string name_;
  JavaType type_;
  JavaDoc doc_;

  JavaVar(const string& name, const JavaType& type)
    : name_(name), type_(type) {}

  friend class Java;
};

/// \brief A definition of a Java class method
///
/// This class defines the signature of a method, including its name, return
/// type and arguments.
class JavaMethod {
 public:
  JavaMethod() = default;
  const string& name() const { return name_; }
  const JavaType& type() const { return type_; }
  const JavaDoc& doc() const { return doc_; }
  JavaDoc* doc_ptr() { return &doc_; }
  JavaMethod& doc(const JavaDoc& doc) { doc_ = doc; return *this; }
  const std::vector<JavaVar>& args() const { return args_; }
  JavaMethod& args(const std::vector<JavaVar>& args) {
    args_.insert(args_.cend(), args.cbegin(), args.cend());
    return *this;
  }
  JavaMethod& arg(const JavaVar& var) { args_.push_back(var); return *this; }
  const std::vector<JavaAnnot>& annotations() const { return annotations_; }
  JavaMethod& annotation(const JavaAnnot& annt) {
    annotations_.push_back(annt);
    return *this;
  }

  /// Scans all types found in the signature of this method.
  template <class TypeScanner>
  void ScanTypes(TypeScanner* scanner, bool scan_return_type) const;

 private:
  string name_;
  JavaType type_;
  std::vector<JavaVar> args_;
  std::vector<JavaAnnot> annotations_;
  JavaDoc doc_;

  explicit JavaMethod(const string& name) : name_(name) {}
  JavaMethod(const string& name, const JavaType& type)
    : name_(name), type_(type) {}

  friend class Java;
};

/// \brief A factory of common Java definitions and other utilities.
class Java {
 public:
  /// Returns the definition of a Java primitive type
  static JavaType Type(const string& name) {
    return JavaType(JavaType::PRIMITIVE, name);
  }
  /// Returns the definition of a Java class
  static JavaType Class(const string& name, const string& package = "") {
    return JavaType(JavaType::CLASS, name, package);
  }
  /// Returns the definition of a Java interface
  static JavaType Interface(const string& name, const string& package = "") {
    return JavaType(JavaType::INTERFACE, name, package);
  }
  /// Returns the definition of Java generic type parameter
  static JavaType Generic(const string& name) {
    return JavaType(JavaType::GENERIC, name);
  }
  /// Returns the definition of a Java wildcard type parameter (<?>)
  static JavaType Wildcard() {
    return JavaType(JavaType::GENERIC);
  }
  /// Returns the definition of a Java annotation
  static JavaAnnot Annot(const string& type_name, const string& pkg = "") {
    return JavaAnnot(JavaType(JavaType::ANNOTATION, type_name, pkg));
  }
  /// Returns the definition of Java variable
  static JavaVar Var(const string& name, const JavaType& type) {
    return JavaVar(name, type);
  }
  /// Returns the definition of Java class method
  static JavaMethod Method(const string& name, const JavaType& return_type) {
    return JavaMethod(name, return_type);
  }
  /// Returns the definition of a Java class constructor
  static JavaMethod ConstructorFor(const JavaType& clazz) {
    return JavaMethod(clazz.name());
  }
  /// Returns the definition of the class of "type" (Class<type>)
  static JavaType ClassOf(const JavaType& type) {
    return Class("Class").param(type);
  }
  /// Returns the definition of a list of "type" (List<type>)
  static JavaType ListOf(const JavaType& type) {
    return Interface("List", "java.util").param(type);
  }
  /// Returns the definition of a iteration of "type" (Iterable<type>)
  static JavaType IterableOf(const JavaType& type) {
    return Interface("Iterable").param(type);
  }
  /// Returns true if "type" is a wildcard type parameter (<?>)
  static bool IsWildcard(const JavaType& type) {
    return type.kind() == JavaType::GENERIC && type.name().empty();
  }
  /// Returns true if "type" is of a known collection type (only a few for now)
  static bool IsCollection(const JavaType& type) {
    return type.name() == "List" || type.name() == "Iterable";
  }
};

/// \brief A function used to collect generic type parameters discovered while
///        scanning an object for types (e.g. JavaMethod::ScanTypes)
class GenericTypeScanner {
 public:
  explicit GenericTypeScanner(std::set<string>* declared_names)
    : declared_names_(declared_names) {}
  const std::vector<const JavaType*>& discoveredTypes() const {
    return discovered_types_;
  }
  void operator()(const JavaType* type) {
    if (type->kind() == JavaType::GENERIC && !type->name().empty()
        && (declared_names_->find(type->name()) == declared_names_->end())) {
      discovered_types_.push_back(type);
      declared_names_->insert(type->name());
    }
  }
 private:
  std::vector<const JavaType*> discovered_types_;
  std::set<string>* declared_names_;
};

// Templates implementation

template <class TypeScanner>
void JavaType::Scan(TypeScanner* scanner) const {
  (*scanner)(this);
  for (std::vector<JavaType>::const_iterator it = params_.cbegin();
      it != params_.cend(); ++it) {
    it->Scan(scanner);
  }
  for (std::vector<JavaAnnot>::const_iterator it = annotations_.cbegin();
      it != annotations_.cend(); ++it) {
    it->type().Scan(scanner);
  }
  for (std::deque<JavaType>::const_iterator it = supertypes_.cbegin();
      it != supertypes_.cend(); ++it) {
    it->Scan(scanner);
  }
}

template <class TypeScanner>
void JavaMethod::ScanTypes(TypeScanner* scanner, bool args_only) const {
  if (!args_only && !type_.empty()) {
    type_.Scan(scanner);
  }
  for (std::vector<JavaVar>::const_iterator arg = args_.cbegin();
      arg != args_.cend(); ++arg) {
    arg->type().Scan(scanner);
  }
}

}  // namespace java
}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_DEFS_H_
