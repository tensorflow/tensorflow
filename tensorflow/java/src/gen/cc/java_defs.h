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
#include <deque>

#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace java {

/// \brief An enumeration of different modifiers commonly used in Java
enum Modifier {
  PUBLIC    = (1 << 0),
  PROTECTED = (1 << 1),
  PRIVATE   = (1 << 2),
  STATIC    = (1 << 3),
  FINAL     = (1 << 4),
};

class Annotation;

/// \brief A definition of any kind of Java type (classes, interfaces...)
///
/// Note that most of the data fields of this class are only useful in specific
/// contexts and are not required in many cases. For example, annotations and
/// supertypes are only useful when declaring a type.
class Type {
 public:
  enum Kind {
    PRIMITIVE, CLASS, INTERFACE, ENUM, GENERIC, ANNOTATION
  };
  struct Comparator {
    bool operator() (const Type& type1, const Type& type2) {
      return type1.name_ < type2.name_ || type1.package_ < type2.package_;
    }
  };
  static Type Primitive(const string& name) {
    return Type(Type::PRIMITIVE, name, "");
  }
  static Type Class(const string& name, const string& package = "") {
    return Type(Type::CLASS, name, package);
  }
  static Type Interface(const string& name, const string& package = "") {
    return Type(Type::INTERFACE, name, package);
  }
  static Type Enum(const string& name, const string& package = "") {
    return Type(Type::ENUM, name, package);
  }
  static Type Generic(const string& name) {
    return Type(Type::GENERIC, name, "");
  }
  static Type Wildcard() {
    return Type(Type::GENERIC, "", "");
  }
  static Type ClassOf(const Type& type) {
    return Class("Class").param(type);
  }
  static Type ListOf(const Type& type) {
    return Interface("List", "java.util").param(type);
  }
  static Type IterableOf(const Type& type) {
    return Interface("Iterable").param(type);
  }
  const Kind& kind() const { return kind_; }
  const string& name() const { return name_; }
  const string& package() const { return package_; }
  const string& descr() const { return descr_; }
  Type& descr(const string& descr) { descr_ = descr; return *this; }
  const std::vector<Type>& params() const { return params_; }
  Type& param(const Type& param) {
    params_.push_back(param);
    return *this;
  }
  const std::vector<Annotation>& annotations() const { return annotations_; }
  Type& annotation(const Annotation& annotation) {
    annotations_.push_back(annotation);
    return *this;
  }
  const std::deque<Type>& supertypes() const { return supertypes_; }
  Type& supertype(const Type& type) {
    if (type.kind_ == CLASS) {
      supertypes_.push_front(type);  // keep superclass at the front of the list
    } else if (type.kind_ == INTERFACE) {
      supertypes_.push_back(type);
    }
    return *this;
  }
  /// Returns true if "type" is of a known collection type (only a few for now)
  bool IsCollection() const {
    return name_ == "List" || name_ == "Iterable";
  }
  /// Returns true if this instance is a wildcard (<?>)
  bool IsWildcard() const {
    return kind_ == GENERIC && name_.empty();
  }
  bool operator==(const Type& type) const {
      return name_ == type.name_ && package_ == type.package_;
  }
  bool operator!=(const Type& type) const { return !(*this == type); }

 protected:
  Type(Kind kind, const string& name, const string& package)
    : kind_(kind), name_(name), package_(package) {}

 private:
  Kind kind_;
  string name_;
  string package_;
  string descr_;
  std::vector<Type> params_;
  std::vector<Annotation> annotations_;
  std::deque<Type> supertypes_;
};

/// \brief Definition of a Java annotation
///
/// This class only defines the usage of an annotation in a specific context,
/// giving optionally a set of attributes to initialize.
class Annotation : public Type {
 public:
  static Annotation Of(const string& type_name, const string& package = "") {
    return Annotation(type_name, package);
  }
  const string& attrs() const { return attrs_; }
  Annotation& attrs(const string& attrs) { attrs_ = attrs; return *this; }

 private:
  string attrs_;

  Annotation(const string& name, const string& package)
    : Type(Kind::ANNOTATION, name, package) {}
};

/// \brief A definition of a Java variable
///
/// This class defines an instance of a type, which could be documented.
class Variable {
 public:
  static Variable Of(const string& name, const Type& type) {
    return Variable(name, type, false);
  }
  static Variable VarArg(const string& name, const Type& type) {
    return Variable(name, type, true);
  }
  const string& name() const { return name_; }
  const Type& type() const { return type_; }
  bool variadic() const { return variadic_; }
  const string& descr() const { return descr_; }
  Variable& descr(const string& descr) { descr_ = descr; return *this; }

 private:
  string name_;
  Type type_;
  bool variadic_;
  string descr_;

  Variable(const string& name, const Type& type, bool variadic)
    : name_(name), type_(type), variadic_(variadic) {}
};

/// \brief A definition of a Java class method
///
/// This class defines the signature of a method, including its name, return
/// type and arguments.
class Method {
 public:
  static Method Of(const string& name, const Type& return_type) {
    return Method(name, return_type, false);
  }
  static Method ConstructorFor(const Type& clazz) {
    return Method(clazz.name(), clazz, true);
  }
  bool constructor() const { return constructor_; }
  const string& name() const { return name_; }
  const Type& ret_type() const { return ret_type_; }
  const string& descr() const { return descr_; }
  Method& descr(const string& descr) { descr_ = descr; return *this; }
  const string& ret_descr() const { return ret_descr_; }
  Method& ret_descr(const string& descr) { ret_descr_ = descr; return *this; }
  const std::vector<Variable>& args() const { return args_; }
  Method& args(const std::vector<Variable>& args) {
    args_.insert(args_.cend(), args.cbegin(), args.cend());
    return *this;
  }
  Method& arg(const Variable& var) { args_.push_back(var); return *this; }
  const std::vector<Annotation>& annotations() const { return annotations_; }
  Method& annotation(const Annotation& annotation) {
    annotations_.push_back(annotation);
    return *this;
  }

 private:
  string name_;
  Type ret_type_;
  bool constructor_;
  string descr_;
  string ret_descr_;
  std::vector<Variable> args_;
  std::vector<Annotation> annotations_;

  Method(const string& name, const Type& ret_type, bool constructor)
    : name_(name), ret_type_(ret_type), constructor_(constructor) {}
};

/// \brief A piece of code to read from a file.
class Snippet {
 public:
  explicit Snippet(const string& fname, Env* env = Env::Default()) {
    TF_CHECK_OK(ReadFileToString(env, fname, &data_));
  }
  const string& data() const { return data_; }

 private:
  string data_;
};

// Templates implementation

template <typename TypeScanner>
void ScanForTypes(const Type& type, TypeScanner* scanner) {
  (*scanner)(type);
  for (auto it = type.params().cbegin(); it != type.params().cend(); ++it) {
    ScanForTypes(*it, scanner);
  }
  for (auto it = type.annotations().cbegin(); it != type.annotations().cend();
      ++it) {
    ScanForTypes(*it, scanner);
  }
  for (auto it = type.supertypes().cbegin(); it != type.supertypes().cend();
      ++it) {
    ScanForTypes(*it, scanner);
  }
}

template <typename TypeScanner>
void ScanForTypes(const Method& method, TypeScanner* scanner) {
  if (!method.constructor()) {
    ScanForTypes(method.ret_type(), scanner);
  }
  for (std::vector<Variable>::const_iterator arg = method.args().cbegin();
      arg != method.args().cend(); ++arg) {
    ScanForTypes(arg->type(), scanner);
  }
}

}  // namespace java
}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_DEFS_H_
