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

#include <list>
#include <map>
#include <string>
#include <utility>

#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace java {

// An enumeration of different modifiers commonly used in Java
enum Modifier {
  PACKAGE = 0,
  PUBLIC = (1 << 0),
  PROTECTED = (1 << 1),
  PRIVATE = (1 << 2),
  STATIC = (1 << 3),
  FINAL = (1 << 4),
};

class Annotation;

// A definition of any kind of Java type (classes, interfaces...)
//
// Note that most of the data fields of this class are only useful in specific
// contexts and are not required in many cases. For example, annotations and
// supertypes are only useful when declaring a type.
class Type {
 public:
  enum Kind {
    PRIMITIVE, CLASS, INTERFACE, ENUM, GENERIC, ANNOTATION
  };
  static const Type Byte() {
    return Type(Type::PRIMITIVE, "byte");
  }
  static const Type Char() {
    return Type(Type::PRIMITIVE, "char");
  }
  static const Type Short() {
    return Type(Type::PRIMITIVE, "short");
  }
  static const Type Int() {
    return Type(Type::PRIMITIVE, "int");
  }
  static const Type Long() {
    return Type(Type::PRIMITIVE, "long");
  }
  static const Type Float() {
    return Type(Type::PRIMITIVE, "float");
  }
  static const Type Double() {
    return Type(Type::PRIMITIVE, "double");
  }
  static const Type Boolean() {
    return Type(Type::PRIMITIVE, "boolean");
  }
  static const Type Void() {
    // For simplicity, we consider 'void' as a primitive type, like the Java
    // Reflection API does
    return Type(Type::PRIMITIVE, "void");
  }
  static Type Generic(const std::string& name) {
    return Type(Type::GENERIC, name);
  }
  static Type Wildcard() { return Type(Type::GENERIC, ""); }
  static Type Class(const std::string& name, const std::string& package = "") {
    return Type(Type::CLASS, name, package);
  }
  static Type Interface(const std::string& name,
                        const std::string& package = "") {
    return Type(Type::INTERFACE, name, package);
  }
  static Type Enum(const std::string& name, const std::string& package = "") {
    return Type(Type::ENUM, name, package);
  }
  static Type ClassOf(const Type& type) {
    return Class("Class").add_parameter(type);
  }
  static Type ListOf(const Type& type) {
    return Interface("List", "java.util").add_parameter(type);
  }
  static Type IterableOf(const Type& type) {
    return Interface("Iterable").add_parameter(type);
  }
  static Type ForDataType(DataType data_type) {
    switch (data_type) {
      case DataType::DT_BOOL:
        return Class("Boolean");
      case DataType::DT_STRING:
        return Class("String");
      case DataType::DT_FLOAT:
        return Class("Float");
      case DataType::DT_DOUBLE:
        return Class("Double");
      case DataType::DT_UINT8:
        return Class("UInt8", "org.tensorflow.types");
      case DataType::DT_INT32:
        return Class("Integer");
      case DataType::DT_INT64:
        return Class("Long");
      case DataType::DT_RESOURCE:
        // TODO(karllessard) create a Resource utility class that could be
        // used to store a resource and its type (passed in a second argument).
        // For now, we need to force a wildcard and we will unfortunately lose
        // track of the resource type.
        // Falling through...
      default:
        // Any other datatypes does not have a equivalent in Java and must
        // remain a wildcard (e.g. DT_COMPLEX64, DT_QINT8, ...)
        return Wildcard();
    }
  }
  const Kind& kind() const { return kind_; }
  const std::string& name() const { return name_; }
  const std::string& package() const { return package_; }
  const std::string canonical_name() const {
    return package_.empty() ? name_ : package_ + "." + name_;
  }
  bool wildcard() const { return name_.empty(); }  // only wildcards has no name
  const std::list<Type>& parameters() const { return parameters_; }
  Type& add_parameter(const Type& parameter) {
    parameters_.push_back(parameter);
    return *this;
  }
  const std::list<Annotation>& annotations() const { return annotations_; }
  Type& add_annotation(const Annotation& annotation) {
    annotations_.push_back(annotation);
    return *this;
  }
  const std::list<Type>& supertypes() const { return supertypes_; }
  Type& add_supertype(const Type& type) {
    if (type.kind_ == CLASS) {
      supertypes_.push_front(type);  // keep superclass at the front of the list
    } else if (type.kind_ == INTERFACE) {
      supertypes_.push_back(type);
    }
    return *this;
  }

 protected:
  Type(Kind kind, const std::string& name, const std::string& package = "")
      : kind_(kind), name_(name), package_(package) {}

 private:
  Kind kind_;
  std::string name_;
  std::string package_;
  std::list<Type> parameters_;
  std::list<Annotation> annotations_;
  std::list<Type> supertypes_;
};

// Definition of a Java annotation
//
// This class only defines the usage of an annotation in a specific context,
// giving optionally a set of attributes to initialize.
class Annotation : public Type {
 public:
  static Annotation Create(const std::string& type_name,
                           const std::string& pkg = "") {
    return Annotation(type_name, pkg);
  }
  const std::string& attributes() const { return attributes_; }
  Annotation& attributes(const std::string& attributes) {
    attributes_ = attributes;
    return *this;
  }

 private:
  std::string attributes_;

  Annotation(const std::string& name, const std::string& package)
      : Type(Kind::ANNOTATION, name, package) {}
};

// A definition of a Java variable
//
// This class declares an instance of a type, such as a class field or a
// method argument, which can be documented.
class Variable {
 public:
  static Variable Create(const std::string& name, const Type& type) {
    return Variable(name, type, false);
  }
  static Variable Varargs(const std::string& name, const Type& type) {
    return Variable(name, type, true);
  }
  const std::string& name() const { return name_; }
  const Type& type() const { return type_; }
  bool variadic() const { return variadic_; }

 private:
  std::string name_;
  Type type_;
  bool variadic_;

  Variable(const std::string& name, const Type& type, bool variadic)
      : name_(name), type_(type), variadic_(variadic) {}
};

// A definition of a Java class method
//
// This class defines the signature of a method, including its name, return
// type and arguments.
class Method {
 public:
  static Method Create(const std::string& name, const Type& return_type) {
    return Method(name, return_type, false);
  }
  static Method ConstructorFor(const Type& clazz) {
    return Method(clazz.name(), clazz, true);
  }
  bool constructor() const { return constructor_; }
  const std::string& name() const { return name_; }
  const Type& return_type() const { return return_type_; }
  const std::list<Variable>& arguments() const { return arguments_; }
  Method& add_argument(const Variable& var) {
    arguments_.push_back(var);
    return *this;
  }
  const std::list<Annotation>& annotations() const { return annotations_; }
  Method& add_annotation(const Annotation& annotation) {
    annotations_.push_back(annotation);
    return *this;
  }

 private:
  std::string name_;
  Type return_type_;
  bool constructor_;
  std::list<Variable> arguments_;
  std::list<Annotation> annotations_;

  Method(const std::string& name, const Type& return_type, bool constructor)
      : name_(name), return_type_(return_type), constructor_(constructor) {}
};

// A definition of a documentation bloc for a Java element (JavaDoc)
class Javadoc {
 public:
  static Javadoc Create(const std::string& brief = "") {
    return Javadoc(brief);
  }
  const std::string& brief() const { return brief_; }
  const std::string& details() const { return details_; }
  Javadoc& details(const std::string& details) {
    details_ = details;
    return *this;
  }
  const std::list<std::pair<std::string, std::string>>& tags() const {
    return tags_;
  }
  Javadoc& add_tag(const std::string& tag, const std::string& text) {
    tags_.push_back(std::make_pair(tag, text));
    return *this;
  }
  Javadoc& add_param_tag(const std::string& name, const std::string& text) {
    return add_tag("param", name + " " + text);
  }

 private:
  std::string brief_;
  std::string details_;
  std::list<std::pair<std::string, std::string>> tags_;

  explicit Javadoc(const std::string& brief) : brief_(brief) {}
};

}  // namespace java
}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_JAVA_DEFS_H_
