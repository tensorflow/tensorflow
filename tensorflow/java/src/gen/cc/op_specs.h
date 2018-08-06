/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_JAVA_SRC_GEN_CC_OP_SPECS_H_
#define TENSORFLOW_JAVA_SRC_GEN_CC_OP_SPECS_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"

namespace tensorflow {
namespace java {

constexpr const char kDefaultEndpointPackage[] = "core";

class EndpointSpec {
 public:
  // A specification for an operation endpoint
  //
  // package: package of this endpoint (from which also derives its package)
  // name: name of this endpoint class
  // javadoc: the endpoint class documentation
  // TODO(annarev): hardcode depcreated to false until deprecated is possible
  EndpointSpec(const string& package, const string& name,
               const Javadoc& javadoc)
      : package_(package), name_(name), javadoc_(javadoc), deprecated_(false) {}

  const string& package() const { return package_; }
  const string& name() const { return name_; }
  const Javadoc& javadoc() const { return javadoc_; }
  bool deprecated() const { return deprecated_; }

 private:
  const string package_;
  const string name_;
  const Javadoc javadoc_;
  const bool deprecated_;
};

class ArgumentSpec {
 public:
  // A specification for an operation argument
  //
  // op_def_name: argument name, as known by TensorFlow core
  // var: a variable to represent this argument in Java
  // type: the tensor type of this argument
  // description: a description of this argument, in javadoc
  // iterable: true if this argument is a list
  ArgumentSpec(const string& op_def_name, const Variable& var, const Type& type,
               const string& description, bool iterable)
      : op_def_name_(op_def_name),
        var_(var),
        type_(type),
        description_(description),
        iterable_(iterable) {}

  const string& op_def_name() const { return op_def_name_; }
  const Variable& var() const { return var_; }
  const Type& type() const { return type_; }
  const string& description() const { return description_; }
  bool iterable() const { return iterable_; }

 private:
  const string op_def_name_;
  const Variable var_;
  const Type type_;
  const string description_;
  const bool iterable_;
};

class AttributeSpec {
 public:
  // A specification for an operation attribute
  //
  // op_def_name: attribute name, as known by TensorFlow core
  // var: a variable to represent this attribute in Java
  // type: the type of this attribute
  // jni_type: the type of this attribute in JNI layer (see OperationBuilder)
  // description: a description of this attribute, in javadoc
  // iterable: true if this attribute is a list
  // has_default_value: true if this attribute has a default value if not set
  AttributeSpec(const string& op_def_name, const Variable& var,
                const Type& type, const Type& jni_type,
                const string& description, bool iterable,
                bool has_default_value)
      : op_def_name_(op_def_name),
        var_(var),
        type_(type),
        description_(description),
        iterable_(iterable),
        jni_type_(jni_type),
        has_default_value_(has_default_value) {}

  const string& op_def_name() const { return op_def_name_; }
  const Variable& var() const { return var_; }
  const Type& type() const { return type_; }
  const string& description() const { return description_; }
  bool iterable() const { return iterable_; }
  const Type& jni_type() const { return jni_type_; }
  bool has_default_value() const { return has_default_value_; }

 private:
  const string op_def_name_;
  const Variable var_;
  const Type type_;
  const string description_;
  const bool iterable_;
  const Type jni_type_;
  const bool has_default_value_;
};

class OpSpec {
 public:
  // Parses an op definition and its API to produce a specification used for
  // rendering its Java wrapper
  //
  // op_def: Op definition
  // api_def: Op API definition
  static OpSpec Create(const OpDef& op_def, const ApiDef& api_def);

  const string& graph_op_name() const { return graph_op_name_; }
  bool hidden() const { return hidden_; }
  const string& deprecation_explanation() const {
    return deprecation_explanation_;
  }
  const std::vector<EndpointSpec> endpoints() const { return endpoints_; }
  const std::vector<ArgumentSpec>& inputs() const { return inputs_; }
  const std::vector<ArgumentSpec>& outputs() const { return outputs_; }
  const std::vector<AttributeSpec>& attributes() const { return attributes_; }
  const std::vector<AttributeSpec>& optional_attributes() const {
    return optional_attributes_;
  }

 private:
  // A specification for an operation
  //
  // graph_op_name: name of this op, as known by TensorFlow core engine
  // hidden: true if this op should not be visible through the Graph Ops API
  // deprecation_explanation: message to show if all endpoints are deprecated
  explicit OpSpec(const string& graph_op_name, bool hidden,
                  const string& deprecation_explanation)
      : graph_op_name_(graph_op_name),
        hidden_(hidden),
        deprecation_explanation_(deprecation_explanation) {}

  const string graph_op_name_;
  const bool hidden_;
  const string deprecation_explanation_;
  std::vector<EndpointSpec> endpoints_;
  std::vector<ArgumentSpec> inputs_;
  std::vector<ArgumentSpec> outputs_;
  std::vector<AttributeSpec> attributes_;
  std::vector<AttributeSpec> optional_attributes_;
};

}  // namespace java
}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_OP_SPECS_H_
