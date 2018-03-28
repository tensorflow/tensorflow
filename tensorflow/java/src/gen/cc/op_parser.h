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

#ifndef TENSORFLOW_JAVA_SRC_GEN_CC_OP_PARSER_H_
#define TENSORFLOW_JAVA_SRC_GEN_CC_OP_PARSER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"

namespace tensorflow {
namespace java {

// Specification of a TensorFlow operation to generate.
//
// This is the result of an operation definition parsing, see OpParser::Parse().
class OpSpec {
 public:
  class Endpoint {
   public:
    Endpoint(const Type& type, const Javadoc& javadoc)
      : type_(type), javadoc_(javadoc) {}
    const Type& type() const { return type_; }
    const Javadoc& javadoc() const { return javadoc_; }

   private:
    Type type_;
    Javadoc javadoc_;
  };

  class Operand {
   public:
    Operand(const string& graph_name, const Variable& var,
        const Type& data_type, const string& description, bool iterable)
     : graph_name_(graph_name), var_(var), data_type_(data_type),
       description_(description), iterable_(iterable) {}
    const string& graph_name() const { return graph_name_; }
    const Variable& var() const { return var_; }
    Variable* var_ptr() { return &var_; }
    const Type& data_type() const { return data_type_; }
    const string& description() const { return description_; }
    bool iterable() const { return iterable_; }

   private:
    string graph_name_;
    Variable var_;
    Type data_type_;
    string description_;
    bool iterable_;
  };

  explicit OpSpec(const string& graph_name) : graph_name_(graph_name) {}
  const string& graph_name() const { return graph_name_; }
  const std::vector<Endpoint> endpoints() const { return endpoints_; }
  void add_endpoint(const Type& type, const Javadoc& javadoc) {
    endpoints_.push_back(Endpoint(type, javadoc));
  }
  const std::vector<Operand>& inputs() const { return inputs_; }
  void add_input(const Operand& input) {
    inputs_.push_back(input);
  }
  const std::vector<Operand>& outputs() const { return outputs_; }
  void add_output(const Operand& output) {
    outputs_.push_back(output);
  }
  const std::vector<Operand>& attributes() const { return attributes_; }
  void add_attribute(const Operand& attribute) {
    attributes_.push_back(attribute);
  }
  const std::vector<Operand>& options() const { return options_; }
  void add_option(const Operand& option) {
    options_.push_back(option);
  }

 private:
  string graph_name_;
  std::vector<Endpoint> endpoints_;
  std::vector<Operand> inputs_;
  std::vector<Operand> outputs_;
  std::vector<Operand> attributes_;
  std::vector<Operand> options_;
};

// A parser of ops proto definitions.
//
// This object parses the definition and the api of an TensorFlow operation to
// produce a specification that can be used for Java source code rendering.
class OpParser {
 public:
  OpParser(const OpDef& op_def, const ApiDef& api_def, const string& lib_name,
      const string& base_package);
  virtual ~OpParser() = default;

  // Produces an operation specification from its proto definitions.
  void Parse(std::unique_ptr<OpSpec>* op_ptr);

 private:
  OpDef op_def_;
  ApiDef op_api_;
  string lib_name_;
  string base_package_;
  std::map<string, Type> visited_attrs_;
  char next_generic_ = 0;

  void BuildEndpoints(OpSpec* op);
  void ParseInput(const OpDef_ArgDef& input_def,
      const ApiDef::Arg& input_api, OpSpec* op);
  void ParseOutput(const OpDef_ArgDef& output_def,
      const ApiDef::Arg& output_api, OpSpec* op);
  void ParseAttribute(const OpDef_AttrDef& attr_def,
      const ApiDef::Attr& attr_api, OpSpec* op);
  Type DataTypeOf(const OpDef_ArgDef& arg_def, bool *iterable_out);
  Type DataTypeOf(const OpDef_AttrDef& attr_def, bool *iterable_out);
  Type GetNextGenericTensorType(const AttrValue& allowed_values);
};

}  // namespace java
}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_OP_PARSER_H_
