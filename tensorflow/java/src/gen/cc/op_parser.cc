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

#include <map>
#include <vector>
#include <utility>
#include <string>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/java/src/gen/cc/op_parser.h"

namespace tensorflow {
namespace java {
namespace {

string SnakeToCamelCase(const string& str, bool upper = false) {
  string result;
  bool cap = upper;
  for (string::const_iterator it = str.begin(); it != str.end(); ++it) {
    const char c = *it;
    if (c == '_') {
      cap = true;
    } else if (cap) {
      result += toupper(c);
      cap = false;
    } else {
      result += c;
    }
  }
  return result;
}

bool IsRealNumber(DataType type) {
  for (DataType dt : RealNumberTypes()) {
    if (type == dt) {
      return true;
    }
  }
  return false;
}

bool IsRealNumbers(const AttrValue& values) {
  if (values.has_list()) {
    for (int i = 0; i < values.list().type_size(); ++i) {
      if (!IsRealNumber(values.list().type(i))) {
        return false;
      }
    }
    return true;
  }
  return IsRealNumber(values.type());
}

string ParseDocumentation(const string& text) {
  std::stringstream javadoc_text;
  string::const_iterator c_iter = text.cbegin();
  bool code = false;
  bool emphasis = false;
  bool list = false;
  while (c_iter != text.cend()) {
    char c = *c_iter++;
    int count = 1;
    switch (c) {
    case '\n':
      if (!code) {
        // consumes all subsequent newlines, if there are more than one,
        // then there are two choices:
        // - if the next line starts with an asterisk, we are enumerating
        //   a list of items
        // - otherwise, we are starting a new paragraph
        for (; c_iter != text.cend() && *c_iter == '\n'; ++count, ++c_iter) {}
        if (c_iter != text.cend()) {
          if (count > 1) {
            if (*c_iter != '*' && list) {
              javadoc_text << "</li>\n</ul>\n";
              list = false;
            } else if (*c_iter == '*' && !list) {
              javadoc_text << "\n<ul>\n<li>";
              list = true;
              c_iter++;
            } else {
              javadoc_text << "\n<p>\n";
            }
          } else if (list && *c_iter == '*') {
            javadoc_text << "</li>\n<li>";
            c_iter++;
          } else {
            javadoc_text << '\n';
          }
        }
      }
      break;
    case '`':
      // consumes all subsequent backquotes, those are use enclose code.
      // if there are more than 3, we are dealing with a pre-formatted block,
      // otherwise it is a single-line code snippet
      for (; c_iter != text.cend() && *c_iter == '`'; ++count, ++c_iter) {}
      if (count >= 3) {
        javadoc_text << (code ? "\n}</pre>" : "<pre>{@code\n");
      } else {
        javadoc_text << (code ? "}" : "{@code ");
      }
      code = !code;
      break;
    case '*':
      if (!code) {
        // consumes all subsequent asterisks, if there are more than one, then
        // we put the text in bold, otherwise in italic
        for (; c_iter != text.cend() && *c_iter == '*'; ++count, ++c_iter) {}
        if (count > 1) {
          javadoc_text << (emphasis ? "</b>" : "<b>");
        } else {
          javadoc_text << (emphasis ? "</i>" : "<i>");
        }
        emphasis = !emphasis;
      } else {
        javadoc_text << '*';
      }
      break;
    default:
      javadoc_text << c;
      break;
    }
  }
  return javadoc_text.str();
}

}  // namespace

OpParser::OpParser(const OpDef& op_def, const ApiDef& api_def,
    const string& lib_name, const string& base_package)
  : op_def_(op_def), op_api_(api_def), lib_name_(lib_name),
    base_package_(base_package) {
}

void OpParser::Parse(std::unique_ptr<OpSpec>* op_ptr) {
  visited_attrs_.clear();
  next_generic_ = 'T';
  op_ptr->reset(new OpSpec(op_api_.graph_op_name()));
  for (const string& next_input_name : op_api_.arg_order()) {
    for (int i = 0; i < op_def_.input_arg().size(); ++i) {
      if (op_def_.input_arg(i).name() == next_input_name) {
        ParseInput(op_def_.input_arg(i), op_api_.in_arg(i), op_ptr->get());
        break;
      }
    }
  }
  for (int i = 0; i < op_def_.attr().size(); ++i) {
    ParseAttribute(op_def_.attr(i), op_api_.attr(i), op_ptr->get());
  }
  for (int i = 0; i < op_def_.output_arg().size(); ++i) {
    ParseOutput(op_def_.output_arg(i), op_api_.out_arg(i), op_ptr->get());
  }
  BuildEndpoints(op_ptr->get());
}

void OpParser::BuildEndpoints(OpSpec* op) {
  Javadoc op_doc = Javadoc::Create(ParseDocumentation(op_api_.summary()))
    .details(ParseDocumentation(op_api_.description()));
  std::vector<Type> op_supertypes;
  op_supertypes.push_back(Type::Class("PrimitiveOp", "org.tensorflow.op"));
  std::map<string, const Type*> op_generics;
  for (const OpSpec::Operand& output : op->outputs()) {
    // declare generic output parameters at the Op class level
    const Type& data_type = output.data_type();
    if (data_type.kind() == Type::GENERIC && !data_type.unknown()
        && op_generics.find(data_type.name()) == op_generics.end()) {
      op_generics.insert(std::make_pair(data_type.name(), &data_type));
      op_doc.add_param_tag("<" + data_type.name() + ">",
          "data type of output '" + output.var().name() + "'");
    }
    // implement the Op as an (iteration of) Operand if it has only one output
    if (op->outputs().size() == 1) {
      Type operand_inf(Type::Interface("Operand", "org.tensorflow"));
      operand_inf.add_parameter(data_type.unknown() ?
          Type::Class("Object") : data_type);
      op_supertypes.push_back(output.iterable() ?
          Type::IterableOf(operand_inf) : operand_inf);
    }
  }
  for (const auto& endpoint_def : op_api_.endpoint()) {
    std::vector<string> name_tokens = str_util::Split(endpoint_def.name(), ".");
    // if the endpoint specifies a package, use it, otherwise derive it from the
    // op library name.
    string name;
    string package;
    if (name_tokens.size() > 1) {
      package = str_util::Lowercase(name_tokens.at(0));
      name = name_tokens.at(1);
    } else {
      package = str_util::StringReplace(lib_name_, "_", "", true);
      name = name_tokens.at(0);
    }
    Type endpoint(Type::Class(name, base_package_ + "." + package));
    Javadoc endpoint_doc(op_doc);
    for (const auto& parameter : op_generics) {
      endpoint.add_parameter(*parameter.second);
    }
    for (const Type& supertype : op_supertypes) {
      endpoint.add_supertype(supertype);
    }
    if (endpoint_def.deprecation_version() > 0) {
      string explanation;
      if (op_api_.endpoint(0).deprecation_version() == 0) {
        explanation = ", use {@link "
            + op->endpoints().at(0).type().full_name()
            + "} instead";
      } else {
        explanation = op_def_.deprecation().explanation();
      }
      endpoint_doc.add_tag("deprecated", explanation);
      endpoint.add_annotation(Annotation::Create("Deprecated"));
    }
    // only visible ops should be annotated for exposure in the Ops Graph API
    if (op_api_.visibility() != ApiDef::HIDDEN) {
      string group_name = SnakeToCamelCase(lib_name_);
      endpoint.add_annotation(
          Annotation::Create("Operator", "org.tensorflow.op.annotation")
            .attributes("group = \"" + group_name + "\""));
    }
    op->add_endpoint(endpoint, endpoint_doc);
  }
}

void OpParser::ParseInput(const OpDef_ArgDef& input_def,
    const ApiDef::Arg& input_api, OpSpec* op) {
  bool iterable = false;
  Type data_type = DataTypeOf(input_def, &iterable);
  Type type = Type::Interface("Operand", "org.tensorflow")
    .add_parameter(data_type);
  if (iterable) {
    type = Type::IterableOf(type);
  }
  op->add_input(OpSpec::Operand(input_api.name(),
      Variable::Create(SnakeToCamelCase(input_api.rename_to()), type),
      data_type,
      ParseDocumentation(input_api.description()),
      iterable));
}

void OpParser::ParseOutput(const OpDef_ArgDef& output_def,
    const ApiDef::Arg& output_api, OpSpec* op) {
  bool iterable = false;
  Type data_type = DataTypeOf(output_def, &iterable);
  Type type = Type::Class("Output", "org.tensorflow")
    .add_parameter(data_type);
  if (iterable) {
    type = Type::ListOf(type);
  }
  op->add_output(OpSpec::Operand(output_api.name(),
      Variable::Create(SnakeToCamelCase(output_api.rename_to()), type),
      data_type,
      ParseDocumentation(output_api.description()),
      iterable));
}

void OpParser::ParseAttribute(const OpDef_AttrDef& attr_def,
    const ApiDef::Attr& attr_api, OpSpec* op) {
  // do not parse attributes already visited, they have probably been inferred
  // before as an input argument type
  if (visited_attrs_.find(attr_def.name()) != visited_attrs_.cend()) {
    return;
  }
  bool iterable = false;
  Type data_type = DataTypeOf(attr_def, &iterable);
  // generic attributes should be passed as an explicit type
  bool explicit_type = data_type.kind() == Type::GENERIC && !iterable;
  Type type = explicit_type ?
      Type::Class("Class").add_parameter(data_type) : data_type;
  if (iterable) {
    type = Type::ListOf(data_type);
  }
  OpSpec::Operand attr(attr_api.name(),
      Variable::Create(SnakeToCamelCase(attr_api.rename_to()), type),
      data_type,
      ParseDocumentation(attr_api.description()),
      iterable);
  // attributes with a default value are optional
  if (attr_api.has_default_value() && !explicit_type) {
    op->add_option(attr);
  } else {
    op->add_attribute(attr);
  }
  visited_attrs_.insert(std::make_pair(attr_api.name(), data_type));
}

Type OpParser::DataTypeOf(const OpDef_ArgDef& arg, bool* iterable_out) {
  if (!arg.number_attr().empty()) {
    visited_attrs_.insert(std::make_pair(arg.number_attr(), Type::Int()));
    *iterable_out = true;
  }
  if (arg.type() != DataType::DT_INVALID) {
    // resolve type from DataType
    switch (arg.type()) {
      case DataType::DT_BOOL:
        return Type::Class("Boolean");

      case DataType::DT_STRING:
        return Type::Class("String");

      case DataType::DT_FLOAT:
        return Type::Class("Float");

      case DataType::DT_DOUBLE:
        return Type::Class("Double");

      case DataType::DT_UINT8:
        return Type::Class("UInt8", "org.tensorflow.types");

      case DataType::DT_INT32:
        return Type::Class("Integer");

      case DataType::DT_INT64:
        return Type::Class("Long");

      case DataType::DT_RESOURCE:
        // TODO(karllessard) create a Resource utility class that could be
        // used to store a resource and its type (passed in a second argument).
        // For now, we need to force a wildcard and we will unfortunately lose
        // track of the resource type.
        return Type::Wildcard();

      default:
        break;
    }
  } else {
    // resolve type from type attribute
    string attr_name = arg.type_attr();
    if (attr_name.empty()) {
      attr_name = arg.type_list_attr();
      if (!attr_name.empty()) {
        *iterable_out = true;
        Type type = Type::Wildcard();
        visited_attrs_.insert(std::make_pair(attr_name, type));
        return type;
      }
    }
    for (const auto& attr : op_def_.attr()) {
      if (attr.name() == attr_name) {
        Type type = DataTypeOf(attr, iterable_out);
        visited_attrs_.insert(std::make_pair(attr_name, type));
        return type;
      }
    }
  }
  LOG(WARNING) << "Data type for arg \"" << arg.name() << "\" is unknown";
  return Type::Wildcard();
}

Type OpParser::DataTypeOf(const OpDef_AttrDef& attr, bool* iterable_out) {
  std::map<string, Type>::const_iterator it = visited_attrs_.find(attr.name());
  if (it != visited_attrs_.cend()) {
    return it->second;
  }
  string attr_type = attr.type();
  if (attr.type().compare(0, 5, "list(") == 0) {
    attr_type = attr_type.substr(5, attr.type().find_last_of(')') - 5);
    *iterable_out = true;
  }
  if (attr_type == "type") {
    if (*iterable_out) {
      return Type::Enum("DataType", "org.tensorflow");
    }
    return GetNextGenericTensorType(attr.allowed_values());
  }
  if (attr_type == "string") {
    return Type::Class("String");
  }
  if (attr_type == "int") {
    return Type::Class("Integer");
  }
  if (attr_type == "float") {
    return Type::Class("Float");
  }
  if (attr_type == "bool") {
    return Type::Class("Boolean");
  }
  if (attr_type == "shape") {
    return Type::Class("Shape", "org.tensorflow");
  }
  if (attr_type == "tensor") {
    return Type::Class("Tensor", "org.tensorflow")
      .add_parameter(Type::Wildcard());
  }
  LOG(WARNING) << "Data type for attribute \"" << attr_type << "\" is unknown";
  return *iterable_out ? Type::Wildcard() : Type::Class("Object");
}

Type OpParser::GetNextGenericTensorType(const AttrValue& allowed_values)  {
  Type generic = Type::Generic(string(1, next_generic_));
  next_generic_ = (next_generic_ == 'Z') ? 'A' : next_generic_ + 1;

  // when only real numbers are allowed, enforce that restriction in the Java by
  // extending the generic from java.lang.Number
  if (IsRealNumbers(allowed_values)) {
    generic.add_supertype(Type::Class("Number"));
  }
  return generic;
}

}  // namespace java
}  // namespace tensorflow
