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

#include <map>
#include <vector>
#include <string>
#include <utility>

#include "re2/re2.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/java/src/gen/cc/op_specs.h"

namespace tensorflow {
namespace java {
namespace {

inline bool IsRealNumbers(const AttrValue& values) {
  if (!values.has_list()) {
    return RealNumberTypes().Contains(values.type());
  }
  for (int i = 0; i < values.list().type_size(); ++i) {
    if (!RealNumberTypes().Contains(values.list().type(i))) {
      return false;
    }
  }
  return true;
}

class TypeResolver {
 public:
  explicit TypeResolver(const OpDef& op_def) : op_def_(op_def) {}

  Type TypeOf(const OpDef_ArgDef& arg_def, bool *iterable_out);
  Type TypeOf(const OpDef_AttrDef& attr_def, bool *iterable_out);
  bool IsAttributeVisited(const string& attr_name) {
    return visited_attrs_.find(attr_name) != visited_attrs_.cend();
  }
 private:
  const OpDef op_def_;
  std::map<std::string, Type> visited_attrs_;
  char next_generic_ = 'T';
};

Type TypeResolver::TypeOf(const OpDef_ArgDef& arg_def,
    bool* iterable_out) {
  *iterable_out = false;
  if (!arg_def.number_attr().empty()) {
    // when number_attr is set, argument has to be a list of tensors
    *iterable_out = true;
    visited_attrs_.insert(std::make_pair(arg_def.number_attr(), Type::Int()));
  }
  Type type = Type::Wildcard();
  if (arg_def.type() != DataType::DT_INVALID) {
    // resolve type from DataType
    switch (arg_def.type()) {
      case DataType::DT_BOOL:
        type = Type::Class("Boolean");
        break;
      case DataType::DT_STRING:
        type = Type::Class("String");
        break;
      case DataType::DT_FLOAT:
        type = Type::Class("Float");
        break;
      case DataType::DT_DOUBLE:
        type = Type::Class("Double");
        break;
      case DataType::DT_UINT8:
        type = Type::Class("UInt8", "org.tensorflow.types");
        break;
      case DataType::DT_INT32:
        type = Type::Class("Integer");
        break;
      case DataType::DT_INT64:
        type = Type::Class("Long");
        break;
      case DataType::DT_RESOURCE:
        // TODO(karllessard) create a Resource utility class that could be
        // used to store a resource and its type (passed in a second argument).
        // For now, we need to force a wildcard and we will unfortunately lose
        // track of the resource type.
        break;
      default:
        // Any other datatypes does not have a equivalent in Java and must
        // remain a wildcard (e.g. DT_COMPLEX64, DT_QINT8, ...)
        break;
    }
  } else if (!arg_def.type_attr().empty()) {
    // resolve type from attribute (if already visited, retrieve its type)
    if (IsAttributeVisited(arg_def.type_attr())) {
      type = visited_attrs_.at(arg_def.type_attr());
    } else {
      for (const auto& attr_def : op_def_.attr()) {
        if (attr_def.name() == arg_def.type_attr()) {
          type = TypeOf(attr_def, iterable_out);
          break;
        }
      }
    }
  } else if (!arg_def.type_list_attr().empty()) {
    // type is a list of tensors that can be of different data types, so leave
    // it as a list of wildcards
    *iterable_out = true;
    visited_attrs_.insert(std::make_pair(arg_def.type_list_attr(), type));

  } else {
    LOG(FATAL) << "Cannot resolve data type of argument \"" << arg_def.name()
        << "\" in operation \"" << op_def_.name() << "\"";
  }
  return type;
}

Type TypeResolver::TypeOf(const OpDef_AttrDef& attr_def,
    bool* iterable_out) {
  *iterable_out = false;
  StringPiece attr_type = attr_def.type();
  if (str_util::ConsumePrefix(&attr_type, "list(")) {
    attr_type.remove_suffix(1);  // remove closing brace
    *iterable_out = true;
  }
  Type type = *iterable_out ? Type::Wildcard() : Type::Class("Object");
  if (attr_type == "type") {
    if (*iterable_out) {
      type = Type::Enum("DataType", "org.tensorflow");
    } else {
      type = Type::Generic(string(1, next_generic_));
      next_generic_ = (next_generic_ == 'Z') ? 'A' : next_generic_ + 1;
      if (IsRealNumbers(attr_def.allowed_values())) {
        // enforce real numbers datasets by extending java.lang.Number
        type.add_supertype(Type::Class("Number"));
      }
    }
  } else if (attr_type == "string") {
    type = Type::Class("String");

  } else if (attr_type == "int") {
    type = Type::Class("Integer");

  } else if (attr_type == "float") {
    type = Type::Class("Float");

  } else if (attr_type == "bool") {
    type = Type::Class("Boolean");

  } else if (attr_type == "shape") {
    type = Type::Class("Shape", "org.tensorflow");

  } else if (attr_type == "tensor") {
    type = Type::Class("Tensor", "org.tensorflow")
        .add_parameter(Type::Wildcard());

  } else {
    LOG(FATAL) << "Cannot resolve data type for attribute \"" << attr_type
        << "\" in operation \"" << op_def_.name() << "\"";
  }
  visited_attrs_.insert(std::make_pair(attr_def.name(), type));
  return type;
}

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

bool FindAndCut(re2::StringPiece* input, const RE2& expr,
    re2::StringPiece* before_match, re2::StringPiece* ret_match = nullptr) {
  re2::StringPiece match;
  if (!expr.Match(*input, 0, input->size(), RE2::UNANCHORED, &match, 1)) {
    return false;
  }
  before_match->set(input->data(), match.begin() - input->begin());
  input->remove_prefix(match.end() - before_match->begin());
  if (ret_match != nullptr) {
    *ret_match = match;
  }
  return true;
}

string ParseDocumentation(re2::StringPiece input) {
  // TODO(karllessard) This is a very minimalist utility method for converting
  // markdown syntax, as found in ops descriptions, to Javadoc/html tags. Check
  // for alternatives to increase the level of support for markups.
  std::stringstream javadoc_text;
  bool in_list = false;
  const RE2 markup_expr(
      "\n+\\*[[:blank:]]+|\n{2,}|`{3,}|`{1,2}|\\*{1,2}\\b|\\[");
  while (true) {
    re2::StringPiece text;
    re2::StringPiece markup;
    if (!FindAndCut(&input, markup_expr, &text, &markup)) {
      javadoc_text << input;
      break;  // end of loop
    }
    javadoc_text << text;
    if (markup.starts_with("\n")) {
      javadoc_text << "\n";
      if (markup.contains("* ")) {
        // starts a list item
        javadoc_text << (in_list ? "</li>\n" : "<ul>\n") << "<li>\n";
        in_list = true;
      } else if (markup.starts_with("\n\n")) {
        if (in_list) {
          // ends the current list
          javadoc_text << "</li>\n</ul>\n";
          in_list = false;
        } else if (!input.starts_with("```")) {
          // starts new paragraph (not required if a <pre> block follows)
          javadoc_text << "<p>\n";
        }
      }
    } else if (markup.starts_with("```") && text.empty()) {
      // create a multiline code block
      re2::StringPiece language;
      RE2::Consume(&input, "[\\w\\+]+", &language);
      if (FindAndCut(&input, markup.ToString() + "\n*", &text)) {
        javadoc_text << "<pre>\n{@code" << text << "}\n</pre>\n";
      } else {
        javadoc_text << markup << language;
      }
    } else if (markup.starts_with("`")) {
      // write inlined code
      if (FindAndCut(&input, markup, &text)) {
        javadoc_text << "{@code " << text << "}";
      } else {
        javadoc_text << markup;
      }
    } else if (markup == "**") {
      // emphase text (strong)
      if (FindAndCut(&input, "\\b\\*{2}", &text)) {
        javadoc_text << "<b>" << ParseDocumentation(text) << "</b>";
      } else {
        javadoc_text << markup;
      }
    } else if (markup == "*") {
      // emphase text (light)
      if (FindAndCut(&input, "\\b\\*{1}", &text)) {
        javadoc_text << "<i>" << ParseDocumentation(text) << "</i>";
      } else {
        javadoc_text << markup;
      }
    } else if (markup == "[") {
      // add an external link
      string label;
      string link;
      if (RE2::Consume(&input, "([^\\[]+)\\]\\((http.+)\\)", &label, &link)) {
        javadoc_text << "<a href=\"" << link << "\">"
            << ParseDocumentation(label)
            << "</a>";
      } else {
        javadoc_text << markup;
      }
    } else {
      javadoc_text << markup;
    }
  }
  return javadoc_text.str();
}

ArgumentSpec CreateInput(const OpDef_ArgDef& input_def,
    const ApiDef::Arg& input_api_def, TypeResolver* type_resolver) {
  bool iterable = false;
  Type type = type_resolver->TypeOf(input_def, &iterable);
  Type var_type = Type::Interface("Operand", "org.tensorflow")
    .add_parameter(type);
  if (iterable) {
    var_type = Type::IterableOf(var_type);
  }
  return ArgumentSpec(input_api_def.name(),
      Variable::Create(SnakeToCamelCase(input_api_def.rename_to()), var_type),
      type,
      ParseDocumentation(input_api_def.description()),
      iterable);
}

AttributeSpec CreateAttribute(const OpDef_AttrDef& attr_def,
    const ApiDef::Attr& attr_api_def, TypeResolver* type_resolver) {
  bool iterable = false;
  Type type = type_resolver->TypeOf(attr_def, &iterable);
  // type attributes must be passed explicitly in methods as a Class<> parameter
  bool is_explicit = type.kind() == Type::GENERIC && !iterable;
  Type var_type = is_explicit ? Type::Class("Class").add_parameter(type) : type;
  if (iterable) {
    var_type = Type::ListOf(type);
  }
  return AttributeSpec(attr_api_def.name(),
      Variable::Create(SnakeToCamelCase(attr_api_def.rename_to()), var_type),
      type,
      ParseDocumentation(attr_api_def.description()),
      iterable,
      attr_api_def.has_default_value() && !is_explicit);
}

ArgumentSpec CreateOutput(const OpDef_ArgDef& output_def,
    const ApiDef::Arg& output_api, TypeResolver* type_resolver) {
  bool iterable = false;
  Type type = type_resolver->TypeOf(output_def, &iterable);
  Type var_type = Type::Class("Output", "org.tensorflow")
    .add_parameter(type);
  if (iterable) {
    var_type = Type::ListOf(var_type);
  }
  return ArgumentSpec(output_api.name(),
      Variable::Create(SnakeToCamelCase(output_api.rename_to()), var_type),
      type,
      ParseDocumentation(output_api.description()),
      iterable);
}

EndpointSpec CreateEndpoint(const OpDef& op_def, const ApiDef& api_def,
    const ApiDef_Endpoint& endpoint_def) {

  std::vector<string> name_tokens = str_util::Split(endpoint_def.name(), ".");
  string package;
  string name;
  if (name_tokens.size() > 1) {
    package = name_tokens.at(0);
    name = name_tokens.at(1);
  } else {
    package = "core";  // generate unclassified ops in the 'core' package
    name = name_tokens.at(0);
  }
  return EndpointSpec(package,
      name,
      Javadoc::Create(ParseDocumentation(api_def.summary()))
          .details(ParseDocumentation(api_def.description())),
      endpoint_def.deprecation_version() > 0);
}

}  // namespace

OpSpec OpSpec::Create(const OpDef& op_def, const ApiDef& api_def) {
  OpSpec op(api_def.graph_op_name(),
      api_def.visibility() == ApiDef::HIDDEN,
      op_def.deprecation().explanation());
  TypeResolver type_resolver(op_def);
  for (const string& next_input_name : api_def.arg_order()) {
    for (int i = 0; i < op_def.input_arg().size(); ++i) {
      if (op_def.input_arg(i).name() == next_input_name) {
        op.inputs_.push_back(CreateInput(op_def.input_arg(i), api_def.in_arg(i),
            &type_resolver));
        break;
      }
    }
  }
  for (int i = 0; i < op_def.attr().size(); ++i) {
    // do not parse attributes already visited, they have probably been inferred
    // before as an input argument type
    if (!type_resolver.IsAttributeVisited(op_def.attr(i).name())) {
      AttributeSpec attr = CreateAttribute(op_def.attr(i), api_def.attr(i),
          &type_resolver);
      // attributes with a default value are optional
      if (attr.optional()) {
        op.optional_attributes_.push_back(attr);
      } else {
        op.attributes_.push_back(attr);
      }
    }
  }
  for (int i = 0; i < op_def.output_arg().size(); ++i) {
    op.outputs_.push_back(CreateOutput(op_def.output_arg(i), api_def.out_arg(i),
        &type_resolver));
  }
  for (const auto& endpoint_def : api_def.endpoint()) {
    op.endpoints_.push_back(CreateEndpoint(op_def, api_def, endpoint_def));
  }
  return op;
}

}  // namespace java
}  // namespace tensorflow
