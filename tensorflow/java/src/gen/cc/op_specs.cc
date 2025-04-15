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

#include "tensorflow/java/src/gen/cc/op_specs.h"

#include <cctype>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "absl/strings/strip.h"
#include "re2/re2.h"
#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"

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

  // Returns the class type of an input/output argument
  //
  // For example, if the argument's datatype is DT_STRING, this method will
  // return "java.lang.String", so the argument can become "Operand<String>"
  // in the Ops API
  Type TypeOf(const OpDef_ArgDef& arg_def, bool* iterable_out);

  // Returns types of an input attribute
  //
  // The first element of the pair is the class type of this attribute while
  // the second is its JNI/primitive type equivalent, required for explicit
  // unboxing.
  //
  // For example, if the attribute is of type "float", this method will return
  // <java.lang.Float, float>, so the attribute can be used as a "Float" object
  // in the Ops API and casted to a "float" when passing through the JNI layer.
  std::pair<Type, Type> TypesOf(const OpDef_AttrDef& attr_def,
                                bool* iterable_out);

  // Returns true if the type of this attribute has already been resolved
  bool IsAttributeVisited(const string& attr_name) {
    return visited_attrs_.find(attr_name) != visited_attrs_.cend();
  }

 private:
  const OpDef op_def_;
  std::map<std::string, Type> visited_attrs_;
  char next_generic_letter_ = 'T';

  std::pair<Type, Type> MakeTypePair(const Type& type, const Type& jni_type) {
    return std::make_pair(type, jni_type);
  }
  std::pair<Type, Type> MakeTypePair(const Type& type) {
    return std::make_pair(type, type);
  }
  Type NextGeneric() {
    char generic_letter = next_generic_letter_++;
    if (next_generic_letter_ > 'Z') {
      next_generic_letter_ = 'A';
    }
    return Type::Generic(string(1, generic_letter));
  }
};

Type TypeResolver::TypeOf(const OpDef_ArgDef& arg_def, bool* iterable_out) {
  *iterable_out = false;
  Type type = Type::Wildcard();
  if (arg_def.type() != DataType::DT_INVALID) {
    type = Type::ForDataType(arg_def.type());

  } else if (!arg_def.type_attr().empty()) {
    // resolve type from attribute (if already visited, retrieve its type)
    if (IsAttributeVisited(arg_def.type_attr())) {
      type = visited_attrs_.at(arg_def.type_attr());
    } else {
      for (const auto& attr_def : op_def_.attr()) {
        if (attr_def.name() == arg_def.type_attr()) {
          type = TypesOf(attr_def, iterable_out).first;
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
  if (!arg_def.number_attr().empty()) {
    // when number_attr is set, argument has to be a list of tensors
    *iterable_out = true;
    visited_attrs_.insert(std::make_pair(arg_def.number_attr(), Type::Int()));
  }
  return type;
}

std::pair<Type, Type> TypeResolver::TypesOf(const OpDef_AttrDef& attr_def,
                                            bool* iterable_out) {
  std::pair<Type, Type> types = MakeTypePair(Type::Wildcard());
  *iterable_out = false;
  absl::string_view attr_type = attr_def.type();
  if (absl::ConsumePrefix(&attr_type, "list(")) {
    attr_type.remove_suffix(1);  // remove closing brace
    *iterable_out = true;
  }
  if (attr_type == "string") {
    types = MakeTypePair(Type::Class("String"));

  } else if (attr_type == "int") {
    types = MakeTypePair(Type::Class("Long"), Type::Long());

  } else if (attr_type == "float") {
    types = MakeTypePair(Type::Class("Float"), Type::Float());

  } else if (attr_type == "bool") {
    types = MakeTypePair(Type::Class("Boolean"), Type::Boolean());

  } else if (attr_type == "shape") {
    types = MakeTypePair(Type::Class("Shape", "org.tensorflow"));

  } else if (attr_type == "tensor") {
    types = MakeTypePair(Type::Class("Tensor", "org.tensorflow")
                             .add_parameter(Type::Wildcard()));

  } else if (attr_type == "type") {
    Type type = *iterable_out ? Type::Wildcard() : NextGeneric();
    if (IsRealNumbers(attr_def.allowed_values())) {
      type.add_supertype(Type::Class("Number"));
    }
    types = MakeTypePair(type, Type::Enum("DataType", "org.tensorflow"));

  } else {
    LOG(FATAL) << "Cannot resolve data type for attribute \"" << attr_type
               << "\" in operation \"" << op_def_.name() << "\"";
  }
  visited_attrs_.insert(std::make_pair(attr_def.name(), types.first));
  return types;
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

bool FindAndCut(string* input, const RE2& expr, string* before_match,
                string* ret_match = nullptr) {
  string match;
  if (!RE2::PartialMatch(*input, expr, &match)) return false;
  *before_match = input->substr(0, input->find(match));
  *input = input->substr(before_match->size() + match.size());
  if (ret_match != nullptr) *ret_match = match;
  return true;
}

string ParseDocumentation(const string& inp) {
  std::stringstream javadoc_text;

  // TODO(karllessard) This is a very minimalist utility method for converting
  // markdown syntax, as found in ops descriptions, to Javadoc/html tags. Check
  // for alternatives to increase the level of support for markups.
  std::vector<string> markups_subexpr;
  markups_subexpr.push_back("\n+\\*\\s+");                // lists
  markups_subexpr.push_back("\n{2,}");                    // paragraphs
  markups_subexpr.push_back("`{3,}\\s*[^\\s\n]*\\s*\n");  // code blocks
  markups_subexpr.push_back("`+");           // inlined code and code blocks
  markups_subexpr.push_back("\\*{1,2}\\b");  // text emphasis
  markups_subexpr.push_back("\\[");          // hyperlinks
  const RE2 markup_expr("(" + absl::StrJoin(markups_subexpr, "|") + ")");

  bool in_list = false;
  string input = inp;
  while (true) {
    string text, markup;
    if (!FindAndCut(&input, markup_expr, &text, &markup)) {
      javadoc_text << input;
      break;  // end of loop
    }
    javadoc_text << text;
    if (absl::StartsWith(markup, "\n")) {
      javadoc_text << "\n";
      if (absl::StrContains(markup, "*")) {
        // new list item
        javadoc_text << (in_list ? "</li>\n" : "<ul>\n") << "<li>\n";
        in_list = true;
      } else if (in_list) {
        // end of list
        javadoc_text << "</li>\n</ul>\n";
        in_list = false;
      } else if (!absl::StartsWith(input, "```")) {
        // new paragraph (not required if a <pre> block follows)
        javadoc_text << "<p>\n";
      }
    } else if (absl::StartsWith(markup, "```")) {
      // code blocks
      if (FindAndCut(&input, "(```\\s*\n*)", &text)) {
        javadoc_text << "<pre>{@code\n" << text << "}</pre>\n";
      } else {
        javadoc_text << markup;
      }
    } else if (absl::StartsWith("(" + markup + ")", "`")) {
      // inlined code
      if (FindAndCut(&input, markup, &text)) {
        javadoc_text << "{@code " << text << "}";
      } else {
        javadoc_text << markup;
      }
    } else if (markup == "**") {
      // text emphasis (strong)
      if (FindAndCut(&input, "(\\b\\*{2})", &text)) {
        javadoc_text << "<b>" << ParseDocumentation(text) << "</b>";
      } else {
        javadoc_text << markup;
      }
    } else if (markup == "*") {
      // text emphasis (normal)
      if (FindAndCut(&input, "(\\b\\*{1})", &text)) {
        javadoc_text << "<i>" << ParseDocumentation(text) << "</i>";
      } else {
        javadoc_text << markup;
      }
    } else if (absl::StartsWith(markup, "[")) {
      // hyperlinks
      string label;
      string link;
      if (RE2::PartialMatch(input, "([^\\[]+)\\]\\((http.+)\\)", &label,
                            &link) &&
          absl::StartsWith(input, label + link)) {
        input = input.substr(label.size() + link.size());
        javadoc_text << "<a href=\"" << link << "\">"
                     << ParseDocumentation(label) << "</a>";
      } else {
        javadoc_text << markup;
      }
    } else {
      // safe fallback
      javadoc_text << markup;
    }
  }
  return javadoc_text.str();
}

ArgumentSpec CreateInput(const OpDef_ArgDef& input_def,
                         const ApiDef::Arg& input_api_def,
                         TypeResolver* type_resolver) {
  bool iterable = false;
  Type type = type_resolver->TypeOf(input_def, &iterable);
  Type var_type =
      Type::Interface("Operand", "org.tensorflow").add_parameter(type);
  if (iterable) {
    var_type = Type::IterableOf(var_type);
  }
  return ArgumentSpec(
      input_api_def.name(),
      Variable::Create(SnakeToCamelCase(input_api_def.rename_to()), var_type),
      type, ParseDocumentation(input_api_def.description()), iterable);
}

AttributeSpec CreateAttribute(const OpDef_AttrDef& attr_def,
                              const ApiDef::Attr& attr_api_def,
                              TypeResolver* type_resolver) {
  bool iterable = false;
  std::pair<Type, Type> types = type_resolver->TypesOf(attr_def, &iterable);
  Type var_type = types.first.kind() == Type::GENERIC
                      ? Type::ClassOf(types.first)
                      : types.first;
  if (iterable) {
    var_type = Type::ListOf(var_type);
  }
  return AttributeSpec(
      attr_api_def.name(),
      Variable::Create(SnakeToCamelCase(attr_api_def.rename_to()), var_type),
      types.first, types.second, ParseDocumentation(attr_api_def.description()),
      iterable,
      attr_def.has_default_value() ? &attr_def.default_value() : nullptr);
}

ArgumentSpec CreateOutput(const OpDef_ArgDef& output_def,
                          const ApiDef::Arg& output_api,
                          TypeResolver* type_resolver) {
  bool iterable = false;
  Type type = type_resolver->TypeOf(output_def, &iterable);
  Type var_type = Type::Class("Output", "org.tensorflow").add_parameter(type);
  if (iterable) {
    var_type = Type::ListOf(var_type);
  }
  return ArgumentSpec(
      output_api.name(),
      Variable::Create(SnakeToCamelCase(output_api.rename_to()), var_type),
      type, ParseDocumentation(output_api.description()), iterable);
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
  return EndpointSpec(package, name,
                      Javadoc::Create(ParseDocumentation(api_def.summary()))
                          .details(ParseDocumentation(api_def.description())));
}

}  // namespace

OpSpec OpSpec::Create(const OpDef& op_def, const ApiDef& api_def) {
  OpSpec op(api_def.graph_op_name(), api_def.visibility() == ApiDef::HIDDEN,
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
      AttributeSpec attr =
          CreateAttribute(op_def.attr(i), api_def.attr(i), &type_resolver);
      // attributes with a default value are optional
      if (attr.has_default_value() && attr.type().kind() != Type::GENERIC) {
        op.optional_attributes_.push_back(attr);
      } else {
        op.attributes_.push_back(attr);
      }
    }
  }
  for (int i = 0; i < op_def.output_arg().size(); ++i) {
    op.outputs_.push_back(
        CreateOutput(op_def.output_arg(i), api_def.out_arg(i), &type_resolver));
  }
  for (const auto& endpoint_def : api_def.endpoint()) {
    op.endpoints_.push_back(CreateEndpoint(op_def, api_def, endpoint_def));
  }
  return op;
}

}  // namespace java
}  // namespace tensorflow
