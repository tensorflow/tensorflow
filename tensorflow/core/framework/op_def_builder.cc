/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op_def_builder.h"

#include <limits>
#include <vector>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

using ::tensorflow::strings::Scanner;

namespace tensorflow {

namespace {

string AttrError(StringPiece orig, const string& op_name) {
  return strings::StrCat(" from Attr(\"", orig, "\") for Op ", op_name);
}

bool ConsumeAttrName(StringPiece* sp, StringPiece* out) {
  return Scanner(*sp)
      .One(Scanner::LETTER)
      .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
      .StopCapture()
      .AnySpace()
      .OneLiteral(":")
      .AnySpace()
      .GetResult(sp, out);
}

bool ConsumeListPrefix(StringPiece* sp) {
  return Scanner(*sp)
      .OneLiteral("list")
      .AnySpace()
      .OneLiteral("(")
      .AnySpace()
      .GetResult(sp);
}

bool ConsumeQuotedString(char quote_ch, StringPiece* sp, StringPiece* out) {
  const string quote_str(1, quote_ch);
  return Scanner(*sp)
      .OneLiteral(quote_str.c_str())
      .RestartCapture()
      .ScanEscapedUntil(quote_ch)
      .StopCapture()
      .OneLiteral(quote_str.c_str())
      .AnySpace()
      .GetResult(sp, out);
}

bool ConsumeAttrType(StringPiece* sp, StringPiece* out) {
  return Scanner(*sp)
      .Many(Scanner::LOWERLETTER_DIGIT)
      .StopCapture()
      .AnySpace()
      .GetResult(sp, out);
}

bool ConsumeAttrNumber(StringPiece* sp, int64* out) {
  Scanner scan(*sp);
  StringPiece match;
  StringPiece remaining;

  scan.AnySpace().RestartCapture();
  if (scan.Peek() == '-') {
    scan.OneLiteral("-");
  }
  if (!scan.Many(Scanner::DIGIT)
           .StopCapture()
           .AnySpace()
           .GetResult(&remaining, &match)) {
    return false;
  }
  int64 value = 0;
  if (!strings::safe_strto64(match, &value)) {
    return false;
  }
  *out = value;
  *sp = remaining;
  return true;
}

#define VERIFY(expr, ...)                                                 \
  do {                                                                    \
    if (!(expr)) {                                                        \
      errors->push_back(                                                  \
          strings::StrCat(__VA_ARGS__, AttrError(orig, op_def->name()))); \
      return;                                                             \
    }                                                                     \
  } while (false)

bool ConsumeCompoundAttrType(StringPiece* sp, StringPiece* out) {
  auto capture_begin = sp->begin();
  if (str_util::ConsumePrefix(sp, "numbertype") ||
      str_util::ConsumePrefix(sp, "numerictype") ||
      str_util::ConsumePrefix(sp, "quantizedtype") ||
      str_util::ConsumePrefix(sp, "realnumbertype") ||
      str_util::ConsumePrefix(sp, "realnumberictype")) {
    *out = StringPiece(capture_begin, sp->begin() - capture_begin);
    return true;
  }
  return false;
}

bool ProcessCompoundType(const StringPiece type_string, AttrValue* allowed) {
  if (type_string == "numbertype" || type_string == "numerictype") {
    for (DataType dt : NumberTypes()) {
      allowed->mutable_list()->add_type(dt);
    }
  } else if (type_string == "quantizedtype") {
    for (DataType dt : QuantizedTypes()) {
      allowed->mutable_list()->add_type(dt);
    }
  } else if (type_string == "realnumbertype" ||
             type_string == "realnumerictype") {
    for (DataType dt : RealNumberTypes()) {
      allowed->mutable_list()->add_type(dt);
    }
  } else {
    return false;
  }
  return true;
}

void FinalizeAttr(StringPiece spec, OpDef* op_def,
                  std::vector<string>* errors) {
  OpDef::AttrDef* attr = op_def->add_attr();
  StringPiece orig(spec);

  // Parse "<name>:" at the beginning.
  StringPiece tmp_name;
  VERIFY(ConsumeAttrName(&spec, &tmp_name), "Trouble parsing '<name>:'");
  attr->set_name(tmp_name.data(), tmp_name.size());

  // Read "<type>" or "list(<type>)".
  bool is_list = ConsumeListPrefix(&spec);
  string type;
  StringPiece type_string;  // Used if type == "type"
  if (str_util::ConsumePrefix(&spec, "string")) {
    type = "string";
  } else if (str_util::ConsumePrefix(&spec, "int")) {
    type = "int";
  } else if (str_util::ConsumePrefix(&spec, "float")) {
    type = "float";
  } else if (str_util::ConsumePrefix(&spec, "bool")) {
    type = "bool";
  } else if (str_util::ConsumePrefix(&spec, "type")) {
    type = "type";
  } else if (str_util::ConsumePrefix(&spec, "shape")) {
    type = "shape";
  } else if (str_util::ConsumePrefix(&spec, "tensor")) {
    type = "tensor";
  } else if (str_util::ConsumePrefix(&spec, "func")) {
    type = "func";
  } else if (ConsumeCompoundAttrType(&spec, &type_string)) {
    type = "type";
    AttrValue* allowed = attr->mutable_allowed_values();
    VERIFY(ProcessCompoundType(type_string, allowed),
           "Expected to see a compound type, saw: ", type_string);
  } else if (str_util::ConsumePrefix(&spec, "{")) {
    // e.g. "{ int32, float, bool }" or "{ \"foo\", \"bar\" }"
    AttrValue* allowed = attr->mutable_allowed_values();
    str_util::RemoveLeadingWhitespace(&spec);
    if (str_util::StartsWith(spec, "\"") || str_util::StartsWith(spec, "'")) {
      type = "string";  // "{ \"foo\", \"bar\" }" or "{ 'foo', 'bar' }"
      while (true) {
        StringPiece escaped_string;
        VERIFY(ConsumeQuotedString('"', &spec, &escaped_string) ||
                   ConsumeQuotedString('\'', &spec, &escaped_string),
               "Trouble parsing allowed string at '", spec, "'");
        string unescaped;
        string error;
        VERIFY(str_util::CUnescape(escaped_string, &unescaped, &error),
               "Trouble unescaping \"", escaped_string,
               "\", got error: ", error);
        allowed->mutable_list()->add_s(unescaped);
        if (str_util::ConsumePrefix(&spec, ",")) {
          str_util::RemoveLeadingWhitespace(&spec);
          if (str_util::ConsumePrefix(&spec, "}"))
            break;  // Allow ending with ", }".
        } else {
          VERIFY(str_util::ConsumePrefix(&spec, "}"),
                 "Expected , or } after strings in list, not: '", spec, "'");
          break;
        }
      }
    } else {  // "{ bool, numbertype, string }"
      type = "type";
      while (true) {
        VERIFY(ConsumeAttrType(&spec, &type_string),
               "Trouble parsing type string at '", spec, "'");
        if (ProcessCompoundType(type_string, allowed)) {
          // Processed a compound type.
        } else {
          DataType dt;
          VERIFY(DataTypeFromString(type_string, &dt),
                 "Unrecognized type string '", type_string, "'");
          allowed->mutable_list()->add_type(dt);
        }
        if (str_util::ConsumePrefix(&spec, ",")) {
          str_util::RemoveLeadingWhitespace(&spec);
          if (str_util::ConsumePrefix(&spec, "}"))
            break;  // Allow ending with ", }".
        } else {
          VERIFY(str_util::ConsumePrefix(&spec, "}"),
                 "Expected , or } after types in list, not: '", spec, "'");
          break;
        }
      }
    }
  } else {  // if spec.Consume("{")
    VERIFY(false, "Trouble parsing type string at '", spec, "'");
  }
  str_util::RemoveLeadingWhitespace(&spec);

  // Write the type into *attr.
  if (is_list) {
    VERIFY(str_util::ConsumePrefix(&spec, ")"),
           "Expected ) to close 'list(', not: '", spec, "'");
    str_util::RemoveLeadingWhitespace(&spec);
    attr->set_type(strings::StrCat("list(", type, ")"));
  } else {
    attr->set_type(type);
  }

  // Read optional minimum constraint at the end.
  if ((is_list || type == "int") && str_util::ConsumePrefix(&spec, ">=")) {
    int64 min_limit = -999;
    VERIFY(ConsumeAttrNumber(&spec, &min_limit),
           "Could not parse integer lower limit after '>=', found '", spec,
           "' instead");
    attr->set_has_minimum(true);
    attr->set_minimum(min_limit);
  }

  // Parse default value, if present.
  if (str_util::ConsumePrefix(&spec, "=")) {
    str_util::RemoveLeadingWhitespace(&spec);
    VERIFY(ParseAttrValue(attr->type(), spec, attr->mutable_default_value()),
           "Could not parse default value '", spec, "'");
  } else {
    VERIFY(spec.empty(), "Extra '", spec, "' unparsed at the end");
  }
}

#undef VERIFY

string InOutError(bool is_output, StringPiece orig, const string& op_name) {
  return strings::StrCat(" from ", is_output ? "Output" : "Input", "(\"", orig,
                         "\") for Op ", op_name);
}

bool ConsumeInOutName(StringPiece* sp, StringPiece* out) {
  return Scanner(*sp)
      .One(Scanner::LOWERLETTER)
      .Any(Scanner::LOWERLETTER_DIGIT_UNDERSCORE)
      .StopCapture()
      .AnySpace()
      .OneLiteral(":")
      .AnySpace()
      .GetResult(sp, out);
}

bool ConsumeInOutRefOpen(StringPiece* sp) {
  return Scanner(*sp)
      .OneLiteral("Ref")
      .AnySpace()
      .OneLiteral("(")
      .AnySpace()
      .GetResult(sp);
}

bool ConsumeInOutRefClose(StringPiece* sp) {
  return Scanner(*sp).OneLiteral(")").AnySpace().GetResult(sp);
}

bool ConsumeInOutNameOrType(StringPiece* sp, StringPiece* out) {
  return Scanner(*sp)
      .One(Scanner::LETTER)
      .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
      .StopCapture()
      .AnySpace()
      .GetResult(sp, out);
}

bool ConsumeInOutTimesType(StringPiece* sp, StringPiece* out) {
  return Scanner(*sp)
      .OneLiteral("*")
      .AnySpace()
      .RestartCapture()
      .One(Scanner::LETTER)
      .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
      .StopCapture()
      .AnySpace()
      .GetResult(sp, out);
}

#define VERIFY(expr, ...)                                             \
  do {                                                                \
    if (!(expr)) {                                                    \
      errors->push_back(strings::StrCat(                              \
          __VA_ARGS__, InOutError(is_output, orig, op_def->name()))); \
      return;                                                         \
    }                                                                 \
  } while (false)

void FinalizeInputOrOutput(StringPiece spec, bool is_output, OpDef* op_def,
                           std::vector<string>* errors) {
  OpDef::ArgDef* arg =
      is_output ? op_def->add_output_arg() : op_def->add_input_arg();

  StringPiece orig(spec);

  // Parse "<name>:" at the beginning.
  StringPiece tmp_name;
  VERIFY(ConsumeInOutName(&spec, &tmp_name), "Trouble parsing 'name:'");
  arg->set_name(tmp_name.data(), tmp_name.size());

  // Detect "Ref(...)".
  if (ConsumeInOutRefOpen(&spec)) {
    arg->set_is_ref(true);
  }

  {  // Parse "<name|type>" or "<name>*<name|type>".
    StringPiece first, second, type_or_attr;
    VERIFY(ConsumeInOutNameOrType(&spec, &first),
           "Trouble parsing either a type or an attr name at '", spec, "'");
    if (ConsumeInOutTimesType(&spec, &second)) {
      arg->set_number_attr(first.data(), first.size());
      type_or_attr = second;
    } else {
      type_or_attr = first;
    }
    DataType dt;
    if (DataTypeFromString(type_or_attr, &dt)) {
      arg->set_type(dt);
    } else {
      const OpDef::AttrDef* attr = FindAttr(type_or_attr, *op_def);
      VERIFY(attr != nullptr, "Reference to unknown attr '", type_or_attr, "'");
      if (attr->type() == "type") {
        arg->set_type_attr(type_or_attr.data(), type_or_attr.size());
      } else {
        VERIFY(attr->type() == "list(type)", "Reference to attr '",
               type_or_attr, "' with type ", attr->type(),
               " that isn't type or list(type)");
        arg->set_type_list_attr(type_or_attr.data(), type_or_attr.size());
      }
    }
  }

  // Closing ) for Ref(.
  if (arg->is_ref()) {
    VERIFY(ConsumeInOutRefClose(&spec),
           "Did not find closing ')' for 'Ref(', instead found: '", spec, "'");
  }

  // Should not have anything else.
  VERIFY(spec.empty(), "Extra '", spec, "' unparsed at the end");

  // Int attrs that are the length of an input or output get a default
  // minimum of 1.
  if (!arg->number_attr().empty()) {
    OpDef::AttrDef* attr = FindAttrMutable(arg->number_attr(), op_def);
    if (attr != nullptr && !attr->has_minimum()) {
      attr->set_has_minimum(true);
      attr->set_minimum(1);
    }
  } else if (!arg->type_list_attr().empty()) {
    // If an input or output has type specified by a list(type) attr,
    // it gets a default minimum of 1 as well.
    OpDef::AttrDef* attr = FindAttrMutable(arg->type_list_attr(), op_def);
    if (attr != nullptr && attr->type() == "list(type)" &&
        !attr->has_minimum()) {
      attr->set_has_minimum(true);
      attr->set_minimum(1);
    }
  }

  // If the arg's dtype is resource we should mark the op as stateful as it
  // likely touches a resource manager. This deliberately doesn't cover inputs /
  // outputs which resolve to resource via Attrs as those mostly operate on
  // resource handles as an opaque type (as opposed to ops which explicitly take
  // / produce resources).
  if (arg->type() == DT_RESOURCE) {
    op_def->set_is_stateful(true);
  }
}

#undef VERIFY

int num_leading_spaces(StringPiece s) {
  size_t i = 0;
  while (i < s.size() && s[i] == ' ') {
    ++i;
  }
  return i;
}

bool ConsumeDocNameColon(StringPiece* sp, StringPiece* out) {
  return Scanner(*sp)
      .One(Scanner::LETTER)
      .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
      .StopCapture()
      .AnySpace()
      .OneLiteral(":")
      .AnySpace()
      .GetResult(sp, out);
}

bool IsDocNameColon(StringPiece s) {
  return ConsumeDocNameColon(&s, nullptr /* out */);
}

void FinalizeDoc(const string& text, OpDef* op_def,
                 std::vector<string>* errors) {
  std::vector<string> lines = str_util::Split(text, '\n');

  // Remove trailing spaces.
  for (string& line : lines) {
    str_util::StripTrailingWhitespace(&line);
  }

  // First non-blank line -> summary.
  int l = 0;
  while (static_cast<size_t>(l) < lines.size() && lines[l].empty()) ++l;
  if (static_cast<size_t>(l) < lines.size()) {
    op_def->set_summary(lines[l]);
    ++l;
  }
  while (static_cast<size_t>(l) < lines.size() && lines[l].empty()) ++l;

  // Lines until we see name: -> description.
  int start_l = l;
  while (static_cast<size_t>(l) < lines.size() && !IsDocNameColon(lines[l])) {
    ++l;
  }
  int end_l = l;
  // Trim trailing blank lines from the description.
  while (start_l < end_l && lines[end_l - 1].empty()) --end_l;
  string desc = str_util::Join(
      gtl::ArraySlice<string>(lines.data() + start_l, end_l - start_l), "\n");
  if (!desc.empty()) op_def->set_description(desc);

  // name: description
  //   possibly continued on the next line
  //   if so, we remove the minimum indent
  StringPiece name;
  std::vector<StringPiece> description;
  while (static_cast<size_t>(l) < lines.size()) {
    description.clear();
    description.push_back(lines[l]);
    ConsumeDocNameColon(&description.back(), &name);
    ++l;
    while (static_cast<size_t>(l) < lines.size() && !IsDocNameColon(lines[l])) {
      description.push_back(lines[l]);
      ++l;
    }
    // Remove any trailing blank lines.
    while (!description.empty() && description.back().empty()) {
      description.pop_back();
    }
    // Compute the minimum indent of all lines after the first.
    int min_indent = -1;
    for (size_t i = 1; i < description.size(); ++i) {
      if (!description[i].empty()) {
        int indent = num_leading_spaces(description[i]);
        if (min_indent < 0 || indent < min_indent) min_indent = indent;
      }
    }
    // Remove min_indent spaces from all lines after the first.
    for (size_t i = 1; i < description.size(); ++i) {
      if (!description[i].empty()) description[i].remove_prefix(min_indent);
    }
    // Concatenate lines into a single string.
    const string complete(str_util::Join(description, "\n"));

    // Find name.
    bool found = false;
    for (int i = 0; !found && i < op_def->input_arg_size(); ++i) {
      if (op_def->input_arg(i).name() == name) {
        op_def->mutable_input_arg(i)->set_description(complete);
        found = true;
      }
    }
    for (int i = 0; !found && i < op_def->output_arg_size(); ++i) {
      if (op_def->output_arg(i).name() == name) {
        op_def->mutable_output_arg(i)->set_description(complete);
        found = true;
      }
    }
    for (int i = 0; !found && i < op_def->attr_size(); ++i) {
      if (op_def->attr(i).name() == name) {
        op_def->mutable_attr(i)->set_description(complete);
        found = true;
      }
    }
    if (!found) {
      errors->push_back(
          strings::StrCat("No matching input/output/attr for name '", name,
                          "' from Doc() for Op ", op_def->name()));
      return;
    }
  }
}

}  // namespace

OpDefBuilder::OpDefBuilder(StringPiece op_name) {
  op_def()->set_name(string(op_name));  // NOLINT
}

OpDefBuilder& OpDefBuilder::Attr(StringPiece spec) {
  attrs_.emplace_back(spec.data(), spec.size());
  return *this;
}

OpDefBuilder& OpDefBuilder::Input(StringPiece spec) {
  inputs_.emplace_back(spec.data(), spec.size());
  return *this;
}

OpDefBuilder& OpDefBuilder::Output(StringPiece spec) {
  outputs_.emplace_back(spec.data(), spec.size());
  return *this;
}

#ifndef TF_LEAN_BINARY
OpDefBuilder& OpDefBuilder::Doc(StringPiece text) {
  if (!doc_.empty()) {
    errors_.push_back(
        strings::StrCat("Extra call to Doc() for Op ", op_def()->name()));
  } else {
    doc_.assign(text.data(), text.size());
  }
  return *this;
}
#endif

OpDefBuilder& OpDefBuilder::SetIsCommutative() {
  op_def()->set_is_commutative(true);
  return *this;
}

OpDefBuilder& OpDefBuilder::SetIsAggregate() {
  op_def()->set_is_aggregate(true);
  return *this;
}

OpDefBuilder& OpDefBuilder::SetIsStateful() {
  op_def()->set_is_stateful(true);
  return *this;
}

OpDefBuilder& OpDefBuilder::SetAllowsUninitializedInput() {
  op_def()->set_allows_uninitialized_input(true);
  return *this;
}

OpDefBuilder& OpDefBuilder::Deprecated(int version, StringPiece explanation) {
  if (op_def()->has_deprecation()) {
    errors_.push_back(
        strings::StrCat("Deprecated called twice for Op ", op_def()->name()));
  } else {
    OpDeprecation* deprecation = op_def()->mutable_deprecation();
    deprecation->set_version(version);
    deprecation->set_explanation(string(explanation));
  }
  return *this;
}

OpDefBuilder& OpDefBuilder::SetShapeFn(
    Status (*fn)(shape_inference::InferenceContext*)) {
  if (op_reg_data_.shape_inference_fn != nullptr) {
    errors_.push_back(
        strings::StrCat("SetShapeFn called twice for Op ", op_def()->name()));
  } else {
    op_reg_data_.shape_inference_fn = OpShapeInferenceFn(fn);
  }
  return *this;
}

Status OpDefBuilder::Finalize(OpRegistrationData* op_reg_data) const {
  std::vector<string> errors = errors_;
  *op_reg_data = op_reg_data_;

  OpDef* op_def = &op_reg_data->op_def;
  for (StringPiece attr : attrs_) {
    FinalizeAttr(attr, op_def, &errors);
  }
  for (StringPiece input : inputs_) {
    FinalizeInputOrOutput(input, false, op_def, &errors);
  }
  for (StringPiece output : outputs_) {
    FinalizeInputOrOutput(output, true, op_def, &errors);
  }
  FinalizeDoc(doc_, op_def, &errors);

  if (errors.empty()) return Status::OK();
  return errors::InvalidArgument(str_util::Join(errors, "\n"));
}

}  // namespace tensorflow
