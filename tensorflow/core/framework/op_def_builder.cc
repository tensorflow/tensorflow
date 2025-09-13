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

#include "absl/strings/escaping.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/errors.h"

using ::tensorflow::strings::Scanner;

namespace tensorflow {

namespace {

string AttrError(absl::string_view orig, const string& op_name) {
  return absl::StrCat(" from Attr(\"", orig, "\") for Op ", op_name);
}

bool ConsumeAttrName(absl::string_view* sp, absl::string_view* out) {
  return Scanner(*sp)
      .One(Scanner::LETTER)
      .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
      .StopCapture()
      .AnySpace()
      .OneLiteral(":")
      .AnySpace()
      .GetResult(sp, out);
}

bool ConsumeListPrefix(absl::string_view* sp) {
  return Scanner(*sp)
      .OneLiteral("list")
      .AnySpace()
      .OneLiteral("(")
      .AnySpace()
      .GetResult(sp);
}

bool ConsumeQuotedString(char quote_ch, absl::string_view* sp,
                         absl::string_view* out) {
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

bool ConsumeAttrType(absl::string_view* sp, absl::string_view* out) {
  return Scanner(*sp)
      .Many(Scanner::LOWERLETTER_DIGIT)
      .StopCapture()
      .AnySpace()
      .GetResult(sp, out);
}

bool ConsumeAttrNumber(absl::string_view* sp, int64_t* out) {
  Scanner scan(*sp);
  absl::string_view match;
  absl::string_view remaining;

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
  int64_t value = 0;
  if (!absl::SimpleAtoi(match, &value)) {
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

bool ConsumeCompoundAttrType(absl::string_view* sp, absl::string_view* out) {
  auto capture_data = sp->data();
  auto capture_begin = sp->begin();
  if (absl::ConsumePrefix(sp, "numbertype") ||
      absl::ConsumePrefix(sp, "numerictype") ||
      absl::ConsumePrefix(sp, "quantizedtype") ||
      absl::ConsumePrefix(sp, "realnumbertype") ||
      absl::ConsumePrefix(sp, "realnumberictype")) {
    *out = absl::string_view(capture_data, sp->begin() - capture_begin);
    return true;
  }
  return false;
}

bool ProcessCompoundType(const absl::string_view type_string,
                         AttrValue* allowed) {
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

void FinalizeAttr(absl::string_view spec, bool allow_attr_type_any,
                  OpDef* op_def, std::vector<string>* errors) {
  OpDef::AttrDef* attr = op_def->add_attr();
  absl::string_view orig(spec);

  // Parse "<name>:" at the beginning.
  absl::string_view tmp_name;
  VERIFY(ConsumeAttrName(&spec, &tmp_name), "Trouble parsing '<name>:'");
  attr->set_name(tmp_name.data(), tmp_name.size());

  // Read "<type>" or "list(<type>)".
  bool is_list = ConsumeListPrefix(&spec);
  string type;
  absl::string_view type_string;  // Used if type == "type"
  if (absl::ConsumePrefix(&spec, "string")) {
    type = "string";
  } else if (absl::ConsumePrefix(&spec, "int")) {
    type = "int";
  } else if (absl::ConsumePrefix(&spec, "float")) {
    type = "float";
  } else if (absl::ConsumePrefix(&spec, "bool")) {
    type = "bool";
  } else if (absl::ConsumePrefix(&spec, "type")) {
    type = "type";
  } else if (absl::ConsumePrefix(&spec, "shape")) {
    type = "shape";
  } else if (absl::ConsumePrefix(&spec, "tensor")) {
    type = "tensor";
  } else if (absl::ConsumePrefix(&spec, "func")) {
    type = "func";
  } else if (absl::ConsumePrefix(&spec, "any") && allow_attr_type_any) {
    type = "any";
  } else if (ConsumeCompoundAttrType(&spec, &type_string)) {
    type = "type";
    AttrValue* allowed = attr->mutable_allowed_values();
    VERIFY(ProcessCompoundType(type_string, allowed),
           "Expected to see a compound type, saw: ", type_string);
  } else if (absl::ConsumePrefix(&spec, "{")) {
    // e.g. "{ int32, float, bool }" or "{ \"foo\", \"bar\" }"
    AttrValue* allowed = attr->mutable_allowed_values();
    str_util::RemoveLeadingWhitespace(&spec);
    if (absl::StartsWith(spec, "\"") || absl::StartsWith(spec, "'")) {
      type = "string";  // "{ \"foo\", \"bar\" }" or "{ 'foo', 'bar' }"
      while (true) {
        absl::string_view escaped_string;
        VERIFY(ConsumeQuotedString('"', &spec, &escaped_string) ||
                   ConsumeQuotedString('\'', &spec, &escaped_string),
               "Trouble parsing allowed string at '", spec, "'");
        string unescaped;
        string error;
        VERIFY(absl::CUnescape(escaped_string, &unescaped, &error),
               "Trouble unescaping \"", escaped_string,
               "\", got error: ", error);
        allowed->mutable_list()->add_s(unescaped);
        if (absl::ConsumePrefix(&spec, ",")) {
          str_util::RemoveLeadingWhitespace(&spec);
          if (absl::ConsumePrefix(&spec, "}"))
            break;  // Allow ending with ", }".
        } else {
          VERIFY(absl::ConsumePrefix(&spec, "}"),
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
        if (absl::ConsumePrefix(&spec, ",")) {
          str_util::RemoveLeadingWhitespace(&spec);
          if (absl::ConsumePrefix(&spec, "}"))
            break;  // Allow ending with ", }".
        } else {
          VERIFY(absl::ConsumePrefix(&spec, "}"),
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
    VERIFY(absl::ConsumePrefix(&spec, ")"),
           "Expected ) to close 'list(', not: '", spec, "'");
    str_util::RemoveLeadingWhitespace(&spec);
    attr->set_type(absl::StrCat("list(", type, ")"));
  } else {
    attr->set_type(type);
  }

  // Read optional minimum constraint at the end.
  if ((is_list || type == "int") && absl::ConsumePrefix(&spec, ">=")) {
    int64_t min_limit = -999;
    VERIFY(ConsumeAttrNumber(&spec, &min_limit),
           "Could not parse integer lower limit after '>=', found '", spec,
           "' instead");
    attr->set_has_minimum(true);
    attr->set_minimum(min_limit);
  }

  // Parse default value, if present.
  if (absl::ConsumePrefix(&spec, "=")) {
    str_util::RemoveLeadingWhitespace(&spec);
    VERIFY(ParseAttrValue(attr->type(), spec, attr->mutable_default_value()),
           "Could not parse default value '", spec, "'");
  } else {
    VERIFY(spec.empty(), "Extra '", spec, "' unparsed at the end");
  }
}

#undef VERIFY

string InOutError(bool is_output, absl::string_view orig,
                  const string& op_name) {
  return strings::StrCat(" from ", is_output ? "Output" : "Input", "(\"", orig,
                         "\") for Op ", op_name);
}

bool ConsumeInOutName(absl::string_view* sp, absl::string_view* out) {
  return Scanner(*sp)
      .One(Scanner::LOWERLETTER)
      .Any(Scanner::LOWERLETTER_DIGIT_UNDERSCORE)
      .StopCapture()
      .AnySpace()
      .OneLiteral(":")
      .AnySpace()
      .GetResult(sp, out);
}

bool ConsumeInOutRefOpen(absl::string_view* sp) {
  return Scanner(*sp)
      .OneLiteral("Ref")
      .AnySpace()
      .OneLiteral("(")
      .AnySpace()
      .GetResult(sp);
}

bool ConsumeInOutRefClose(absl::string_view* sp) {
  return Scanner(*sp).OneLiteral(")").AnySpace().GetResult(sp);
}

bool ConsumeInOutNameOrType(absl::string_view* sp, absl::string_view* out) {
  return Scanner(*sp)
      .One(Scanner::LETTER)
      .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
      .StopCapture()
      .AnySpace()
      .GetResult(sp, out);
}

bool ConsumeInOutTimesType(absl::string_view* sp, absl::string_view* out) {
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

bool ConsumeControlOutName(absl::string_view* sp, absl::string_view* out) {
  return Scanner(*sp)
      .One(Scanner::LETTER)
      .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
      .StopCapture()
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

void FinalizeInputOrOutput(absl::string_view spec, bool is_output,
                           OpDef* op_def, std::vector<string>* errors) {
  OpDef::ArgDef* arg =
      is_output ? op_def->add_output_arg() : op_def->add_input_arg();

  absl::string_view orig(spec);

  // Parse "<name>:" at the beginning.
  absl::string_view tmp_name;
  VERIFY(ConsumeInOutName(&spec, &tmp_name), "Trouble parsing 'name:'");
  arg->set_name(tmp_name.data(), tmp_name.size());

  // Detect "Ref(...)".
  if (ConsumeInOutRefOpen(&spec)) {
    arg->set_is_ref(true);
  }

  {  // Parse "<name|type>" or "<name>*<name|type>".
    absl::string_view first, second, type_or_attr;
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

string ControlOutError(absl::string_view orig, const string& op_name) {
  return absl::StrCat(" from ControlOutput(\"", orig, "\") for Op ", op_name);
}

void FinalizeControlOutput(absl::string_view name, OpDef* op_def,
                           std::vector<string>* errors) {
  absl::string_view orig(name);

  // Parse control output name.
  absl::string_view tmp_name;
  if (!ConsumeControlOutName(&orig, &tmp_name)) {
    errors->push_back(absl::StrCat("Trouble parsing 'name:'",
                                   ControlOutError(orig, op_def->name())));
  }

  *op_def->add_control_output() = string(tmp_name.data(), tmp_name.size());
}

int num_leading_spaces(absl::string_view s) {
  size_t i = 0;
  while (i < s.size() && s[i] == ' ') {
    ++i;
  }
  return i;
}

bool ConsumeDocNameColon(absl::string_view* sp, absl::string_view* out) {
  return Scanner(*sp)
      .One(Scanner::LETTER)
      .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
      .StopCapture()
      .AnySpace()
      .OneLiteral(":")
      .AnySpace()
      .GetResult(sp, out);
}

bool IsDocNameColon(absl::string_view s) {
  return ConsumeDocNameColon(&s, nullptr /* out */);
}

void FinalizeDoc(const string& text, OpDef* op_def,
                 std::vector<string>* errors) {
  std::vector<string> lines = str_util::Split(text, '\n');

  // Remove trailing spaces.
  for (string& line : lines) {
    absl::StripTrailingAsciiWhitespace(&line);
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
  string desc = absl::StrJoin(
      absl::Span<const string>(lines.data() + start_l, end_l - start_l), "\n");
  if (!desc.empty()) op_def->set_description(desc);

  // name: description
  //   possibly continued on the next line
  //   if so, we remove the minimum indent
  absl::string_view name;
  std::vector<absl::string_view> description;
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
    const string complete(absl::StrJoin(description, "\n"));

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
      errors->push_back(absl::StrCat("No matching input/output/attr for name '",
                                     name, "' from Doc() for Op ",
                                     op_def->name()));
      return;
    }
  }
}

}  // namespace

OpDefBuilder::OpDefBuilder(string op_name) {
  op_def()->set_name(std::move(op_name));
}

OpDefBuilder& OpDefBuilder::Attr(string spec) {
  attrs_.push_back(std::move(spec));
  return *this;
}

OpDefBuilder& OpDefBuilder::Input(string spec) {
  inputs_.push_back(std::move(spec));
  return *this;
}

OpDefBuilder& OpDefBuilder::Output(string spec) {
  outputs_.push_back(std::move(spec));
  return *this;
}

OpDefBuilder& OpDefBuilder::ControlOutput(string name) {
  control_outputs_.push_back(std::move(name));
  return *this;
}

OpDefBuilder& OpDefBuilder::Doc(string text) {
#ifndef TF_LEAN_BINARY
  if (!doc_.empty()) {
    errors_.push_back(
        absl::StrCat("Extra call to Doc() for Op ", op_def()->name()));
  } else {
    doc_ = std::move(text);
  }
#endif
  return *this;
}

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

OpDefBuilder& OpDefBuilder::SetIsDistributedCommunication() {
  op_def()->set_is_distributed_communication(true);
  return *this;
}

OpDefBuilder& OpDefBuilder::Deprecated(int version, string explanation) {
  if (op_def()->has_deprecation()) {
    errors_.push_back(
        absl::StrCat("Deprecated called twice for Op ", op_def()->name()));
  } else {
    OpDeprecation* deprecation = op_def()->mutable_deprecation();
    deprecation->set_version(version);
    deprecation->set_explanation(std::move(explanation));
  }
  return *this;
}

OpDefBuilder& OpDefBuilder::SetTypeConstructor(OpTypeConstructor c) {
  op_reg_data_.type_ctor = c;
  return *this;
}

OpDefBuilder& OpDefBuilder::SetForwardTypeFn(TypeInferenceFn f) {
  op_reg_data_.fwd_type_fn = f;
  return *this;
}

OpDefBuilder& OpDefBuilder::SetReverseTypeFn(int input_number,
                                             TypeInferenceFn f) {
  op_reg_data_.rev_type_fn = f;
  op_reg_data_.rev_type_input = input_number;
  return *this;
}

OpDefBuilder& OpDefBuilder::SetShapeFn(OpShapeInferenceFn fn) {
  if (op_reg_data_.shape_inference_fn != nullptr) {
    errors_.push_back(
        absl::StrCat("SetShapeFn called twice for Op ", op_def()->name()));
  } else {
    op_reg_data_.shape_inference_fn = OpShapeInferenceFn(fn);
  }
  return *this;
}

OpDefBuilder& OpDefBuilder::AllowAttrTypeAny() {
  allow_attr_type_any_ = true;
  return *this;
}

absl::Status OpDefBuilder::Finalize(OpRegistrationData* op_reg_data) const {
  std::vector<string> errors = errors_;
  *op_reg_data = op_reg_data_;

  OpDef* op_def = &op_reg_data->op_def;
  for (absl::string_view attr : attrs_) {
    FinalizeAttr(attr, allow_attr_type_any_, op_def, &errors);
  }
  for (absl::string_view input : inputs_) {
    FinalizeInputOrOutput(input, false, op_def, &errors);
  }
  for (absl::string_view output : outputs_) {
    FinalizeInputOrOutput(output, true, op_def, &errors);
  }
  for (absl::string_view control_output : control_outputs_) {
    FinalizeControlOutput(control_output, op_def, &errors);
  }
  FinalizeDoc(doc_, op_def, &errors);

  if (op_reg_data->type_ctor != nullptr) {
    TF_RETURN_IF_ERROR(op_reg_data->type_ctor(op_def));
  }

  if (errors.empty()) return absl::OkStatus();
  return errors::InvalidArgument(absl::StrJoin(errors, "\n"));
}

}  // namespace tensorflow
