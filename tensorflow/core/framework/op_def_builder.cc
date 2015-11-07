#include "tensorflow/core/framework/op_def_builder.h"

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {

namespace {

bool RE2Consume(StringPiece* sp, const char* pattern) {
  RegexpStringPiece base_sp = ToRegexpStringPiece(*sp);
  bool r = RE2::Consume(&base_sp, pattern);
  *sp = FromRegexpStringPiece(base_sp);
  return r;
}

bool RE2Consume(StringPiece* sp, const char* pattern, StringPiece* out) {
  RegexpStringPiece base_sp = ToRegexpStringPiece(*sp);
  RegexpStringPiece base_out;
  bool r = RE2::Consume(&base_sp, pattern, &base_out);
  *sp = FromRegexpStringPiece(base_sp);
  *out = FromRegexpStringPiece(base_out);
  return r;
}

bool RE2Consume(StringPiece* sp, const char* pattern, int64* out) {
  RegexpStringPiece base_sp = ToRegexpStringPiece(*sp);
  bool r = RE2::Consume(&base_sp, pattern, out);
  *sp = FromRegexpStringPiece(base_sp);
  return r;
}

string AttrError(StringPiece orig, const string& op_name) {
  return strings::StrCat(" from Attr(\"", orig, "\") for Op ", op_name);
}

#define VERIFY(expr, ...)                                                 \
  do {                                                                    \
    if (!(expr)) {                                                        \
      errors->push_back(                                                  \
          strings::StrCat(__VA_ARGS__, AttrError(orig, op_def->name()))); \
      return;                                                             \
    }                                                                     \
  } while (false)

void FinalizeAttr(StringPiece spec, OpDef* op_def,
                  std::vector<string>* errors) {
  OpDef::AttrDef* attr = op_def->add_attr();
  StringPiece orig(spec);

  // Parse "<name>:" at the beginning.
  StringPiece tmp_name;
  VERIFY(RE2Consume(&spec, "([a-zA-Z][a-zA-Z0-9_]*)\\s*:\\s*", &tmp_name),
         "Trouble parsing '<name>:'");
  attr->set_name(tmp_name.data(), tmp_name.size());

  // Read "<type>" or "list(<type>)".
  bool is_list = RE2Consume(&spec, "list\\s*\\(\\s*");
  string type;
  if (spec.Consume("string")) {
    type = "string";
  } else if (spec.Consume("int")) {
    type = "int";
  } else if (spec.Consume("float")) {
    type = "float";
  } else if (spec.Consume("bool")) {
    type = "bool";
  } else if (spec.Consume("type")) {
    type = "type";
  } else if (spec.Consume("shape")) {
    type = "shape";
  } else if (spec.Consume("tensor")) {
    type = "tensor";
  } else if (spec.Consume("func")) {
    type = "func";
  } else if (spec.Consume("numbertype") || spec.Consume("numerictype")) {
    type = "type";
    AttrValue* allowed = attr->mutable_allowed_values();
    for (DataType dt : NumberTypes()) {
      allowed->mutable_list()->add_type(dt);
    }
  } else if (spec.Consume("quantizedtype")) {
    type = "type";
    AttrValue* allowed = attr->mutable_allowed_values();
    for (DataType dt : QuantizedTypes()) {
      allowed->mutable_list()->add_type(dt);
    }
  } else if (spec.Consume("realnumbertype") ||
             spec.Consume("realnumerictype")) {
    type = "type";
    AttrValue* allowed = attr->mutable_allowed_values();
    for (DataType dt : RealNumberTypes()) {
      allowed->mutable_list()->add_type(dt);
    }
  } else if (spec.Consume("{")) {
    // e.g. "{ int32, float, bool }" or "{ \"foo\", \"bar\" }"
    RE2Consume(&spec, "\\s*");
    AttrValue* allowed = attr->mutable_allowed_values();
    if (spec.starts_with("\"") || spec.starts_with("'")) {
      type = "string";  // "{ \"foo\", \"bar\" }" or "{ 'foo', 'bar' }"
      while (true) {
        StringPiece escaped_string;
        VERIFY((RE2Consume(&spec, R"xx("((?:[^"\\]|\\.)*)"\s*)xx",
                           &escaped_string) ||
                RE2Consume(&spec, R"xx('((?:[^'\\]|\\.)*)'\s*)xx",
                           &escaped_string)),
               "Trouble parsing allowed string at '", spec, "'");
        string unescaped;
        string error;
        VERIFY(str_util::CUnescape(escaped_string, &unescaped, &error),
               "Trouble unescaping \"", escaped_string, "\", got error: ",
               error);
        allowed->mutable_list()->add_s(unescaped);
        if (spec.Consume(",")) {
          RE2Consume(&spec, "\\s*");
          if (spec.Consume("}")) break;  // Allow ending with ", }".
        } else {
          VERIFY(spec.Consume("}"),
                 "Expected , or } after strings in list, not: '", spec, "'");
          break;
        }
      }
    } else {  // "{ int32, float, bool }"
      type = "type";
      while (true) {
        StringPiece type_string;
        VERIFY(RE2Consume(&spec, "([a-z0-9]+)\\s*", &type_string),
               "Trouble parsing type string at '", spec, "'");
        DataType dt;
        VERIFY(DataTypeFromString(type_string, &dt),
               "Unrecognized type string '", type_string, "'");
        allowed->mutable_list()->add_type(dt);
        if (spec.Consume(",")) {
          RE2Consume(&spec, "\\s*");
          if (spec.Consume("}")) break;  // Allow ending with ", }".
        } else {
          VERIFY(spec.Consume("}"),
                 "Expected , or } after types in list, not: '", spec, "'");
          break;
        }
      }
    }
  } else {
    VERIFY(false, "Trouble parsing type string at '", spec, "'");
  }
  RE2Consume(&spec, "\\s*");

  // Write the type into *attr.
  if (is_list) {
    VERIFY(spec.Consume(")"), "Expected ) to close 'list(', not: '", spec, "'");
    RE2Consume(&spec, "\\s*");
    attr->set_type(strings::StrCat("list(", type, ")"));
  } else {
    attr->set_type(type);
  }

  // Read optional minimum constraint at the end.
  if ((is_list || type == "int") && spec.Consume(">=")) {
    int64 min_limit = -999;
    VERIFY(RE2Consume(&spec, "\\s*(-?\\d+)\\s*", &min_limit),
           "Could not parse integer lower limit after '>=', found '", spec,
           "' instead");
    attr->set_has_minimum(true);
    attr->set_minimum(min_limit);
  }

  // Parse default value, if present.
  if (spec.Consume("=")) {
    RE2Consume(&spec, "\\s*");
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
  VERIFY(RE2Consume(&spec, "([a-z][a-z0-9_]*)\\s*:\\s*", &tmp_name),
         "Trouble parsing 'name:'");
  arg->set_name(tmp_name.data(), tmp_name.size());

  // Detect "Ref(...)".
  if (RE2Consume(&spec, "Ref\\s*\\(\\s*")) {
    arg->set_is_ref(true);
  }

  {  // Parse "<name|type>" or "<name>*<name|type>".
    StringPiece first, second, type_or_attr;
    VERIFY(RE2Consume(&spec, "([a-zA-Z][a-zA-Z0-9_]*)\\s*", &first),
           "Trouble parsing either a type or an attr name at '", spec, "'");
    if (RE2Consume(&spec, "[*]\\s*([a-zA-Z][a-zA-Z0-9_]*)\\s*", &second)) {
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
    VERIFY(RE2Consume(&spec, "\\)\\s*"),
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
}

#undef VERIFY

int num_leading_spaces(StringPiece s) {
  size_t i = 0;
  while (i < s.size() && s[i] == ' ') {
    ++i;
  }
  return i;
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
  while (static_cast<size_t>(l) < lines.size() &&
         !RE2::PartialMatch(lines[l], "^[a-zA-Z][a-zA-Z0-9_]*\\s*:")) {
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
    RE2Consume(&description.back(), "([a-zA-Z][a-zA-Z0-9_]*)\\s*:\\s*", &name);
    ++l;
    while (static_cast<size_t>(l) < lines.size() &&
           !RE2::PartialMatch(lines[l], "^[a-zA-Z][a-zA-Z0-9_]*\\s*:")) {
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
  op_def_.set_name(op_name.ToString());  // NOLINT
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

OpDefBuilder& OpDefBuilder::Doc(StringPiece text) {
  if (!doc_.empty()) {
    errors_.push_back(
        strings::StrCat("Extra call to Doc() for Op ", op_def_.name()));
  } else {
    doc_.assign(text.data(), text.size());
  }
  return *this;
}

OpDefBuilder& OpDefBuilder::SetIsCommutative() {
  op_def_.set_is_commutative(true);
  return *this;
}

OpDefBuilder& OpDefBuilder::SetIsAggregate() {
  op_def_.set_is_aggregate(true);
  return *this;
}

OpDefBuilder& OpDefBuilder::SetIsStateful() {
  op_def_.set_is_stateful(true);
  return *this;
}

OpDefBuilder& OpDefBuilder::SetAllowsUninitializedInput() {
  op_def_.set_allows_uninitialized_input(true);
  return *this;
}

Status OpDefBuilder::Finalize(OpDef* op_def) const {
  std::vector<string> errors = errors_;
  *op_def = op_def_;

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
