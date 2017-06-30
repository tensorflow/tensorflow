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

#include "tensorflow/core/framework/op_def_util.h"

#include <set>
#include <unordered_map>
#include <unordered_set>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/op_def.pb_text.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {  // ------ Helper functions ------

bool HasAttrStyleType(const OpDef::ArgDef& arg) {
  return arg.type() != DT_INVALID || !arg.type_attr().empty() ||
         !arg.type_list_attr().empty();
}

Status AllowedTypeValue(DataType dt, const OpDef::AttrDef& attr) {
  const AttrValue& allowed_values(attr.allowed_values());
  for (auto allowed : allowed_values.list().type()) {
    if (dt == allowed) {
      return Status::OK();
    }
  }
  string allowed_str;
  for (int i = 0; i < allowed_values.list().type_size(); ++i) {
    if (!allowed_str.empty()) {
      strings::StrAppend(&allowed_str, ", ");
    }
    strings::StrAppend(&allowed_str,
                       DataTypeString(allowed_values.list().type(i)));
  }
  return errors::InvalidArgument(
      "Value for attr '", attr.name(), "' of ", DataTypeString(dt),
      " is not in the list of allowed values: ", allowed_str);
}

Status AllowedStringValue(const string& str, const OpDef::AttrDef& attr) {
  const AttrValue& allowed_values(attr.allowed_values());
  for (const auto& allowed : allowed_values.list().s()) {
    if (str == allowed) {
      return Status::OK();
    }
  }
  string allowed_str;
  for (const string& allowed : allowed_values.list().s()) {
    if (!allowed_str.empty()) {
      strings::StrAppend(&allowed_str, ", ");
    }
    strings::StrAppend(&allowed_str, "\"", allowed, "\"");
  }
  return errors::InvalidArgument(
      "Value for attr '", attr.name(), "' of \"", str,
      "\" is not in the list of allowed values: ", allowed_str);
}

}  // namespace

// Requires: attr has already been validated.
Status ValidateAttrValue(const AttrValue& attr_value,
                         const OpDef::AttrDef& attr) {
  // Is it a valid value?
  TF_RETURN_WITH_CONTEXT_IF_ERROR(AttrValueHasType(attr_value, attr.type()),
                                  " for attr '", attr.name(), "'");

  // Does the value satisfy the minimum constraint in the AttrDef?
  if (attr.has_minimum()) {
    if (attr.type() == "int") {
      if (attr_value.i() < attr.minimum()) {
        return errors::InvalidArgument(
            "Value for attr '", attr.name(), "' of ", attr_value.i(),
            " must be at least minimum ", attr.minimum());
      }
    } else {
      int length = -1;
      if (attr.type() == "list(string)") {
        length = attr_value.list().s_size();
      } else if (attr.type() == "list(int)") {
        length = attr_value.list().i_size();
      } else if (attr.type() == "list(float)") {
        length = attr_value.list().f_size();
      } else if (attr.type() == "list(bool)") {
        length = attr_value.list().b_size();
      } else if (attr.type() == "list(type)") {
        length = attr_value.list().type_size();
      } else if (attr.type() == "list(shape)") {
        length = attr_value.list().shape_size();
      } else if (attr.type() == "list(tensor)") {
        length = attr_value.list().tensor_size();
      }
      if (length < attr.minimum()) {
        return errors::InvalidArgument(
            "Length for attr '", attr.name(), "' of ", length,
            " must be at least minimum ", attr.minimum());
      }
    }
  }

  // Does the value satisfy the allowed_value constraint in the AttrDef?
  if (attr.has_allowed_values()) {
    if (attr.type() == "type") {
      TF_RETURN_IF_ERROR(AllowedTypeValue(attr_value.type(), attr));
    } else if (attr.type() == "list(type)") {
      for (int dt : attr_value.list().type()) {
        TF_RETURN_IF_ERROR(AllowedTypeValue(static_cast<DataType>(dt), attr));
      }
    } else if (attr.type() == "string") {
      TF_RETURN_IF_ERROR(AllowedStringValue(attr_value.s(), attr));
    } else if (attr.type() == "list(string)") {
      for (const string& str : attr_value.list().s()) {
        TF_RETURN_IF_ERROR(AllowedStringValue(str, attr));
      }
    } else {
      return errors::Unimplemented(
          "Support for allowed_values not implemented for type ", attr.type());
    }
  }
  return Status::OK();
}

const OpDef::AttrDef* FindAttr(StringPiece name, const OpDef& op_def) {
  for (int i = 0; i < op_def.attr_size(); ++i) {
    if (op_def.attr(i).name() == name) {
      return &op_def.attr(i);
    }
  }
  return nullptr;
}

OpDef::AttrDef* FindAttrMutable(StringPiece name, OpDef* op_def) {
  for (int i = 0; i < op_def->attr_size(); ++i) {
    if (op_def->attr(i).name() == name) {
      return op_def->mutable_attr(i);
    }
  }
  return nullptr;
}

#define VALIDATE(EXPR, ...)                                          \
  do {                                                               \
    if (!(EXPR)) {                                                   \
      return errors::InvalidArgument(__VA_ARGS__, "; in OpDef: ",    \
                                     ProtoShortDebugString(op_def)); \
    }                                                                \
  } while (false)

static Status ValidateArg(const OpDef::ArgDef& arg, const OpDef& op_def,
                          bool output, std::set<string>* names) {
  const string suffix = strings::StrCat(
      output ? " for output '" : " for input '", arg.name(), "'");
  VALIDATE(gtl::InsertIfNotPresent(names, arg.name()), "Duplicate name: ",
           arg.name());
  VALIDATE(HasAttrStyleType(arg), "Missing type", suffix);

  if (!arg.number_attr().empty()) {
    const OpDef::AttrDef* attr = FindAttr(arg.number_attr(), op_def);
    VALIDATE(attr != nullptr, "No attr with name '", arg.number_attr(), "'",
             suffix);
    VALIDATE(attr->type() == "int", "Attr '", attr->name(), "' used as length",
             suffix, " has type ", attr->type(), " != int");
    VALIDATE(attr->has_minimum(), "Attr '", attr->name(), "' used as length",
             suffix, " must have minimum");
    VALIDATE(attr->minimum() >= 0, "Attr '", attr->name(), "' used as length",
             suffix, " must have minimum >= 0");
    VALIDATE(arg.type_list_attr().empty(),
             "Can't have both number_attr and type_list_attr", suffix);
    VALIDATE((arg.type() != DT_INVALID ? 1 : 0) +
                     (!arg.type_attr().empty() ? 1 : 0) ==
                 1,
             "Exactly one of type, type_attr must be set", suffix);
  } else {
    const int num_type_fields = (arg.type() != DT_INVALID ? 1 : 0) +
                                (!arg.type_attr().empty() ? 1 : 0) +
                                (!arg.type_list_attr().empty() ? 1 : 0);
    VALIDATE(num_type_fields == 1,
             "Exactly one of type, type_attr, type_list_attr must be set",
             suffix);
  }

  if (!arg.type_attr().empty()) {
    const OpDef::AttrDef* attr = FindAttr(arg.type_attr(), op_def);
    VALIDATE(attr != nullptr, "No attr with name '", arg.type_attr(), "'",
             suffix);
    VALIDATE(attr->type() == "type", "Attr '", attr->name(),
             "' used as type_attr", suffix, " has type ", attr->type(),
             " != type");
  } else if (!arg.type_list_attr().empty()) {
    const OpDef::AttrDef* attr = FindAttr(arg.type_list_attr(), op_def);
    VALIDATE(attr != nullptr, "No attr with name '", arg.type_list_attr(), "'",
             suffix);
    VALIDATE(attr->type() == "list(type)", "Attr '", attr->name(),
             "' used as type_list_attr", suffix, " has type ", attr->type(),
             " != list(type)");
  } else {
    // All argument types should be non-reference types at this point.
    // ArgDef.is_ref is set to true for reference arguments.
    VALIDATE(!IsRefType(arg.type()), "Illegal use of ref type '",
             DataTypeString(arg.type()), "'. Use 'Ref(type)' instead", suffix);
  }

  return Status::OK();
}

Status ValidateOpDef(const OpDef& op_def) {
  using ::tensorflow::strings::Scanner;

  if (!StringPiece(op_def.name()).starts_with("_")) {
    VALIDATE(Scanner(op_def.name())
                 .One(Scanner::UPPERLETTER)
                 .Any(Scanner::LETTER_DIGIT)
                 .Eos()
                 .GetResult(),
             "Invalid name: ", op_def.name(), " (Did you use CamelCase?)");
  }

  std::set<string> names;  // for detecting duplicate names
  for (const auto& attr : op_def.attr()) {
    // Validate name
    VALIDATE(gtl::InsertIfNotPresent(&names, attr.name()), "Duplicate name: ",
             attr.name());
    DataType dt;
    VALIDATE(!DataTypeFromString(attr.name(), &dt), "Attr can't have name ",
             attr.name(), " that matches a data type");

    // Validate type
    StringPiece type(attr.type());
    bool is_list = type.Consume("list(");
    bool found = false;
    for (StringPiece valid : {"string", "int", "float", "bool", "type", "shape",
                              "tensor", "func"}) {
      if (type.Consume(valid)) {
        found = true;
        break;
      }
    }
    VALIDATE(found, "Unrecognized type '", type, "' in attr '", attr.name(),
             "'");
    if (is_list) {
      VALIDATE(type.Consume(")"), "'list(' is missing ')' in attr ",
               attr.name(), "'s type ", attr.type());
    }
    VALIDATE(type.empty(), "Extra '", type, "' at the end of attr ",
             attr.name(), "'s type ", attr.type());

    // Validate minimum
    if (attr.has_minimum()) {
      VALIDATE(attr.type() == "int" || is_list, "Attr '", attr.name(),
               "' has minimum for unsupported type ", attr.type());
      if (is_list) {
        VALIDATE(attr.minimum() >= 0, "Attr '", attr.name(),
                 "' with list type must have a non-negative minimum, not ",
                 attr.minimum());
      }
    } else {
      VALIDATE(attr.minimum() == 0, "Attr '", attr.name(),
               "' with has_minimum = false but minimum ", attr.minimum(),
               " not equal to default of 0");
    }

    // Validate allowed_values
    if (attr.has_allowed_values()) {
      const string list_type =
          is_list ? attr.type() : strings::StrCat("list(", attr.type(), ")");
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          AttrValueHasType(attr.allowed_values(), list_type), " for attr '",
          attr.name(), "' in Op '", op_def.name(), "'");
    }

    // Validate default_value (after we have validated the rest of the attr,
    // so we can use ValidateAttrValue()).
    if (attr.has_default_value()) {
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          ValidateAttrValue(attr.default_value(), attr), " in Op '",
          op_def.name(), "'");
    }
  }

  for (const auto& arg : op_def.input_arg()) {
    TF_RETURN_IF_ERROR(ValidateArg(arg, op_def, false, &names));
  }

  for (const auto& arg : op_def.output_arg()) {
    TF_RETURN_IF_ERROR(ValidateArg(arg, op_def, true, &names));
  }

  return Status::OK();
}

#undef VALIDATE

Status CheckOpDeprecation(const OpDef& op_def, int graph_def_version) {
  if (op_def.has_deprecation()) {
    const OpDeprecation& dep = op_def.deprecation();
    if (graph_def_version >= dep.version()) {
      return errors::Unimplemented(
          "Op ", op_def.name(), " is not available in GraphDef version ",
          graph_def_version, ". It has been removed in version ", dep.version(),
          ". ", dep.explanation(), ".");
    } else {
      // Warn only once for each op name, and do it in a threadsafe manner.
      static mutex mu;
      static std::unordered_set<string> warned;
      bool warn;
      {
        mutex_lock lock(mu);
        warn = warned.insert(op_def.name()).second;
      }
      if (warn) {
        LOG(WARNING) << "Op " << op_def.name() << " is deprecated."
                     << " It will cease to work in GraphDef version "
                     << dep.version() << ". " << dep.explanation() << ".";
      }
    }
  }
  return Status::OK();
}

namespace {

string SummarizeArgs(const protobuf::RepeatedPtrField<OpDef::ArgDef>& args) {
  string ret;
  for (const OpDef::ArgDef& arg : args) {
    if (!ret.empty()) strings::StrAppend(&ret, ", ");
    strings::StrAppend(&ret, arg.name(), ":");
    if (arg.is_ref()) strings::StrAppend(&ret, "Ref(");
    if (!arg.number_attr().empty()) {
      strings::StrAppend(&ret, arg.number_attr(), "*");
    }
    if (arg.type() != DT_INVALID) {
      strings::StrAppend(&ret, DataTypeString(arg.type()));
    } else {
      strings::StrAppend(&ret, arg.type_attr());
    }
    if (arg.is_ref()) strings::StrAppend(&ret, ")");
  }
  return ret;
}

}  // namespace

string SummarizeOpDef(const OpDef& op_def) {
  string ret = strings::StrCat("Op<name=", op_def.name());
  strings::StrAppend(&ret, "; signature=", SummarizeArgs(op_def.input_arg()),
                     " -> ", SummarizeArgs(op_def.output_arg()));
  for (int i = 0; i < op_def.attr_size(); ++i) {
    strings::StrAppend(&ret, "; attr=", op_def.attr(i).name(), ":",
                       op_def.attr(i).type());
    if (op_def.attr(i).has_default_value()) {
      strings::StrAppend(&ret, ",default=",
                         SummarizeAttrValue(op_def.attr(i).default_value()));
    }
    if (op_def.attr(i).has_minimum()) {
      strings::StrAppend(&ret, ",min=", op_def.attr(i).minimum());
    }
    if (op_def.attr(i).has_allowed_values()) {
      strings::StrAppend(&ret, ",allowed=",
                         SummarizeAttrValue(op_def.attr(i).allowed_values()));
    }
  }
  if (op_def.is_commutative()) {
    strings::StrAppend(&ret, "; is_commutative=true");
  }
  if (op_def.is_aggregate()) {
    strings::StrAppend(&ret, "; is_aggregate=true");
  }
  if (op_def.is_stateful()) {
    strings::StrAppend(&ret, "; is_stateful=true");
  }
  if (op_def.allows_uninitialized_input()) {
    strings::StrAppend(&ret, "; allows_uninitialized_input=true");
  }
  strings::StrAppend(&ret, ">");
  return ret;
}

namespace {

// Returns true if every element of `sub` is contained in `super`.
template <class T>
bool IsSubsetOf(const T& sub, const T& super) {
  for (const auto& o : sub) {
    bool found = false;
    for (const auto& n : super) {
      if (o == n) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

bool MoreRestrictive(const OpDef::AttrDef& old_attr,
                     const OpDef::AttrDef& new_attr) {
  // Anything -> no restriction : not more restrictive.
  if (!new_attr.has_allowed_values()) return false;
  // No restriction -> restriction : more restrictive.
  if (!old_attr.has_allowed_values()) return true;
  // If anything that was previously allowed is no longer allowed:
  // more restrictive.
  if (!IsSubsetOf(old_attr.allowed_values().list().type(),
                  new_attr.allowed_values().list().type())) {
    return true;
  }
  if (!IsSubsetOf(old_attr.allowed_values().list().s(),
                  new_attr.allowed_values().list().s())) {
    return true;
  }
  return false;
}

string AllowedStr(const OpDef::AttrDef& attr) {
  if (!attr.has_allowed_values()) return "no restriction";
  return SummarizeAttrValue(attr.allowed_values());
}

bool HigherMinimum(const OpDef::AttrDef& old_attr,
                   const OpDef::AttrDef& new_attr) {
  // Anything -> no restriction : not more restrictive.
  if (!new_attr.has_minimum()) return false;
  // No restriction -> restriction : more restrictive.
  if (!old_attr.has_minimum()) return true;
  // If anything that was previously allowed is no longer allowed:
  // more restrictive.
  return new_attr.minimum() > old_attr.minimum();
}

string MinStr(const OpDef::AttrDef& attr) {
  if (!attr.has_minimum()) return "no minimum";
  return strings::StrCat(attr.minimum());
}

typedef std::unordered_map<string, const OpDef::AttrDef*> AttrMap;
void FillAttrMap(const OpDef& op_def, AttrMap* attr_map) {
  for (const auto& attr : op_def.attr()) {
    (*attr_map)[attr.name()] = &attr;
  }
}

// Add a comma to *s every call but the first (*add_comma should be
// initialized to false).
void AddComma(string* s, bool* add_comma) {
  if (*add_comma) {
    strings::StrAppend(s, ", ");
  } else {
    *add_comma = true;
  }
}

// Will add the `name` from arg if name is true.
void AddName(string* s, bool name, const OpDef::ArgDef& arg) {
  if (name) {
    strings::StrAppend(s, arg.name(), ":");
  }
}

// Compute a signature for either inputs or outputs that will be the
// same for both the old and new OpDef if they are compatible.  We
// assume that new_attrs is a superset of old_attrs, and that any attr
// in the difference has a default.  Our strategy is to make a list of
// types, where the types are things like:
// * "int32", "float", etc.,
// * "T" for some attr "T" in old_attrs, or
// * "N * type" for "N" either some attr in old_attrs.
//
// We get the types by either using the attrs in args if they are in
// old_attrs, or substituting the default value from new_attrs.
string ComputeArgSignature(
    const protobuf::RepeatedPtrField<OpDef::ArgDef>& args,
    const AttrMap& old_attrs, const AttrMap& new_attrs, std::vector<bool>* ref,
    bool names) {
  string s;
  bool add_comma = false;
  for (const OpDef::ArgDef& arg : args) {
    if (!arg.type_list_attr().empty()) {
      const OpDef::AttrDef* old_attr =
          gtl::FindPtrOrNull(old_attrs, arg.type_list_attr());
      if (old_attr) {
        // Both old and new have the list(type) attr, so can use it directly.
        AddComma(&s, &add_comma);
        AddName(&s, names, arg);
        strings::StrAppend(&s, arg.type_list_attr());
        ref->push_back(arg.is_ref());
      } else {
        // Missing the list(type) attr in the old, so use the default
        // value for the attr from new instead.
        const OpDef::AttrDef* new_attr =
            gtl::FindPtrOrNull(new_attrs, arg.type_list_attr());
        const auto& type_list = new_attr->default_value().list().type();
        if (type_list.empty()) continue;
        for (int i = 0; i < type_list.size(); ++i) {
          AddComma(&s, &add_comma);
          AddName(&s, names, arg);
          strings::StrAppend(
              &s, DataTypeString(static_cast<DataType>(type_list.Get(i))));
          ref->push_back(arg.is_ref());
        }
      }
    } else {
      int num = 1;  // How many input/outputs does this represent?
      string type;  // What is the type of this arg?
      AddName(&type, names, arg);
      if (!arg.number_attr().empty()) {
        // N * type case.
        const OpDef::AttrDef* old_attr =
            gtl::FindPtrOrNull(old_attrs, arg.number_attr());
        if (old_attr) {
          // Both old and new have the number attr, so can use it directly.
          strings::StrAppend(&type, arg.number_attr(), " * ");
        } else {
          // Missing the number attr in the old, so use the default
          // value for the attr from new instead.
          const OpDef::AttrDef* new_attr =
              gtl::FindPtrOrNull(new_attrs, arg.number_attr());
          num = new_attr->default_value().i();
        }
      }

      if (arg.type() != DT_INVALID) {
        // int32, float, etc. case
        strings::StrAppend(&type, DataTypeString(arg.type()));
      } else {
        const OpDef::AttrDef* old_attr =
            gtl::FindPtrOrNull(old_attrs, arg.type_attr());
        if (old_attr) {
          // Both old and new have the type attr, so can use it directly.
          strings::StrAppend(&type, arg.type_attr());
        } else {
          // Missing the type attr in the old, so use the default
          // value for the attr from new instead.
          const OpDef::AttrDef* new_attr =
              gtl::FindPtrOrNull(new_attrs, arg.type_attr());
          strings::StrAppend(&type,
                             DataTypeString(new_attr->default_value().type()));
        }
      }

      // Record `num` * `type` in the signature.
      for (int i = 0; i < num; ++i) {
        AddComma(&s, &add_comma);
        strings::StrAppend(&s, type);
        ref->push_back(arg.is_ref());
      }
    }
  }

  return s;
}

}  // namespace

Status OpDefCompatible(const OpDef& old_op, const OpDef& new_op) {
#define VALIDATE(CONDITION, ...)                                            \
  if (!(CONDITION)) {                                                       \
    return errors::InvalidArgument("Incompatible Op change: ", __VA_ARGS__, \
                                   "; old: ", SummarizeOpDef(old_op),       \
                                   "; new: ", SummarizeOpDef(new_op));      \
  }

  VALIDATE(old_op.name() == new_op.name(), "Name mismatch");

  AttrMap new_attrs, old_attrs;
  FillAttrMap(old_op, &old_attrs);
  FillAttrMap(new_op, &new_attrs);
  for (const auto& old_attr : old_op.attr()) {
    const OpDef::AttrDef* new_attr =
        gtl::FindPtrOrNull(new_attrs, old_attr.name());
    VALIDATE(new_attr != nullptr, "Attr '", old_attr.name(), "' removed");
    VALIDATE(old_attr.type() == new_attr->type(), "Attr '", old_attr.name(),
             "' changed type '", old_attr.type(), "' -> '", new_attr->type(),
             "'");
    VALIDATE(!MoreRestrictive(old_attr, *new_attr), "Attr '", old_attr.name(),
             "' has a stricter set of allowed values; from ",
             AllowedStr(old_attr), " to ", AllowedStr(*new_attr));
    VALIDATE(!HigherMinimum(old_attr, *new_attr), "Attr '", old_attr.name(),
             "' has a higher minimum; from ", MinStr(old_attr), " to ",
             MinStr(*new_attr));
  }

  for (const auto& new_attr : new_op.attr()) {
    const OpDef::AttrDef* old_attr =
        gtl::FindPtrOrNull(old_attrs, new_attr.name());
    VALIDATE(old_attr != nullptr || new_attr.has_default_value(), "Attr '",
             new_attr.name(), "' added without default");
  }

  std::vector<bool> old_in_ref, new_in_ref, old_out_ref, new_out_ref;
  const string old_in_sig = ComputeArgSignature(
      old_op.input_arg(), old_attrs, new_attrs, &old_in_ref, false /* names */);
  const string new_in_sig = ComputeArgSignature(
      new_op.input_arg(), old_attrs, new_attrs, &new_in_ref, false /* names */);
  VALIDATE(old_in_sig == new_in_sig, "Input signature mismatch '", old_in_sig,
           "' vs. '", new_in_sig, "'");
  VALIDATE(old_in_ref.size() == new_in_ref.size(),  // Should not happen
           "Unexpected change in input ref lists.");
  for (int i = 0; i < old_in_ref.size(); ++i) {
    // Allowed to remove "ref" from an input (or leave it unchanged).
    VALIDATE(old_in_ref[i] || !new_in_ref[i], "Input ", i,
             " changed from non-ref to ref");
  }

  const string old_out_sig =
      ComputeArgSignature(old_op.output_arg(), old_attrs, new_attrs,
                          &old_out_ref, true /* names */);
  const string new_out_sig =
      ComputeArgSignature(new_op.output_arg(), old_attrs, new_attrs,
                          &new_out_ref, true /* names */);
  VALIDATE(old_out_sig == new_out_sig, "Output signature mismatch '",
           old_out_sig, "' vs. '", new_out_sig, "'");
  VALIDATE(old_out_ref.size() == new_out_ref.size(),  // Should not happen
           "Unexpected change in output ref lists");
  for (int i = 0; i < old_out_ref.size(); ++i) {
    // Allowed to add "ref" to an output (or leave it unchanged).
    VALIDATE(!old_out_ref[i] || new_out_ref[i], "Output ", i,
             " changed from ref to non-ref");
  }

  return Status::OK();
}

Status OpDefAddedDefaultsUnchanged(const OpDef& old_op,
                                   const OpDef& penultimate_op,
                                   const OpDef& new_op) {
  AttrMap new_attrs, old_attrs;
  FillAttrMap(old_op, &old_attrs);
  FillAttrMap(new_op, &new_attrs);

  for (const auto& penultimate_attr : penultimate_op.attr()) {
    const OpDef::AttrDef* old_attr =
        gtl::FindPtrOrNull(old_attrs, penultimate_attr.name());
    if (old_attr != nullptr) continue;  // attr wasn't added
    const OpDef::AttrDef* new_attr =
        gtl::FindPtrOrNull(new_attrs, penultimate_attr.name());

    // These shouldn't happen if the op passed OpDefCompatible().
    if (new_attr == nullptr) {
      return errors::InvalidArgument("Missing attr '", penultimate_attr.name(),
                                     "' in op: ", SummarizeOpDef(new_op));
    }
    if (!penultimate_attr.has_default_value() ||
        !new_attr->has_default_value()) {
      return errors::InvalidArgument("Missing default for attr '",
                                     penultimate_attr.name(), "' in op: ",
                                     SummarizeOpDef(new_op));
    }

    // Actually test that the attr's default value hasn't changed.
    if (!AreAttrValuesEqual(penultimate_attr.default_value(),
                            new_attr->default_value())) {
      return errors::InvalidArgument(
          "Can't change default value for attr '", penultimate_attr.name(),
          "' from ", SummarizeAttrValue(penultimate_attr.default_value()),
          " in op: ", SummarizeOpDef(new_op));
    }
  }

  return Status::OK();
}

void RemoveNonDeprecationDescriptionsFromOpDef(OpDef* op_def) {
  for (int i = 0; i < op_def->input_arg_size(); ++i) {
    op_def->mutable_input_arg(i)->clear_description();
  }
  for (int i = 0; i < op_def->output_arg_size(); ++i) {
    op_def->mutable_output_arg(i)->clear_description();
  }
  for (int i = 0; i < op_def->attr_size(); ++i) {
    op_def->mutable_attr(i)->clear_description();
  }
  op_def->clear_summary();
  op_def->clear_description();
}

void RemoveDescriptionsFromOpDef(OpDef* op_def) {
  RemoveNonDeprecationDescriptionsFromOpDef(op_def);
  if (op_def->has_deprecation()) {
    op_def->mutable_deprecation()->clear_explanation();
  }
}

void RemoveDescriptionsFromOpList(OpList* op_list) {
  for (int i = 0; i < op_list->op_size(); ++i) {
    OpDef* op_def = op_list->mutable_op(i);
    RemoveDescriptionsFromOpDef(op_def);
  }
}

}  // namespace tensorflow
