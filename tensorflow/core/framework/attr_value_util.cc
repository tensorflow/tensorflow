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

#include "tensorflow/core/framework/attr_value_util.h"

#include <vector>
#include "tensorflow/core/framework/attr_value.pb_text.h"
#include "tensorflow/core/framework/tensor.pb_text.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb_text.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

namespace {

string SummarizeString(const string& str) {
  return strings::StrCat("\"", str_util::CEscape(str), "\"");
}

string SummarizeTensor(const TensorProto& tensor_proto) {
  Tensor t;
  if (!t.FromProto(tensor_proto)) {
    return strings::StrCat(
        "<Invalid TensorProto: ", ProtoShortDebugString(tensor_proto), ">");
  }
  return t.DebugString();
}

string SummarizeFunc(const NameAttrList& func) {
  std::vector<string> entries;
  for (auto p : func.attr()) {
    entries.push_back(
        strings::StrCat(p.first, "=", SummarizeAttrValue(p.second)));
  }
  std::sort(entries.begin(), entries.end());
  return strings::StrCat(func.name(), "[", str_util::Join(entries, ", "), "]");
}

}  // namespace

string SummarizeAttrValue(const AttrValue& attr_value) {
  switch (attr_value.value_case()) {
    case AttrValue::kS:
      return SummarizeString(attr_value.s());
    case AttrValue::kI:
      return strings::StrCat(attr_value.i());
    case AttrValue::kF:
      return strings::StrCat(attr_value.f());
    case AttrValue::kB:
      return attr_value.b() ? "true" : "false";
    case AttrValue::kType:
      return EnumName_DataType(attr_value.type());
    case AttrValue::kShape:
      return PartialTensorShape::DebugString(attr_value.shape());
    case AttrValue::kTensor:
      return SummarizeTensor(attr_value.tensor());
    case AttrValue::kList: {
      string ret = "[";
      if (attr_value.list().s_size() > 0) {
        for (int i = 0; i < attr_value.list().s_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(&ret, SummarizeString(attr_value.list().s(i)));
        }
      } else if (attr_value.list().i_size() > 0) {
        for (int i = 0; i < attr_value.list().i_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(&ret, attr_value.list().i(i));
        }
      } else if (attr_value.list().f_size() > 0) {
        for (int i = 0; i < attr_value.list().f_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(&ret, attr_value.list().f(i));
        }
      } else if (attr_value.list().b_size() > 0) {
        for (int i = 0; i < attr_value.list().b_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(&ret, attr_value.list().b(i) ? "true" : "false");
        }
      } else if (attr_value.list().type_size() > 0) {
        for (int i = 0; i < attr_value.list().type_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(&ret,
                             EnumName_DataType(attr_value.list().type(i)));
        }
      } else if (attr_value.list().shape_size() > 0) {
        for (int i = 0; i < attr_value.list().shape_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(
              &ret, TensorShape::DebugString(attr_value.list().shape(i)));
        }
      } else if (attr_value.list().tensor_size() > 0) {
        for (int i = 0; i < attr_value.list().tensor_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(&ret,
                             SummarizeTensor(attr_value.list().tensor(i)));
        }
      } else if (attr_value.list().func_size() > 0) {
        for (int i = 0; i < attr_value.list().func_size(); ++i) {
          if (i > 0) strings::StrAppend(&ret, ", ");
          strings::StrAppend(&ret, SummarizeFunc(attr_value.list().func(i)));
        }
      }

      strings::StrAppend(&ret, "]");
      return ret;
    }
    case AttrValue::kFunc: {
      return SummarizeFunc(attr_value.func());
    }
    case AttrValue::kPlaceholder:
      return strings::StrCat("$", attr_value.placeholder());
    case AttrValue::VALUE_NOT_SET:
      return "<Unknown AttrValue type>";
  }
  return "<Unknown AttrValue type>";  // Prevent missing return warning
}

Status AttrValueHasType(const AttrValue& attr_value, StringPiece type) {
  int num_set = 0;

#define VALIDATE_FIELD(name, type_string, oneof_case)                         \
  do {                                                                        \
    if (attr_value.has_list()) {                                              \
      if (attr_value.list().name##_size() > 0) {                              \
        if (type != "list(" type_string ")") {                                \
          return errors::InvalidArgument(                                     \
              "AttrValue had value with type 'list(" type_string ")' when '", \
              type, "' expected");                                            \
        }                                                                     \
        ++num_set;                                                            \
      }                                                                       \
    } else if (attr_value.value_case() == AttrValue::oneof_case) {            \
      if (type != type_string) {                                              \
        return errors::InvalidArgument(                                       \
            "AttrValue had value with type '" type_string "' when '", type,   \
            "' expected");                                                    \
      }                                                                       \
      ++num_set;                                                              \
    }                                                                         \
  } while (false)

  VALIDATE_FIELD(s, "string", kS);
  VALIDATE_FIELD(i, "int", kI);
  VALIDATE_FIELD(f, "float", kF);
  VALIDATE_FIELD(b, "bool", kB);
  VALIDATE_FIELD(type, "type", kType);
  VALIDATE_FIELD(shape, "shape", kShape);
  VALIDATE_FIELD(tensor, "tensor", kTensor);
  VALIDATE_FIELD(func, "func", kFunc);

#undef VALIDATE_FIELD

  if (attr_value.value_case() == AttrValue::kPlaceholder) {
    return errors::InvalidArgument(
        "AttrValue had value with unexpected type 'placeholder'");
  }

  // If the attr type is 'list', we expect attr_value.has_list() to be
  // true.  However, proto3's attr_value.has_list() can be false when
  // set to an empty list for GraphDef versions <= 4. So we simply
  // check if has_list is false and some other field in attr_value is
  // set to flag the error.  This test can be made more strict once
  // support for GraphDef versions <= 4 is dropped.
  if (StringPiece(type).starts_with("list(") && !attr_value.has_list()) {
    if (num_set) {
      return errors::InvalidArgument(
          "AttrValue missing value with expected type '", type, "'");
    } else {
      // Indicate that we have a list, but an empty one.
      ++num_set;
    }
  }

  // Okay to have an empty list, but not to be missing a non-list value.
  if (num_set == 0 && !StringPiece(type).starts_with("list(")) {
    return errors::InvalidArgument(
        "AttrValue missing value with expected type '", type, "'");
  }

  // Ref types and DT_INVALID are illegal, and DataTypes must
  // be a valid enum type.
  if (type == "type") {
    if (!DataType_IsValid(attr_value.type())) {
      return errors::InvalidArgument("AttrValue has invalid DataType enum: ",
                                     attr_value.type());
    }
    if (IsRefType(attr_value.type())) {
      return errors::InvalidArgument(
          "AttrValue must not have reference type value of ",
          DataTypeString(attr_value.type()));
    }
    if (attr_value.type() == DT_INVALID) {
      return errors::InvalidArgument("AttrValue has invalid DataType");
    }
  } else if (type == "list(type)") {
    for (auto as_int : attr_value.list().type()) {
      const DataType dtype = static_cast<DataType>(as_int);
      if (!DataType_IsValid(dtype)) {
        return errors::InvalidArgument("AttrValue has invalid DataType enum: ",
                                       as_int);
      }
      if (IsRefType(dtype)) {
        return errors::InvalidArgument(
            "AttrValue must not have reference type value of ",
            DataTypeString(dtype));
      }
      if (dtype == DT_INVALID) {
        return errors::InvalidArgument("AttrValue contains invalid DataType");
      }
    }
  }

  return Status::OK();
}

bool ParseAttrValue(StringPiece type, StringPiece text, AttrValue* out) {
  // Parse type.
  string field_name;
  bool is_list = type.Consume("list(");
  if (type.Consume("string")) {
    field_name = "s";
  } else if (type.Consume("int")) {
    field_name = "i";
  } else if (type.Consume("float")) {
    field_name = "f";
  } else if (type.Consume("bool")) {
    field_name = "b";
  } else if (type.Consume("type")) {
    field_name = "type";
  } else if (type.Consume("shape")) {
    field_name = "shape";
  } else if (type.Consume("tensor")) {
    field_name = "tensor";
  } else if (type.Consume("func")) {
    field_name = "func";
  } else if (type.Consume("placeholder")) {
    field_name = "placeholder";
  } else {
    return false;
  }
  if (is_list && !type.Consume(")")) {
    return false;
  }

  // Construct a valid text proto message to parse.
  string to_parse;
  if (is_list) {
    // TextFormat parser considers "i: 7" to be the same as "i: [7]",
    // but we only want to allow list values with [].
    StringPiece cleaned = text;
    str_util::RemoveLeadingWhitespace(&cleaned);
    str_util::RemoveTrailingWhitespace(&cleaned);
    if (cleaned.size() < 2 || cleaned[0] != '[' ||
        cleaned[cleaned.size() - 1] != ']') {
      return false;
    }
    cleaned.remove_prefix(1);
    str_util::RemoveLeadingWhitespace(&cleaned);
    if (cleaned.size() == 1) {
      // User wrote "[]", so return empty list without invoking the TextFormat
      // parse which returns an error for "i: []".
      out->Clear();
      out->mutable_list();
      return true;
    }
    to_parse = strings::StrCat("list { ", field_name, ": ", text, " }");
  } else {
    to_parse = strings::StrCat(field_name, ": ", text);
  }

  return ProtoParseFromString(to_parse, out);
}

void SetAttrValue(const AttrValue& value, AttrValue* out) { *out = value; }

#define DEFINE_SET_ATTR_VALUE_ONE(ARG_TYPE, FIELD) \
  void SetAttrValue(ARG_TYPE value, AttrValue* out) { out->set_##FIELD(value); }

#define DEFINE_SET_ATTR_VALUE_LIST(ARG_TYPE, FIELD)                       \
  void SetAttrValue(ARG_TYPE value, AttrValue* out) {                     \
    out->mutable_list()->Clear(); /* create list() even if value empty */ \
    for (const auto& v : value) {                                         \
      out->mutable_list()->add_##FIELD(v);                                \
    }                                                                     \
  }

#define DEFINE_SET_ATTR_VALUE_BOTH(ARG_TYPE, FIELD) \
  DEFINE_SET_ATTR_VALUE_ONE(ARG_TYPE, FIELD)        \
  DEFINE_SET_ATTR_VALUE_LIST(gtl::ArraySlice<ARG_TYPE>, FIELD)

DEFINE_SET_ATTR_VALUE_ONE(const string&, s)
DEFINE_SET_ATTR_VALUE_LIST(gtl::ArraySlice<string>, s)
DEFINE_SET_ATTR_VALUE_BOTH(const char*, s)
DEFINE_SET_ATTR_VALUE_BOTH(int64, i)
DEFINE_SET_ATTR_VALUE_BOTH(int32, i)
DEFINE_SET_ATTR_VALUE_BOTH(float, f)
DEFINE_SET_ATTR_VALUE_BOTH(double, f)
DEFINE_SET_ATTR_VALUE_BOTH(bool, b)
DEFINE_SET_ATTR_VALUE_LIST(const std::vector<bool>&, b)
DEFINE_SET_ATTR_VALUE_LIST(std::initializer_list<bool>, b)
DEFINE_SET_ATTR_VALUE_BOTH(DataType, type)

void SetAttrValue(StringPiece value, AttrValue* out) {
  out->set_s(value.data(), value.size());
}

void SetAttrValue(const gtl::ArraySlice<StringPiece> value, AttrValue* out) {
  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    out->mutable_list()->add_s(v.data(), v.size());
  }
}

void SetAttrValue(const TensorShape& value, AttrValue* out) {
  value.AsProto(out->mutable_shape());
}

void SetAttrValue(const TensorShapeProto& value, AttrValue* out) {
  *out->mutable_shape() = value;
}

void SetAttrValue(const PartialTensorShape& value, AttrValue* out) {
  value.AsProto(out->mutable_shape());
}

void SetAttrValue(const gtl::ArraySlice<TensorShape> value, AttrValue* out) {
  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    v.AsProto(out->mutable_list()->add_shape());
  }
}

void SetAttrValue(gtl::ArraySlice<TensorShapeProto> value, AttrValue* out) {
  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    *out->mutable_list()->add_shape() = v;
  }
}

void SetAttrValue(const gtl::ArraySlice<PartialTensorShape> value,
                  AttrValue* out) {
  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    v.AsProto(out->mutable_list()->add_shape());
  }
}

void SetAttrValue(const Tensor& value, AttrValue* out) {
  if (value.NumElements() > 1) {
    value.AsProtoTensorContent(out->mutable_tensor());
  } else {
    value.AsProtoField(out->mutable_tensor());
  }
}

void SetAttrValue(const gtl::ArraySlice<Tensor> value, AttrValue* out) {
  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    if (v.NumElements() > 1) {
      v.AsProtoTensorContent(out->mutable_list()->add_tensor());
    } else {
      v.AsProtoField(out->mutable_list()->add_tensor());
    }
  }
}

void SetAttrValue(const TensorProto& value, AttrValue* out) {
  *out->mutable_tensor() = value;
}

void SetAttrValue(const gtl::ArraySlice<TensorProto> value, AttrValue* out) {
  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    *out->mutable_list()->add_tensor() = v;
  }
}

void SetAttrValue(const NameAttrList& value, AttrValue* out) {
  *out->mutable_func() = value;
}

void SetAttrValue(gtl::ArraySlice<NameAttrList> value, AttrValue* out) {
  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    *out->mutable_list()->add_func() = v;
  }
}

// Wrapper around protocol buffer serialization that requests deterministic
// serialization, in particular for Map fields, which serialize in a random
// order by default. Returns true on success.
template <typename T>
static bool DeterministicSerialization(const T& t, string* result) {
  const int size = t.ByteSize();
  *result = string(size, '\0');
  ::tensorflow::protobuf::io::ArrayOutputStream array_stream(&(*result)[0],
                                                             size);
  ::tensorflow::protobuf::io::CodedOutputStream output_stream(&array_stream);
  output_stream.SetSerializationDeterministic(true);
  t.SerializeWithCachedSizes(&output_stream);
  return !output_stream.HadError() && size == output_stream.ByteCount();
}

bool AreAttrValuesEqual(const AttrValue& a, const AttrValue& b) {
  string a_str, b_str;
  DeterministicSerialization(a, &a_str);
  DeterministicSerialization(b, &b_str);
  // Note: it should be safe to compare proto serializations of the attr
  // values since at most one field should be set in each (indeed, it
  // must be the same field if they are to compare equal).
  // Exception: there are multiple equivalent representations of
  // TensorProtos.  So a return value of true implies a == b, but not the
  // converse.
  // TODO(phawkins): this is incorrect for NameAttrList attributes that may
  // contain nested AttrValue maps.
  return a_str == b_str;
}

bool HasPlaceHolder(const AttrValue& val) {
  switch (val.value_case()) {
    case AttrValue::kList: {
      for (const NameAttrList& func : val.list().func()) {
        for (const auto& p : func.attr()) {
          if (HasPlaceHolder(p.second)) {
            return true;
          }
        }
      }
      break;
    }
    case AttrValue::kFunc:
      for (const auto& p : val.func().attr()) {
        if (HasPlaceHolder(p.second)) {
          return true;
        }
      }
      break;
    case AttrValue::kPlaceholder:
      return true;
    default:
      break;
  }
  return false;
}

bool SubstitutePlaceholders(SubstituteFunc substitute, AttrValue* value) {
  switch (value->value_case()) {
    case AttrValue::kList: {
      for (NameAttrList& func : *value->mutable_list()->mutable_func()) {
        for (auto& p : *func.mutable_attr()) {
          if (!SubstitutePlaceholders(substitute, &p.second)) {
            return false;
          }
        }
      }
      break;
    }
    case AttrValue::kFunc:
      for (auto& p : *(value->mutable_func()->mutable_attr())) {
        if (!SubstitutePlaceholders(substitute, &p.second)) {
          return false;
        }
      }
      break;
    case AttrValue::kPlaceholder:
      return substitute(value->placeholder(), value);
    case AttrValue::VALUE_NOT_SET:
      return false;
    default:
      break;
  }
  return true;
}

}  // namespace tensorflow
