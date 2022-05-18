/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/ops/gen/cpp/views/attr_view.h"

#include <string>

#include "absl/strings/substitute.h"
#include "tensorflow/c/experimental/ops/gen/common/case_format.h"
#include "tensorflow/c/experimental/ops/gen/common/view_util.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace generator {
namespace cpp {

string AttrView::VariableName() const { return attr_.name(); }

string AttrView::VariableType() const {
  if (attr_.full_type() == "string") {
    return "const char*";
  }
  if (attr_.full_type() == "type") {
    return "DataType";
  }
  if (attr_.full_type() == "shape") {
    return "const PartialTensorShape";
  }
  if (attr_.full_type() == "list(string)") {
    return "absl::Span<string const>";
  }
  return attr_.full_type();
}

string AttrView::AttrNameString() const { return Quoted(attr_.name()); }

string AttrView::DefaultValue() const {
  const AttrValue &attr_value = attr_.default_value();
  switch (attr_value.value_case()) {
    case AttrValue::VALUE_NOT_SET:
      return "";
    case AttrValue::kType:
      return DataType_Name(attr_value.type());
    case AttrValue::kS:
      return "\"" + attr_value.s() + "\"";
    case AttrValue::kI:
      return std::to_string(attr_value.i());
    case AttrValue::kF:
      return std::to_string(attr_value.f());
    case AttrValue::kB:
      return attr_value.b() ? "true" : "false";
    case AttrValue::kList:
      if (attr_.full_type() == "list(string)" &&
          attr_value.list().s_size() == 0) {
        return "{}";
      }
      LOG(WARNING) << "Unimplemented: default value of list-typed attribute.";
      return "/* UNIMPLEMENTED */";
    case AttrValue::kShape:
    case AttrValue::kTensor:
    case AttrValue::kFunc:
    case AttrValue::kPlaceholder:
      LOG(ERROR) << "Unexpected non-primitive attribute value.";
      return "/* ERROR */";
  }
}

string AttrView::VariableStrLen() const {
  return Call("strlen", {VariableName()});
}

string AttrView::InputArg(bool with_default_value) const {
  string default_value = DefaultValue();
  if (!with_default_value || default_value.empty()) {
    return absl::Substitute("$0 $1", VariableType(), attr_.name());
  }
  return absl::Substitute("$0 $1 = $2", VariableType(), attr_.name(),
                          default_value);
}

string AttrView::SetterMethod() const {
  if (!attr_.is_list()) {
    return absl::StrCat("SetAttr", toUpperCamel(attr_.full_type()));
  } else {
    return absl::StrCat("SetAttr", toUpperCamel(attr_.base_type()), "List");
  }
}

std::vector<string> AttrView::SetterArgs() const {
  if (attr_.full_type() != "string") {
    return {AttrNameString(), VariableName()};
  } else {
    return {AttrNameString(), VariableName(), VariableStrLen()};
  }
}

}  // namespace cpp
}  // namespace generator
}  // namespace tensorflow
