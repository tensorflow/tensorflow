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

#include "tensorflow/core/framework/kernel_def_util.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/kernel_def.pb_text.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

namespace {
// Helper for KernelAttrsMatch().
bool InTypeList(DataType dt, const AttrValue& type_list) {
  for (int in_list : type_list.list().type()) {
    if (dt == in_list) return true;
  }
  return false;
}
}  // namespace

Status KernelAttrsMatch(const KernelDef& kernel_def, AttrSlice attrs,
                        bool* match) {
  *match = false;
  for (const auto& constraint : kernel_def.constraint()) {
    if (constraint.allowed_values().list().type_size() == 0) {
      return errors::Unimplemented(
          "KernelDef '", ProtoShortDebugString(kernel_def),
          " has constraint on attr '", constraint.name(),
          "' with unsupported type: ",
          SummarizeAttrValue(constraint.allowed_values()));
    }

    const AttrValue* found = attrs.Find(constraint.name());
    if (found) {
      if (found->type() != DT_INVALID) {
        if (!InTypeList(found->type(), constraint.allowed_values())) {
          return Status::OK();
        }
      } else {
        if (!AttrValueHasType(*found, "list(type)").ok()) {
          return errors::InvalidArgument(
              "KernelDef '", ProtoShortDebugString(kernel_def),
              "' has constraint on attr '", constraint.name(),
              "' that has value '", SummarizeAttrValue(*found),
              "' that does not have type 'type' or 'list(type)' in NodeDef "
              "'",
              attrs.SummarizeNode(), "'");
        }

        for (int t : found->list().type()) {
          if (!InTypeList(static_cast<DataType>(t),
                          constraint.allowed_values())) {
            return Status::OK();
          }
        }
      }
    } else {
      return errors::InvalidArgument(
          "OpKernel '", kernel_def.op(), "' has constraint on attr '",
          constraint.name(), "' not in NodeDef '", attrs.SummarizeNode(),
          "', KernelDef: '", ProtoShortDebugString(kernel_def), "'");
    }
  }
  *match = true;
  return Status::OK();
}

}  // namespace tensorflow
