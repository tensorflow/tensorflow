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

#include "tensorflow/core/util/padding.h"

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

Status GetNodeAttr(const NodeDef& node_def, StringPiece attr_name,
                   Padding* value) {
  string str_value;
  TF_RETURN_IF_ERROR(GetNodeAttr(node_def, attr_name, &str_value));
  if (str_value == "SAME") {
    *value = SAME;
  } else if (str_value == "VALID") {
    *value = VALID;
  } else {
    return errors::NotFound(str_value, " is not an allowed padding type");
  }
  return Status::OK();
}

string GetPaddingAttrString() { return "padding: {'SAME', 'VALID'}"; }

}  // end namespace tensorflow
