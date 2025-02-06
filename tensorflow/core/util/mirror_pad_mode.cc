/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/mirror_pad_mode.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {

absl::Status GetNodeAttr(const NodeDef& node_def, absl::string_view attr_name,
                         MirrorPadMode* value) {
  string str_value;
  TF_RETURN_IF_ERROR(GetNodeAttr(node_def, attr_name, &str_value));
  if (str_value == "REFLECT") {
    *value = MirrorPadMode::REFLECT;
  } else if (str_value == "SYMMETRIC") {
    *value = MirrorPadMode::SYMMETRIC;
  } else {
    return errors::NotFound(str_value, " is not an allowed padding mode.");
  }
  return absl::OkStatus();
}

string GetMirrorPadModeAttrString() { return "mode: {'REFLECT', 'SYMMETRIC'}"; }

}  // end namespace tensorflow
