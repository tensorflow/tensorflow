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

#include "tensorflow/core/util/tie_strategy.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

Status GetNodeAttr(const NodeDef& node_def, StringPiece attr_name,
                   TieStrategy* value) {
  string str_value;
  TF_RETURN_IF_ERROR(GetNodeAttr(node_def, attr_name, &str_value));
  if (str_value == "SAMPLE") {
    *value = TieStrategy::SAMPLE;
  } else if (str_value == "INCLUDE") {
    *value = TieStrategy::INCLUDE;
  } else if (str_value == "EXCLUDE") {
    *value = TieStrategy::EXCLUDE;
  } else {
    return errors::NotFound(str_value, " is not an allowed tie strategy.");
  }
  return Status::OK();
}

string GetTieStrategyAttrString() { return "handle_ties: {'SAMPLE', 'INCLUDE', 'EXCLUDE'} = 'SAMPLE' "; }

}  // end namespace tensorflow
