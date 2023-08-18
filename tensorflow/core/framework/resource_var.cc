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

#include "tensorflow/core/framework/resource_var.h"

#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/graph/graph_def_builder.h"

namespace tensorflow {

Status Var::AsGraphDef(GraphDefBuilder* builder, Node** out) const {
  // Set a shared_name so that the created resource can outlive the graph that
  // created it.
  Node* var = ops::SourceOp(
      "VarHandleOp",
      builder->opts()
          .WithAttr("dtype", tensor_.dtype())
          .WithAttr("shape", tensor_.shape())
          .WithAttr("shared_name", ResourceHandle::ANONYMOUS_NAME));
  Node* value = ops::SourceOp("Const", builder->opts()
                                           .WithAttr("dtype", tensor_.dtype())
                                           .WithAttr("value", tensor_));
  Node* assign =
      ops::BinaryOp("AssignVariableOp", var, value,
                    builder->opts().WithAttr("dtype", tensor_.dtype()));
  *out =
      ops::UnaryOp("Identity", var, builder->opts().WithControlInput(assign));
  return OkStatus();
}

std::string Var::MakeRefCountingHandleName(int64_t resource_id) const {
  // Use the resource id to ensure uniqueness.
  std::string handle_name = absl::StrFormat("%s%d", debug_name_, resource_id);
  return handle_name;
}
}  //  end namespace tensorflow
