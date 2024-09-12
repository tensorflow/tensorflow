/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/test_util.h"

#include "xla/status_macros.h"
#include "tensorflow/core/framework/node_def.pb.h"

namespace tensorflow {

Status InstantiateFunctionForTest(const string& name,
                                  const FunctionLibraryDefinition& library,
                                  InstantiationResultForTest* result) {
  const FunctionDef* fdef = library.Find(name);
  TF_RET_CHECK(fdef != nullptr);

  auto get_func_sig = [&library](const string& op, const OpDef** sig) {
    return library.LookUpOpDef(op, sig);
  };
  InstantiationResult inst;
  TF_RETURN_IF_ERROR(
      InstantiateFunction(*fdef, AttrSlice(), get_func_sig, &inst));
  result->arg_types = inst.arg_types;
  result->ret_types = inst.ret_types;
  for (NodeDef& n : inst.nodes) {
    *result->gdef.add_node() = std::move(n);
  }
  return absl::OkStatus();
}

}  // namespace tensorflow
