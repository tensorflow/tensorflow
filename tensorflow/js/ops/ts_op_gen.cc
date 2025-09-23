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

#include "tensorflow/js/ops/ts_op_gen.h"

#include <memory>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

static bool IsListAttr(const OpDef_ArgDef& arg) {
  return !arg.type_list_attr().empty() || !arg.number_attr().empty();
}

// Struct to hold a combo OpDef and ArgDef for a given Op argument:
struct ArgDefs {
  ArgDefs(const OpDef::ArgDef& op_def_arg, const ApiDef::Arg& api_def_arg)
      : op_def_arg(op_def_arg), api_def_arg(api_def_arg) {}

  const OpDef::ArgDef& op_def_arg;
  const ApiDef::Arg& api_def_arg;
};

// Struct to hold a combo OpDef::AttrDef and ApiDef::Attr for an Op.
struct OpAttrs {
  OpAttrs(const OpDef::AttrDef& op_def_attr, const ApiDef::Attr& api_def_attr)
      : op_def_attr(op_def_attr), api_def_attr(api_def_attr) {}

  const OpDef::AttrDef& op_def_attr;
  const ApiDef::Attr& api_def_attr;
};

// Helper class to generate TypeScript code for a given OpDef:
class GenTypeScriptOp {
 public:
  GenTypeScriptOp(const OpDef& op_def, const ApiDef& api_def);
  ~GenTypeScriptOp();

  // Returns the generated code as a string:
  string Code();

 private:
  void ProcessArgs();
  void ProcessAttrs();
  void AddAttrForArg(const string& attr, int arg_index);
  string InputForAttr(const OpDef::AttrDef& op_def_attr);

  void AddMethodSignature();
  void AddOpAttrs();
  void AddMethodReturnAndClose();

  const OpDef& op_def_;
  const ApiDef& api_def_;

  // Placeholder string for all generated code:
  string result_;

  // Holds in-order vector of Op inputs:
  std::vector<ArgDefs> input_op_args_;

  // Holds in-order vector of Op attributes:
  std::vector<OpAttrs> op_attrs_;

  // Stores attributes-to-arguments by name:
  typedef std::unordered_map<string, std::vector<int>> AttrArgIdxMap;
  AttrArgIdxMap attr_arg_idx_map_;

  // Holds number of outputs:
  int num_outputs_;
};

GenTypeScriptOp::GenTypeScriptOp(const OpDef& op_def, const ApiDef& api_def)
    : op_def_(op_def), api_def_(api_def), num_outputs_(0) {}

GenTypeScriptOp::~GenTypeScriptOp() = default;

string GenTypeScriptOp::Code() {
  ProcessArgs();
  ProcessAttrs();

  // Generate exported function for Op:
  AddMethodSignature();
  AddOpAttrs();
  AddMethodReturnAndClose();

  absl::StrAppend(&result_, "\n");
  return result_;
}

void GenTypeScriptOp::ProcessArgs() {
  for (int i = 0; i < api_def_.arg_order_size(); i++) {
    auto op_def_arg = FindInputArg(api_def_.arg_order(i), op_def_);
    if (op_def_arg == nullptr) {
      LOG(WARNING) << "Could not find OpDef::ArgDef for "
                   << api_def_.arg_order(i);
      continue;
    }
    auto api_def_arg = FindInputArg(api_def_.arg_order(i), api_def_);
    if (api_def_arg == nullptr) {
      LOG(WARNING) << "Could not find ApiDef::Arg for "
                   << api_def_.arg_order(i);
      continue;
    }

    // Map attr names to arg indexes:
    if (!op_def_arg->type_attr().empty()) {
      AddAttrForArg(op_def_arg->type_attr(), i);
    } else if (!op_def_arg->type_list_attr().empty()) {
      AddAttrForArg(op_def_arg->type_list_attr(), i);
    }
    if (!op_def_arg->number_attr().empty()) {
      AddAttrForArg(op_def_arg->number_attr(), i);
    }

    input_op_args_.push_back(ArgDefs(*op_def_arg, *api_def_arg));
  }

  num_outputs_ = api_def_.out_arg_size();
}

void GenTypeScriptOp::ProcessAttrs() {
  for (int i = 0; i < op_def_.attr_size(); i++) {
    op_attrs_.push_back(OpAttrs(op_def_.attr(i), api_def_.attr(i)));
  }
}

void GenTypeScriptOp::AddAttrForArg(const string& attr, int arg_index) {
  // Keep track of attributes-to-arguments by name. These will be used for
  // construction Op attributes that require information about the inputs.
  auto iter = attr_arg_idx_map_.find(attr);
  if (iter == attr_arg_idx_map_.end()) {
    attr_arg_idx_map_.insert(AttrArgIdxMap::value_type(attr, {arg_index}));
  } else {
    iter->second.push_back(arg_index);
  }
}

string GenTypeScriptOp::InputForAttr(const OpDef::AttrDef& op_def_attr) {
  string inputs;
  auto arg_list = attr_arg_idx_map_.find(op_def_attr.name());
  if (arg_list != attr_arg_idx_map_.end()) {
    for (auto iter = arg_list->second.begin(); iter != arg_list->second.end();
         ++iter) {
      absl::StrAppend(&inputs, input_op_args_[*iter].op_def_arg.name());
    }
  }
  return inputs;
}

void GenTypeScriptOp::AddMethodSignature() {
  absl::StrAppend(&result_, "export function ", api_def_.endpoint(0).name(),
                  "(");

  bool is_first = true;
  for (auto& in_arg : input_op_args_) {
    if (is_first) {
      is_first = false;
    } else {
      absl::StrAppend(&result_, ", ");
    }

    auto op_def_arg = in_arg.op_def_arg;

    absl::StrAppend(&result_, op_def_arg.name(), ": ");
    if (IsListAttr(op_def_arg)) {
      absl::StrAppend(&result_, "tfc.Tensor[]");
    } else {
      absl::StrAppend(&result_, "tfc.Tensor");
    }
  }

  if (num_outputs_ == 1) {
    absl::StrAppend(&result_, "): tfc.Tensor {\n");
  } else {
    absl::StrAppend(&result_, "): tfc.Tensor[] {\n");
  }
}

void GenTypeScriptOp::AddOpAttrs() {
  absl::StrAppend(&result_, "  const opAttrs = [\n");

  bool is_first = true;
  for (auto& attr : op_attrs_) {
    if (is_first) {
      is_first = false;
    } else {
      absl::StrAppend(&result_, ",\n");
    }

    // Append 4 spaces to start:
    absl::StrAppend(&result_, "    ");

    if (attr.op_def_attr.type() == "type") {
      // Type OpAttributes can be generated from a helper function:
      strings::StrAppend(&result_, "createTensorsTypeOpAttr('",
                         attr.op_def_attr.name(), "', ",
                         InputForAttr(attr.op_def_attr), ")");
    } else if (attr.op_def_attr.type() == "int") {
      absl::StrAppend(&result_, "{name: '", attr.op_def_attr.name(), "', ");
      absl::StrAppend(&result_, "type: nodeBackend().binding.TF_ATTR_INT, ");
      absl::StrAppend(&result_, "value: ", InputForAttr(attr.op_def_attr),
                      ".length}");
    }
  }
  absl::StrAppend(&result_, "\n  ];\n");
}

void GenTypeScriptOp::AddMethodReturnAndClose() {
  absl::StrAppend(&result_, "  return null;\n}\n");
}

void WriteTSOp(const OpDef& op_def, const ApiDef& api_def, WritableFile* ts) {
  GenTypeScriptOp ts_op(op_def, api_def);
  TF_CHECK_OK(ts->Append(GenTypeScriptOp(op_def, api_def).Code()));
}

void StartFile(WritableFile* ts_file) {
  const string header =
      R"header(/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// This file is MACHINE GENERATED! Do not edit

import * as tfc from '@tensorflow/tfjs-core';
import {createTensorsTypeOpAttr, nodeBackend} from './op_utils';

)header";

  TF_CHECK_OK(ts_file->Append(header));
}

}  // namespace

void WriteTSOps(const OpList& ops, const ApiDefMap& api_def_map,
                const string& ts_filename) {
  Env* env = Env::Default();

  std::unique_ptr<WritableFile> ts_file = nullptr;
  TF_CHECK_OK(env->NewWritableFile(ts_filename, &ts_file));

  StartFile(ts_file.get());

  for (const auto& op_def : ops.op()) {
    // Skip deprecated ops
    if (op_def.has_deprecation() &&
        op_def.deprecation().version() <= TF_GRAPH_DEF_VERSION) {
      continue;
    }

    const auto* api_def = api_def_map.GetApiDef(op_def.name());
    if (api_def->visibility() == ApiDef::VISIBLE) {
      WriteTSOp(op_def, *api_def, ts_file.get());
    }
  }

  TF_CHECK_OK(ts_file->Close());
}

}  // namespace tensorflow
