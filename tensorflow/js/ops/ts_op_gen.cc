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
#include <unordered_map>

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

// Helper class to generate TypeScript code for a given OpDef:
class GenTypeScriptOp {
 public:
  GenTypeScriptOp(const OpDef& op_def, const ApiDef& api_def);
  ~GenTypeScriptOp();

  // Returns the generated code as a string:
  string Code();

 private:
  void ProcessArgs();

  void AddMethodSignature();
  void AddMethodReturnAndClose();

  const OpDef& op_def_;
  const ApiDef& api_def_;

  // Placeholder string for all generated code:
  string result_;

  // Holds in-order vector of Op inputs:
  std::vector<ArgDefs> input_op_args_;

  // Holds number of outputs:
  int num_outputs_;
};

GenTypeScriptOp::GenTypeScriptOp(const OpDef& op_def, const ApiDef& api_def)
    : op_def_(op_def), api_def_(api_def), num_outputs_(0) {}

GenTypeScriptOp::~GenTypeScriptOp() {}

string GenTypeScriptOp::Code() {
  ProcessArgs();

  // Generate exported function for Op:
  AddMethodSignature();
  AddMethodReturnAndClose();

  strings::StrAppend(&result_, "\n");
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
    input_op_args_.push_back(ArgDefs(*op_def_arg, *api_def_arg));
  }

  num_outputs_ = api_def_.out_arg_size();
}

void GenTypeScriptOp::AddMethodSignature() {
  strings::StrAppend(&result_, "export function ", api_def_.endpoint(0).name(),
                     "(");

  bool is_first = true;
  for (auto& in_arg : input_op_args_) {
    if (is_first) {
      is_first = false;
    } else {
      strings::StrAppend(&result_, ", ");
    }

    auto op_def_arg = in_arg.op_def_arg;

    strings::StrAppend(&result_, op_def_arg.name(), ": ");
    if (IsListAttr(op_def_arg)) {
      strings::StrAppend(&result_, "tfc.Tensor[]");
    } else {
      strings::StrAppend(&result_, "tfc.Tensor");
    }
  }

  if (num_outputs_ == 1) {
    strings::StrAppend(&result_, "): tfc.Tensor {\n");
  } else {
    strings::StrAppend(&result_, "): tfc.Tensor[] {\n");
  }
}

void GenTypeScriptOp::AddMethodReturnAndClose() {
  strings::StrAppend(&result_, "  return null;\n}\n");
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
import {createTypeOpAttr, getTFDTypeForInputs, nodeBackend} from './op_utils';

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
