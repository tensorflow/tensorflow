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
#include "tensorflow/c/experimental/ops/gen/common/controller.h"

#include <vector>

#include "absl/log/check.h"
#include "absl/strings/substitute.h"
#include "tensorflow/c/experimental/ops/gen/common/path_config.h"
#include "tensorflow/c/experimental/ops/gen/common/source_code.h"
#include "tensorflow/c/experimental/ops/gen/model/op_spec.h"
#include "xla/tsl/platform/status.h"
#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"

namespace tensorflow {
namespace generator {

Controller::Controller(PathConfig path_config, Env* env)
    : env_(env), path_config_(path_config) {
  // Load the Op and API definitions
  InitializeOpApi();

  // Convert the Op and API definitions to the internal data model
  BuildModel();
}
Controller::~Controller() { delete api_def_map_; }

const void Controller::WriteFile(const string& file_path,
                                 const SourceCode& code) const {
  TF_CHECK_OK(WriteStringToFile(env_, file_path, code.Render())) << file_path;
}

const std::vector<OpSpec>& Controller::GetModelOps() const {
  return operators_;
}

void Controller::InitializeOpApi() {
  OpRegistry::Global()->Export(false, &op_list_);

  // Load matching API defs for each Op. Paths are visited in order, allowing
  // python/api_def_Xyz.pbtxt to override base/api_def_Xyz.pbtxt, for example.
  api_def_map_ = new ApiDefMap(op_list_);
  for (const auto& op : op_list_.op()) {
    for (const auto& dir : path_config_.api_dirs) {
      const string file_name = absl::Substitute("api_def_$0.pbtxt", op.name());
      const string file_path = io::JoinPath(dir, file_name);
      if (env_->FileExists(file_path).ok()) {
        TF_CHECK_OK(api_def_map_->LoadFile(env_, file_path)) << file_path;
      } else {
        // API defs are currently used for only optional pieces.
      }
    }
  }

  // Doc strings (summary, description) typically come from the API def.
  api_def_map_->UpdateDocs();
}

void Controller::BuildModel() {
  // Build the internal data model for the requested ops
  for (const auto& op_name : path_config_.op_names) {
    const OpDef* op_def = nullptr;
    TF_CHECK_OK(OpRegistry::Global()->LookUpOpDef(op_name, &op_def));
    CHECK(op_def != nullptr);  // Crash OK

    const ApiDef* api_def = api_def_map_->GetApiDef(op_name);
    CHECK(api_def != nullptr);  // Crash OK

    operators_.push_back(OpSpec::Create(*op_def, *api_def));
  }
}

}  // namespace generator
}  // namespace tensorflow
