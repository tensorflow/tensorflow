/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/tfr/integration/graph_decompose_pass.h"

#include "tensorflow/compiler/mlir/tfr/integration/tfr_decompose_ctx.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

bool GraphDecomposePass::IsEnabled(const ConfigProto& config_proto) const {
  const char* tfr_lib_env_val = getenv(std::string(kTFRLibEnv).c_str());
  return tfr_lib_env_val != nullptr;
}

Status GraphDecomposePass::Run(const ConfigProto& config_proto,
                               mlir::ModuleOp module) {
  if (!IsEnabled(config_proto)) {
    VLOG(1) << "Skipping Graph Decomposition Pass, decompositin library was "
               "not found";
    return Status::OK();
  }
  VLOG(1) << "Run Graph Decomposition Passes";
  TF_ASSIGN_OR_RETURN(ctx_, TFRDecomposeContext::Get(module.getContext()));
  TF_RETURN_IF_ERROR(ctx_->Decompose(module));
  return ctx_->Destroy();
}

namespace {
constexpr int kMlirGraphDecomposePassPriority = -1;

static mlir_pass_registration::MlirOptimizationPassRegistration
    register_mlir_graph_decompose_pass(kMlirGraphDecomposePassPriority,
                                       std::make_unique<GraphDecomposePass>());
}  // namespace

}  // namespace tensorflow
