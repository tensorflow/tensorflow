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
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

Status GraphDecomposePass::Run(const ConfigProto& config_proto,
                               mlir::ModuleOp module) {
  TF_ASSIGN_OR_RETURN(ctx_, LoadDecompositionLib(module.getContext()));
  TF_RETURN_IF_ERROR(ctx_->Decompose(module));
  return ctx_->Destroy();
}

StatusOr<std::unique_ptr<TFRDecomposeContext>>
GraphDecomposePass::LoadDecompositionLib(mlir::MLIRContext* mlir_ctx) {
  Env* env = Env::Default();
  std::string tfr_lib_dir;
  TF_RETURN_IF_ERROR(ReadStringFromEnvVar(
      "TF_MLIR_TFR_LIB_DIR", "tensorflow/compiler/mlir/tfr/resources",
      &tfr_lib_dir));
  string composite_mlir_dir = io::JoinPath(env->GetRunfilesDir(), tfr_lib_dir);
  std::vector<string> files;
  TF_RETURN_IF_ERROR(env->GetChildren(composite_mlir_dir, &files));
  std::string tfr_raw_text;
  for (const auto& file : files) {
    string fullpath = io::JoinPath(composite_mlir_dir, file);
    if (env->MatchPath(fullpath, io::JoinPath(composite_mlir_dir, "*.mlir"))) {
      std::string text;
      TF_RETURN_IF_ERROR(ReadFileToString(env, fullpath, &text));
      tfr_raw_text.append(text);
    }
  }

  auto ctx = TFRDecomposeContext::Get(tfr_raw_text, mlir_ctx);
  if (!ctx) {
    return errors::Internal(absl::StrCat(
        "Failed to load the imported decomposition lib: ", tfr_raw_text));
  }
  return ctx;
}

namespace {
constexpr int kMlirGraphDecomposePassPriority = 1;

static mlir_pass_registration::MlirOptimizationPassRegistration
    register_mlir_graph_decompose_pass(kMlirGraphDecomposePassPriority,
                                       std::make_unique<GraphDecomposePass>());
}  // namespace

}  // namespace tensorflow
