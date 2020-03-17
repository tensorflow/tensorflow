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

#include "tensorflow/compiler/mlir/tensorflow/transforms/graph_optimization_pass.h"

namespace tensorflow {

Status MlirGraphOptimizationPass::Run(const ConfigProto& config_proto,
                                      mlir::ModuleOp module) {
  if (!config_proto.experimental().enable_mlir_graph_optimization()) {
    VLOG(1) << "Skipping MLIR Graph Optimization Pass"
            << ", session flag not enabled";
    return Status::OK();
  }

  // TODO(ezhulenev): Add something here.

  return Status::OK();
}

}  // namespace tensorflow
