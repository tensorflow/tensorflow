/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_pass.h"

#include "mlir/Analysis/Verifier.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"

namespace tensorflow {

Status MlirRoundtripPass::Run(const GraphOptimizationPassOptions& options) {
  // TODO(fengliuai): get debug info at runtime.
  GraphDebugInfo debug_info;
  mlir::MLIRContext context;
  NodeSpecs specs;
  ExporterConfigs confs;
  TF_ASSIGN_OR_RETURN(auto module,
                      ConvertGraphToMlir(**options.graph, debug_info,
                                         *options.flib_def, specs, &context));
  if (failed(mlir::verify(*module))) {
    // TODO(jpienaar): Remove, just simple verification that this works.
    module->dump();
    return errors::Internal("Verifier failed on MLIR import for the graph");
  }
  auto status =
      ConvertMlirToGraph(*module, confs, options.graph, options.flib_def);
  if (!status.ok()) module->dump();
  return status;
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 0,
                      MlirRoundtripPass);

}  // namespace tensorflow
