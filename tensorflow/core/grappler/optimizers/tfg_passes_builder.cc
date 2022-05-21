/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/tfg_passes_builder.h"

#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/transforms/pass_registration.h"
#include "tensorflow/core/util/util.h"

namespace mlir {
namespace tfg {

// The default pipeline only does shape inference now.
void DefaultGrapplerPipeline(PassManager& manager) {
  // Turn certain shape attrs into types to give better knowledge for shape
  // inference.
  manager.addPass(CreateConsolidateAttributesPass());
  // Infer the shape of operation if possible. The TFG importer doesn't do shape
  // inference for almost all operations.
  // TODO(chiahungduan): Temporarily disable it because it hangs on certain
  // operations. We need to set a proper limit of iteration.
  // manager.addPass(CreateShapeInferencePass());
  // Contruct the shape attrs back from types.
  // TODO(chiahungduan): This will be the required pass before exporting, remove
  // this instance when the exporter has handled it.
  manager.addPass(CreatePrepareAttributesForExportPass());
}

// Run the consolidate attributes pass. Convert the whole module to region
// control-flow and run control-flow sinking. Convert the whole module back to
// functional control-flow and prepare the attributes for export.
void DefaultModuleGrapplerPipeline(PassManager& manager,
                                   const tensorflow::RewriterConfig& config) {
  // TODO(chiahungduan): This will be the default pass for TFG pipeline, remove
  // this when the default pipeline has added it.
  manager.addPass(CreateConsolidateAttributesPass());
  manager.addPass(CreateFunctionalToRegionPass());
  if (config.experimental_conditional_code_motion() !=
      tensorflow::RewriterConfig::OFF)
    manager.addNestedPass<GraphFuncOp>(CreateControlFlowSinkPass());
  manager.addPass(CreateRegionToFunctionalPass(/*force_control_capture=*/true));
  manager.addPass(CreateLiftLegacyCallPass());
  // Graph will be lifted as a function, don't mark it as private.
  manager.addPass(createSymbolPrivatizePass(
      {TFGraphDialect::getLiftedGraphFuncNameKey().str()}));
  manager.addPass(createSymbolDCEPass());
  // TODO(chiahungduan): This will be the required pass before exporting, remove
  // this instance when the exporter has handled it.
  manager.addPass(CreatePrepareAttributesForExportPass());
}

void RemapperPassBuilder(PassManager& manager) {
  manager.addPass(
      CreateRemapperPass(/*enable_mkl_patterns=*/tensorflow::IsMKLEnabled()));
}

}  // namespace tfg
}  // namespace mlir
