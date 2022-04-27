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

#include "tensorflow/core/transforms/cf_sink/cf_sink.h"
#include "tensorflow/core/transforms/consolidate_attrs/pass.h"
#include "tensorflow/core/transforms/functional_to_region/pass.h"
#include "tensorflow/core/transforms/pass_registration.h"
#include "tensorflow/core/transforms/region_to_functional/pass.h"
#include "tensorflow/core/transforms/remapper/pass.h"
#include "tensorflow/core/util/util.h"

namespace mlir {
namespace tfg {

// The default pipeline is empty.
void DefaultGrapplerPipeline(PassManager& mgr) {}

// Run the consolidate attributes pass. Convert the whole module to region
// control-flow and run control-flow sinking. Convert the whole module back to
// functional control-flow and prepare the attributes for export.
void DefaultModuleGrapplerPipeline(PassManager& mgr) {
  mgr.addPass(CreateConsolidateAttributesPass());
  mgr.addPass(CreateFunctionalToRegionPass());
  // TODO(b/228618345): Enable control-flow sinking.
  // mgr.addNestedPass<GraphFuncOp>(CreateControlFlowSinkPass());
  mgr.addPass(CreateRegionToFunctionalPass(/*force_control_capture=*/true));
  mgr.addPass(CreatePrepareAttributesForExportPass());
}

void RemapperPassBuilder(PassManager& mgr) {
  mgr.addPass(
      CreateRemapperPass(/*enable_mkl_patterns=*/tensorflow::IsMKLEnabled()));
}

}  // namespace tfg
}  // namespace mlir
