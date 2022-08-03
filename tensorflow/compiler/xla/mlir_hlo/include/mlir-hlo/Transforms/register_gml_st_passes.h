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

#ifndef MLIR_HLO_TRANSFORMS_REGISTER_GML_ST_PASSES_H
#define MLIR_HLO_TRANSFORMS_REGISTER_GML_ST_PASSES_H

#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir-hlo/Dialect/gml_st/transforms/test_passes.h"
#include "mlir-hlo/Transforms/gml_st_pipeline.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace gml_st {
inline void registerAllGmlStPasses() {
  registerGmlStPasses();
  registerGmlStTestPasses();
  PassPipelineRegistration<GmlStPipelineOptions>(
      "gml-st-pipeline", "Pipeline to transform HLO to GmlSt and Linalg.",
      createGmlStPipeline);
}
}  // namespace gml_st
}  // namespace mlir

#endif  // MLIR_HLO_TRANSFORMS_REGISTER_GML_ST_PASSES_H
