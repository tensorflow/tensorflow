/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_STABLEHLO_EXPORT_H_
#define XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_STABLEHLO_EXPORT_H_

#include "llvm/Support/CommandLine.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"

namespace xla {
namespace sdy {

struct StablehloExportPipelineOptions
    : public mlir::PassPipelineOptions<StablehloExportPipelineOptions> {
  Option<bool> keepHloShardingConstraints{
      *this, "keep-hlo-sharding-constraints",
      llvm::cl::desc(
          "Whether to convert SDY sharding constraints to @Sharding custom "
          "calls - the HLO sharding constraint op. Else export "
          "them to MHLO copy ops. By default, export to MHLO copy ops."),
      llvm::cl::init(false)};
  Option<bool> dedupFunctionsFully{
      *this, "dedup-functions-fully",
      llvm::cl::desc(
          "Whether to deduplicate functions fully, regardless of the input and "
          "output shardings of functions, and it keeps one callee function for "
          "each caller function. The default is false, meaning it will "
          "deduplicate only if the input and output shardings are the same."),
      llvm::cl::init(false)};
  Option<bool> enableNativeNonFlatSupport{
      *this, "enable-native-non-flat-support",
      llvm::cl::desc("Whether to propagate shardings directly on a non-flat "
                     "graph without flattening it. The default is false, "
                     "meaning it will flatten the graph and then propagate."),
      llvm::cl::init(false)};
  Option<bool> addMissingShardingToControlFlow{
      *this, "add-missing-sharding-to-control-flow",
      llvm::cl::desc(
          "Whether to add a sharding to a control flow op without one."),
      llvm::cl::init(true)};
  Option<bool> enableHloShardingV3{
      *this, "enable-hlo-sharding-v3",
      llvm::cl::desc("Whether to enable HloShardingV3 which is the mesh and "
                     "axis based sharding representation."),
      llvm::cl::init(false)};
};

// Register the xla-sdy-stablehlo-export-pipeline.
void registerStablehloExportPipeline();

// Add the xla-sdy-stablehlo-export-pipeline in `pm`. The pipeline, including a
// sequence of passes, exports the Shardy dialect into an StableHLO module meant
// for the XLA compiler with HLO shardings.
void addStablehloExportPipeline(mlir::OpPassManager& pm,
                                const StablehloExportPipelineOptions& options =
                                    StablehloExportPipelineOptions());

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_STABLEHLO_EXPORT_H_
