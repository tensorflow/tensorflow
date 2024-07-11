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

#include "xla/service/spmd/shardy/round_trip_common/pipeline_passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/service/spmd/shardy/round_trip_common/convert_sharding_custom_calls.h"
#include "xla/service/spmd/shardy/round_trip_common/identity_to_pass_through_while_args.h"
#include "xla/service/spmd/shardy/round_trip_common/import_constants.h"
#include "xla/service/spmd/shardy/round_trip_common/shard_map_import.h"

namespace xla {
namespace sdy {

using ::mlir::func::FuncOp;

void addCommonPreImportPasses(mlir::OpPassManager& pm) {
  pm.addPass(mlir::createSymbolDCEPass());
  // TODO(b/333505182): remove when partitioning is done in SDY.
  // We call prepare-for-export pass before SDY propagation, so that all IR
  // changes happen before shardings are added to operations, to ensure the
  // correct shardings are added and that they are not lost by this pass.
  pm.addNestedPass<FuncOp>(mlir::mhlo::createPrepareForExportPass());
  // The prepare-for-export pass lifts `mhlo::WhileOp` free variables, and added
  // them as additional operands of the op whose corresponding block arguments
  // are directly returned by the body of the op (pass-through). To prevent
  // canonicalization from undoing this, we add identity ops.
  pm.addNestedPass<FuncOp>(createAddIdentityToPassThroughWhileArgsPass());

  // We import `mhlo.constant` ops to `sdy.constant` ops so that constants
  // aren't folded in greedy pattern rewriters, which would lift them outside of
  // nested regions (this undoes `WhileLoopConstantSinking` HLO pass).
  // Therefore, this pass needs to be applied after any mhlo pass that expects
  // `mhlo.constant`, and before any pass that has a greedy pattern rewriter.
  pm.addNestedPass<FuncOp>(createImportConstantsPass());

  pm.addNestedPass<FuncOp>(mlir::mhlo::createFlattenTuplePass());
  // We need to canonicalize redundant mhlo::GetTupleElementOp and
  // mhlo::GetTupleOp.
  pm.addPass(mlir::createCanonicalizerPass());
}

void addCommonPostImportPasses(mlir::OpPassManager& pm) {
  pm.addPass(createShardMapImportPass());
  pm.addPass(createConvertShardingCustomCallsPass());
}

}  // namespace sdy
}  // namespace xla
