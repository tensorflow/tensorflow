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

#ifndef XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_EXPORT_SHARDINGS_H_
#define XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_EXPORT_SHARDINGS_H_

#include <functional>
#include <memory>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "xla/hlo/ir/hlo_sharding.h"

namespace xla {
namespace sdy {

// Convert the `sdySharding` into an `xla::HloSharding`.
HloSharding convertToHloSharding(
    mlir::sdy::TensorShardingAttr sdySharding,
    std::function<mlir::sdy::MeshAttr(mlir::sdy::TensorShardingAttr)>
        getMeshAttr,
    mlir::ArrayRef<mlir::StringAttr> manualAxes = {});

// Convert the `shardings` into a `StringAttr` representing `xla::HloSharding`
// for the given `op`.
mlir::StringAttr convertToHloShardingAttr(
    mlir::Operation* op,
    mlir::ArrayRef<mlir::sdy::TensorShardingAttr> shardings,
    std::function<mlir::sdy::MeshAttr(mlir::sdy::TensorShardingAttr)>
        getMeshAttr,
    std::function<mlir::StringAttr(const HloSharding&)> getStringAttr,
    mlir::ArrayRef<mlir::StringAttr> manualAxes = {});

// Creates a pass that converts the shardings from `kShardingAttr` to
// `kXlaShardingAttr` and removes mesh symbols. Fully or partially manual
// shardings are processed in `ShardMapExportPass`.
//
// If `addMissingShardingToControlFlow` is true, the pass will add a replicated
// hlo sharding to control flow ops (while, case, if) that have no sdy sharding.
std::unique_ptr<mlir::Pass> createExportStablehloShardingsPass(
    bool addMissingShardingToControlFlow = false);

// Register the xla-sdy-stablehlo-export-shardings pass.
void registerStablehloExportShardingsPass();

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_EXPORT_SHARDINGS_H_
