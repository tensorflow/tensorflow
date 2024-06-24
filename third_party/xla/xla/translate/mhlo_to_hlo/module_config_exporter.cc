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

#include "xla/translate/mhlo_to_hlo/module_config_exporter.h"

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "xla/service/hlo_module_config.h"

namespace mlir {
namespace mhlo {
namespace {
constexpr char kConfigNumPartitions[] = "mhlo.num_partitions";
constexpr char kConfigNumReplicas[] = "mhlo.num_replicas";
}  // namespace

void ExportHloModuleConfig(xla::HloModuleConfig& config,
                           mlir::ModuleOp module) {
  if (auto num_partitions =
          module->getAttrOfType<mlir::IntegerAttr>(kConfigNumPartitions)) {
    config.set_num_partitions(num_partitions.getInt());
  }
  if (auto num_replicas =
          module->getAttrOfType<mlir::IntegerAttr>(kConfigNumReplicas)) {
    config.set_replica_count(num_replicas.getInt());
  }
}

}  // namespace mhlo
}  // namespace mlir
