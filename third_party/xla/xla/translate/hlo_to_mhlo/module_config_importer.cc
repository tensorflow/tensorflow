/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/translate/hlo_to_mhlo/module_config_importer.h"

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "xla/service/hlo_module_config.h"

namespace xla {

namespace {

constexpr char kConfigNumPartitions[] = "mhlo.num_partitions";
constexpr char kConfigNumReplicas[] = "mhlo.num_replicas";

}  // namespace

void ImportHloModuleConfig(const HloModuleConfig& config,
                           mlir::ModuleOp module) {
  mlir::Builder builder(module.getContext());
  if (config.num_partitions() != 1) {
    module->setAttr(kConfigNumPartitions,
                    builder.getI32IntegerAttr(config.num_partitions()));
  }
  if (config.replica_count() != 1) {
    module->setAttr(kConfigNumReplicas,
                    builder.getI32IntegerAttr(config.replica_count()));
  }
}

}  // namespace xla
