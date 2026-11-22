/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/export/passes.h"

#include <string>
#include <utility>

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.pb.h"
#include "xla/python/ifrt/ir/version.h"

namespace xla {
namespace ifrt {

void createIfrtToVersionedPipeline(mlir::OpPassManager& pm,
                                   std::string ifrt_target_version,
                                   std::string vhlo_target_version,
                                   std::string sdy_target_version,
                                   IfrtIrProgramProto& ifrt_ir_program) {
  pm.addPass(createIfrtRemoveAttrsFromOtherDialectsPass());
  pm.addPass(createIfrtAtomProgramsToVhloPass(
      ifrt_ir_program.mutable_atom_programs(), std::move(vhlo_target_version),
      std::move(sdy_target_version)));
  pm.addPass(createIfrtLegalizeToVifrtPass());
  pm.addPass(createVifrtToVersionPass({std::move(ifrt_target_version)}));
  // Run symbol DCE to remove atom programs that have been legalized to VHLO.
  pm.addPass(mlir::createSymbolDCEPass());
}

void createIfrtFromVersionedPipeline(
    mlir::OpPassManager& pm, const IfrtIrProgramProto& ifrt_ir_program) {
  // Converts from given VIFRT version to the current VIFRT version.
  pm.addPass(
      createVifrtToVersionPass({Version::getCurrentVersion().toString()}));
  // Deserializes atom programs (including VHLO serialized version to VHLO
  // current conversion), and inserts them to the IFRT IR program ModuleOp.
  pm.addPass(
      createIfrtAtomProgramsFromVhloPass(ifrt_ir_program.atom_programs()));
  // Converts VIFRT current to IFRT.
  pm.addPass(createVifrtLegalizeToIfrtPass());
}

}  // namespace ifrt
}  // namespace xla
