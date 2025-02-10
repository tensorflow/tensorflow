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

#include <memory>
#include <string>

#include "absl/strings/str_cat.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/Version.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/ir/version.h"
#include "xla/python/ifrt/ir/vifrt_dialect.h"
#include "xla/python/ifrt/serdes.h"

namespace xla {
namespace ifrt {

// NOLINTNEXTLINE
llvm::cl::opt<bool> strip_debug_info_option(
    "strip_debuginfo", llvm::cl::desc("Strip debug info from all operations"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<std::string> ifrt_version_option(
    "ifrt_version", llvm::cl::desc("Target version for IFRT IR serialization"),
    llvm::cl::init("current"));

// NOLINTNEXTLINE
llvm::cl::opt<std::string> atom_program_version_option(
    "atom_program_version",
    llvm::cl::desc("Target version for atom program serialization"),
    llvm::cl::init("current"));

mlir::TranslateFromMLIRRegistration serializeRegistration(
    "serialize", "Serialize IFRT IR program into a VIFRT artifact",
    [](mlir::ModuleOp module, llvm::raw_ostream &os) -> mlir::LogicalResult {
      std::string ifrt_version = ifrt_version_option.getValue();
      if (ifrt_version == "current") {
        ifrt_version = Version::getCurrentVersion().toString();
      }
      std::string atom_program_version = atom_program_version_option.getValue();
      if (atom_program_version == "current") {
        atom_program_version =
            ::mlir::vhlo::Version::getCurrentVersion().toString();
      }
      if (strip_debug_info_option) {
        mlir::PassManager pm(module->getContext());
        pm.addPass(mlir::createStripDebugInfoPass());
        if (mlir::failed(pm.run(module)))
          return module.emitError("failed to strip debuginfo");
      }

      auto program = std::make_unique<IfrtIRProgram>(module);
      auto serialized_or =
          Serialize(*program, std::make_unique<SerializeIfrtIRProgramOptions>(
                                  ifrt_version, atom_program_version));
      if (serialized_or.ok()) {
        os << serialized_or->SerializeAsString();
        return mlir::success();
      } else {
        module.emitError(absl::StrCat("failed to serialize: ",
                                      serialized_or.status().message()));
        return mlir::failure();
      }
    },
    [](mlir::DialectRegistry &registry) {
      mlir::registerAllDialects(registry);
      mlir::stablehlo::registerAllDialects(registry);
      registry.insert<xla::ifrt::IfrtDialect>();
      registry.insert<xla::ifrt::VifrtDialect>();
    });

mlir::TranslateToMLIRRegistration deserializeRegistration(
    "deserialize", "Deserialize VIFRT into an IFRT IR program",
    [](llvm::StringRef input,
       mlir::MLIRContext *context) -> mlir::OwningOpRef<mlir::ModuleOp> {
      Serialized serialized_proto;
      if (!serialized_proto.ParseFromString(std::string(input))) {
        return nullptr;
      }
      auto deserialized_program_or = Deserialize<IfrtIRProgram>(
          serialized_proto,
          std::make_unique<DeserializeIfrtIRProgramOptions>(context));
      if (deserialized_program_or.ok()) {
        return mlir::OwningOpRef<mlir::ModuleOp>(
            deserialized_program_or.value()->mlir_module);
      } else {
        llvm::dbgs() << "failed to deserialize: "
                     << deserialized_program_or.status().message() << "\n";
        return nullptr;
      }
    },
    [](mlir::DialectRegistry &registry) {
      mlir::registerAllDialects(registry);
      mlir::stablehlo::registerAllDialects(registry);
      registry.insert<xla::ifrt::IfrtDialect>();
      registry.insert<xla::ifrt::VifrtDialect>();
    });

}  // namespace ifrt
}  // namespace xla

int main(int argc, char **argv) {
  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "IFRT IR translate driver\n"));
}
