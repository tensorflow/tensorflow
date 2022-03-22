/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TAC_MODULE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TAC_MODULE_H_

#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/tac_importer_exporter.h"

namespace mlir {
namespace TFL {
namespace tac {

// Main class for using Target Aware Conversion (TAC).
// To run TAC:
// 1) users should create object form this class, with desired options
//    (TacModule::Options).
// 2) Use SetImporter/SetExporter to the desired importer
//    and exporter.
// 3) Call Run()
//
// The module fetches all TargetHardware backends registered in the binary
// and only create TargetHardware requested in Options.
//
// This class is not thread safe.
class TacModule {
 public:
  // TAC options. Contains knobs to configure TAC as needed.
  struct Options {
    // List of names for the requested Target hardware.
    std::vector<std::string> hardware_backends;
    // Debug mode.
    // This will output different alternative subgraphs in mlir format for debug
    // purpose.
    bool debug_mode = false;
    // Whether to enable inliner passes or not.
    bool enable_inliner = false;
    // Whether to legalize ops to TFLite ops before exporting.
    bool legalize_to_tflite_ops = false;
  };

  virtual ~TacModule() {}

  explicit TacModule(const Options& options) : options_(options) {}

  void SetImporter(std::unique_ptr<TacImporter> importer) {
    importer_ = std::move(importer);
  }

  void SetExporter(std::unique_ptr<TacExporter> exporter) {
    exporter_ = std::move(exporter);
  }

  // Returns pointer to the TargetHardware that is identified by 'hardware_name'
  // Returns NULL If no hardware with this name found.
  const tac::TargetHardware* GetTargetHardware(
      const std::string& hardware_name) const;

  // Runs the TAC workflow, configured as in the options provided during
  // construction.
  // SetImporter/SetExporter should be called prior to invoking `Run`.
  // Returns Status of the Run.
  virtual absl::Status Run();

  // Returns all available hardware backends registered in this module
  // instance.
  const std::vector<const tac::TargetHardware*>& GetAvailableHardwares() const {
    return const_backends_;
  }

  // Registers all dialects in 'registry' with the module.
  // This to allow clients to register extra dialects required.
  void RegisterExtraDialects(mlir::DialectRegistry& registry);

 protected:
  // Adds TAC passes to the 'pass_manager'.
  virtual void AddTACPass(mlir::OpPassManager* pass_manager,
                          llvm::ArrayRef<std::string> device_specs);

 private:
  // Runs all TAC passes on the provided module.
  absl::Status RunTacPasses(mlir::ModuleOp* module, bool debug_mode = false);

  // Create instances of all registered hardwares.
  std::vector<std::unique_ptr<tac::TargetHardware>> InstantiateBackends();

  std::unique_ptr<TacImporter> importer_;
  std::unique_ptr<TacExporter> exporter_;
  // Owned list of all target hardware backends.
  std::vector<std::unique_ptr<tac::TargetHardware>> backends_;
  // Holder for const pointers for the data in 'backends_'
  std::vector<const tac::TargetHardware*> const_backends_;
  // Extra dialects requested by the user.
  mlir::DialectRegistry registry_;

  const Options options_;
};

}  // namespace tac
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TAC_MODULE_H_
