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

include "mlir/Pass/PassBase.td"

def PrepareForExportPass : FunctionPass<"xla-prepare-for-export"> {
  let summary = "Prepare for XLA export";

  let description = [{
    This pass transforms functions in preparation for exporting to XLA. This

    * converts splat constants to constants and broadcasts to reduce size of
      and speedup the creation of the generated proto during export.

    Note: The result of this pass need not be a module in canonical form and
    canonicalization may undo transformations.
  }];

  let constructor = "::mlir::mhlo::CreatePrepareForExport()";
}

def LegalizeTFPass : FunctionPass<"xla-legalize-tf-with-tf2xla"> {
  let summary = "Legalize from TensorFlow to the HLO dialect using tf2xla kernels";
  let dependentDialects = ["mhlo::MhloDialect"];
  let constructor = "::mlir::mhlo::createLegalizeTfWithTf2XlaPass()";
  let options = [
  // TODO(hinsu): Support finer grained device type assignment instead of a
  // global device type for all TensorFlow ops.
  Option<"device_type_", "device-type", "std::string", "",
         "XLA device type for execution of TensorFlow ops.">,
  Option<"prefer_tf2xla_", "prefer-tf2xla", "bool", "",
         "Enable legalization when it is not in the list of "
                     "MLIR-legalized ops.">,
  Option<"legalize_test_only_ops_", "legalize-test-only-ops", "bool", "",
         "Enable tf2xla legalizations for some ops that are "
                     "enabled only for testing.">
  ];
}

def LegalizeTFCommunicationPass : Pass<"xla-legalize-tf-communication", "ModuleOp"> {
  let summary = "Legalize TF/XLA communication ops (TensorFlow dialect) to the HLO "
           "dialect";
  let dependentDialects = ["mhlo::MhloDialect"];
  let constructor = "::mlir::mhlo::CreateLegalizeTFCommunicationPass()";
}
